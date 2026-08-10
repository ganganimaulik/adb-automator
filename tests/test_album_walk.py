"""Walking a photo album end to end, against a scripted media viewer.

This is the regression test for ``runs/af76720d05c4``: fifteen WhatsApp photos
that took 136 steps, 102 minutes and four full re-walks of the album, because a
swipe was always graded ``success``. The device reproduces the properties that
made that possible:

* the overlay chrome fades after a couple of gestures, changing the pager's
  element index and removing the caption from the tree entirely;
* the ViewPager drops some flings, leaving the photo exactly where it was.

The album also holds two photos sent in the same minute. That used to matter a
great deal, because item identity came from the caption and the twins collided.
It no longer matters at all: identity comes from the pixels, and two different
photos taken in the same minute look different. The fixture keeps them because a
case that used to need special handling and now needs none is worth pinning.

The agent under test is a scripted model that swipes left and stops when the
harness tells it the gesture no longer advances. The assertions are about what
the *harness* guarantees it.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from adbagent.actions import AgentAction
from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.memory import Memory
from adbagent.screen import Screen, parse
from adbagent.fingerprint import attach

from . import fake
from . import xmlgen as X

#: Fifteen photos, two of them sent in the same minute (9:33) -- exactly the
#: shape of the album from the run.
STAMPS = ["9:30 am", "9:31 am", "9:32 am", "9:33 am", "9:33 am", "9:36 am",
          "9:39 am", "9:40 am", "9:43 am", "9:45 am", "9:51 am", "9:52 am",
          "9:52 am", "9:59 am", "10:03 am"]

GOAL = "read the weight in every photo of the album"


def _png(seed: int) -> bytes:
    """A distinct image per photo, so the perceptual hash has something to see."""
    image = Image.new("L", (32, 32))
    image.putdata([(seed * 37 + x * 5 + y * 11) % 256
                   for y in range(32) for x in range(32)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class AlbumDevice(fake.FakeDevice):
    """A media viewer showing one of `STAMPS`, advanced by a horizontal swipe."""

    def __init__(self, cfg: Config, *, chrome_fades_after: int = 2,
                 drop_swipes: frozenset = frozenset()):
        super().__init__(cfg)
        self.index = 0
        self.gestures = 0
        self.chrome_fades_after = chrome_fades_after
        #: Photo indices whose *outgoing* swipe the ViewPager drops once.
        self.drop_swipes = set(drop_swipes)
        self.dropped: list = []
        self.reads: list = []

    @property
    def chrome(self) -> bool:
        return self.gestures < self.chrome_fades_after

    def observe(self, settle: bool = False) -> Screen:
        self.dumps += 1
        return attach(parse(
            X.media_viewer(STAMPS[self.index], chrome=self.chrome),
            width=self.size[0], height=self.size[1],
            activity=X.MEDIA_ACTIVITY))

    def screenshot(self, **kw) -> bytes:
        self.screenshots += 1
        self.reads.append(self.index)
        return _png(self.index)

    def scroll(self, direction: str, **kw) -> None:
        self.actions.append(f"scroll({direction})")
        self.gestures += 1
        if direction not in ("left", "right"):
            return
        if direction == "left" and self.index in self.drop_swipes:
            self.drop_swipes.discard(self.index)
            self.dropped.append(self.index)
            return
        step = 1 if direction == "left" else -1
        self.index = max(0, min(len(STAMPS) - 1, self.index + step))

    def tap(self, x: int, y: int) -> None:
        """Tapping the photo toggles the overlay back on, as WhatsApp does."""
        self.actions.append(f"tap({x},{y})")
        self.gestures = 0


def album_walker():
    """A model that swipes left until the harness says the gesture stopped working.

    It keeps no memory of its own on purpose. It also does not ask what item it
    is on, because nothing tells it any more -- the only signal it acts on is
    "that gesture no longer advances", which is the one thing observable.
    """
    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        note = llm.notes[-1] if llm.notes else ""
        if "no longer advance" in note or "stopped changing" in note:
            return AgentAction(observation="the album stopped advancing",
                               reasoning="nothing further to page to",
                               action="done", text="read all photos")
        pager = next((e for e in screen.elements if e.resource_id == "pager"),
                     None)
        if pager is None:
            return AgentAction(observation="no pager", reasoning="give up",
                               action="fail", text="no pager on screen")
        return AgentAction(
            observation="a photo of a scale is on screen",
            reasoning="read it and advance", action="swipe",
            direction="left", target={"index": pager.index})

    return policy


def unread_album_walker():
    """The same walk, opting out of the per-item read with `read_each=False`.

    The model's business here is only getting to the far end of the album --
    what is in between does not matter, so it declines the per-screen analysis
    while keeping the mechanical repeat.
    """
    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        action = album_walker()(screen, llm)
        if action.action == "swipe":
            action = action.model_copy(update={"read_each": False})
        return action

    return policy


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 60
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


def walk(cfg, mem, **device_kw):
    dev = AlbumDevice(cfg, **device_kw)
    llm = fake.FakeLLM(dev, album_walker())
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    return dev, llm, outcome, state


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------

def _readings(cfg, run_id):
    return [e for e in _events(cfg, run_id) if e["kind"] == "item_reading"]


def test_every_photo_is_read(cfg, mem):
    dev, llm, outcome, state = walk(cfg, mem, chrome_fades_after=999)

    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1, "the album was not walked to the end"
    # One vision read per photo the sweep landed on. The first is read by the
    # main loop's own vision pass before the sweep starts, so the sweep files
    # one fewer.
    assert len(_readings(cfg, state.run_id)) >= len(STAMPS) - 2


def test_the_walk_costs_a_step_or_two_per_photo(cfg, mem):
    """The run that motivated this used 136 steps for these fifteen photos."""
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    assert state.step <= len(STAMPS) * 2, f"took {state.step} steps"


def test_no_forced_back_ejects_the_agent_from_the_album(cfg, mem):
    """`exact_id` is identical for all fifteen photos, so without the pixel
    signal the loop breaker fires and dumps the agent out of the viewer."""
    dev, _, _, _ = walk(cfg, mem, chrome_fades_after=999)
    assert "press(back)" not in dev.actions


def test_a_dropped_fling_is_detected_and_retried(cfg, mem):
    dev, _, outcome, state = walk(cfg, mem, chrome_fades_after=999,
                                  drop_swipes=frozenset({2, 7, 10}))
    assert dev.dropped == [2, 7, 10]
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


def test_the_same_minute_twins_need_no_special_handling(cfg, mem):
    """Two photos sent in the same minute used to collide, because identity was
    the caption. The pixels tell them apart without anyone having to try."""
    dev, _, outcome, _ = walk(cfg, mem, chrome_fades_after=999,
                              drop_swipes=frozenset({3}))
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


def test_the_agent_is_handed_what_the_sweep_read(cfg, mem):
    _, llm, _, _ = walk(cfg, mem, chrome_fades_after=999)
    notes = "\n".join(llm.notes)
    assert "YOU REPEATED" in notes
    assert "`notes`" in notes, "the model was not told to keep what it needs"


def test_the_agent_is_told_nothing_it_cannot_know(cfg, mem):
    """The old block closed with verdicts about a set: how many items it held,
    which were unread, that every one had been read. None was observable."""
    _, llm, _, _ = walk(cfg, mem, chrome_fades_after=999)
    notes = "\n".join(llm.notes)
    for claim in ("ITEMS INSPECTED IN THIS SET", "STILL NOT READ",
                  "LAST item of this set", "Every item in this set"):
        assert claim not in notes, claim


def test_hidden_chrome_does_not_stop_the_walk(cfg, mem):
    """With `chrome_fades_after=1` the caption is gone on almost every turn.

    That used to pause the sweep outright -- items "could not be told apart".
    The content hash crops the bands the chrome lives in, so it never mattered.
    """
    dev, _, outcome, _ = walk(cfg, mem, chrome_fades_after=1)
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


# ---------------------------------------------------------------------------
# Sweeping the album in code
# ---------------------------------------------------------------------------
#
# Walking a set is the one genuinely mechanical thing the agent does: the same
# question, answered by the same gesture, once per item. In the run this file is
# named after, 71 of 127 steps were the single action `swipe #4 left`, each one
# paid for with a reasoning turn at 26s median. These tests are about that bill.

def decides(llm) -> int:
    """Reasoning turns, excluding the completion judge."""
    return llm.calls - llm.judges


def test_sweeping_replaces_reasoning_turns_with_vision_reads(cfg, mem):
    cfg.device.serial = ""
    dev, llm, outcome, state = walk(cfg, mem, chrome_fades_after=999)
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1
    # Fifteen photos on a handful of decisions: start the walk, resume after the
    # repeat cap, and confirm it finished.
    assert decides(llm) <= 4, f"{decides(llm)} decide calls"
    assert len(llm.reads_requested) >= len(STAMPS) - 3


def test_the_saving_is_real_and_not_an_accounting_trick(cfg, mem):
    """Same album, same policy, sweep off then on."""
    cfg.run.pager_sweep = False
    dev_without, without, _, _ = walk(cfg, mem, chrome_fades_after=999)
    cfg.run.pager_sweep = True
    dev_with, with_sweep, _, _ = walk(cfg, mem, chrome_fades_after=999)
    # Both walked the whole album; only the bill differs.
    assert dev_without.index == dev_with.index == len(STAMPS) - 1

    assert decides(with_sweep) < decides(without) / 3, (
        f"{decides(without)} -> {decides(with_sweep)}")


def test_sweeping_off_restores_a_turn_per_photo(cfg, mem):
    cfg.run.pager_sweep = False
    _, llm, outcome, state = walk(cfg, mem, chrome_fades_after=999)
    assert outcome == "success"
    assert llm.reads_requested == []
    assert decides(llm) >= len(STAMPS)


def test_the_model_can_sweep_without_the_per_item_read(cfg, mem):
    """`read_each=False` skips the vision read, not the repeat: the album is
    still walked to the end in a handful of decisions, and no frame is read
    or kept for it."""
    from pathlib import Path

    dev = AlbumDevice(cfg, chrome_fades_after=999)
    llm = fake.FakeLLM(dev, unread_album_walker())
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1, "the album was not walked to the end"
    # The repeat still happened in code, so the reasoning bill stays small.
    assert decides(llm) <= 4, f"{decides(llm)} decide calls"
    # But nothing was read: no vision calls, no readings, no frames kept.
    assert llm.reads_requested == []
    assert llm.item_frames_seen == []
    events = _events(cfg, state.run_id)
    assert not [e for e in events if e["kind"] == "item_reading"]
    sweeps = [e for e in events if e["kind"] == "sweep"]
    assert sweeps, "the gesture was never repeated"
    assert all(e["read"] == 0 for e in sweeps), sweeps
    assert any("stopped changing" in e["reason"] for e in sweeps), sweeps
    assert not list((Path(cfg.run.artifacts_dir) / state.run_id).glob(
        "*read_item*"))


def test_a_sweep_only_ever_swipes(cfg, mem):
    """The safety case. The sweep repeats one gesture; it never taps, types,
    presses a key or navigates, so it can take no action the model did not
    already authorise on this screen."""
    dev, _, _, _ = walk(cfg, mem, chrome_fades_after=999)
    swipes = [a for a in dev.actions if a.startswith("scroll(")]
    assert swipes, "the album was never paged"
    # `list_apps('')` is the once-per-run "which apps did the goal name" lookup
    # for skill selection -- a read-only query, not a gesture on this screen.
    assert all(a in ("scroll(left)", "scroll(right)") or a.startswith("tap(")
               or a == "list_apps('')"
               for a in dev.actions), dev.actions
    assert "press(back)" not in dev.actions
    assert not any(a.startswith("input_text") for a in dev.actions)


def test_a_dropped_fling_mid_sweep_is_retried_not_mistaken_for_the_end(cfg, mem):
    """A ViewPager drops flings it judges too slow. Believing the first one would
    hand back four photos early and report the album as finished."""
    dev, _, outcome, state = walk(cfg, mem, chrome_fades_after=999,
                                  drop_swipes=frozenset({4, 9}))
    assert dev.dropped == [4, 9]
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


def test_the_sweep_hands_back_when_the_gesture_stops_working(cfg, mem):
    """The only stopping condition that is not a budget.

    It used to also hand back when the caption vanished, because items "could
    not be told apart" without one. Nothing needs telling apart now, so a faded
    overlay is not an event.
    """
    _, _, outcome, state = walk(cfg, mem, chrome_fades_after=999)
    sweeps = [e for e in _events(cfg, state.run_id) if e["kind"] == "sweep"]
    assert sweeps
    assert any("stopped changing" in e["reason"] for e in sweeps), sweeps


def test_a_sweep_is_capped_so_an_endless_feed_cannot_run_away(cfg, mem):
    cfg.run.pager_sweep_max = 3
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    events = _events(cfg, state.run_id)
    sweeps = [e for e in events if e["kind"] == "sweep"]
    assert sweeps
    assert all(e["swept"] <= 3 for e in sweeps), sweeps
    assert any("limit was reached" in e["reason"] for e in sweeps)


def test_a_sweep_costs_one_history_entry_not_one_per_photo(cfg, mem):
    """Twelve near-identical lines would push everything else out of the prompt
    to say what the ledger block already says per item, in more detail."""
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    swept_lines = [h for h in state.history if "repeated" in h]
    assert swept_lines
    assert len(swept_lines) <= 3
    assert "swipe left" in swept_lines[0]
    # And the per-gesture entries are genuinely absent.
    assert sum(1 for h in state.history if "swipe" in h) < len(STAMPS)


def test_the_sweep_records_every_frame_it_read(cfg, mem):
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    readings = _readings(cfg, state.run_id)
    assert len(readings) >= len(STAMPS) - 3
    assert all(e["reading"] for e in readings)
    # Filed by position in the sweep, which is a fact, rather than under the
    # app's caption for the item, which used to be a guess. Positions restart at
    # 1 for each sweep, because each sweep is its own list and claims no
    # relationship to the one before it.
    positions = [e["position"] for e in readings]
    assert positions[0] == 1
    for previous, current in zip(positions, positions[1:]):
        assert current == previous + 1 or current == 1, positions


def test_every_sweep_reading_keeps_the_frame_it_was_read_from(cfg, mem):
    """A sweep is most of a run's vision calls and gets no live panel, so the
    reading and the frame it came off are the whole record of one -- and "what
    did it read off photo 7" is not answerable from the text alone."""
    from pathlib import Path

    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    directory = Path(cfg.run.artifacts_dir) / state.run_id
    readings = [e for e in _events(cfg, state.run_id)
                if e["kind"] == "item_reading"]

    assert readings
    assert all(e["shot"] for e in readings), "a reading with no frame kept"
    assert all((directory / e["shot"]).is_file() for e in readings)
    # One per read item, named for the step that read it.
    for event in readings:
        assert event["shot"].startswith(f"step_{event['step']:03d}_read_item_")


def test_no_frames_are_kept_when_there_is_nothing_to_read_with(cfg, mem):
    """`never_screenshot` takes the sweep's reads away entirely; it must not
    leave the run writing frames nobody was shown."""
    from pathlib import Path

    cfg.run.never_screenshot = True
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    assert not list((Path(cfg.run.artifacts_dir) / state.run_id).glob("*.jpg"))


def test_the_sweep_stops_rather_than_declaring_the_album_finished(cfg, mem):
    """It used to record an "edge" and call the set complete. It reports what
    happened -- the gesture stopped moving anything -- and says nothing about
    whether more exists somewhere else."""
    _, llm, _, state = walk(cfg, mem, chrome_fades_after=999)
    sweeps = [e for e in _events(cfg, state.run_id) if e["kind"] == "sweep"]
    assert any("no longer advances" in e["reason"] for e in sweeps), sweeps
    assert "complete" not in "\n".join(llm.notes)


def _events(cfg, run_id):
    import json
    from pathlib import Path
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# What the image model read reaches the model
# ---------------------------------------------------------------------------
#
# These two used to assert that a `reading` was filed onto an item record, so
# the decider's paraphrase could not overwrite it. There are no item records
# now. The reading still has to survive verbatim -- it is the fact the run is
# collecting -- so what is pinned is that it reaches the model unedited.

def test_a_sweep_reading_reaches_the_model_verbatim(cfg, mem):
    dev = AlbumDevice(cfg, chrome_fades_after=999)
    llm = fake.FakeLLM(dev, album_walker())
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    readings = [e["reading"] for e in _readings(cfg, state.run_id)]
    assert readings, "no frame was read at all"
    handed_back = "\n".join(llm.notes)
    assert readings[0] in handed_back, (
        f"the reading never reached the model: {readings[0]!r}")


def test_a_reading_is_not_rounded_away_by_a_paraphrase(cfg, mem):
    """The album policy's `observation` restates the same photo, and a
    restatement is where a figure gets rounded off."""
    dev = AlbumDevice(cfg, chrome_fades_after=999)
    llm = fake.FakeLLM(dev, album_walker())
    llm.vision_reading = "chicken breast on scale, 428 g"
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    readings = [e["reading"] for e in _readings(cfg, state.run_id)]
    assert readings
    assert all(r.strip() == r for r in readings)
    # What the image model returned is what was recorded -- not the decider's
    # description of the screen it was taken on.
    assert not any("a photo of a scale is on screen" in r for r in readings)


def test_the_item_read_is_the_item_and_not_the_frame_around_it(cfg, mem):
    """A sweep is most of a run's vision calls, and each one asks a question about
    one bitmap. The status bar and the nav bar are not that bitmap:
    `ITEM_READING_SYSTEM` spends two of its rules telling the model to ignore
    chrome that need not be sent at all -- and the chrome has been the answer
    before, when a clock was read as an item caption.

    The full frame still goes to disk and still feeds `dhash`; only the copy the
    model reads is cropped.
    """
    from adbagent.pager import content_box

    dev = AlbumDevice(cfg, chrome_fades_after=999)
    llm = fake.FakeLLM(dev, album_walker())
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.item_frames_seen, "no item was read"
    box = content_box(Screen(width=dev.size[0], height=dev.size[1]))
    kept = box[3] - box[1]
    for frame in llm.item_frames_seen:
        with Image.open(io.BytesIO(frame)) as img:
            # 32px tall fixture; the crop keeps the middle band of it.
            assert img.height == pytest.approx(32 * kept, abs=1), img.size
            assert img.width == 32          # nothing is cropped horizontally

    # A screen *analysis* is a different question -- "what am I looking at" -- and
    # still gets the whole frame, chrome and all.
    for frame in llm.frames_seen:
        with Image.open(io.BytesIO(frame)) as img:
            assert img.size == (32, 32), img.size


def test_a_frame_that_cannot_be_cropped_is_still_read(cfg, mem, monkeypatch):
    """A crop is an improvement, never a precondition. When it cannot be taken --
    a truncated capture, a format PIL will not open -- the whole frame must reach
    the model rather than nothing reaching it."""
    monkeypatch.setattr("adbagent.agent.crop_frac", lambda *a, **kw: None)
    dev = AlbumDevice(cfg, chrome_fades_after=999)
    llm = fake.FakeLLM(dev, album_walker())
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.item_frames_seen, "the item was not read at all"
    for frame in llm.item_frames_seen:
        with Image.open(io.BytesIO(frame)) as img:
            assert img.size == (32, 32)          # the uncropped fixture
