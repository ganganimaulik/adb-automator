"""Walking a photo album end to end, against a scripted media viewer.

This is the regression test for ``runs/af76720d05c4``: fifteen WhatsApp photos
that took 136 steps, 102 minutes and four full re-walks of the album, because a
swipe was always graded ``success`` and nothing recorded which photos had been
looked at. The device here reproduces the three properties that made that
possible:

* the album contains two photos sent in the same minute, so their captions are
  identical;
* the overlay chrome fades after a couple of gestures, which changes the pager's
  element index and removes the caption from the tree entirely;
* the ViewPager drops some flings, leaving the photo exactly where it was.

The agent under test is a scripted model that does the sensible thing -- read
what the NOTE block tells it, swipe left -- and the assertions are about what the
*harness* guarantees it: every photo read exactly once, no re-walks.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from adbagent.actions import AgentAction
from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.memory import Memory
from adbagent.pager import pager_element
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
    """A model that reads the current photo and swipes left, one item per turn.

    It keeps no memory of its own on purpose -- if the album gets fully covered,
    that is the harness's ledger doing the work, not the policy's.
    """
    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        note = llm.notes[-1] if llm.notes else ""
        if "Every item in this set has been read" in note:
            return AgentAction(observation="album finished",
                               reasoning="every photo is read",
                               action="done", text="read all photos")
        if "did NOT change" in note:
            return AgentAction(observation="the swipe was dropped",
                               reasoning="flick harder", action="swipe",
                               direction="left", scroll_amount=2, duration=0.12)
        pager = pager_element(screen)
        if pager is None:
            return AgentAction(observation="no pager", reasoning="give up",
                               action="fail", text="no pager on screen")
        if not screen.item_label:
            return AgentAction(observation="the caption is hidden",
                               reasoning="reveal the title bar",
                               action="tap", target={"index": pager.index})
        return AgentAction(
            observation=f"photo {screen.item_label} shows a scale",
            reasoning="record it and advance", action="swipe",
            direction="left", target={"index": pager.index})

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

def test_every_photo_is_read_exactly_once(cfg, mem):
    dev, llm, outcome, state = walk(cfg, mem)

    assert outcome == "success"
    assert state.items.read_count == len(STAMPS), (
        f"read {state.items.read_count} of {len(STAMPS)} photos")
    # One vision pass per photo, and the ledger's count proves none was skipped
    # or silently merged into its same-minute twin.
    assert len(state.items.items) == len(STAMPS)
    assert dev.index == len(STAMPS) - 1


def test_the_same_minute_twins_are_not_merged(cfg, mem):
    _, _, _, state = walk(cfg, mem)
    labels = [record.label for record in state.items.items.values()]
    assert labels.count("Today, 9:33 am") == 1
    assert "Today, 9:33 am (#2)" in labels
    assert "Today, 9:52 am (#2)" in labels


def test_the_walk_costs_a_step_or_two_per_photo(cfg, mem):
    """The run that motivated this used 136 steps for these fifteen photos.

    The budget here is two steps each: one to read and advance, plus the turns
    spent tapping the overlay back on. This device fades its chrome every two
    gestures, which is harsher than the real app's few-second timeout.
    """
    _, _, _, state = walk(cfg, mem)
    assert state.step <= len(STAMPS) * 2, f"took {state.step} steps"


def test_no_forced_back_ejects_the_agent_from_the_album(cfg, mem):
    """`exact_id` is identical for all fifteen photos, so without item-aware
    loop detection the loop breaker fires and dumps the agent out of the viewer."""
    dev, _, _, _ = walk(cfg, mem)
    assert "press(back)" not in dev.actions


def test_a_dropped_fling_is_detected_and_retried(cfg, mem):
    dev, _, outcome, state = walk(cfg, mem, drop_swipes=frozenset({2, 7, 10}))
    assert dev.dropped == [2, 7, 10]
    assert outcome == "success"
    assert state.items.read_count == len(STAMPS)


def test_a_dropped_fling_between_the_twins_still_advances(cfg, mem):
    """The hardest case: the swipe out of the first 9:33 is dropped, so the
    caption is unchanged for a reason that is *not* a second photo."""
    dev, _, outcome, state = walk(cfg, mem, drop_swipes=frozenset({3}))
    labels = [record.label for record in state.items.items.values()]
    assert labels.count("Today, 9:33 am") == 1
    assert "Today, 9:33 am (#2)" in labels
    assert state.items.read_count == len(STAMPS)


def test_the_agent_is_told_where_it_is_and_what_it_has_read(cfg, mem):
    _, llm, _, _ = walk(cfg, mem)
    notes = "\n".join(llm.notes)
    assert "CAROUSEL: this screen shows ONE item of a set" in notes
    assert "ITEMS INSPECTED IN THIS SET" in notes
    assert "you are here" in notes

    # The last turn's block is the durable memory the model no longer has to
    # keep by hand: every photo, marked read, with what was read off it. A photo
    # the sweep passed carries the vision model's reading rather than the
    # decider's own description of the screen it was on.
    final = llm.notes[-1]
    for stamp in ("9:30 am", "9:45 am", "10:03 am"):
        line = next((l for l in final.splitlines() if f"Today, {stamp}" in l), "")
        assert line, f"{stamp} is missing from the ledger block"
        assert "[read]" in line, line
        assert " -- " in line, f"{stamp} has no reading attached: {line}"
    assert "Every item in this set has been read" in final


def test_a_photo_the_agent_never_saw_is_reported_as_unread(cfg, mem):
    """The ledger's whole point: an item sighted but not looked at still counts
    as outstanding, and says so."""
    cfg.run.never_screenshot = True         # no vision, so nothing can be read
    cfg.run.max_steps = 12
    _, llm, _, state = walk(cfg, mem)
    assert state.items.read_count == 0
    assert len(state.items.items) > 1
    assert "STILL NOT READ" in llm.notes[-1]


def test_hidden_chrome_does_not_lose_the_agents_place(cfg, mem):
    """With `chrome_fades_after=1` the caption is gone on most turns."""
    dev, _, outcome, state = walk(cfg, mem, chrome_fades_after=1)
    assert outcome == "success"
    assert state.items.read_count == len(STAMPS)


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
    assert state.items.read_count == len(STAMPS)
    # Fifteen photos on three decisions: open the walk, and confirm it finished.
    assert decides(llm) <= 4, f"{decides(llm)} decide calls"
    assert len(llm.reads_requested) >= len(STAMPS) - 3


def test_the_saving_is_real_and_not_an_accounting_trick(cfg, mem):
    """Same album, same policy, sweep off then on."""
    cfg.run.pager_sweep = False
    _, without, _, state_without = walk(cfg, mem, chrome_fades_after=999)
    cfg.run.pager_sweep = True
    _, with_sweep, _, state_with = walk(cfg, mem, chrome_fades_after=999)

    assert state_without.items.read_count == len(STAMPS)
    assert state_with.items.read_count == len(STAMPS)
    assert decides(with_sweep) < decides(without) / 3, (
        f"{decides(without)} -> {decides(with_sweep)}")


def test_sweeping_off_restores_a_turn_per_photo(cfg, mem):
    cfg.run.pager_sweep = False
    _, llm, outcome, state = walk(cfg, mem, chrome_fades_after=999)
    assert outcome == "success"
    assert llm.reads_requested == []
    assert decides(llm) >= len(STAMPS)


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
    assert state.items.read_count == len(STAMPS)


def test_the_sweep_hands_back_when_the_caption_disappears(cfg, mem):
    """With chrome fading every two gestures the sweep can only ever cover a
    couple of photos before items stop being distinguishable -- so it stops, the
    model taps to bring the caption back, and the album still gets fully read."""
    dev, llm, outcome, state = walk(cfg, mem, chrome_fades_after=2)
    assert outcome == "success"
    assert state.items.read_count == len(STAMPS)
    assert decides(llm) > 4          # it really did keep asking the model
    assert "tap(" in " ".join(dev.actions)


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
    swept_lines = [h for h in state.history if "swept" in h]
    assert swept_lines
    assert len(swept_lines) <= 3
    assert "item(s) left through the carousel" in swept_lines[0]
    # And the per-gesture entries are genuinely absent.
    assert sum(1 for h in state.history if "swipe" in h) < len(STAMPS)


def test_the_sweep_records_every_item_it_read(cfg, mem):
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    events = _events(cfg, state.run_id)
    readings = [e for e in events if e["kind"] == "item_reading"]
    assert len(readings) >= len(STAMPS) - 3
    assert all(e["reading"] for e in readings)
    # Each reading is filed against the item it was taken of, not the one the
    # swipe landed on.
    labels = {e["item"] for e in readings}
    assert len(labels) == len(readings)


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


def test_the_sweep_marks_the_end_of_the_album(cfg, mem):
    _, _, _, state = walk(cfg, mem, chrome_fades_after=999)
    assert "left" in state.items.edges
    assert state.items.complete


def _events(cfg, run_id):
    import json
    from pathlib import Path
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# What the image model read goes into the ledger, not just into the prompt
# ---------------------------------------------------------------------------

def test_a_vision_reading_is_filed_against_the_item_it_was_taken_from(cfg, mem):
    """`reading` is the fact the run is collecting, so it belongs on the item
    record -- routing it through prose for the decider to re-extract loses it the
    moment the decider paraphrases.

    Only the items the *decider* looks at come through this path; the ones the
    sweep walks are read by `read_item`, which was already terse.
    """
    dev = AlbumDevice(cfg)
    llm = fake.FakeLLM(dev, album_walker())
    llm.vision_reading = "chicken breast on scale, 428 g"
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    details = [r.detail for r in state.items.items.values() if r.read]
    assert details, "no item was read at all"
    assert any("428 g" in detail for detail in details), (
        f"the vision reading reached no item record: {details}")
    assert all(detail for detail in details), "an item was read but recorded nothing"


def test_the_decider_does_not_overwrite_the_reading_with_its_paraphrase(cfg, mem):
    """The album policy's `observation` is a restatement of the same photo, and a
    restatement is where a figure gets rounded away."""
    dev = AlbumDevice(cfg)
    llm = fake.FakeLLM(dev, album_walker())
    llm.vision_reading = "428 g"
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    details = [r.detail for r in state.items.items.values() if r.read]
    assert any(detail == "428 g" for detail in details), (
        f"the direct reading was replaced by a paraphrase: {details}")
