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
It no longer matters at all: whether a swipe moved anything is asked of the
pixels, and two different photos taken in the same minute look different.

The agent under test is a scripted model that swipes left, once per turn, and
stops when the harness tells it the gesture no longer advances. Every swipe
that reaches the phone is one it chose: the harness used to go on paging in
code once a swipe had moved the content, and to re-send a swipe it judged
dropped -- and on a card deck a swipe is a decision about the card. The
assertions are about what the *harness* guarantees with only the model acting.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from adbagent.actions import AgentAction
from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.screen import Screen, parse
from adbagent.fingerprint import attach

import pytest

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


def album_walker(swipes: list):
    """A model that swipes left until the harness says the gesture stopped
    working, appending each swipe it decides to `swipes`.

    It keeps no memory of its own on purpose: the only signal it acts on is
    what the harness tells it about the last gesture. Told that a swipe may
    have been dropped, it does what it is told -- once more, faster.
    """
    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        note = llm.notes[-1] if llm.notes else ""
        if "no longer advance" in note:
            return AgentAction(observation="the album stopped advancing",
                               reasoning="nothing further to page to",
                               action="done", text="read all photos")
        pager = next((e for e in screen.elements if e.resource_id == "pager"),
                     None)
        if pager is None:
            return AgentAction(observation="no pager", reasoning="give up",
                               action="fail", text="no pager on screen")
        faster = "once more, faster" in note
        swipes.append("faster" if faster else "normal")
        return AgentAction(
            observation="a photo of a scale is on screen",
            reasoning="read it and advance", action="swipe",
            direction="left", target={"index": pager.index},
            **({"duration": 0.12, "scroll_amount": 2} if faster else {}))

    return policy


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 60
    c.safety.unattended = True
    return c


def walk(cfg, **device_kw):
    dev = AlbumDevice(cfg, **device_kw)
    swipes: list = []
    llm = fake.FakeLLM(dev, album_walker(swipes))
    outcome, state = Agent(dev, llm, cfg).run(GOAL)
    return dev, llm, outcome, state, swipes


def _events(cfg, run_id):
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _gestures(dev) -> list:
    return [a for a in dev.actions if a.startswith("scroll(")]


# ---------------------------------------------------------------------------
# The headline claims
# ---------------------------------------------------------------------------

def test_every_photo_is_reached(cfg):
    dev, _, outcome, _, _ = walk(cfg, chrome_fades_after=999)
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1, "the album was not walked to the end"


def test_every_swipe_on_the_phone_is_one_the_model_chose(cfg):
    """The safety case. Nothing repeats a gesture in code and nothing re-sends
    one: the count of swipes that reached the phone is the count of swipes the
    model decided, dropped flings and the end of the album included."""
    dev, _, _, state, swipes = walk(cfg, chrome_fades_after=999,
                                    drop_swipes=frozenset({2, 7, 10}))
    assert len(_gestures(dev)) == len(swipes)
    assert "press(back)" not in dev.actions
    assert not any(a.startswith(("tap(", "input_text")) for a in dev.actions)
    kinds = {e["kind"] for e in _events(cfg, state.run_id)}
    assert not kinds & {"sweep", "sweep_step", "pager_retry", "dismiss",
                        "loop_break"}, kinds


def test_the_walk_costs_a_turn_or_two_per_photo(cfg):
    """The run that motivated this used 136 steps for these fifteen photos."""
    _, _, _, state, _ = walk(cfg, chrome_fades_after=999)
    assert state.step <= len(STAMPS) * 2, f"took {state.step} steps"


def test_paging_the_album_is_not_called_a_loop(cfg):
    """`exact_id` is identical for all fifteen photos, and the walk is one swipe
    repeated fifteen times. That is browsing: no back press, and no telling the
    model it is repeating itself."""
    dev, llm, _, _, _ = walk(cfg, chrome_fades_after=999)
    assert "press(back)" not in dev.actions
    assert not any("You are repeating yourself" in note for note in llm.notes)


# ---------------------------------------------------------------------------
# A dropped fling, and the end of the album
# ---------------------------------------------------------------------------

def test_a_dropped_fling_is_retried_by_the_model_not_the_harness(cfg):
    """A ViewPager drops a fling it judges too short or too slow. Believing the
    first miss would end the walk four photos early; the harness used to answer
    it with a harder swipe of its own. Now the gesture is left open and the model
    is told to try once more, faster -- and it is the model's swipe that goes."""
    dev, _, outcome, state, swipes = walk(cfg, chrome_fades_after=999,
                                          drop_swipes=frozenset({2, 7, 10}))
    assert dev.dropped == [2, 7, 10]
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1
    # One faster retry per dropped fling, and one at the real end.
    assert swipes.count("faster") == 4, swipes


def test_the_end_of_the_album_takes_two_misses(cfg):
    """One miss on a screen that has been paging may be a dropped swipe, so it
    does not close the gesture; a second in a row does, and the model is told
    the gesture no longer advances."""
    dev, llm, outcome, _, swipes = walk(cfg, chrome_fades_after=999)
    assert outcome == "success"
    # Fourteen swipes to reach the last photo, then the two that found its end.
    assert len(swipes) == len(STAMPS) - 1 + 2, swipes
    assert swipes[-1] == "faster"
    notes = "\n".join(llm.notes)
    assert "once more, faster" in notes
    assert "no longer advances" in notes


def test_the_same_minute_twins_need_no_special_handling(cfg):
    """Two photos sent in the same minute used to collide, because identity was
    the caption. The pixels tell them apart without anyone having to try."""
    dev, _, outcome, _, _ = walk(cfg, chrome_fades_after=999,
                                 drop_swipes=frozenset({3}))
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


def test_hidden_chrome_does_not_stop_the_walk(cfg):
    """With `chrome_fades_after=1` the caption is gone on almost every turn.

    The content hash crops the bands the chrome lives in, so it never mattered.
    """
    dev, _, outcome, _, _ = walk(cfg, chrome_fades_after=1)
    assert outcome == "success"
    assert dev.index == len(STAMPS) - 1


def test_the_agent_is_told_nothing_it_cannot_know(cfg):
    """The old block closed with verdicts about a set: how many items it held,
    which were unread, that every one had been read. None was observable."""
    _, llm, _, _, _ = walk(cfg, chrome_fades_after=999)
    notes = "\n".join(llm.notes)
    for claim in ("ITEMS INSPECTED IN THIS SET", "STILL NOT READ",
                  "LAST item of this set", "Every item in this set",
                  "complete"):
        assert claim not in notes, claim
