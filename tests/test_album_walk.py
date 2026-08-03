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
    # keep by hand: every photo, with what was read off it.
    final = llm.notes[-1]
    for stamp in ("9:30 am", "9:45 am", "10:03 am"):
        assert f"photo Today, {stamp} shows a scale" in final, stamp
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
