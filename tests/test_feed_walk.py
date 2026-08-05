"""Walking a vertical, endless feed -- the case the carousel model could not see.

This is the regression test for ``runs/2521862d7a23``: "open instagram explore
page and like top 5 posts", which spent 45 steps in a two-cycle and never liked
a post. Three things in the old pager conspired:

* it classified the screen from the *largest* full-bleed horizontal scroller,
  which on this layout is the Home/Search/Reels/Profile tab strip, not the reel;
* it accepted only ``left``/``right`` as paging, so a feed that advances upward
  could never authorise a sweep;
* it minted item identities from a caption, found none, fell through to a bare
  clock pattern and matched the *status bar* -- so the ledger filled with items
  named after the minute they were seen in.

None of those questions is asked any more. The fixture keeps the layout that
provoked them -- the tab pager is deliberately larger than the content pager,
and there is no caption anywhere in the tree -- so that a design that started
guessing again would fail here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

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

GOAL = "scroll the reels feed and tell me what the first few are about"

#: Long enough that the sweep cap, not the content, is what stops it -- an
#: endless feed is the point.
AUTHORS = [f"creator{i}" for i in range(40)]


def _png(seed: int) -> bytes:
    image = Image.new("L", (32, 32))
    image.putdata([(seed * 53 + x * 7 + y * 13) % 256
                   for y in range(32) for x in range(32)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FeedDevice(fake.FakeDevice):
    """An endless vertical feed. Swiping UP advances; nothing else moves it."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.index = 0

    def observe(self, settle: bool = False) -> Screen:
        self.dumps += 1
        return attach(parse(X.video_feed(AUTHORS[self.index % len(AUTHORS)]),
                            width=self.size[0], height=self.size[1],
                            activity=X.REELS_ACTIVITY))

    def screenshot(self, **kw) -> bytes:
        self.screenshots += 1
        return _png(self.index)

    def scroll(self, direction: str, **kw) -> None:
        self.actions.append(f"scroll({direction})")
        if direction == "up":
            self.index += 1
        elif direction == "down":
            self.index = max(0, self.index - 1)
        # left/right do nothing at all: this feed does not page sideways.


def feed_walker():
    """Swipe up, and stop once the harness says the run has seen enough."""
    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        note = llm.notes[-1] if llm.notes else ""
        if "YOU REPEATED" in note:
            return AgentAction(observation="the feed has been sampled",
                               reasoning="enough reels seen",
                               action="done", text="described the reels")
        target = next((e for e in screen.elements
                       if e.resource_id == "clips_viewer_view_pager"), None)
        return AgentAction(
            observation="a reel is playing",
            reasoning="advance to the next reel", action="swipe",
            direction="up",
            target={"index": target.index} if target else None)

    return policy


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 40
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


def walk(cfg, mem):
    dev = FeedDevice(cfg)
    llm = fake.FakeLLM(dev, feed_walker())
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    return dev, llm, outcome, state


def _events(cfg, run_id):
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------

def test_a_vertical_feed_is_swept(cfg, mem):
    """The headline. The old gate accepted only left/right, so this feed -- the
    surface where most real swiping happens -- was never swept at all."""
    dev, llm, outcome, state = walk(cfg, mem)
    sweeps = [e for e in _events(cfg, state.run_id) if e["kind"] == "sweep"]
    assert sweeps, "an upward-paging feed authorised no sweep"
    assert all(e["gesture"] == "swipe up" for e in sweeps), sweeps
    assert dev.index > 1, "the feed never advanced"


def test_the_sweep_never_flings_sideways_on_a_feed_that_pages_up(cfg, mem):
    """The old code retargeted horizontal swipes onto the largest horizontal
    scroller -- here the tab strip -- so "next reel" became "next tab"."""
    dev, _, _, _ = walk(cfg, mem)
    assert "scroll(left)" not in dev.actions
    assert "scroll(right)" not in dev.actions


def test_an_endless_feed_is_capped_and_never_called_finished(cfg, mem):
    """A feed has no end, so nothing may claim it reached one."""
    cfg.run.pager_sweep_max = 5
    _, llm, _, state = walk(cfg, mem)
    sweeps = [e for e in _events(cfg, state.run_id) if e["kind"] == "sweep"]
    assert sweeps
    assert all(e["swept"] <= 5 for e in sweeps), sweeps
    assert any("limit was reached" in e["reason"] for e in sweeps)

    handed_back = "\n".join(llm.notes)
    for claim in ("LAST item", "Every item", "complete", "STILL NOT READ"):
        assert claim not in handed_back, claim


def test_the_status_bar_clock_names_nothing_on_a_captionless_feed(cfg, mem):
    """There is no caption anywhere in this tree. The old code fell through to a
    bare clock pattern and matched the status bar, so every frame was "an item"
    named after the minute it was seen in."""
    _, llm, _, state = walk(cfg, mem)
    readings = [e for e in _events(cfg, state.run_id)
                if e["kind"] == "item_reading"]
    assert readings, "nothing was read"
    assert all("position" in e for e in readings)
    assert not any("9:41" in json.dumps(e) for e in readings)


def test_the_loop_detector_does_not_eject_the_agent_from_the_feed(cfg, mem):
    """Every reel has the same `exact_id`, so a loop detector counting that
    alone concludes the agent is stuck and presses back."""
    dev, _, outcome, _ = walk(cfg, mem)
    assert "press(back)" not in dev.actions
