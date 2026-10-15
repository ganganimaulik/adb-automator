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

None of those questions is asked any more, and nothing pages the feed in code
either: every swipe is the model's, one turn each. The fixture keeps the layout
that provoked the old failures -- the tab pager is deliberately larger than the
content pager, and there is no caption anywhere in the tree -- so that a design
that started guessing again would fail here.
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
from adbagent.screen import Screen, parse
from adbagent.fingerprint import attach

from . import fake
from . import xmlgen as X

GOAL = "scroll the reels feed and tell me what the first few are about"

#: Longer than any walk here gets through -- an endless feed is the point.
AUTHORS = [f"creator{i}" for i in range(40)]

#: How many reels the scripted model watches before it reports. More than the
#: nine repeats that make one action on one screen count as a loop, so a walk
#: this long would be called one if paging were not exempt.
REELS = 12


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
    """Swipe up, one reel per turn, and report after `REELS` of them."""
    swiped = {"n": 0}

    def policy(screen: Screen, llm: fake.FakeLLM) -> AgentAction:
        if swiped["n"] >= REELS:
            return AgentAction(observation="the feed has been sampled",
                               reasoning="enough reels seen",
                               action="done", text="described the reels")
        swiped["n"] += 1
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
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 40
    c.safety.unattended = True
    return c




def walk(cfg):
    dev = FeedDevice(cfg)
    llm = fake.FakeLLM(dev, feed_walker())
    outcome, state = Agent(dev, llm, cfg).run(GOAL)
    return dev, llm, outcome, state


def _events(cfg, run_id):
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------

def test_a_vertical_feed_is_paged_one_model_decision_per_swipe(cfg):
    """Every swipe that reaches the phone is one the model chose. The harness
    used to go on swiping in code once a swipe had moved the content."""
    dev, llm, outcome, state = walk(cfg)
    swipes = [a for a in dev.actions if a.startswith("scroll(")]
    assert outcome == "success"
    assert dev.index == REELS, "the feed did not advance once per swipe"
    assert swipes == ["scroll(up)"] * REELS
    kinds = {e["kind"] for e in _events(cfg, state.run_id)}
    assert not kinds & {"sweep", "sweep_step", "pager_retry"}, kinds


def test_the_agent_never_flings_sideways_on_a_feed_that_pages_up(cfg):
    """The old code retargeted horizontal swipes onto the largest horizontal
    scroller -- here the tab strip -- so "next reel" became "next tab"."""
    dev, _, _, _ = walk(cfg)
    assert "scroll(left)" not in dev.actions
    assert "scroll(right)" not in dev.actions


def test_an_endless_feed_is_never_called_finished(cfg):
    """A feed has no end, so nothing may claim it reached one."""
    _, llm, _, _ = walk(cfg)
    handed_back = "\n".join(llm.notes)
    for claim in ("LAST item", "Every item", "complete", "STILL NOT READ"):
        assert claim not in handed_back, claim


def test_paging_a_feed_is_not_called_a_loop(cfg):
    """Every reel has the same `exact_id` and the same skeleton, and the walk is
    one swipe repeated more than nine times. That is browsing: no back press,
    and no telling the model it is repeating itself."""
    dev, llm, outcome, _ = walk(cfg)
    assert outcome == "success"
    assert "press(back)" not in dev.actions
    assert not any("You are repeating yourself" in note for note in llm.notes)
