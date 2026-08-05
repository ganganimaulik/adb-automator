"""A scripted Android phone and a scripted model.

`FakeDevice` is a state machine whose states emit real, well-formed u2 XML and
whose transitions are driven by tap coordinates -- so the loop under test does
the same parsing, pruning, fingerprinting and anchoring it would do against a
real phone, deterministically and in milliseconds.

`FakeLLM` answers with whatever policy the test gives it and counts its calls,
keeping decide calls apart from the vision reads a sweep makes. Counting them is
the point: what a change to the loop is usually for is spending fewer reasoning
turns on the same goal, and that is an assertion about these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from adbagent.actions import AgentAction
from adbagent.config import Config
from adbagent.fingerprint import attach
from adbagent.llm import Call, Ledger, ScreenAnalysis, Strategy, Verdict
from adbagent.screen import Screen, parse

from . import xmlgen as X


#: What `pm list packages` would report. `com.android.settings` is the app the
#: scripted screens below actually belong to; the rest give a name-resolving
#: caller something to pick wrongly from.
INSTALLED = ["com.android.settings", "com.whatsapp", "com.spotify.music"]


@dataclass
class FakeScreen:
    xml: str
    #: label (or resource-id) -> the state tapping it leads to.
    taps: Dict[str, str] = field(default_factory=dict)
    #: where `back` goes; empty means the app exits to the launcher.
    back: str = ""


def _settings_home(checked: bool = False) -> str:
    return X.settings_screen(rows=7, checked_row=0 if checked else -1,
                             labels=["Wi-Fi", "Bluetooth", "Mobile network",
                                     "Hotspot", "Data usage", "Airplane mode",
                                     "VPN"])


def build_app(checked: bool = False) -> Dict[str, FakeScreen]:
    return {
        "home": FakeScreen(
            xml=_settings_home(checked),
            taps={"Wi-Fi": "wifi", "Bluetooth": "bluetooth"},
        ),
        "wifi": FakeScreen(xml=X.detail_screen(), back="home"),
        "bluetooth": FakeScreen(
            xml=X.settings_screen(rows=3, title="Bluetooth",
                                  labels=["Pair new device", "Previously connected",
                                          "Bluetooth settings"]),
            back="home"),
        "launcher": FakeScreen(xml=X.dump(
            X.N("android.widget.FrameLayout", (0, 0, X.W, X.H), rid="workspace",
                package="com.android.launcher", children=[
                    X.N("android.widget.TextView", (100, 1000, 400, 1100),
                        text="Settings", rid="app_icon", package="com.android.launcher",
                        clickable=True)]))),
    }


class FakeDevice:
    """Implements exactly the surface `Agent` uses."""

    def __init__(self, cfg: Optional[Config] = None, start: str = "home",
                 app: Optional[Dict[str, FakeScreen]] = None,
                 locked: bool = False):
        self.cfg = cfg or Config()
        self.app = app or build_app()
        self.installed = list(INSTALLED)
        #: Which of `installed` the user put there rather than the vendor.
        self.third_party = ["com.whatsapp", "com.spotify.music"]
        self.state = start
        self.checked = False
        self.locked = locked
        self.size = (X.W, X.H)
        self.taps: List[Tuple[int, int]] = []
        self.actions: List[str] = []
        self.shell_calls: List[str] = []
        self.shell_replies: Dict[str, str] = {}
        self.screenshots = 0
        self.dumps = 0

    # -- observation -------------------------------------------------------

    def _xml(self) -> str:
        if self.state == "home":
            return _settings_home(self.checked)
        return self.app[self.state].xml

    def observe(self, settle: bool = False) -> Screen:
        self.dumps += 1
        return attach(parse(self._xml(), width=self.size[0], height=self.size[1],
                            activity=f".{self.state.title()}Activity"))

    def screenshot(self, **kw) -> bytes:
        self.screenshots += 1
        return b"\xff\xd8\xff\xdb fake jpeg"

    # -- actions -----------------------------------------------------------

    def _hit(self, x: int, y: int):
        screen = self.observe()
        candidates = [el for el in screen.elements
                      if el.bounds[0] <= x <= el.bounds[2]
                      and el.bounds[1] <= y <= el.bounds[3]]
        if not candidates:
            return None
        # Innermost wins, as a real hit test would.
        return min(candidates, key=lambda e: e.area)

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))
        self.actions.append(f"tap({x},{y})")
        el = self._hit(x, y)
        if el is None:
            return
        if el.checkable:
            self.checked = not self.checked
            return
        label = el.best_text
        target = self.app[self.state].taps.get(label)
        if target is None and el.resource_id:
            target = self.app[self.state].taps.get(el.resource_id)
        if target:
            self.state = target

    def long_press(self, x: int, y: int, duration: float = 0.6) -> None:
        self.actions.append(f"long_press({x},{y})")

    def swipe(self, *a, **kw) -> None:
        self.actions.append("swipe")

    def scroll(self, direction: str, **kw) -> None:
        self.actions.append(f"scroll({direction})")

    def press(self, key: str) -> None:
        self.actions.append(f"press({key})")
        if key == "back":
            self.state = self.app[self.state].back or self.state
        elif key == "home":
            self.state = "launcher"

    def input_text(self, text: str, clear: bool = True, press_enter: bool = False) -> None:
        self.actions.append(f"input_text({text!r},clear={clear},press_enter={press_enter})")
        if press_enter:
            self.actions.append("press(enter)")

    def get_clipboard(self) -> str:
        self.actions.append("get_clipboard")
        return getattr(self, "clipboard_text", "sample clipboard text")

    def set_clipboard(self, text: str) -> None:
        self.actions.append(f"set_clipboard({text!r})")
        self.clipboard_text = text

    def open_app(self, package: str) -> None:
        self.actions.append(f"open_app({package})")
        # A real `app_start` is fire-and-forget: it cannot launch what is not
        # installed, and it reports nothing when it fails to. Same here, so a
        # caller that does not verify the foreground gets caught.
        if package in self.installed:
            self.state = "home"

    def list_apps(self, query: str = "", third_party_only: bool = False) -> List[str]:
        self.actions.append(f"list_apps({query!r})")
        pkgs = [p for p in self.installed
                if not third_party_only or p in self.third_party]
        if query:
            q = query.lower()
            pkgs = [p for p in pkgs if q in p.lower()]
        return sorted(pkgs)

    # -- lock state --------------------------------------------------------

    def is_awake(self) -> bool:
        return True

    def is_locked(self) -> bool:
        return self.locked

    def wake(self) -> None:
        self.actions.append("wake")

    # -- misc --------------------------------------------------------------

    def shell(self, command: str, timeout: float = 20.0, allow_meta: bool = False) -> str:
        self.shell_calls.append(command)
        return self.shell_replies.get(command, "")

    def recover(self, tier: int = 1) -> bool:
        self.actions.append(f"recover({tier})")
        return True

    def close(self) -> None:
        pass


Policy = Callable[[Screen, "FakeLLM"], AgentAction]


class FakeLLM:
    """Answers from a policy and counts how often it was consulted."""

    def __init__(self, dev: FakeDevice, policy: Policy,
                 judge_result: bool = True):
        self.dev = dev
        self.policy = policy
        self.calls = 0
        self.judges = 0
        self.judge_result = judge_result
        self.model = "fake/model"
        self.model_small = "fake/model-small"
        self.model_image = "fake/model-image"
        self.ledger = Ledger()
        self.seen_screenshots = 0
        self.notes: List[str] = []
        #: The rendered collected-data ledger as each turn saw it.
        self.scratchpads: List[str] = []
        #: Item labels the sweep asked to have read, in order.
        self.reads_requested: List[str] = []
        #: How many separate vision passes were made, and what they return.
        self.analyses = 0
        self.vision_reading = ""
        self.vision_label = ""
        #: Tier 3 of the stall ladder. Counted apart from `calls` for the same
        #: reason the sweep's reads are: a test asserting how many reasoning
        #: turns a change costs must not have a rescue call folded into it.
        self.replans = 0
        #: The `tried` list each replan was shown, so a test can assert the call
        #: was given the evidence it is supposed to reason from.
        self.replans_seen: List[tuple] = []
        self.replan_strategy = "use the search box instead of the grid"
        self.replan_abandon = False

    @property
    def needs_vision_pass(self) -> bool:
        return True

    def analyze_image(self, screenshot: bytes, *, goal: str = "", rendered: str = "",
                      **kwargs) -> ScreenAnalysis:
        self.ledger.record(Call(model=self.model_image, prompt_tokens=500,
                                completion_tokens=100, purpose="analyze_image"))
        self.analyses += 1
        return ScreenAnalysis(reading=self.vision_reading,
                              item_label=self.vision_label,
                              notable="fake visual analysis")

    def read_item(self, screenshot: bytes, *, goal: str = "", label: str = "",
                  **kwargs) -> str:
        """One compact reading of one gallery item, as the sweep asks for.

        Counted separately from `calls`: the point of sweeping is that these
        replace *decide* calls, so a test that asserts the saving needs the two
        numbers apart.
        """
        self.reads_requested.append(label)
        self.ledger.record(Call(model=self.model_image, prompt_tokens=400,
                                completion_tokens=30, purpose="read_item"))
        return f"reading of {label or 'an unlabelled item'}"

    def decide(self, *, goal: str, rendered: str, history, width: int, height: int,
               package: str = "", screenshot: Optional[bytes] = None,
               note: str = "", scratchpad: str = "",
               progress: str = "", image_analysis: Optional[str] = None, **kwargs) -> AgentAction:
        self.calls += 1
        if screenshot:
            self.seen_screenshots += 1
            if not image_analysis:
                image_analysis = self.analyze_image(screenshot, goal=goal, rendered=rendered)
        self.notes.append(note)
        self.scratchpads.append(scratchpad)
        self.ledger.record(Call(model=self.model, prompt_tokens=1000,
                                completion_tokens=50, purpose="decide"))
        return self.policy(self.dev.observe(), self)

    def replan(self, *, goal: str, rendered: str, tried=(), stalled: int = 0,
               scratchpad: str = "", progress: str = "", packages=(),
               screenshot: Optional[bytes] = None,
               image_analysis: Optional[str] = None, **kwargs) -> Strategy:
        """One different approach, asked for when the run has stopped moving.

        Returns whatever the test set on `replan_strategy` / `replan_abandon`.
        The default is a plausible strategy rather than an empty one, because a
        fake that returns nothing would make every stalled test look like the
        ladder had failed when it had merely been mocked away.
        """
        self.replans += 1
        self.replans_seen.append(tuple(tried))
        self.ledger.record(Call(model=self.model, prompt_tokens=800,
                                completion_tokens=60, purpose="replan"))
        return Strategy(assessment="the same control keeps being tapped",
                        strategy=self.replan_strategy,
                        abandon=self.replan_abandon)

    def judge(self, *, goal: str, rendered: str, history,
              screenshot: Optional[bytes] = None,
              scratchpad: str = "",
              progress: str = "", image_analysis: Optional[str] = None, **kwargs) -> Verdict:
        self.judges += 1
        self.calls += 1
        if screenshot and not image_analysis:
            image_analysis = self.analyze_image(screenshot, goal=goal, rendered=rendered)
        return Verdict(satisfied=self.judge_result,
                       evidence="fake judge" if self.judge_result else "not yet")



# ---------------------------------------------------------------------------
# Ready-made policies
# ---------------------------------------------------------------------------

def tap_label(label: str) -> Policy:
    """Tap the element with this label; say done once it is gone."""

    def policy(screen: Screen, llm: FakeLLM) -> AgentAction:
        for el in screen.elements:
            if el.best_text == label and el.interactive:
                return AgentAction(observation="found it", reasoning="tap it",
                                   action="tap",
                                   target={"index": el.index})
        return AgentAction(observation="not here", reasoning="finished",
                           action="done", text=f"{label} was opened")

    return policy


def reach_state(dev: FakeDevice, wanted: str, path: List[str]) -> Policy:
    """Walk a fixed path of labels, then declare done on arrival."""

    def policy(screen: Screen, llm: FakeLLM) -> AgentAction:
        if dev.state == wanted:
            return AgentAction(observation="arrived", reasoning="goal reached",
                               action="done", text=f"reached {wanted}")
        for label in path:
            for el in screen.elements:
                if el.best_text == label and el.interactive:
                    return AgentAction(observation=f"can see {label}",
                                       reasoning="step towards the goal",
                                       action="tap", target={"index": el.index})
        return AgentAction(observation="lost", reasoning="back out",
                           action="press_key", key="back")

    return policy
