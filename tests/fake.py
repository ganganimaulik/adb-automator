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

#: The date this scripted phone's clock reads. Fixed, so a test can assert the
#: exact sentence the prompt carries. A Thursday, so "yesterday" lands on an
#: ordinary weekday rather than across a weekend or a month boundary -- the
#: boundaries are `prompts.date_facts`' own tests to make.
TODAY = "2026-08-06"


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
        self.shot_edges: List[int] = []
        self.dumps = 0
        #: What this phone says the date is, and how many times it was asked.
        #: A run reads it once and puts it in the prompt above the goal, so a
        #: test can check both that the model was told and that the answer was
        #: not re-bought every turn.
        self.date = TODAY
        self.date_reads = 0

    # -- observation -------------------------------------------------------

    def _xml(self) -> str:
        if self.state == "home":
            return _settings_home(self.checked)
        return self.app[self.state].xml

    def observe(self, settle: bool = False) -> Screen:
        self.dumps += 1
        return attach(parse(self._xml(), width=self.size[0], height=self.size[1],
                            activity=f".{self.state.title()}Activity"))

    def screenshot(self, max_long_edge: int = 1280, **kw) -> bytes:
        self.screenshots += 1
        # The long edge asked for, in order. A sharper re-read is only a re-read
        # if it actually asks for more pixels than the frame it is doubting, and
        # the bytes differ so a test can tell which frame reached the model.
        self.shot_edges.append(max_long_edge)
        return b"\xff\xd8\xff\xdb fake jpeg @%d" % max_long_edge

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

    def today(self) -> str:
        self.date_reads += 1
        return self.date

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
        #: How often the harness asked "is the goal already met?", and with what.
        self.goal_checks = 0
        self.goal_checks_seen: List[tuple] = []
        #: What that check answers. False by default, so a test that is not about
        #: this feature runs exactly as it did. May be a callable taking the step
        #: number, for a run that becomes finished part-way through.
        self.goal_check_result = False
        self.model = "fake/model"
        self.model_small = "fake/model-small"
        self.model_image = "fake/model-image"
        self.ledger = Ledger()
        self.seen_screenshots = 0
        self.notes: List[str] = []
        #: The rendered collected-data ledger as each turn saw it.
        self.scratchpads: List[str] = []
        #: The budget line each turn was shown -- where the run stood against its
        #: ceilings. Nothing carried this before; see `prompts.budget_line`.
        self.budgets: List[str] = []
        #: Item labels the sweep asked to have read, in order.
        self.reads_requested: List[str] = []
        #: How many separate vision passes were made, and what they return.
        self.analyses = 0
        self.vision_reading = ""
        self.vision_label = ""
        #: True to answer every vision pass as the real client does when the call
        #: fails: an empty analysis flagged `unavailable`.
        self.vision_fails = False
        #: The frames each vision pass was handed, in order, so a test can assert
        #: a re-read was given different pixels rather than the same ones twice.
        self.frames_seen: List[bytes] = []
        #: The frames each *item* read was handed. Kept apart from `frames_seen`
        #: because the two calls are shown different pixels on purpose: a screen
        #: analysis gets the whole frame, an item read gets the item.
        self.item_frames_seen: List[bytes] = []
        #: Reading for the second and later passes. A sharper capture that answers
        #: what the downscaled one could not is the whole point of re-reading.
        self.vision_reading_after_reread: Optional[str] = None
        #: The rendered analysis each decide turn was handed. Apart from `notes`
        #: because they answer different questions: what the model was told about
        #: the screen, against what it was told to do about it.
        self.analyses_seen: List[str] = []
        #: The phone's date each decide turn was told, one per call. A goal
        #: bounded in time -- "today and yesterday" -- is unreadable without it.
        self.dates_seen: List[str] = []
        #: What `locate` answers a text-mode tap_at with: (x, y) fractions, or
        #: None for "not on screen". None by default, the same shape as
        #: `goal_check_result`: a test that is not about grounding runs exactly
        #: as it did, and one that is sets the point it wants tapped.
        self.location: Optional[Tuple[float, float]] = None
        self.locates = 0
        #: The control descriptions each locate was asked to find, in order.
        self.locates_seen: List[str] = []
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
        self.frames_seen.append(screenshot)
        if self.vision_fails:
            failed = ScreenAnalysis()
            failed._failed = True
            return failed
        reading = self.vision_reading
        if self.analyses > 1 and self.vision_reading_after_reread is not None:
            reading = self.vision_reading_after_reread
        return ScreenAnalysis(reading=reading,
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
        self.item_frames_seen.append(screenshot)
        self.ledger.record(Call(model=self.model_image, prompt_tokens=400,
                                completion_tokens=30, purpose="read_item"))
        return f"reading of {label or 'an unlabelled item'}"

    def locate(self, screenshot: bytes, description: str, *, goal: str = "",
               step: int = 0, **kwargs) -> Optional[Tuple[float, float]]:
        """Grounds a named control, as the real client does with the vision model.

        Counted apart from `calls`: a locate is a vision read, not a reasoning
        turn, the same distinction `analyze_image` keeps.
        """
        self.locates += 1
        self.locates_seen.append(description)
        self.ledger.record(Call(model=self.model_image, prompt_tokens=400,
                                completion_tokens=20, purpose="locate"))
        return self.location

    def decide(self, *, goal: str, rendered: str, history, width: int, height: int,
               package: str = "", today: str = "", screenshot: Optional[bytes] = None,
               note: str = "", scratchpad: str = "",
               progress: str = "", budget: str = "",
               image_analysis: Optional[str] = None, **kwargs) -> AgentAction:
        self.calls += 1
        self.dates_seen.append(today)
        self.budgets.append(budget)
        if screenshot:
            self.seen_screenshots += 1
            # `is None`, mirroring the real client: "" means a pass ran and found
            # nothing, and treating it as "no pass ran" buys a second one.
            if image_analysis is None:
                image_analysis = self.analyze_image(screenshot, goal=goal, rendered=rendered)
        self.notes.append(note)
        self.analyses_seen.append(image_analysis or "")
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
        if screenshot and image_analysis is None:     # `is None`, as `decide` does
            image_analysis = self.analyze_image(screenshot, goal=goal, rendered=rendered)
        return Verdict(satisfied=self.judge_result,
                       evidence="fake judge" if self.judge_result else "not yet")

    def goal_check(self, *, goal: str, history=(), rendered: str = "",
                   scratchpad: str = "", progress: str = "", step: int = 0,
                   **kwargs) -> Verdict:
        """"Is the run already finished?", asked mid-run by the harness.

        Answers `goal_check_result`, which defaults to False -- the same default
        the real prompt is written around, and the one that leaves every test not
        interested in this feature behaving exactly as it did.

        Counted separately from `calls`. The point of this check is that it costs
        no wall clock because it runs inside a device round trip, and a test
        asserting how many *reasoning turns* a goal took must not have those
        numbers moved by it.
        """
        self.goal_checks += 1
        self.goal_checks_seen.append((step, scratchpad))
        self.ledger.record(Call(model=self.model, prompt_tokens=600,
                                completion_tokens=30, purpose="goal_check"))
        satisfied = self.goal_check_result
        if callable(satisfied):
            satisfied = satisfied(step)
        return Verdict(satisfied=bool(satisfied),
                       evidence=("everything the goal asked for is recorded"
                                 if satisfied else "there is more to collect"))



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
