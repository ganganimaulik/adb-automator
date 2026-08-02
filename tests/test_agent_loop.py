"""The whole loop, end to end, against a scripted phone.

The agent always uses the LLM for decisions. These tests verify the core loop:
perceive → ask the LLM → guard → act → verify → learn → repeat.
"""

from __future__ import annotations

import pytest

from adbagent.actions import AgentAction
from adbagent.agent import Agent, Oracle, RunState, needs_screenshot
from adbagent.config import Config
from adbagent.memory import Memory

from . import fake
from . import xmlgen as X


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    c.safety.unattended = True      # never block a test on input()
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


GOAL = "open the Wi-Fi settings screen"


def run(dev, mem, cfg, policy, **kw):
    llm = fake.FakeLLM(dev, policy)
    agent = Agent(dev, mem, llm, cfg, **kw)
    outcome, state = agent.run(GOAL)
    return outcome, state, llm


# ---------------------------------------------------------------------------
# Basic operation
# ---------------------------------------------------------------------------

def test_run_uses_the_llm_and_succeeds(cfg, mem):
    dev = fake.FakeDevice(cfg)
    outcome, state, llm = run(dev, mem, cfg,
                              fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.calls >= 2          # at least one decide plus the completion judge


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def test_a_programmatic_assertion_ends_the_run_with_no_judge(cfg, mem):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    outcome, state = Agent(dev, mem, llm, cfg,
                           oracle=Oracle(text="Forget network")).run(GOAL)
    assert outcome == "success"
    assert llm.judges == 0


def test_shell_assertion(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.shell_replies["settings get global wifi_on"] = "1"
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    oracle = Oracle(shell="settings get global wifi_on", equals="1")
    outcome, _ = Agent(dev, mem, llm, cfg, oracle=oracle).run(GOAL)
    assert outcome == "success"


def test_premature_done_is_rejected_by_the_judge(cfg, mem):
    """Claiming success too early is a documented failure of every mobile agent."""
    dev = fake.FakeDevice(cfg)

    calls = {"n": 0}

    def policy(screen, llm):
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentAction(observation="home", reasoning="claiming early",
                               action="done", text="I think it's done")
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="really do it",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="there", reasoning="arrived",
                           action="done", text="done for real")

    llm = fake.FakeLLM(dev, policy, judge_result=False)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert llm.judges >= 1
    assert outcome in ("failed", "success")
    assert any("rejected" in line for line in state.history)


def test_assertion_overrules_a_premature_done(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        if dev.state == "wifi":
            return AgentAction(observation="arrived", reasoning="ok",
                               action="done", text="done")
        if llm.calls <= 1:
            return AgentAction(observation="home", reasoning="too early",
                               action="done", text="premature")
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="go",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="?", reasoning="back", action="press_key",
                           key="back")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg,
                           oracle=Oracle(text="Forget network")).run(GOAL)
    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.judges == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_credential_screen_hands_over_and_learns_nothing(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.app["home"] = fake.FakeScreen(xml=X.dump(
        X.N("android.widget.FrameLayout", (0, 0, X.W, X.H), rid="content", children=[
            X.N("android.widget.EditText", (48, 700, 1030, 820), rid="password",
                hint="Password", password=True, clickable=True, focusable=True),
            X.N("android.widget.Button", (48, 900, 520, 1020), text="Sign in",
                rid="signin", clickable=True)])))
    dev._xml = lambda: dev.app[dev.state].xml  # type: ignore[assignment]

    llm = fake.FakeLLM(dev, fake.tap_label("Sign in"))
    outcome, state = Agent(dev, mem, llm, cfg).run("log in")
    assert outcome == "needs_user"
    assert llm.calls == 0, "the model must never even see a credential screen"


def test_irreversible_action_is_refused_when_unattended(cfg, mem):
    dev = fake.FakeDevice(cfg, start="wifi")
    llm = fake.FakeLLM(dev, fake.tap_label("Forget network"))
    outcome, state = Agent(dev, mem, llm, cfg).run("forget this network")
    assert "Forget network" not in "".join(dev.actions)
    assert any("refused" in line for line in state.history)


def test_interstitial_is_dismissed_without_an_llm_call(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.app["home"] = fake.FakeScreen(
        xml=X.settings_screen(extra_roots=[
            X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
                package="com.android.vending", rid="nag", children=[
                    X.N("android.widget.TextView", (100, 860, 980, 980),
                        package="com.android.vending", text="Rate this app!"),
                    X.N("android.widget.Button", (620, 1240, 980, 1380),
                        package="com.android.vending", text="Not now",
                        rid="dismiss", clickable=True)])]),
        taps={"Wi-Fi": "wifi"})
    dev._xml = lambda: dev.app[dev.state].xml  # type: ignore[assignment]

    llm = fake.FakeLLM(dev, fake.tap_label("Wi-Fi"))
    Agent(dev, mem, llm, cfg).run(GOAL)
    # The dismiss happened before any model call.
    assert dev.taps, "the nag should have been tapped"


def test_loop_breaker_stops_a_stuck_agent(cfg, mem):
    """A model that keeps choosing a dud action must not burn the whole budget."""
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 12

    def useless(screen, llm):
        el = next(e for e in screen.elements if e.best_text == "Data usage")
        return AgentAction(observation="home", reasoning="press it again",
                           action="tap", target={"index": el.index})

    llm = fake.FakeLLM(dev, useless)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "failed"
    assert state.step <= cfg.run.max_steps
    # The dud action was banned rather than retried forever.
    assert any(state.loops.bans_for(k) for k in state.loops.banned)


def test_step_budget_is_enforced(cfg, mem):
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 3
    cfg.run.max_consecutive_failures = 99

    def wander(screen, llm):
        return AgentAction(observation="x", reasoning="y", action="scroll",
                           direction="down")

    llm = fake.FakeLLM(dev, wander)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "failed" and state.step == 3


def test_dry_run_touches_nothing(cfg, mem):
    cfg.run.dry_run = True
    cfg.run.max_steps = 3
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.tap_label("Wi-Fi"))
    Agent(dev, mem, llm, cfg).run(GOAL)
    assert dev.taps == []
    assert dev.state == "home"


# ---------------------------------------------------------------------------
# Screenshot policy
# ---------------------------------------------------------------------------

def test_xml_first_no_screenshot_on_a_normal_screen(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    state = RunState(goal="g", run_id="r", intent_id="i")
    want, _ = needs_screenshot(state, screen, cfg)
    assert not want


def test_screenshot_for_a_degenerate_tree(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.webview_screen(), width=X.W, height=X.H))
    want, note = needs_screenshot(RunState(goal="g", run_id="r", intent_id="i"),
                                  screen, cfg)
    assert want and "WebView" in note


def test_screenshot_after_a_failure(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    state = RunState(goal="g", run_id="r", intent_id="i", consecutive_failures=1)
    want, _ = needs_screenshot(state, screen, cfg)
    assert want


def test_never_screenshot_wins(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    cfg.run.never_screenshot = True
    screen = attach(parse(X.webview_screen(), width=X.W, height=X.H))
    want, _ = needs_screenshot(RunState(goal="g", run_id="r", intent_id="i"),
                               screen, cfg)
    assert not want


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def test_run_writes_artifacts(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)
    events = (tmp_path / "runs" / state.run_id / "events.jsonl")
    assert events.exists()
    lines = [l for l in events.read_text().splitlines() if l.strip()]
    kinds = {__import__("json").loads(l)["kind"] for l in lines}
    assert {"run_start", "decide", "verify", "run_end"} <= kinds


def test_recorder_dump_messages(cfg, tmp_path):
    from adbagent.agent import Recorder
    rec = Recorder(cfg, "run_test_123")
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ABCDEF"}}]}
    ]
    path_str = rec.dump_messages(1, msgs, purpose="decide")
    from pathlib import Path
    import json
    p = Path(path_str)
    assert p.exists()
    assert p.name == "step_001_decide_messages.json"
    data = json.loads(p.read_text())
    assert data[0]["content"] == "sys prompt"
    assert "base64 image payload" in data[1]["content"][1]["image_url"]["url"]
    rec.close()
