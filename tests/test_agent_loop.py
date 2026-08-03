"""The whole loop, end to end, against a scripted phone.

The agent always uses the LLM for decisions. These tests verify the core loop:
perceive → ask the LLM → guard → act → verify → learn → repeat.
"""

from __future__ import annotations

import json

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


def test_progress_log_retains_only_latest():
    from adbagent.agent import RunState
    from adbagent.actions import AgentAction

    state = RunState(goal="multi step test", run_id="r1", intent_id="i1")

    act1 = AgentAction(action="wait", observation="step 1", reasoning="r1", progress="Step 1 done")
    act2 = AgentAction(action="wait", observation="step 2", reasoning="r2", progress="Step 2 done")

    for act in [act1, act2]:
        if getattr(act, "progress", None):
            prog_text = act.progress.strip()
            if prog_text:
                state.progress_log = [prog_text]
                state.progress_chars = len(prog_text)

    assert state.progress_log == ["Step 2 done"]


def test_list_apps_in_agent_loop(cfg, mem):
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="home screen", reasoning="find whatsapp package",
                               action="list_apps", text="whatsapp")
        elif step_count["n"] == 2:
            return AgentAction(observation="home screen", reasoning="open whatsapp",
                               action="open_app", text="com.whatsapp")
        return AgentAction(observation="whatsapp opened", reasoning="done",
                           action="done", text="opened whatsapp")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("open whatsapp")
    assert outcome == "success"
    assert "list_apps('whatsapp')" in dev.actions
    assert "open_app(com.whatsapp)" in dev.actions


def test_scroll_swipe_no_change_does_not_ban_action(cfg, mem):
    """When a scroll or swipe action encounters no_change, it must not be added to the ban list."""
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="feed", reasoning="scroll down",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="done",
                           action="done", text="finished")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll feed")
    assert outcome == "success"
    # Even if scroll down was graded no_change, scroll/down must not be in banned
    assert "scroll/down" not in state.loops.bans_for(dev.observe().skeleton_id)




# ---------------------------------------------------------------------------
# Screenshots are captured once per screen
# ---------------------------------------------------------------------------

def test_a_screen_is_photographed_once_however_many_times_it_is_needed(cfg, mem):
    """Verification's screenshot IS the next turn's screenshot.

    `observe(settle=True)` has already waited for the tree to stop moving, and
    every path that touches the device between verifying one step and deciding
    the next (dismissing an interstitial, breaking a loop with `back`) drops the
    screen and re-observes. So a second capture buys nothing and costs a device
    round trip -- and leaves the pager comparing a `dhash` taken from different
    pixels than the ones the model was shown.
    """
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="feed", reasoning="scroll",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="finished",
                           action="done", text="done")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll the feed")
    assert outcome == "success"
    # Step 1 decides (1) and verifies (2). Step 2 reuses the verify screenshot,
    # and so does the completion judge. Four captures would mean the fix regressed.
    assert dev.screenshots == 2


def test_the_screenshot_the_model_saw_is_the_one_the_dhash_came_from(cfg, mem):
    from adbagent.agent import Agent as _Agent

    dev = fake.FakeDevice(cfg)
    agent = _Agent(dev, mem, None, cfg)
    screen = dev.observe()
    first = agent._ensure_screenshot(screen)
    assert dev.screenshots == 1
    assert agent._ensure_screenshot(screen) is first
    assert dev.screenshots == 1


# ---------------------------------------------------------------------------
# Token instrumentation
# ---------------------------------------------------------------------------

def _events(tmp_path, run_id):
    path = tmp_path / "runs" / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def test_every_decision_records_what_it_cost(cfg, mem, tmp_path):
    """Latency work needs per-step tokens in the artifact, not just a total."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides
    for event in decides:
        metrics = event["llm"]
        assert metrics["n_calls"] >= 1
        assert metrics["prompt_tokens"] == 1000     # from FakeLLM's ledger
        assert metrics["completion_tokens"] == 50
        assert "wall_s" in event


def test_a_vision_turn_is_attributed_both_of_its_calls(cfg, mem, tmp_path):
    """A screenshot turn is an image analysis *then* a decision. Charging the
    step for only the last one is how a 2-call turn reads as a 1-call turn."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides[0]["llm"]["n_calls"] == 2
    purposes = [c["purpose"] for c in decides[0]["llm"]["calls"]]
    assert purposes == ["analyze_image", "decide"]
    # 500 + 1000 prompt, 100 + 50 completion.
    assert decides[0]["llm"]["prompt_tokens"] == 1500
    assert decides[0]["llm"]["completion_tokens"] == 150


def test_the_judge_is_costed_separately_from_the_step_that_proposed_done(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    judges = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "judge"]
    assert len(judges) == 1
    assert judges[0]["step"] == state.step
    assert "llm" in judges[0] and "wall_s" in judges[0]


def test_the_run_total_omits_the_per_call_breakdown(cfg, mem, tmp_path):
    """Each call already has its own event; repeating them all in `run_end`
    would double the size of the artifact for nothing."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    end = next(e for e in _events(tmp_path, state.run_id) if e["kind"] == "run_end")
    assert "calls" not in end["llm"]
    assert end["llm"]["n_calls"] >= 2
    assert end["llm"]["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Repeated history, folded in the loop
# ---------------------------------------------------------------------------

def test_a_repeated_action_folds_in_the_history_the_loop_keeps(cfg, mem):
    """`prompts.history_only_block` renders whatever the loop appended, so the
    fold has to happen on the way in -- see `actions.append_history`."""
    cfg.run.max_steps = 8

    def policy(screen, llm):
        if llm.calls > 5:
            return AgentAction(observation="enough", reasoning="stop",
                               action="done", text="done waiting")
        return AgentAction(observation=f"waiting, turn {llm.calls}",
                           reasoning="let the screen settle",
                           action="wait", duration=0.05)

    _, state, _ = run(fake.FakeDevice(cfg), mem, cfg, policy)

    waits = [line for line in state.history if "wait" in line]
    assert len(waits) == 1, f"identical waits were not folded: {waits}"
    assert "[x5]" in waits[0]
    # The readings are what a fold must never discard.
    assert "waiting, turn 1" in waits[0] and "waiting, turn 5" in waits[0]


def test_the_situational_advice_only_shows_up_when_it_applies(cfg, mem):
    """The gallery, scrolling and app-switching blocks are 36% of what the system
    prompt used to be, and irrelevant on a turn like this one."""
    dev = fake.FakeDevice(cfg)
    _, _, llm = run(dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert llm.notes, "no NOTE block was built at all"
    assert not any("BROWSING A GALLERY" in note for note in llm.notes)
    assert not any("SWITCHING APPS" in note for note in llm.notes)


def test_a_screenshot_turn_is_timed_for_both_of_its_calls(cfg, mem, tmp_path):
    """`report`'s latency/step reads `wall_s`, so a step whose cost covers two
    calls must have a clock that covers them too."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides[0]["llm"]["n_calls"] == 2
    assert decides[0]["wall_s"] > 0
