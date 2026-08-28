"""The supervisor loop: when it spends a pass, when it sleeps, and that it lives.

The loop's whole job is to outlive things, so most of these tests are about what
happens when something goes wrong rather than when it goes right.
"""

from __future__ import annotations

import pytest

from adbagent.config import Config
from adbagent.device import DeviceLost
from adbagent.fingerprint import attach
from adbagent.ledger import ReplyLedger
from adbagent.screen import parse
from adbagent.watch import Anchor, Watch, load_policy, screen_digest

from . import xmlgen as X


def chat(**kw):
    return attach(parse(X.chat_thread(**kw), width=X.W, height=X.H))


def settings():
    return attach(parse(X.settings_screen(), width=X.W, height=X.H))


class StubLedgerHolder:
    """Minimal stand-in for LLMClient.ledger."""

    def __init__(self):
        self.total_usd = 0.0


class StubLLM:
    def __init__(self):
        self.ledger = StubLedgerHolder()


class FakeClock:
    """A monotonic clock that only moves when the loop sleeps.

    Which is the truth about this loop: between probes it is either sleeping or
    doing something with a stub that takes no time at all. Without it every
    wall-clock ceiling -- the sweep, the rolling spend window -- would sit at
    zero elapsed forever while the test ran through a day of probes.
    """

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds

    def __call__(self) -> float:
        return self.t


class StubDevice:
    """Hands back scripted screens; raises whatever is scripted instead."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.observations = 0

    def observe(self, settle: bool = False):
        self.observations += 1
        frame = self.frames[min(self.observations - 1, len(self.frames) - 1)]
        if isinstance(frame, Exception):
            raise frame
        return frame


class StubState:
    """Just the fields a finished pass is asked for: what
    `TraceCollector.finish` reads, and whether a person took the phone during
    it -- which the watch latches across passes."""

    def __init__(self, step, took_over=False):
        self.step = step
        self.scratchpad = None
        self.took_over = took_over


class StubAgent:
    """Returns scripted outcomes; records the goals it was given."""

    def __init__(self, outcomes, seen_goals, spend=0.0, llm=None,
                 takeovers=()):
        self.outcomes = outcomes
        self.seen_goals = seen_goals
        self.spend = spend
        self.llm = llm
        #: Which pass numbers (1-based) hand the phone to a person.
        self.takeovers = set(takeovers)

    def run(self, goal, run_id="", resume=None):
        self.seen_goals.append(goal)
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        if self.llm is not None:
            self.llm.ledger.total_usd += self.spend
        # BaseException, not Exception: KeyboardInterrupt is not an Exception,
        # and testing that Ctrl-C escapes the loop depends on raising it here.
        if isinstance(outcome, BaseException):
            raise outcome
        n = len(self.seen_goals)
        return outcome, StubState(n, took_over=n in self.takeovers)


def distinct(n: int):
    """`n` screens that never match each other, so every probe finds work."""
    return [chat(messages=[f"message number {i}"]) for i in range(n)]


@pytest.fixture()
def cfg():
    c = Config()
    c.watch.interval_s = 45.0
    c.watch.backoff_initial_s = 30.0
    c.watch.backoff_max_s = 900.0
    return c


def build(cfg, frames, outcomes=(), spend=0.0, ledger_path=None, tmp_path=None,
          takeovers=()):
    """A Watch wired to stubs, plus the lists that record what it did."""
    llm = StubLLM()
    clock, goals = FakeClock(), []
    agent = StubAgent(list(outcomes), goals, spend=spend, llm=llm,
                      takeovers=takeovers)
    watch = Watch(StubDevice(frames), None, llm, cfg,
                  policy="be brief",
                  ledger=ReplyLedger(ledger_path or (tmp_path / "l.jsonl")),
                  make_agent=lambda: agent,
                  sleep=clock.sleep, clock=clock)
    return watch, clock.slept, goals


# -- the novelty signal -----------------------------------------------------

def test_screen_digest_ignores_a_ticking_clock():
    assert screen_digest(chat(stamp="2m")) == screen_digest(chat(stamp="3m"))


def test_screen_digest_notices_a_new_message():
    assert screen_digest(chat()) != \
           screen_digest(chat(messages=["hey", "you around?", "hello?"]))


def test_anchor_matches_only_the_same_package_and_content():
    a = Anchor.of(chat())
    assert a.matches(chat())
    assert not a.matches(chat(messages=["different"]))
    assert not a.matches(settings())          # wandered off to another app


def test_an_empty_anchor_never_matches():
    """Nothing is anchored yet, so the first pass must always run."""
    assert not Anchor().matches(chat())


# -- when a pass is spent ---------------------------------------------------

def test_first_probe_always_runs_a_pass(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()], ["success"], tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    assert len(goals) == 1
    assert watch.stats.passes == 1


def test_unchanged_screen_costs_no_pass(cfg, tmp_path):
    """The whole point: a quiet inbox spends a UI dump, not a model call."""
    watch, slept, goals = build(cfg, [chat()], ["success"], tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)      # anchors on chat()
    watch.run("watch instagram dms", max_passes=2)      # one more probe
    assert len(goals) == 1, "a second pass ran against an unchanged screen"
    assert watch.stats.skipped == 1
    assert cfg.watch.interval_s in slept


def test_new_content_spends_a_pass(cfg, tmp_path):
    frames = [chat(), chat(), chat(messages=["hey", "you around?", "hello?"])]
    watch, slept, goals = build(cfg, frames, ["success", "success"],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    assert watch.anchor.digest == screen_digest(chat())
    watch.run("watch instagram dms", max_passes=2)
    assert len(goals) == 2, "a new message did not trigger a pass"


def test_the_sweep_is_off_unless_it_is_asked_for(cfg, tmp_path):
    """The default has to stay reactive: a quiet app costs a dump, not a pass."""
    assert cfg.watch.sweep_s == 0
    watch, slept, goals = build(cfg, [chat()], ["success"], tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=6)
    assert len(goals) == 1


def test_a_sweep_spends_a_pass_on_an_unchanged_screen(cfg, tmp_path):
    """Work that does not announce itself -- a feed, a queue, a periodic check.

    The screen the last pass left behind is exactly the screen that is there now,
    and there is still something to do, so novelty cannot be the only trigger.
    """
    cfg.watch.sweep_s = 120.0
    watch, slept, goals = build(cfg, [chat()], ["success"] * 2,
                                tmp_path=tmp_path)
    # 45s of probing at a time: passes at t=0 and t=135, skips in between.
    watch.run("watch instagram dms", max_passes=5)
    assert len(goals) == 2, "the sweep never came due"
    assert watch.stats.skipped == 3, "it swept more often than it was asked to"


def test_a_new_message_does_not_wait_for_the_sweep(cfg, tmp_path):
    """The sweep is a second trigger, not a floor on how often a pass may run."""
    cfg.watch.sweep_s = 3600.0
    frames = [chat(), chat(), chat(messages=["hey", "you around?", "hello?"])]
    watch, slept, goals = build(cfg, frames, ["success", "success"],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    watch.run("watch instagram dms", max_passes=2)
    assert len(goals) == 2


def test_wandering_off_the_app_spends_a_pass(cfg, tmp_path):
    """A phone left on the launcher must be brought back, not watched."""
    frames = [chat(), chat(), settings()]
    watch, slept, goals = build(cfg, frames, ["success", "success"],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    watch.run("watch instagram dms", max_passes=2)
    assert len(goals) == 2


# -- the goal it hands over -------------------------------------------------

def test_the_pass_goal_carries_the_iteration_contract(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()], ["success"], tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    goal = goals[0]
    assert goal.startswith("watch instagram dms")
    assert "ONE PASS" in goal
    assert "At most one reply per conversation" in goal
    assert "Finish on the screen you worked from" in goal


def test_the_iteration_contract_does_not_assume_an_inbox(cfg, tmp_path):
    """It frames a pass; the goal and the policy say what the work is.

    Naming conversations and incoming messages here made it a second, competing
    description of the task for any goal that was not an inbox sweep.
    """
    watch, slept, goals = build(cfg, [chat()], ["success"], tmp_path=tmp_path)
    watch.run("work through my feed", max_passes=1)
    contract = goals[0].split("ONE PASS", 1)[1]
    assert "the goal works from" in contract
    assert "whatever the goal counts as work" in contract


# -- surviving failure ------------------------------------------------------

def test_a_failed_pass_backs_off_and_keeps_going(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()], ["failed", "failed"],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=2)
    assert watch.stats.failures == 2
    assert slept[:2] == [30.0, 60.0], "backoff did not double"


def test_backoff_is_capped(cfg, tmp_path):
    cfg.watch.backoff_initial_s = 600.0
    cfg.watch.backoff_max_s = 900.0
    watch, slept, goals = build(cfg, [chat()], ["failed"] * 4,
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=4)
    assert max(slept) == 900.0


def test_a_failed_pass_drops_the_anchor(cfg, tmp_path):
    """A failed pass left the screen somewhere unknown; do not trust it."""
    watch, slept, goals = build(cfg, distinct(6), ["success", "failed"],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    assert watch.anchor.package
    watch.run("watch instagram dms", max_passes=2)
    assert watch.anchor == Anchor()


def test_a_crashing_pass_does_not_kill_the_watch(cfg, tmp_path):
    """An unhandled bug in one pass is not a reason for the watch to be gone."""
    watch, slept, goals = build(cfg, [chat()],
                                [RuntimeError("boom"), "success"],
                                tmp_path=tmp_path)
    stats = watch.run("watch instagram dms", max_passes=2)
    assert stats.failures == 1
    assert stats.passes == 2


def test_a_device_loss_in_a_pass_is_survived(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()],
                                [DeviceLost("unplugged"), "success"],
                                tmp_path=tmp_path)
    stats = watch.run("watch instagram dms", max_passes=2)
    assert stats.failures == 1
    assert stats.passes == 2


def test_an_unreachable_device_retries_instead_of_exiting(cfg, tmp_path):
    """The probe itself failing must back off, not end the watch."""
    frames = [DeviceLost("gone"), DeviceLost("still gone"), chat()]
    watch, slept, goals = build(cfg, frames, ["success"], tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=1)
    assert slept[:2] == [30.0, 60.0]
    assert len(goals) == 1, "it never got to a pass once the device came back"


def test_keyboard_interrupt_propagates(cfg, tmp_path):
    """Ctrl-C is the one thing that stops a watch."""
    watch, slept, goals = build(cfg, [chat()], [KeyboardInterrupt()],
                                tmp_path=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        watch.run("watch instagram dms", max_passes=1)


def test_needs_user_is_not_a_failure(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()], ["needs_user"],
                                tmp_path=tmp_path)
    stats = watch.run("watch instagram dms", max_passes=1)
    assert stats.failures == 0
    assert watch.anchor.package, "it should still anchor and keep watching"


# -- ceilings ---------------------------------------------------------------

def test_passes_are_bounded_by_watch_max_steps(cfg, tmp_path):
    cfg.run.max_steps = 60
    cfg.watch.max_steps = 25
    build(cfg, [chat()], tmp_path=tmp_path)
    assert cfg.run.max_steps == 25


def test_rolling_spend_ceiling_pauses_the_loop(cfg, tmp_path):
    cfg.watch.max_usd_per_hour = 0.10
    watch, slept, goals = build(cfg, distinct(12), ["success"] * 6, spend=0.06,
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=3)
    # Two passes at $0.06 clear the $0.10 ceiling, so the third probe must pause
    # rather than start a pass -- even though it is finding fresh work.
    assert len(goals) == 2
    assert watch.stats.paused == 1
    assert any(s > cfg.watch.interval_s for s in slept)


def test_the_spend_ceiling_pauses_and_does_not_end_the_loop(cfg, tmp_path):
    """It is a ceiling per rolling hour, so an hour later there is room again."""
    cfg.watch.max_usd_per_hour = 0.10
    watch, slept, goals = build(cfg, distinct(12), ["success"] * 6, spend=0.06,
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=4)
    assert len(goals) == 3, "the loop never came back from the pause"


def test_no_spend_ceiling_by_default(cfg, tmp_path):
    assert cfg.watch.max_usd_per_hour == 0.0
    watch, slept, goals = build(cfg, distinct(12), ["success"] * 4, spend=5.0,
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=4)
    assert len(goals) == 4
    assert watch.stats.paused == 0


def test_stop_ends_the_loop(cfg, tmp_path):
    watch, slept, goals = build(cfg, [chat()], ["success"] * 5,
                                tmp_path=tmp_path)
    watch.stop()
    watch.run("watch instagram dms")
    assert goals == []


# -- the policy file --------------------------------------------------------

def test_load_policy_reads_the_file(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("only reply to people I follow\n", encoding="utf-8")
    assert load_policy(str(p)) == "only reply to people I follow"


def test_a_missing_policy_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(str(tmp_path / "nope.md"))


def test_an_empty_policy_is_fatal(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(str(p))


# -- learning on stop -------------------------------------------------------

def test_the_last_pass_state_is_kept_for_the_trace(cfg, tmp_path):
    """Whoever closes the trace off on stop needs a state to read."""
    watch, slept, goals = build(cfg, distinct(8), ["success", "success"],
                                tmp_path=tmp_path)
    assert watch.last_state is None
    watch.run("watch instagram dms", max_passes=1)
    assert watch.last_state.step == 1
    watch.run("watch instagram dms", max_passes=2)
    assert watch.last_state.step == 2, "it kept the first pass's state, not the last"


def test_a_crashing_pass_leaves_the_previous_state_alone(cfg, tmp_path):
    """A pass that raised has no state; the last good one must survive."""
    watch, slept, goals = build(cfg, distinct(8),
                                ["success", RuntimeError("boom")],
                                tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=2)
    assert watch.last_state.step == 1


def test_a_trace_accumulates_across_passes(cfg, tmp_path):
    """One collector for the whole watch, so the skill is learned from all of it.

    Not per pass: rewriting the file the next pass obeys every 45 seconds, mostly
    from passes that did nothing, is the churn this avoids.
    """
    from adbagent.skills import AppTrace, TraceCollector

    class Dev:
        def screenshot(self):
            return b""

    trace = TraceCollector(Dev(), AppTrace(tasks="watch dms"))
    # Two passes over the same two screens, as a watch really does.
    for _pass in range(2):
        for s in (chat(), chat(messages=["a", "b"])):
            trace(kind="step", step=1, screen=s, action=None)
    per_app = trace.app_traces()
    assert per_app[0].package == "com.instagram.android"
    # Steps counted across both passes...
    assert per_app[0].steps == 4
    # ...screens deduped on the content-free skeleton, so they stay bounded.
    assert len(per_app[0].screens) < 4


def test_the_action_list_is_capped_when_asked(cfg, tmp_path):
    """A week of passes must not grow one list per step without bound."""
    from adbagent.actions import AgentAction
    from adbagent.skills import AppTrace, TraceCollector

    class Dev:
        def screenshot(self):
            return b""

    act = AgentAction(observation="x", reasoning="y", action="press_key",
                      key="back")
    trace = TraceCollector(Dev(), AppTrace(), max_actions=10)
    screen = chat()
    for i in range(50):
        trace(kind="step", step=i, screen=screen, action=act)
    assert len(trace.trace.actions) == 10
    assert all(len(v) <= 10 for v in trace.actions_in.values())


def test_no_cap_by_default_so_runs_are_unchanged():
    from adbagent.actions import AgentAction
    from adbagent.skills import AppTrace, TraceCollector

    class Dev:
        def screenshot(self):
            return b""

    act = AgentAction(observation="x", reasoning="y", action="press_key",
                      key="back")
    trace = TraceCollector(Dev(), AppTrace())
    screen = chat()
    for i in range(30):
        trace(kind="step", step=i, screen=screen, action=act)
    assert len(trace.trace.actions) == 30


# -- a person taking the phone, across passes --------------------------------

def test_a_takeover_in_one_pass_is_remembered_by_the_watch(cfg, tmp_path):
    """A watch has one `RunState` per pass and one trace across all of them.

    So the flag cannot live on the state the way it does for a goal run: asking
    the last pass whether a person took the phone asks the wrong pass. Here the
    takeover is in pass 1 and three more passes run after it.
    """
    frames = [chat(messages=[f"m{i}"]) for i in range(8)]
    watch, _slept, goals = build(cfg, frames, ["success"] * 4,
                                 tmp_path=tmp_path, takeovers=[1])
    watch.run("watch instagram dms", max_passes=4)
    assert len(goals) >= 2, "the watch never ran a pass after the takeover"
    assert watch.took_over is True


def test_a_watch_nobody_touched_stays_learnable(cfg, tmp_path):
    frames = [chat(messages=[f"m{i}"]) for i in range(8)]
    watch, _slept, _goals = build(cfg, frames, ["success"] * 4,
                                  tmp_path=tmp_path)
    watch.run("watch instagram dms", max_passes=4)
    assert watch.took_over is False


def test_the_trace_takes_the_watchs_word_over_the_last_passs(cfg, tmp_path):
    """`finish` is given both, and the watch's is the one that spans the run."""
    from adbagent.skills import AppTrace, TraceCollector

    class Dev:
        def screenshot(self):
            return b""

    collector = TraceCollector(Dev(), AppTrace(package="com.instagram.android"))
    last_pass = StubState(40, took_over=False)      # forty passes later

    collector.finish("stopped", last_pass, took_over=True)
    assert collector.trace.took_over is True
    # And every app the watch worked in inherits it.
    collector.steps_in = {"com.instagram.android": 30, "com.whatsapp": 5}
    collector.screens_in = {"com.instagram.android": ["a"], "com.whatsapp": ["b"]}
    collector.actions_in = {"com.instagram.android": ["x"], "com.whatsapp": ["y"]}
    collector.shots_in = {}
    assert [t.took_over for t in collector.app_traces()] == [True, True]

    # Without it, an untouched watch is still learnable.
    clean = TraceCollector(Dev(), AppTrace(package="com.instagram.android"))
    clean.finish("stopped", StubState(40))
    assert clean.trace.took_over is False
