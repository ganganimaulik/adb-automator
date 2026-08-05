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


class StubAgent:
    """Returns scripted outcomes; records the goals it was given."""

    def __init__(self, outcomes, seen_goals, spend=0.0, llm=None):
        self.outcomes = outcomes
        self.seen_goals = seen_goals
        self.spend = spend
        self.llm = llm

    def run(self, goal, run_id="", resume=None):
        self.seen_goals.append(goal)
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        if self.llm is not None:
            self.llm.ledger.total_usd += self.spend
        # BaseException, not Exception: KeyboardInterrupt is not an Exception,
        # and testing that Ctrl-C escapes the loop depends on raising it here.
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, None


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


def build(cfg, frames, outcomes=(), spend=0.0, ledger_path=None, tmp_path=None):
    """A Watch wired to stubs, plus the lists that record what it did."""
    llm = StubLLM()
    slept, goals = [], []
    agent = StubAgent(list(outcomes), goals, spend=spend, llm=llm)
    watch = Watch(StubDevice(frames), None, llm, cfg,
                  policy="be brief",
                  ledger=ReplyLedger(ledger_path or (tmp_path / "l.jsonl")),
                  make_agent=lambda: agent,
                  sleep=slept.append)
    return watch, slept, goals


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
    assert "Finish on the conversation list" in goal


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
    watch.run("watch instagram dms", max_passes=4)
    # Two passes at $0.06 clear the $0.10 ceiling, so the loop must pause rather
    # than start a third -- even though every probe is finding fresh work.
    assert len(goals) == 2
    assert watch.stats.paused > 0
    assert any(s > cfg.watch.interval_s for s in slept)


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
