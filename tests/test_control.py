"""Holding a run that is already going.

Everything a run says travelled outward through files and nothing came back:
the whole vocabulary from outside was a SIGINT. These tests pin the file that
carries a command in, the point in the loop that reads it, and the two things
that make a pause honest rather than a hang -- the time it gives back, and the
stop that still works while it is held.
"""

from __future__ import annotations

import json

import pytest

from adbagent import checkpoint, control, runlog
from adbagent.actions import AgentAction
from adbagent.agent import Agent, RunState
from adbagent.config import Config
from adbagent.memory import Memory

from . import fake

GOAL = "open the Wi-Fi settings screen"


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


def _events(tmp_path, run_id):
    path = tmp_path / "runs" / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def test_a_command_survives_the_trip(tmp_path):
    control.send(tmp_path, "pause", 1)
    command = control.read(tmp_path)
    assert command is not None
    assert (command.cmd, command.seq) == ("pause", 1)


def test_nothing_to_read_is_not_an_error(tmp_path):
    assert control.read(tmp_path) is None
    (tmp_path / control.NAME).write_text("{half a fi", encoding="utf-8")
    # A read landing between the write and the rename sees a partial file. The
    # answer is to look again in a quarter second, not to fail a run over it.
    assert control.read(tmp_path) is None
    (tmp_path / control.NAME).write_text('{"seq": 1, "cmd": "explode"}',
                                         encoding="utf-8")
    assert control.read(tmp_path) is None


def test_send_refuses_a_command_nobody_implements(tmp_path):
    with pytest.raises(ValueError):
        control.send(tmp_path, "explode", 1)
    assert control.read(tmp_path) is None


def test_send_leaves_no_temporary_file_behind(tmp_path):
    control.send(tmp_path, "pause", 1)
    control.send(tmp_path, "run", 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [control.NAME]


def test_clear_is_safe_on_a_directory_with_nothing_in_it(tmp_path):
    control.clear(tmp_path)          # must not raise
    control.send(tmp_path, "pause", 1)
    control.clear(tmp_path)
    assert control.read(tmp_path) is None


# ---------------------------------------------------------------------------
# The waiting
# ---------------------------------------------------------------------------

class Clock:
    """A sleep that does not, and a monotonic that moves when it is called."""

    def __init__(self, on_sleep=None):
        self.slept = 0.0
        self._on_sleep = on_sleep

    def sleep(self, seconds):
        self.slept += seconds
        if self._on_sleep is not None:
            self._on_sleep(self.slept)


def test_an_untouched_run_never_waits(tmp_path):
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    clock = Clock()
    said = []
    ctl = control.Control(tmp_path, sleep=clock.sleep)
    for _ in range(3):
        ctl.wait(state, said.append)
    assert clock.slept == 0.0
    assert said == []                # nothing happened, so nothing is announced
    assert state.paused_s == 0.0


def test_a_pause_holds_until_it_is_let_go(tmp_path):
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []

    def release(total):
        if total >= 1.0:
            control.send(tmp_path, "run", 2)

    clock = Clock(on_sleep=release)
    ctl = control.Control(tmp_path, sleep=clock.sleep)
    control.send(tmp_path, "pause", 1)
    ctl.wait(state, said.append)

    assert said == ["pause", "run"]
    assert clock.slept >= 1.0
    assert ctl.mode == "run"


def test_time_spent_held_is_given_back(tmp_path):
    """The wall-clock budget bounds how long a run may *work*. A run held for
    five minutes by the person reading it has not worked for five minutes, and
    without this, resuming one is how you find out it was killed meanwhile."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")

    def release(total):
        control.send(tmp_path, "run", 2)

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=release).sleep)
    control.send(tmp_path, "pause", 1)

    before = state.elapsed
    ctl.wait(state, lambda _m: None)
    assert state.paused_s > 0
    # Real time passed, and none of it counted against the budget.
    assert state.elapsed <= before + 0.05


def test_a_step_runs_one_and_holds_again(tmp_path):
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []
    ctl = control.Control(tmp_path, sleep=Clock().sleep)

    control.send(tmp_path, "step", 1)
    ctl.wait(state, said.append)       # spends it and returns at once
    assert said == ["step"]

    # And the call after it is the one that holds. Released from inside so the
    # test does not depend on a real clock.
    def release(_total):
        control.send(tmp_path, "run", 2)

    ctl._sleep = Clock(on_sleep=release).sleep
    ctl.wait(state, said.append)
    assert said == ["step", "pause", "run"]


def test_two_steps_in_a_row_are_two_things_that_happened(tmp_path):
    """A state machine that deduplicated them would show the second step as
    nothing happening at all.

    Each one comes back to the hold it was taken from, and the trace says so:
    stepping through a run is a sequence of holds with one step between each,
    which is what it looks like from the outside too.
    """
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []
    n = {"i": 0}

    def step_again(_total):
        n["i"] += 1
        control.send(tmp_path, "step", 10 + n["i"])

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=step_again).sleep)
    control.send(tmp_path, "step", 1)
    ctl.wait(state, said.append)
    ctl.wait(state, said.append)
    ctl.wait(state, said.append)
    assert said == ["step", "pause", "step", "pause", "step"]
    assert said.count("step") == 3


def test_a_command_is_obeyed_once(tmp_path):
    """The file stays on disk after it is read, and is read again every quarter
    second. Re-announcing it would fill the trace with a pause nobody asked for
    twice."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []

    def release(_total):
        control.send(tmp_path, "run", 2)

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=release).sleep)
    control.send(tmp_path, "pause", 1)
    ctl.wait(state, said.append)
    assert said == ["pause", "run"]
    for _ in range(3):
        ctl.wait(state, said.append)   # the same `run` file, still there
    assert said == ["pause", "run"]


def test_a_stale_command_does_not_reach_the_next_sitting(tmp_path):
    """A resumed run reuses its directory. What the last sitting was told is
    not an instruction to this one."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    control.send(tmp_path, "pause", 7)
    ctl = control.Control(tmp_path, sleep=Clock().sleep)
    ctl.seen = 7                      # as if this sitting had already read it
    ctl.wait(state, lambda _m: None)
    assert state.paused_s == 0.0


def test_a_pause_can_still_be_interrupted(tmp_path):
    """Stopping is what restores the phone's keyboard, animations and rotation,
    so it has to reach a run that is holding -- which, with a plain sleep loop
    and nothing catching it, it does."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")

    def interrupt(_total):
        raise KeyboardInterrupt

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=interrupt).sleep)
    control.send(tmp_path, "pause", 1)
    with pytest.raises(KeyboardInterrupt):
        ctl.wait(state, lambda _m: None)
    # And the time it was held still gets handed back on the way out.
    assert state.paused_s >= 0.0


# ---------------------------------------------------------------------------
# In the loop
# ---------------------------------------------------------------------------

def test_the_loop_reads_its_control_file_and_says_so(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    run_dir = tmp_path / "runs"
    seen = {"paused": False}

    def policy(screen, llm):
        # Ask for a pause on the first turn; release it from inside the sleep.
        if llm.calls == 1:
            for d in run_dir.iterdir():
                control.send(d, "pause", 1)
        return fake.reach_state(dev, "wifi", ["Wi-Fi"])(screen, llm)

    def release(_seconds):
        seen["paused"] = True
        for d in run_dir.iterdir():
            control.send(d, "run", 2)

    original = control.Control.__init__

    def patched(self, directory, sleep=None):
        original(self, directory, sleep=release)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(control.Control, "__init__", patched)
        outcome, state = Agent(dev, mem, fake.FakeLLM(dev, policy), cfg).run(GOAL)

    assert outcome == "success"
    assert seen["paused"], "the loop never held"
    modes = [e["mode"] for e in _events(tmp_path, state.run_id)
             if e["kind"] == "control"]
    assert "pause" in modes and "run" in modes


def test_a_run_clears_its_control_file_at_both_ends(cfg, mem, tmp_path):
    """On the way in, so a resume does not inherit what the last sitting was
    told; on the way out, so a run stopped while held leaves no `pause` for the
    next one to find."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert control.read(runlog.run_dir(cfg, state.run_id)) is None

    # Now leave one behind and resume: the second sitting must not act on it.
    run_dir = runlog.run_dir(cfg, state.run_id)
    control.send(run_dir, "pause", 99)
    data = checkpoint.load(run_dir) or {"goal": GOAL, "step": 1}
    outcome, resumed = Agent(dev, mem, fake.FakeLLM(
        dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])), cfg).run(
            GOAL, run_id=state.run_id, resume=data)
    assert outcome == "success"
    assert resumed.paused_s == 0.0


def test_the_run_reports_how_long_it_was_held(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    end = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "run_end"][-1]
    # Recorded on every run, held or not: otherwise the wall clock in the trace
    # and the one on the page disagree with nothing in the file to say why.
    assert end["paused_s"] == 0.0


# ---------------------------------------------------------------------------
# Giving the phone back
# ---------------------------------------------------------------------------

def test_release_is_a_way_of_being_held(tmp_path):
    """A run that carried on while somebody was typing on the phone would be
    reading their taps as its own, so the two cannot come apart."""
    assert "release" in control.COMMANDS
    assert control.HELD == {"pause", "release"}

    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []

    def release_then_resume(total):
        control.send(tmp_path, "run", 3) if total >= 1.0 else None

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=release_then_resume).sleep)
    control.send(tmp_path, "release", 2)
    ctl.wait(state, said.append)
    assert said == ["release", "run"]
    assert state.paused_s > 0        # the time it was theirs is not the run's


def test_moving_between_the_two_holds_is_said_out_loud(tmp_path):
    """Taking the phone back without letting the run go is a real state change:
    the session reopens, and the handler that reopens it runs off this."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    said = []
    seq = {"n": 2}

    def script(_total):
        seq["n"] += 1
        control.send(tmp_path, "pause" if seq["n"] == 3 else "run", seq["n"])

    ctl = control.Control(tmp_path, sleep=Clock(on_sleep=script).sleep)
    control.send(tmp_path, "release", 2)
    ctl.wait(state, said.append)
    assert said == ["release", "pause", "run"]


def test_the_loop_closes_and_reopens_the_session_around_a_takeover(cfg, mem,
                                                                   tmp_path):
    dev = fake.FakeDevice(cfg)
    run_dir = tmp_path / "runs"

    def policy(screen, llm):
        if llm.calls == 1:
            for d in run_dir.iterdir():
                control.send(d, "release", 1)
        return fake.reach_state(dev, "wifi", ["Wi-Fi"])(screen, llm)

    def give_it_back(_seconds):
        for d in run_dir.iterdir():
            control.send(d, "run", 2)

    original = control.Control.__init__

    def patched(self, directory, sleep=None):
        original(self, directory, sleep=give_it_back)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(control.Control, "__init__", patched)
        outcome, state = Agent(dev, mem, fake.FakeLLM(dev, policy), cfg).run(GOAL)

    assert outcome == "success"
    # Closed to hand it over -- which is what puts the keyboard, the animations
    # and the rotation back -- and reopened to take it on again.
    assert dev.closes >= 1 and dev.opens >= 1
    # And it ends holding the phone rather than having given it away: the run
    # carried on for a step after the handback, which it could not have done
    # without the session. (Closing it for good is the caller's `with Device`,
    # which this test does not go through.)
    assert dev.session_open is True
    kinds = [e["kind"] for e in _events(tmp_path, state.run_id)]
    assert "released" in kinds and "reclaimed" in kinds
    # And the run knows it cannot be learned from.
    assert state.took_over is True


def test_a_takeover_tells_the_model_it_happened(cfg, mem, tmp_path):
    """Otherwise the model reads the new screen as the result of its own last
    action, and concludes that whatever it did worked."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    state.step = 4
    state.steps_since_progress = 6
    dev = fake.FakeDevice(cfg)
    agent = Agent(dev, mem, fake.FakeLLM(dev, lambda s, l: None), cfg)

    class Rec:
        def __init__(self): self.events = []
        def event(self, kind, **kw): self.events.append((kind, kw))

    rec = Rec()
    agent._released = True
    agent._reclaim_phone(state, rec)

    assert any("took the phone" in line for line in state.history)
    # And the stall ladder resets: somebody intervening is the most likely thing
    # to have unstuck a run, and coming back to tier four would have the harness
    # break the loop it was just rescued from.
    assert state.steps_since_progress == 0
    assert agent._reobserve is True


def test_a_run_somebody_drove_teaches_the_skill_nothing(cfg, mem, tmp_path):
    """The trace records every step as the agent's, so the steps around a
    takeover describe a path the agent never found."""
    from adbagent.skills import AppTrace, SkillRegistry, learn_from_run

    trace = AppTrace(package="com.example.app", steps=20,
                     screens=["a", "b", "c"], actions=["tap #1", "tap #2"],
                     took_over=True)
    registry = SkillRegistry(str(tmp_path / "skills"))

    def explode(*a, **kw):
        raise AssertionError("the synthesis must not be reached")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("adbagent.skills.SkillGenerator.generate_from_exploration",
                   explode)
        assert learn_from_run(trace, object(), registry, goal=GOAL) is None


def test_the_takeover_flag_reaches_every_app_the_run_touched(cfg):
    """The person may have opened one app to unblock another, and neither
    trace can say which of its steps were theirs."""
    from adbagent.skills import AppTrace, TraceCollector

    collector = TraceCollector(None)
    collector.trace = AppTrace(took_over=True)
    collector.steps_in = {"com.one.app": 5, "com.two.app": 3}
    collector.screens_in = {"com.one.app": ["a"], "com.two.app": ["b"]}
    collector.actions_in = {"com.one.app": ["x"], "com.two.app": ["y"]}
    collector.shots_in = {}
    assert [t.took_over for t in collector.app_traces()] == [True, True]


def test_an_ordinary_run_still_teaches(cfg, mem, tmp_path):
    """The guard must be about takeovers and nothing else."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert state.took_over is False


def test_a_terminal_action_is_not_a_step_the_loop_can_be_held_before(cfg, mem,
                                                                     tmp_path):
    """The read happens at the top of a step, so a run that ends on its first
    decision is never held -- and must not hang waiting to be."""
    dev = fake.FakeDevice(cfg)

    def gives_up(screen, llm):
        return AgentAction(observation="stuck", reasoning="cannot",
                           action="fail", text="giving up")

    outcome, state = Agent(dev, mem, fake.FakeLLM(dev, gives_up), cfg).run(GOAL)
    assert outcome == "failed"
    assert state.paused_s == 0.0
