"""Continuing a failed run where it stopped.

A run keeps everything it knows in `RunState`: the history the model reads, the
scratchpad of collected data, the gallery ledger, the loop detector's bans. A
failure used to throw all of it away. These tests pin the serialisation that
saves it to `runs/<id>/checkpoint.json`, and the resume that puts it back.
"""

from __future__ import annotations

import json

import pytest

from adbagent import checkpoint, runlog
from adbagent.actions import AgentAction
from adbagent.agent import Agent, RunState
from adbagent.config import Config
from adbagent.memory import Memory
from adbagent.pager import ItemRecord

from . import fake

GOAL = "open the Wi-Fi settings screen"


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


def populated_state() -> RunState:
    """A RunState with something in every ledger the run keeps."""
    state = RunState(goal=GOAL, run_id="abc123def456", intent_id="intent1")
    state.step = 7
    state.llm_calls = 9
    state.consecutive_failures = 2
    state.remember("1. tapped 'Wi-Fi'")
    state.remember("2. swiped left on the pager")
    state.visits["skel1"] = 3
    state.visits["skel2"] = 1
    state.loops.record("exact1", "tap/#4")
    state.loops.ban("skel1", "scroll/#2/down")
    state.loops.record_element_action("skel1", 4, "tap/#4", "tap #4 'Wi-Fi'")
    state.loops.record_scroll("down")
    state.loops.record_scroll("up")
    state.loops.consecutive_backs = 1
    state.want_screenshot = True
    state.last_failure = "tap #4 failed: nothing changed"
    state.scroll_warnings = 1
    state.scratchpad.update([{"key": "9:30", "value": "water 275g"}], step=3)
    state.scratchpad.update([{"key": "9:30", "value": "water 280g"}], step=5)
    state.progress_log.append("read 3 of 5 items")
    state.packages.add("com.android.settings")
    state.package_steps["com.android.settings"] = 6
    state.items.items["photo1"] = ItemRecord(
        key="photo1", label="first photo", first_step=2, last_step=2,
        read=True, detail="a cat on a scale", visits=1)
    state.items.items["photo2#2"] = ItemRecord(
        key="photo2", label="second photo (#2)", first_step=3, last_step=4,
        read=False, detail="", visits=2)
    state.items.total = 5
    state.items.set_id = "set-1"
    state.items.cursor = "photo2#2"
    state.items.edges.add("left")
    return state


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_round_trip_preserves_everything_the_loop_needs(cfg):
    state = populated_state()
    checkpoint.save(cfg, state)

    data = checkpoint.load(runlog.run_dir(cfg, state.run_id))
    assert data is not None
    assert data["goal"] == GOAL
    assert data["run_id"] == state.run_id

    fresh = RunState(goal=GOAL, run_id=state.run_id, intent_id=state.intent_id)
    checkpoint.restore(fresh, data)

    assert fresh.step == 7
    assert fresh.llm_calls == 9
    assert fresh.consecutive_failures == 2
    assert fresh.history == state.history
    assert fresh.visits == {"skel1": 3, "skel2": 1}
    assert fresh.loops.history == [("exact1", "tap/#4")]
    assert fresh.loops.banned == {"skel1": {"scroll/#2/down"}}
    assert fresh.loops.element_actions == {
        "skel1": [(4, "tap/#4", "tap #4 'Wi-Fi'")]}
    assert fresh.loops.scroll_dir_log == ["down", "up"]
    assert fresh.loops.total_scroll_count == 2
    assert fresh.loops.consecutive_backs == 1
    assert fresh.want_screenshot is True
    assert fresh.last_failure == state.last_failure
    assert fresh.scroll_warnings == 1
    assert fresh.progress_log == ["read 3 of 5 items"]
    assert fresh.packages == {"com.android.settings"}
    assert fresh.package_steps == {"com.android.settings": 6}

    # The scratchpad keeps both the current value and the one it replaced.
    rendered = fresh.scratchpad.plain()
    assert "9:30: water 280g" in rendered
    assert "water 275g" in rendered          # the superseded reading survives

    # The item ledger keeps its keys verbatim: a `#2` suffix is how two items
    # sharing a caption stay apart, and losing it would merge them.
    assert set(fresh.items.items) == {"photo1", "photo2#2"}
    assert fresh.items.items["photo1"].read is True
    assert fresh.items.items["photo1"].detail == "a cat on a scale"
    assert fresh.items.items["photo2#2"].key == "photo2"
    assert fresh.items.items["photo2#2"].visits == 2
    assert fresh.items.total == 5
    assert fresh.items.set_id == "set-1"
    assert fresh.items.cursor == "photo2#2"
    assert fresh.items.edges == {"left"}


def test_load_returns_none_when_there_is_nothing_to_load(cfg, tmp_path):
    assert checkpoint.load(tmp_path / "no-such-run") is None
    bare = tmp_path / "bare"
    bare.mkdir()
    assert checkpoint.load(bare) is None
    (bare / checkpoint.NAME).write_text("{not json")
    assert checkpoint.load(bare) is None


def test_restore_tolerates_a_sparse_checkpoint(cfg):
    """A checkpoint can outlive the code that wrote it; missing keys are not
    a reason to refuse the resume."""
    state = RunState(goal=GOAL, run_id="r1", intent_id="i1")
    checkpoint.restore(state, {"goal": GOAL, "step": 3})
    assert state.step == 3
    assert state.history == []
    assert len(state.scratchpad) == 0


def test_latest_resumable_picks_the_newest_checkpoint(cfg, tmp_path):
    runs = tmp_path / "runs"
    assert checkpoint.latest_resumable(runs) is None
    older, newer, done = runs / "older", runs / "newer", runs / "done"
    for d in (older, newer, done):
        d.mkdir(parents=True)
    (older / checkpoint.NAME).write_text("{}")
    (newer / checkpoint.NAME).write_text("{}")
    import os
    os.utime(older / checkpoint.NAME, (1, 1))
    # A run that succeeded has no checkpoint, so it is not a candidate even
    # when its directory is the newest thing on disk.
    assert checkpoint.latest_resumable(runs) == newer


# ---------------------------------------------------------------------------
# The loop, end to end
# ---------------------------------------------------------------------------

def test_a_failed_run_leaves_a_checkpoint_and_success_clears_it(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def gives_up(screen, llm):
        return AgentAction(observation="stuck", reasoning="cannot",
                           action="fail", text="giving up")

    outcome, state = Agent(dev, mem, fake.FakeLLM(dev, gives_up), cfg).run(GOAL)
    assert outcome == "failed"
    run_dir = runlog.run_dir(cfg, state.run_id)
    data = checkpoint.load(run_dir)
    assert data is not None
    assert data["goal"] == GOAL

    # Continue the same run, with a model that now knows the way.
    resumed_outcome, resumed = Agent(
        dev, mem, fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])),
        cfg).run(GOAL, run_id=state.run_id, resume=data)
    assert resumed_outcome == "success"
    assert resumed.step > data["step"]       # step numbers continue, not restart
    assert not (run_dir / checkpoint.NAME).exists()


def test_a_successful_run_leaves_no_checkpoint(cfg, mem):
    dev = fake.FakeDevice(cfg)
    outcome, state = Agent(
        dev, mem, fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])),
        cfg).run(GOAL)
    assert outcome == "success"
    assert not (runlog.run_dir(cfg, state.run_id) / checkpoint.NAME).exists()


def test_resume_restores_what_the_run_had_learned(cfg, mem):
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 2        # tiny budget, so the first sitting fails

    def wanders(screen, llm):
        return AgentAction(observation="browsing", reasoning="not done yet",
                           action="scroll", direction="down")

    outcome, state = Agent(dev, mem, fake.FakeLLM(dev, wanders), cfg).run(GOAL)
    assert outcome == "failed"
    assert state.step == 2

    data = checkpoint.load(runlog.run_dir(cfg, state.run_id))
    cfg.run.max_steps = 25
    outcome, resumed = Agent(
        dev, mem, fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])),
        cfg).run(GOAL, run_id=state.run_id, resume=data)
    assert outcome == "success"
    # The resumed run still holds the failed sitting's history: the model is
    # shown it, rather than rediscovering the first two steps.
    assert resumed.history[:len(data["history"])] == data["history"]
    assert len(resumed.history) > len(data["history"])
    assert resumed.step > 2
    assert resumed.llm_calls > data["llm_calls"]


def test_a_resumed_run_appends_to_its_own_trace(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def gives_up(screen, llm):
        return AgentAction(observation="stuck", reasoning="cannot",
                           action="fail", text="giving up")

    _, state = Agent(dev, mem, fake.FakeLLM(dev, gives_up), cfg).run(GOAL)
    data = checkpoint.load(runlog.run_dir(cfg, state.run_id))
    Agent(dev, mem, fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])),
          cfg).run(GOAL, run_id=state.run_id, resume=data)

    events = [json.loads(line) for line in
              (runlog.run_dir(cfg, state.run_id) / "events.jsonl")
              .read_text().splitlines() if line.strip()]
    kinds = [e["kind"] for e in events]
    # One run, one file: the sittings join at run_resume, and the final
    # run_end is the one report shows.
    assert kinds[0] == "run_start"
    assert "run_resume" in kinds
    assert kinds.count("run_end") == 2
    ends = [e for e in events if e["kind"] == "run_end"]
    assert ends[-1]["outcome"] == "success"
    assert ends[-1]["steps"] >= ends[0]["steps"]


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------

def parse(argv):
    from adbagent.cli import build_parser
    return build_parser().parse_args(argv)


def test_resume_parses_with_and_without_a_target():
    args = parse(["run", "--resume"])
    assert args.resume == "latest"
    assert args.goal is None
    args = parse(["run", "--resume", "abc123"])
    assert args.resume == "abc123"
    args = parse(["run", "a goal", "--resume", "abc123"])
    assert args.goal == "a goal" and args.resume == "abc123"


def test_resolve_resume(tmp_path, monkeypatch):
    from adbagent.cli import _resolve_resume

    runs = tmp_path / "runs"
    run_dir = runs / "abc123"
    run_dir.mkdir(parents=True)
    (run_dir / checkpoint.NAME).write_text("{}")

    assert _resolve_resume("abc123", str(runs)) == run_dir
    assert _resolve_resume(str(run_dir), str(runs)) == run_dir
    assert _resolve_resume("latest", str(runs)) == run_dir
    assert _resolve_resume("missing", str(runs)) is None
    monkeypatch.chdir(tmp_path)


def test_cmd_run_resume_reports_missing_checkpoint(tmp_path, capsys):
    from adbagent.cli import cmd_run

    run_dir = tmp_path / "runs" / "abc123"
    run_dir.mkdir(parents=True)      # a run with no checkpoint in it
    args = parse(["run", "--resume", str(run_dir),
                  "--artifacts-dir", str(tmp_path / "runs")])
    assert cmd_run(args) == 1
    assert "no checkpoint" in capsys.readouterr().out


def test_cmd_run_resume_latest_with_nothing_resumable(tmp_path, capsys):
    from adbagent.cli import cmd_run

    args = parse(["run", "--resume", "--artifacts-dir", str(tmp_path / "runs")])
    assert cmd_run(args) == 1
    assert "no resumable run" in capsys.readouterr().out


def test_report_shows_the_final_outcome_of_a_resumed_run(tmp_path, capsys):
    from adbagent.cli import cmd_report

    run_dir = tmp_path / "runs" / "res"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "g", "model": "m"},
        {"t": 2, "kind": "run_end", "outcome": "failed", "steps": 3,
         "llm_calls": 3, "usd": 0.001},
        {"t": 3, "kind": "run_resume", "goal": "g", "resumed_at_step": 3},
        {"t": 4, "kind": "run_end", "outcome": "success", "steps": 5,
         "llm_calls": 6, "usd": 0.002},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    out = capsys.readouterr().out
    assert "SUCCESS" in out
    assert "5 steps" in out


def test_history_counts_a_resumed_runs_packages_once():
    """Two run_end events are one run, not two: the first sitting's steps are
    inside the second's totals already. Summing both would double the apparent
    time in the first app and push the second under the dominance threshold."""
    from adbagent.history import packages_in

    events = [
        # The first sitting: 30 steps, all in com.b, then the budget ran out.
        {"kind": "run_end", "packages": ["com.b"],
         "package_steps": {"com.b": 30}},
        # The resumed sitting finished in com.a; its counts are cumulative.
        {"kind": "run_end", "packages": ["com.a", "com.b"],
         "package_steps": {"com.a": 10, "com.b": 30}},
    ]
    assert packages_in(events) == {"com.a", "com.b"}
