"""What the run answered, and whether anyone can see it.

The text of the terminal action is the answer to every "read X and tell me"
goal, and it used to live in exactly one place: inside the last line of the step
feed, formatted like the forty tap lines above it. On anything longer than a
couple of steps that has scrolled off by the time the run ends, so the visible
ending of a run was the outcome word and the bill.

These tests hold the answer to three promises: the run carries it out of
`Agent.run`, it is written into `run_end` so a report a week later still has it,
and both the live command and `report` print it under a heading of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adbagent.actions import AgentAction
from adbagent.agent import Agent, Oracle
from adbagent.cli import Out, _result_block, build_parser, cmd_report
from adbagent.config import Config
from adbagent.memory import Memory

from . import fake


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


GOAL = "open the Wi-Fi settings screen and tell me what it says"


def parse(argv):
    return build_parser().parse_args(argv)


def events_of(cfg, run_id):
    path = Path(cfg.run.artifacts_dir) / run_id / "events.jsonl"
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The run carries its answer out
# ---------------------------------------------------------------------------

def test_the_run_keeps_what_it_answered(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="go",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="arrived", reasoning="read it",
                           action="done",
                           text="Wi-Fi is on and connected to Home-5G.")

    outcome, state = Agent(dev, mem, fake.FakeLLM(dev, policy), cfg).run(GOAL)
    assert outcome == "success"
    assert state.result == "Wi-Fi is on and connected to Home-5G."
    assert state.evidence == "fake judge"


def test_the_answer_is_written_into_run_end(cfg, mem):
    """A report a week later reconstructs the run from this file alone."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    end = [e for e in events_of(cfg, state.run_id) if e["kind"] == "run_end"][-1]
    assert end["result"] == "reached wifi"
    assert end["evidence"] == "fake judge"


def test_a_rejected_done_is_not_the_answer(cfg, mem):
    """A completion the judge threw out is not what the run concluded.

    Keeping it would have a run that went on for another thirty steps and then
    ran out of budget report the summary it was told was wrong -- with the
    outcome word FAILED next to it.
    """
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 3

    def policy(screen, llm):
        return AgentAction(observation="home", reasoning="claiming early",
                           action="done", text="all finished, nothing to do")

    llm = fake.FakeLLM(dev, policy, judge_result=False)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "failed"
    assert state.result == ""
    # The rejection is what there is to say instead, and it is kept.
    assert "rejected" in state.evidence


def test_giving_up_says_why(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        return AgentAction(observation="home", reasoning="no way through",
                           action="fail",
                           text="the Wi-Fi row is not on this screen")

    outcome, state = Agent(dev, mem, fake.FakeLLM(dev, policy), cfg).run(GOAL)
    assert outcome == "failed"
    assert state.result == "the Wi-Fi row is not on this screen"


def test_an_assertion_that_settles_it_says_which_check_passed(cfg, mem, capsys):
    """The oracle ends the run at the top of the loop, before the model is asked
    anything, so there is no summary and never will be. The condition that
    passed is the whole answer this path has -- without it the ending is the
    word SUCCESS over an empty block."""
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="go",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="?", reasoning="back",
                           action="press_key", key="back")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg,
                           oracle=Oracle(text="Forget network")).run(GOAL)
    assert outcome == "success"
    assert llm.judges == 0
    assert state.result == ""
    assert "Forget network" in state.evidence

    capsys.readouterr()                      # drop whatever the run printed
    _result_block(Out(), outcome, state.result, state.evidence)
    text = capsys.readouterr().out
    assert "success check settled it" in text
    assert "Forget network" in text


# ---------------------------------------------------------------------------
# Showing it
# ---------------------------------------------------------------------------

def test_the_block_prints_the_answer_under_its_own_heading(capsys):
    _result_block(Out(), "success", "The cheapest is Amul, 62 rupees a litre.",
                  "the price is visible on the product card")
    text = capsys.readouterr().out
    assert "── Result ──" in text
    assert "The cheapest is Amul, 62 rupees a litre." in text
    assert "the price is visible on the product card" in text


def test_a_long_answer_is_wrapped_rather_than_run_off_the_screen(capsys):
    _result_block(Out(), "success", " ".join(["word"] * 200))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) > 3                       # it did not print one long line
    assert all(len(line) <= 100 for line in lines)


def test_a_multi_line_answer_keeps_its_lines(capsys):
    _result_block(Out(), "success", "Pousali: two messages today\nYV: one, Wednesday")
    text = capsys.readouterr().out
    assert "  Pousali: two messages today" in text.splitlines()
    assert "  YV: one, Wednesday" in text.splitlines()


def test_a_run_that_never_answered_says_so_and_falls_back(capsys):
    """An empty heading is worse than no heading: it reads like a bug."""
    _result_block(Out(), "failed", "", progress="Done: opened Settings.",
                  problem="the search field would not take text")
    text = capsys.readouterr().out
    assert "without an answer" in text
    assert "ran out of steps" in text
    assert "last progress: Done: opened Settings." in text
    assert "last problem: the search field would not take text" in text


def test_report_shows_the_recorded_answer(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "answered"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "what is the wifi name", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1, "action": {"action": "done"}},
        {"t": 3, "kind": "run_end", "outcome": "success", "steps": 1,
         "llm_calls": 2, "usd": 0.001, "result": "The network is Home-5G.",
         "evidence": "the SSID is on screen"},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    text = capsys.readouterr().out
    assert "── Result ──" in text
    assert "The network is Home-5G." in text
    assert "the SSID is on screen" in text


def test_report_recovers_the_answer_from_a_run_recorded_before_run_end_had_it(
        tmp_path, capsys):
    """Every run already on disk. The text was always in the terminal action."""
    run_dir = tmp_path / "runs" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "g", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1, "action": {"action": "tap"}},
        {"t": 3, "kind": "decide", "step": 2,
         "action": {"action": "done", "text": "Battery is at 41 percent."}},
        {"t": 4, "kind": "run_end", "outcome": "success", "steps": 2,
         "llm_calls": 2, "usd": 0.001},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    assert "Battery is at 41 percent." in capsys.readouterr().out


def test_report_on_a_run_that_stopped_without_answering(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "stuck"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "g", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1,
         "action": {"action": "tap", "progress": "Done: found the list."}},
        {"t": 3, "kind": "judge", "step": 1, "satisfied": False,
         "evidence": "the value is still not on screen"},
        {"t": 4, "kind": "run_end", "outcome": "failed", "steps": 1,
         "llm_calls": 2, "usd": 0.001, "result": "", "evidence": ""},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    text = capsys.readouterr().out
    assert "without an answer" in text
    assert "last progress: Done: found the list." in text
    assert "the value is still not on screen" in text
