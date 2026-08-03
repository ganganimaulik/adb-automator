"""The replay harness: turning recorded runs into a regression set.

These tests build run directories by hand rather than driving the agent, because
the harness's contract is with the *artifacts* -- `events.jsonl` plus the
`step_NNN_*_messages.json` dumps -- and that contract has to hold for runs
recorded by earlier versions too.
"""

from __future__ import annotations

import json

import pytest

from adbagent.actions import AgentAction
from adbagent import replay as rp


# ---------------------------------------------------------------------------
# Fixtures: a run on disk
# ---------------------------------------------------------------------------

def messages(screen_text: str = "#1 [Button] \"Wi-Fi\"") -> list:
    return [
        {"role": "system", "content": "RECORDED SYSTEM PROMPT"},
        {"role": "user", "content": "Device: 720x1600 px"},
        {"role": "user", "content": "GOAL: open wifi"},
        {"role": "user", "content": [{"type": "text",
                                      "text": f"CURRENT SCREEN:\n{screen_text}"}]},
    ]


def action(name: str = "tap", index: int = 1, **kw) -> dict:
    return AgentAction(observation="o", reasoning="r", action=name,
                       target={"index": index} if index else None,
                       **kw).model_dump()


def write_run(tmp_path, steps, *, purpose: str = "decide", torn: bool = False):
    """`steps` is a list of (step, action_dict, grade). grade "" means unverified."""
    run = tmp_path / "runs" / "abc123"
    run.mkdir(parents=True)
    lines = [json.dumps({"t": 0, "kind": "run_start", "goal": "open wifi"})]
    for step, act, grade in steps:
        lines.append(json.dumps({"t": step, "kind": "decide", "step": step,
                                 "action": act}))
        if grade:
            lines.append(json.dumps({"t": step, "kind": "verify", "step": step,
                                     "grade": grade, "reason": ""}))
        (run / f"step_{step:03d}_{purpose}_messages.json").write_text(
            json.dumps(messages(f"#1 [Button] \"screen for step {step}\"")))
    body = "\n".join(lines) + "\n"
    if torn:
        body += '{"kind": "decide", "step": 99, "act'  # killed mid-write
    (run / "events.jsonl").write_text(body)
    return run


@pytest.fixture
def run(tmp_path):
    return write_run(tmp_path, [(1, action("tap", 1), "success"),
                                (2, action("scroll", 0, direction="down"), "no_change"),
                                (3, action("tap", 4), "success")])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_cases_pair_each_dump_with_the_action_it_produced(run):
    cases = rp.load_cases(run)
    assert [c.step for c in cases] == [1, 2, 3]
    assert cases[0].recorded["action"] == "tap"
    assert cases[1].recorded["direction"] == "down"
    assert cases[0].messages[0]["content"] == "RECORDED SYSTEM PROMPT"


def test_the_grade_the_recorded_action_earned_comes_along(run):
    by_step = {c.step: c for c in rp.load_cases(run)}
    assert by_step[1].grade == "success"
    assert by_step[2].grade == "no_change"
    # Diverging from a step that worked is a regression risk; diverging from one
    # that changed nothing is the point of the exercise.
    assert by_step[1].recorded_was_good
    assert not by_step[2].recorded_was_good


def test_a_terminal_step_has_no_grade_but_still_counts_as_good(tmp_path):
    run = write_run(tmp_path, [(1, action("done", 0, text="did it"), "")])
    case = rp.load_cases(run)[0]
    assert case.grade == ""
    assert case.recorded_was_good


def test_dumps_of_other_purposes_are_left_alone(tmp_path):
    run = write_run(tmp_path, [(1, action(), "success")])
    (run / "step_001_judge_messages.json").write_text(json.dumps(messages()))
    (run / "step_001_analyze_image_messages.json").write_text(json.dumps(messages()))
    assert [c.step for c in rp.load_cases(run, purpose="decide")] == [1]
    assert rp.load_cases(run, purpose="judge")[0].path.name.endswith(
        "judge_messages.json")


def test_a_dump_with_no_recorded_action_is_not_a_case(tmp_path):
    run = write_run(tmp_path, [(1, action(), "success")])
    # The call was made but the reply never validated: no baseline to diff.
    (run / "step_007_decide_messages.json").write_text(json.dumps(messages()))
    assert [c.step for c in rp.load_cases(run)] == [1]


def test_a_torn_final_event_line_does_not_lose_the_run(tmp_path):
    run = write_run(tmp_path, [(1, action(), "success"), (2, action(), "success")],
                    torn=True)
    assert [c.step for c in rp.load_cases(run)] == [1, 2]


def test_missing_events_is_a_clear_error(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(rp.ReplayError):
        rp.load_cases(empty)


def test_steps_can_be_selected(run):
    assert [c.step for c in rp.load_cases(run, steps=[1, 3])] == [1, 3]


def test_a_limit_samples_across_the_run_rather_than_truncating(tmp_path):
    run = write_run(tmp_path, [(i, action("tap", i), "success")
                               for i in range(1, 21)])
    steps = [c.step for c in rp.load_cases(run, limit=4)]
    assert len(steps) == 4
    assert steps[0] == 1
    # The interesting part of a long run is at the end. Truncating to the first
    # four would replay nothing but the opening navigation.
    assert steps[-1] >= 15


def test_events_jsonl_can_be_named_directly(run):
    assert len(rp.load_cases(run / "events.jsonl")) == 3


# ---------------------------------------------------------------------------
# Stubbed images
# ---------------------------------------------------------------------------

def test_a_dumped_image_is_recognised_as_a_placeholder():
    stubbed = messages()
    stubbed[3]["content"].append(
        {"type": "image_url",
         "image_url": {"url": "[base64 image payload: 91234 chars]"}})
    assert rp.has_stub_image(stubbed)
    assert not rp.has_stub_image(messages())


def test_a_real_data_url_is_not_a_placeholder():
    real = messages()
    real[3]["content"].append(
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4A"}})
    assert not rp.has_stub_image(real)


def test_cases_with_stubbed_images_are_skipped_not_sent(tmp_path):
    run = write_run(tmp_path, [(1, action(), "success"), (2, action(), "success")])
    stubbed = messages()
    stubbed[3]["content"].append(
        {"type": "image_url",
         "image_url": {"url": "[base64 image payload: 500 chars]"}})
    (run / "step_002_decide_messages.json").write_text(json.dumps(stubbed))

    sent = []

    def decide(msgs):
        sent.append(msgs)
        return AgentAction(observation="o", reasoning="r", action="tap",
                           target={"index": 1}), {}

    report = rp.replay(rp.load_cases(run), decide)
    assert report.skipped == [2]
    assert report.n == 1
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_identical_decisions_match_even_when_the_prose_differs():
    recorded = action("tap", 3)
    fresh = AgentAction(observation="totally different wording",
                        reasoning="also different", action="tap",
                        target={"index": 3}, notes="new notes")
    assert rp.compare(recorded, fresh) == "match"


def test_a_different_target_is_the_same_action():
    fresh = AgentAction(observation="o", reasoning="r", action="tap",
                        target={"index": 9})
    assert rp.compare(action("tap", 3), fresh) == "same_action"


def test_a_different_action_differs():
    fresh = AgentAction(observation="o", reasoning="r", action="press_key",
                        key="back")
    assert rp.compare(action("tap", 3), fresh) == "differs"


def test_the_same_scroll_the_other_way_is_not_a_match():
    recorded = action("scroll", 0, direction="down")
    fresh = AgentAction(observation="o", reasoning="r", action="scroll",
                        direction="up")
    assert rp.compare(recorded, fresh) == "same_action"


def test_scroll_distance_is_not_part_of_the_comparison():
    """`scroll_amount` is a tuning knob, not a decision: a faster scroll in the
    same direction is the same choice."""
    recorded = action("scroll", 0, direction="down", scroll_amount=1)
    fresh = AgentAction(observation="o", reasoning="r", action="scroll",
                        direction="down", scroll_amount=4)
    assert rp.compare(recorded, fresh) == "match"


def test_targets_named_differently_but_meaning_the_same_still_differ():
    """A recorded `#3` and a fresh `text="Wi-Fi"` may well resolve to the same
    element, but nothing here can know that -- there is no live screen. Reported
    as a divergence rather than guessed at."""
    fresh = AgentAction(observation="o", reasoning="r", action="tap",
                        target={"text": "Wi-Fi"})
    assert rp.compare(action("tap", 3), fresh) == "same_action"


def test_describe_action_is_short_and_comparable():
    assert rp.describe_action(action("tap", 3)) == "tap #3"
    assert rp.describe_action(
        action("scroll", 0, direction="up")) == "scroll up"
    assert rp.describe_action(
        action("press_key", 0, key="back")) == "press_key back"


# ---------------------------------------------------------------------------
# Rebuilding the system prompt
# ---------------------------------------------------------------------------

def test_rebuilding_swaps_the_system_prompt_and_nothing_else():
    original = messages()
    rebuilt = rp.rebuild_system(original)
    assert rebuilt[0]["content"] != "RECORDED SYSTEM PROMPT"
    assert "You are driving a real Android phone" in rebuilt[0]["content"]
    assert rebuilt[1:] == original[1:]
    assert original[0]["content"] == "RECORDED SYSTEM PROMPT"  # not mutated


def test_rebuilding_inserts_a_system_prompt_when_there_was_none():
    rebuilt = rp.rebuild_system([{"role": "user", "content": "GOAL: x"}])
    assert rebuilt[0]["role"] == "system"
    assert len(rebuilt) == 2


def test_verbatim_mode_sends_the_recorded_system_prompt(run):
    seen = []

    def decide(msgs):
        seen.append(msgs[0]["content"])
        return AgentAction(observation="o", reasoning="r", action="tap",
                           target={"index": 1}), {}

    rp.replay(rp.load_cases(run), decide)
    assert set(seen) == {"RECORDED SYSTEM PROMPT"}

    seen.clear()
    rp.replay(rp.load_cases(run), decide, rebuild_system_prompt=True)
    assert all("driving a real Android phone" in s for s in seen)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

def _replay_all(run, answer):
    def decide(msgs):
        return answer, {"latency_s": 2.0, "completion_tokens": 400,
                        "reasoning_tokens": 350, "usd": 0.001}
    return rp.replay(rp.load_cases(run), decide)


def test_a_run_that_reproduces_itself_scores_full_agreement(run):
    def decide(msgs):
        step = int(msgs[3]["content"][0]["text"].rsplit(" ", 1)[1].rstrip('"'))
        recorded = {1: ("tap", 1), 2: ("scroll", 0), 3: ("tap", 4)}[step]
        kw = {"direction": "down"} if recorded[0] == "scroll" else {}
        return AgentAction(observation="o", reasoning="r", action=recorded[0],
                           target={"index": recorded[1]} if recorded[1] else None,
                           **kw), {}

    report = rp.replay(rp.load_cases(run), decide)
    assert report.n == 3
    assert report.agreement == 1.0
    assert report.regressions == []


def test_diverging_from_a_step_that_worked_is_flagged(run):
    always_back = AgentAction(observation="o", reasoning="r",
                              action="press_key", key="back")
    report = _replay_all(run, always_back)
    assert report.count("differs") == 3
    # Steps 1 and 3 succeeded; step 2 changed nothing, so leaving it behind is
    # not a regression.
    assert [r.step for r in report.regressions] == [1, 3]


def test_an_exception_on_one_case_does_not_end_the_replay(run):
    calls = {"n": 0}

    def decide(msgs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("model never produced a valid AgentAction")
        return AgentAction(observation="o", reasoning="r", action="tap",
                           target={"index": 1}), {}

    report = rp.replay(rp.load_cases(run), decide)
    assert report.n == 3
    assert report.count("error") == 1
    errored = next(r for r in report.results if r.verdict == "error")
    assert "never produced" in errored.error
    # Step 2's recording was a no_change, so failing to reproduce it is not
    # counted against the change under test.
    assert errored.step == 2
    assert report.regressions == []


def test_metrics_roll_up_for_the_latency_question(run):
    report = _replay_all(run, AgentAction(observation="o", reasoning="r",
                                          action="tap", target={"index": 1}))
    assert report.median("latency_s") == 2.0
    assert report.median("reasoning_tokens") == 350
    assert report.totals("completion_tokens") == 1200


def test_the_json_report_carries_everything_a_diff_needs(run):
    report = _replay_all(run, AgentAction(observation="o", reasoning="r",
                                          action="press_key", key="back"))
    data = report.to_dict()
    assert data["cases"] == 3
    assert data["verdicts"]["differs"] == 3
    assert data["regressions"] == [1, 3]
    assert data["median"]["reasoning_tokens"] == 350
    assert data["results"][0]["recorded"] == "tap #1"
    assert data["results"][0]["replayed"] == "press_key back"
    # Round-trips as JSON, since --json pipes it into other tools.
    assert json.loads(json.dumps(data))["cases"] == 3


def test_an_empty_report_does_not_divide_by_zero():
    report = rp.Report()
    assert report.agreement == 0.0
    assert report.median("latency_s") == 0.0
    assert report.to_dict()["cases"] == 0
