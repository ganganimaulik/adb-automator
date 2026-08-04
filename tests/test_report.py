"""The cost summary `adbagent report` ends with.

Its whole job is to say where the wall clock went, which means the numbers have
to be grouped by the kind of call that spent it. A sweep read and a reasoning
turn differ by two orders of magnitude in output tokens; pooled into one median
they describe neither.
"""

from __future__ import annotations

import json

from adbagent.cli import Out, _cost_summary, build_parser, cmd_report


def parse(argv):
    return build_parser().parse_args(argv)


DECIDE = {"n_calls": 1, "prompt_tokens": 5500, "cached_tokens": 3700,
          "completion_tokens": 4400, "reasoning_tokens": 4200,
          "reasoning_chars": 0, "latency_s": 26.0, "usd": 0.002}
READ = {"n_calls": 1, "prompt_tokens": 1400, "cached_tokens": 0,
        "completion_tokens": 25, "reasoning_tokens": 0,
        "reasoning_chars": 0, "latency_s": 1.6, "usd": 0.0006}


def events(*, decides: int = 2, sweeps: int = 0):
    out = [{"t": 0, "kind": "run_start", "goal": "g", "model": "m"}]
    step = 0
    for _ in range(decides):
        step += 1
        out.append({"t": step, "kind": "decide", "step": step, "wall_s": 26.0,
                    "llm": dict(DECIDE), "action": {"action": "swipe"}})
    for _ in range(sweeps):
        step += 1
        out.append({"t": step, "kind": "sweep_step", "step": step,
                    "direction": "left", "moved": True, "llm": dict(READ)})
    return out


def summary(capsys, **kw) -> str:
    _cost_summary(Out(), events(**kw))
    return capsys.readouterr().out


def test_a_run_with_no_sweeps_is_reported_as_one_block(capsys):
    text = summary(capsys, decides=3)
    assert "Cost of thinking" in text
    # No group label when there is only one group -- it would be noise.
    assert "decisions (" not in text
    assert "67% served from cache" in text
    assert "95% of output" in text


def test_sweep_reads_are_counted_but_kept_apart(capsys):
    """Pooled, thirteen 25-token reads would drag the median output tokens from
    4,400 to 25 and make the reasoning turns invisible."""
    text = summary(capsys, decides=3, sweeps=13)
    assert "decisions (3)" in text
    assert "sweep reads (13)" in text
    assert "4400 median" in text        # the decisions kept their own median
    assert "  25 median" in text        # and the reads theirs
    assert "13200 total" in text        # 3 x 4400
    assert "18200 total" in text        # 13 x 1400


def test_a_sweep_is_never_reported_as_free(capsys):
    """A swept item is a real vision call. Totalling only the reasoning turns
    would show the sweep as costing nothing at all."""
    without = summary(capsys, decides=3)
    with_sweep = summary(capsys, decides=3, sweeps=13)
    assert "18200 total" not in without
    assert "18200 total" in with_sweep


def test_a_one_shot_vision_read_is_not_scolded_for_a_cold_cache(capsys):
    """The advice is about the prompt prefix. A read has no prefix to reuse, so
    its 0% is correct rather than a finding."""
    text = summary(capsys, decides=3, sweeps=13)
    assert "0% served from cache" in text          # reported
    assert "changing every turn" not in text       # but not diagnosed


def test_the_cache_advice_still_fires_on_the_reasoning_turns(capsys):
    cold = dict(DECIDE, cached_tokens=0)
    _cost_summary(Out(), [{"t": 1, "kind": "decide", "step": 1, "wall_s": 26.0,
                           "llm": cold, "action": {}}])
    assert "changing every turn" in capsys.readouterr().out


def test_a_group_with_no_thinking_omits_the_thinking_line(capsys):
    text = summary(capsys, decides=0, sweeps=5)
    assert "sweep reads" not in text     # sole group, so unlabelled
    assert "of which think" not in text
    assert "prompt tokens" in text


def test_a_run_recorded_before_metrics_existed_prints_nothing(capsys):
    _cost_summary(Out(), [
        {"t": 1, "kind": "run_start", "goal": "g"},
        {"t": 2, "kind": "decide", "step": 1, "action": {"action": "tap"}},
    ])
    assert capsys.readouterr().out == ""


def test_report_renders_the_summary_end_to_end(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "swept"
    run_dir.mkdir(parents=True)
    rows = events(decides=2, sweeps=4) + [
        {"t": 99, "kind": "run_end", "outcome": "success", "steps": 6,
         "llm_calls": 2, "usd": 0.0064}]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in rows))

    assert cmd_report(parse(["report", str(run_dir)])) == 0
    text = capsys.readouterr().out
    assert "decisions (2)" in text and "sweep reads (4)" in text
    assert "SUCCESS" in text
    # The metrics block is summarised, never dumped as a dict.
    assert "'reasoning_tokens'" not in text


# ---------------------------------------------------------------------------
# Reasoning depth
# ---------------------------------------------------------------------------

def depth_events(*, escalated: int, shallow: int):
    """Decide events with a recorded reasoning depth, deep ones costing more."""
    out = [{"t": 0, "kind": "run_start", "goal": "g", "model": "m"}]
    step = 0
    for _ in range(escalated):
        step += 1
        out.append({"t": step, "kind": "decide", "step": step, "wall_s": 26.0,
                    "effort": "high", "hard_because": "the last action failed",
                    "llm": dict(DECIDE), "action": {"action": "tap"}})
    for _ in range(shallow):
        step += 1
        out.append({"t": step, "kind": "decide", "step": step, "wall_s": 2.1,
                    "effort": "none", "hard_because": "",
                    "llm": dict(DECIDE, completion_tokens=200,
                               reasoning_tokens=60, latency_s=2.1),
                    "action": {"action": "swipe"}})
    return out


def test_the_split_between_deep_and_shallow_turns_is_reported(capsys):
    _cost_summary(Out(), depth_events(escalated=4, shallow=8))
    text = capsys.readouterr().out
    assert "thinking depth  4 of 12 turn(s) escalated, 8 at the floor" in text


def test_a_run_with_no_depth_recorded_says_nothing_about_it(capsys):
    """Runs from before the setting existed must not grow a misleading line."""
    _cost_summary(Out(), events(decides=3))
    assert "thinking depth" not in capsys.readouterr().out


def test_a_capped_run_is_not_told_to_cap_it(capsys):
    """The advice existed before the setting did. Repeating it to someone who has
    already taken it is how a report teaches people to stop reading it."""
    _cost_summary(Out(), depth_events(escalated=1, shallow=11))
    text = capsys.readouterr().out
    assert "setting llm.reasoning_effort" not in text
    assert "Most turns escalated" not in text        # only 1 of 12 did


def test_escalating_on_most_turns_points_at_the_policy_instead(capsys):
    """A cap that never applies is not the cap's fault, and telling someone to
    lower it further would be the wrong advice."""
    _cost_summary(Out(), depth_events(escalated=11, shallow=1))
    text = capsys.readouterr().out
    assert "Most turns escalated" in text
    assert "setting llm.reasoning_effort" not in text


def test_an_uncapped_run_is_still_told_to_cap_it(capsys):
    _cost_summary(Out(), events(decides=3))
    assert "setting llm.reasoning_effort" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Models that do not reason
# ---------------------------------------------------------------------------

def reasoning_report(**llm) -> str:
    import io, contextlib
    from adbagent.cli import _report_reasoning
    from adbagent.config import Config

    cfg = Config()
    cfg.llm.reasoning_effort = "none"
    cfg.llm.reasoning_effort_hard = "high"
    for key, value in llm.items():
        setattr(cfg.llm, key, value)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _report_reasoning(Out(), cfg)
    return buffer.getvalue()


def test_a_non_reasoning_model_is_not_reported_as_a_problem():
    """Most models do not reason. That is the normal case, not a misconfiguration,
    and it needs no action."""
    text = reasoning_report(model="llama-v3p3-70b-instruct")
    assert "does not reason -- nothing to cap" in text
    assert "WARN" not in text
    assert "reasoning_style" not in text        # forcing one would break it


def test_an_unfamiliar_model_says_which_of_the_two_it_might_be():
    """Advice that assumes it reasons would break every call if it does not."""
    text = reasoning_report(model="brand-new-model-2027")
    assert "WARN" in text
    assert "if it does reason, set llm.reasoning_style" in text
    assert "if it does not, nothing to fix" in text


def test_a_mixed_setup_is_reported_per_model():
    """A reasoning decider next to a vision model that does not think is a
    perfectly ordinary setup, so one verdict for the whole config would be wrong."""
    text = reasoning_report(model="deepseek-v4-flash",
                            model_image="llama-v3p3-70b-instruct")
    assert "deepseek-v4-flash (deciding/judging/skills)" in text
    assert '"thinking": true' in text
    assert "llama-v3p3-70b-instruct (vision) does not reason" in text


def test_nothing_is_said_about_bodies_when_none_were_printed():
    text = reasoning_report(model="llama-v3p3-70b-instruct")
    assert "confirm the bodies above" not in text


def test_the_depths_a_model_will_actually_be_asked_for_are_the_ones_shown():
    """The decider varies turn to turn; a vision model never reasons at all."""
    text = reasoning_report(model="deepseek-v4-flash")
    decider = text.split("deciding")[1]
    assert '"thinking": false' in decider      # routine
    assert '"thinking": true' in decider       # when stuck


def test_the_feature_being_off_is_stated_plainly():
    assert "left to the model" in reasoning_report(
        model="deepseek-v4-flash", reasoning_effort="")
