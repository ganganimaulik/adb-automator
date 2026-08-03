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
