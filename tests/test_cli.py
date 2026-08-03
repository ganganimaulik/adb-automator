"""CLI surface: argument parsing, config precedence, and the commands that
work without a device.
"""

from __future__ import annotations

import json

import pytest

from adbagent.cli import build_config, build_parser, cmd_report, main


def parse(argv):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_every_subcommand_parses():
    for argv in (
        ["doctor"],
        ["devices"],
        ["pair", "10.0.0.5:37115", "--code", "123456"],
        ["models", "--vision", "--search", "kimi"],
        ["dump", "--raw"],
        ["dump", "--detail", "4"],
        ["run", "turn on wifi"],
        ["report", "runs/abc"],
        ["report"],
        ["replay"],
        ["replay", "runs/abc", "--rebuild-system", "--limit", "10"],
        ["replay", "runs/abc", "--steps", "4", "7", "--json"],
        ["apps", "--search", "whatsapp", "-3"],
    ):
        assert parse(argv).func is not None


def test_a_goal_is_required():
    with pytest.raises(SystemExit):
        parse(["run"])


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        parse(["teleport"])


# ---------------------------------------------------------------------------
# Config precedence
# ---------------------------------------------------------------------------

def test_cli_flags_win_over_defaults():
    cfg = build_config(parse(["run", "g", "--model", "m", "--max-steps", "7",
                              "--budget-usd", "0.5"]))
    assert cfg.llm.model == "m"
    assert cfg.run.max_steps == 7
    assert cfg.safety.budget_usd == 0.5


def test_cli_flags_win_over_the_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "from-file", "rpm": 30},
                                "run": {"max_steps": 99}}))
    cfg = build_config(parse(["run", "g", "-c", str(path), "--model", "from-cli"]))
    assert cfg.llm.model == "from-cli"      # flag wins
    assert cfg.llm.rpm == 30                # file still applies
    assert cfg.run.max_steps == 99


def test_config_file_wins_over_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADBAGENT_MODEL", "from-env")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"model": "from-file"}}))
    assert build_config(parse(["run", "g", "-c", str(path)])).llm.model == "from-file"

    path.unlink()
    assert build_config(parse(["run", "g"])).llm.model == "from-env"


def test_empty_config_string_does_not_overwrite_env_var(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANDROID_SERIAL", "192.168.1.50:5555")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device": {"serial": ""}}))
    assert build_config(parse(["run", "g", "-c", str(path)])).device.serial == "192.168.1.50:5555"


def test_no_app_flag_means_unrestricted():
    assert build_config(parse(["run", "g"])).allowed_packages() == []


def test_bad_config_file_warns_but_does_not_crash(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    cfg = build_config(parse(["run", "g", "-c", str(path)]))
    assert cfg.llm.provider == "fireworks"          # defaults survive
    assert "config:" in capsys.readouterr().err


def test_unknown_config_key_warns(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"nonexistent": 1}}))
    build_config(parse(["run", "g", "-c", str(path)]))
    assert "unknown config key" in capsys.readouterr().err


def test_api_key_comes_only_from_the_environment(monkeypatch, tmp_path):
    """A key must never be readable out of a config file that might be committed."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"api_key": "sk-should-be-ignored"}}))
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-from-env")
    cfg = build_config(parse(["run", "g", "-c", str(path)]))
    assert cfg.api_key() == "fw-from-env"
    assert not hasattr(cfg.llm, "api_key")


# ---------------------------------------------------------------------------
# Commands that need no device
# ---------------------------------------------------------------------------

def test_report_reads_a_run(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "turn on wifi", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1, "source": "llm",
         "action": {"action": "tap", "target": {"index": 3},
                    "observation": "Settings list screen",
                    "reasoning": "Tapping Wi-Fi settings element"}},
        {"t": 3, "kind": "verify", "step": 1, "grade": "success", "reason": ""},
        {"t": 4, "kind": "decide", "step": 2, "source": "llm",
         "action": {"action": "done"}},
        {"t": 5, "kind": "run_end", "outcome": "success", "steps": 2,
         "llm_calls": 2, "usd": 0.0031},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    out = capsys.readouterr().out
    assert "turn on wifi" in out
    assert "SUCCESS" in out and "0.0031" in out
    assert "Obs:       Settings list screen" in out
    assert "Reasoning: Tapping Wi-Fi settings element" in out
    # A run recorded before per-call metrics existed has nothing to summarise.
    assert "Cost of thinking" not in out


def _metric_run(tmp_path, **over):
    """A run directory whose decide events carry per-call metrics."""
    metrics = {"n_calls": 1, "prompt_tokens": 5500, "cached_tokens": 3100,
               "completion_tokens": 4400, "reasoning_tokens": 4200,
               "reasoning_chars": 16800, "latency_s": 26.0, "usd": 0.002}
    metrics.update(over)
    run_dir = tmp_path / "runs" / "metrics"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "browse the album", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1, "wall_s": 26.0, "llm": metrics,
         "action": {"action": "swipe", "direction": "left"}},
        {"t": 3, "kind": "verify", "step": 1, "grade": "success", "reason": ""},
        {"t": 4, "kind": "decide", "step": 2, "wall_s": 96.0, "llm": metrics,
         "action": {"action": "done"}},
        {"t": 5, "kind": "run_end", "outcome": "success", "steps": 2,
         "llm_calls": 2, "usd": 0.004},
    ]))
    return run_dir


def test_report_summarises_where_the_time_went(tmp_path, capsys):
    run_dir = _metric_run(tmp_path)
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    out = capsys.readouterr().out
    assert "Cost of thinking" in out
    assert "56% served from cache" in out       # 6200 of 11000
    assert "95% of output" in out               # 8400 of 8800
    assert "lower reasoning effort" in out      # the conclusion, spelled out
    # The metrics block is summarised, never dumped as a dict.
    assert "'reasoning_chars'" not in out


def test_report_flags_a_cold_prompt_cache(tmp_path, capsys):
    run_dir = _metric_run(tmp_path, cached_tokens=0)
    cmd_report(parse(["report", str(run_dir)]))
    out = capsys.readouterr().out
    assert "0% served from cache" in out
    assert "changing every turn" in out


def test_report_estimates_thinking_when_the_provider_will_not_say(tmp_path, capsys):
    """Not every provider reports `reasoning_tokens`, but the thinking still
    arrives on the wire, so the characters we counted stand in for it."""
    run_dir = _metric_run(tmp_path, reasoning_tokens=0, reasoning_chars=16800)
    cmd_report(parse(["report", str(run_dir)]))
    out = capsys.readouterr().out
    assert "est. from streamed text" in out
    assert "4200 median" in out                 # 16800 chars / 4


def test_report_defaults_to_the_most_recent_run(tmp_path, capsys, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    older = tmp_path / "runs" / "older"
    older.mkdir(parents=True)
    (older / "events.jsonl").write_text(json.dumps(
        {"t": 1, "kind": "run_start", "goal": "the older run", "model": "m"}))
    newer = tmp_path / "runs" / "newer"
    newer.mkdir(parents=True)
    (newer / "events.jsonl").write_text(json.dumps(
        {"t": 1, "kind": "run_start", "goal": "the newer run", "model": "m"}))
    os.utime(older, (1, 1))

    assert cmd_report(parse(["report"])) == 0
    assert "the newer run" in capsys.readouterr().out


def test_report_says_so_when_there_are_no_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cmd_report(parse(["report"])) == 1
    assert "no runs found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _replay_run(tmp_path):
    from adbagent.actions import AgentAction

    run_dir = tmp_path / "runs" / "rep"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "open wifi", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1,
         "action": AgentAction(observation="o", reasoning="r", action="tap",
                               target={"index": 3}).model_dump()},
        {"t": 3, "kind": "verify", "step": 1, "grade": "success", "reason": ""},
    ]))
    (run_dir / "step_001_decide_messages.json").write_text(json.dumps([
        {"role": "system", "content": "OLD SYSTEM PROMPT"},
        {"role": "user", "content": "GOAL: open wifi"},
    ]))
    return run_dir


def _fake_client(monkeypatch, answer):
    """Stand in for LLMClient so no network or API key is needed."""
    from adbagent.llm import Call, Ledger

    class FakeClient:
        seen = []

        def __init__(self, cfg, run_id=""):
            self.cfg = cfg
            self.model = cfg.llm.model
            self.ledger = Ledger()

        def structured(self, messages, model_cls, **kw):
            FakeClient.seen.append(messages)
            self.ledger.record(Call(model=self.model, prompt_tokens=100,
                                    completion_tokens=200, reasoning_tokens=150,
                                    latency_s=1.5, purpose="replay"))
            if isinstance(answer, Exception):
                raise answer
            return answer

    FakeClient.seen = []
    monkeypatch.setattr("adbagent.llm.LLMClient", FakeClient)
    return FakeClient


def test_replay_reports_agreement_and_exits_clean(tmp_path, capsys, monkeypatch):
    from adbagent.actions import AgentAction
    from adbagent.cli import cmd_replay

    run_dir = _replay_run(tmp_path)
    _fake_client(monkeypatch, AgentAction(observation="different words",
                                          reasoning="also different",
                                          action="tap", target={"index": 3}))
    code = cmd_replay(parse(["replay", str(run_dir), "--model", "m"]))
    out = capsys.readouterr().out
    assert code == 0
    assert "1/1 identical" in out
    assert "no divergence from a step that had worked" in out


def test_replay_exits_nonzero_when_a_working_step_changed(tmp_path, capsys, monkeypatch):
    from adbagent.actions import AgentAction
    from adbagent.cli import cmd_replay

    run_dir = _replay_run(tmp_path)
    _fake_client(monkeypatch, AgentAction(observation="o", reasoning="r",
                                          action="press_key", key="back"))
    code = cmd_replay(parse(["replay", str(run_dir), "--model", "m"]))
    out = capsys.readouterr().out
    assert code == 1                       # so CI can gate on it
    assert "diverged from a step that had worked: 1" in out
    assert "tap #3" in out and "press_key back" in out


def test_replay_sends_the_recording_verbatim_by_default(tmp_path, capsys, monkeypatch):
    from adbagent.actions import AgentAction
    from adbagent.cli import cmd_replay

    run_dir = _replay_run(tmp_path)
    client = _fake_client(monkeypatch, AgentAction(
        observation="o", reasoning="r", action="tap", target={"index": 3}))

    cmd_replay(parse(["replay", str(run_dir), "--model", "m"]))
    assert client.seen[0][0]["content"] == "OLD SYSTEM PROMPT"

    client.seen.clear()
    cmd_replay(parse(["replay", str(run_dir), "--model", "m", "--rebuild-system"]))
    assert "driving a real Android phone" in client.seen[0][0]["content"]


def test_replay_json_is_machine_readable(tmp_path, capsys, monkeypatch):
    from adbagent.actions import AgentAction
    from adbagent.cli import cmd_replay

    run_dir = _replay_run(tmp_path)
    _fake_client(monkeypatch, AgentAction(observation="o", reasoning="r",
                                          action="tap", target={"index": 3}))
    cmd_replay(parse(["replay", str(run_dir), "--model", "m", "--json"]))
    data = json.loads(capsys.readouterr().out)
    assert data["cases"] == 1
    assert data["agreement"] == 1.0
    assert data["median"]["reasoning_tokens"] == 150


def test_replay_survives_a_model_that_will_not_answer(tmp_path, capsys, monkeypatch):
    from adbagent.cli import cmd_replay
    from adbagent.llm import LLMError

    run_dir = _replay_run(tmp_path)
    _fake_client(monkeypatch, LLMError("never produced a valid AgentAction"))
    code = cmd_replay(parse(["replay", str(run_dir), "--model", "m"]))
    out = capsys.readouterr().out
    assert code == 1
    assert "error" in out


def test_replay_needs_a_model(tmp_path, capsys, monkeypatch):
    from adbagent.cli import cmd_replay

    monkeypatch.chdir(tmp_path)          # no config.json to supply one
    run_dir = _replay_run(tmp_path)
    assert cmd_replay(parse(["replay", str(run_dir)])) == 1
    assert "no model chosen" in capsys.readouterr().out


def test_replay_says_so_when_a_run_has_no_cases(tmp_path, capsys, monkeypatch):
    from adbagent.cli import cmd_replay

    _fake_client(monkeypatch, None)
    run_dir = tmp_path / "runs" / "bare"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(json.dumps(
        {"t": 1, "kind": "run_start", "goal": "g", "model": "m"}))
    assert cmd_replay(parse(["replay", str(run_dir), "--model", "m"])) == 1
    assert "no replayable decide cases" in capsys.readouterr().out


def test_live_reporter_displays_llm_reasoning(capsys):
    from adbagent.actions import AgentAction
    from adbagent.cli import Out, _live_reporter
    from unittest.mock import MagicMock

    out = Out(quiet=False)
    reporter = _live_reporter(out)

    action = AgentAction(
        observation="Main settings screen",
        reasoning="Tap element #5 to open Wi-Fi settings",
        action="tap",
        target={"index": 5}
    )
    state = MagicMock(step=1)

    reporter("step", state=state, action=action, source="llm")
    captured = capsys.readouterr().out
    assert "tap #5" in captured
    assert "Obs:       Main settings screen" in captured
    assert "Reasoning: Tap element #5 to open Wi-Fi settings" in captured


def test_live_reporter_displays_screenshot_indicator(capsys):
    from adbagent.actions import AgentAction
    from adbagent.cli import Out, _live_reporter
    from unittest.mock import MagicMock

    out = Out(quiet=False)
    reporter = _live_reporter(out)

    action = AgentAction(
        observation="Screen",
        reasoning="Tap element",
        action="tap",
        target={"index": 2}
    )
    state = MagicMock(step=1)

    reporter("step", state=state, action=action, source="llm", screenshot=True)
    captured = capsys.readouterr().out
    assert "+img" in captured


def test_report_on_a_missing_run():
    assert cmd_report(parse(["report", "/nonexistent/run"])) == 1


def test_main_returns_an_exit_code(tmp_path):
    # Use report on a non-existent path -- returns 1 but doesn't crash.
    assert main(["report", str(tmp_path / "nonexistent")]) == 1


def test_model_image_cli_and_env_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADBAGENT_MODEL_IMAGE", "vision-from-env")
    assert build_config(parse(["run", "g"])).llm.model_image == "vision-from-env"

    cfg = build_config(parse(["run", "g", "--model-image", "vision-from-cli"]))
    assert cfg.llm.model_image == "vision-from-cli"


def test_max_tokens_cli_and_env_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADBAGENT_MAX_TOKENS", "2500")
    assert build_config(parse(["run", "g"])).llm.max_tokens == 2500

    cfg = build_config(parse(["run", "g", "--max-tokens", "4000"]))
    assert cfg.llm.max_tokens == 4000


def test_live_reporter_terminal_output(capsys):
    from adbagent.cli import Out, _live_reporter
    from adbagent.actions import AgentAction, Target
    from types import SimpleNamespace

    out = Out()
    reporter = _live_reporter(out, max_steps=20)

    state = SimpleNamespace(step=3)
    action = AgentAction(
        observation="Settings screen",
        reasoning="Looking for dark mode",
        action="tap",
        target=Target(index=1),
        confidence="low",
        progress="Step 2 of 5 done",
    )

    reporter("step", state=state, action=action, screenshot=True)
    reporter("loop_warning", message="step 3: stuck in a loop; going back")
    reporter("safety_warning", message="step 3: irreversible action 'delete' in com.example")

    captured = capsys.readouterr()
    assert "[ 3/20] +img tap" in captured.out
    assert "(confidence: low)" in captured.out
    assert "Obs:       Settings screen" in captured.out
    assert "Reasoning: Looking for dark mode" in captured.out
    assert "Progress:  Step 2 of 5 done" in captured.out
    assert "[Loop Warning] step 3: stuck in a loop; going back" in captured.out
    assert "[Safety Warning] step 3: irreversible action 'delete' in com.example" in captured.out


def test_service_tier_cli_and_env_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADBAGENT_SERVICE_TIER", "priority")
    assert build_config(parse(["run", "g"])).llm.service_tier == "priority"

    cfg = build_config(parse(["run", "g", "--service-tier", "default"]))
    assert cfg.llm.service_tier == "default"


def test_cmd_apps(monkeypatch, capsys):
    from unittest.mock import MagicMock
    from adbagent.cli import cmd_apps

    mock_dev = MagicMock()
    mock_dev.list_apps.return_value = ["com.whatsapp", "com.whatsapp.w4b"]
    mock_dev.__enter__.return_value = mock_dev

    monkeypatch.setattr("adbagent.cli._ensure_device", lambda args, cfg, out: None)
    monkeypatch.setattr("adbagent.device.Device", lambda cfg, serial: mock_dev)

    args = parse(["apps", "--search", "whatsapp", "-3"])
    code = cmd_apps(args)
    assert code == 0

    mock_dev.list_apps.assert_called_once_with(query="whatsapp", third_party_only=True)
    captured = capsys.readouterr()
    assert "Installed 3rd-Party Apps matching 'whatsapp' (2)" in captured.out
    assert "- com.whatsapp" in captured.out


def test_live_reporter_displays_notes(capsys):
    from adbagent.cli import Out, _live_reporter
    from adbagent.actions import AgentAction, Target
    from types import SimpleNamespace

    out = Out()
    reporter = _live_reporter(out)
    state = SimpleNamespace(step=1)
    action = AgentAction(
        observation="Screen",
        reasoning="Reading item",
        action="tap",
        target=Target(index=1),
        notes="Item price is $15",
    )
    reporter("step", state=state, action=action)
    captured = capsys.readouterr()
    assert "Notes:     Item price is $15" in captured.out


def test_cmd_scratchpad(tmp_path, monkeypatch, capsys):
    from adbagent.cli import cmd_scratchpad

    runs_dir = tmp_path / "runs" / "run1"
    runs_dir.mkdir(parents=True)
    events = runs_dir / "events.jsonl"
    events.write_text(json.dumps({"kind": "decide", "action": {"notes": "Collected data XYZ"}}) + "\n" +
                      json.dumps({"kind": "image_analysis", "model": "vision-1", "result": "Digital scale showing 275g water"}) + "\n")

    monkeypatch.chdir(tmp_path)
    args = parse(["scratchpad", "latest"])
    assert cmd_scratchpad(args) == 0
    captured = capsys.readouterr()
    assert "Notes: Collected data XYZ" in captured.out
    assert "Latest Vision Analysis: Digital scale showing 275g water" in captured.out


def test_live_reporter_displays_image_analysis(capsys):
    from adbagent.cli import Out, _live_reporter

    out = Out()
    reporter = _live_reporter(out)
    reporter("llm_start", purpose="analyze_image", model="vision-v1", screenshot=True)
    reporter("image_analysis", model="vision-v1", result="Photo shows 100g oats on scale")
    captured = capsys.readouterr()
    assert "calling LLM image analyzer (vision-v1 +img)..." in captured.out
    assert "Vision (vision-v1): Photo shows 100g oats on scale" in captured.out


def test_live_reporter_llm_stream(capsys):
    from adbagent.cli import Out, _live_reporter

    out = Out()
    reporter = _live_reporter(out)
    reporter("llm_start", purpose="decide", model="deepseek-r1")
    reporter("llm_stream", stream_type="thinking", text="Analyzing element 5...")
    reporter("llm_stream", stream_type="content", text='{"action": "tap"}')
    reporter("llm_end", purpose="decide", elapsed=1.2)

    captured = capsys.readouterr()
    assert "calling LLM (deepseek-r1)..." in captured.out
    assert "[Thinking]" in captured.out
    assert "Analyzing element 5..." in captured.out
    assert "[Response]" in captured.out
    assert '{"action": "tap"}' in captured.out
    assert "LLM responded in 1.20s" in captured.out


def test_live_reporter_long_thinking_tailing(capsys, monkeypatch):
    from adbagent.cli import Out, _live_reporter

    out = Out()
    reporter = _live_reporter(out)
    reporter("llm_start", purpose="decide", model="deepseek-r1")
    
    # Stream 50 lines of thinking
    long_thinking = "\n".join([f"Step {i}: thinking deeply..." for i in range(1, 51)])
    reporter("llm_stream", stream_type="thinking", text=long_thinking)

    # Trigger end
    reporter("llm_end", purpose="decide", elapsed=2.5)

    captured = capsys.readouterr()
    assert "calling LLM (deepseek-r1)..." in captured.out
    assert "LLM responded in 2.50s" in captured.out


def _render_stream_panel(thinking, content, width=80, height=24):
    """The stream panel as the rows a `width` x `height` terminal would show."""
    from rich.console import Console

    from adbagent.cli import _stream_panel

    console = Console(width=width, height=height, force_terminal=False,
                      legacy_windows=False)
    with console.capture() as cap:
        console.print(_stream_panel(thinking, content, width, height), end="")
    return cap.get().rstrip("\n").split("\n")


def test_stream_panel_tails_long_thinking():
    # Long reasoning must keep showing its newest lines. Before the panel was
    # tailed it outgrew the terminal, and rich cropped it to the *oldest* lines
    # -- so the stream looked frozen.
    thinking = "\n".join(f"Step {i}: thinking" for i in range(1, 201))
    rows = _render_stream_panel(thinking, "", height=24)

    assert len(rows) <= 23, "panel must leave the live region room on screen"
    body = "\n".join(rows)
    assert "Step 200: thinking" in body
    assert "Step 1: thinking\n" not in body
    assert "scrolled off" in body


def test_stream_panel_tails_within_one_long_paragraph():
    # Reasoning often arrives as a single unbroken paragraph, so tailing has to
    # count wrapped rows rather than newlines.
    paragraph = "".join(f"I should check element {i} first. " for i in range(1, 400))
    rows = _render_stream_panel(paragraph, "", height=24)

    assert len(rows) <= 23
    assert "element 399" in "\n".join(rows)


def test_stream_panel_fits_every_terminal_size():
    thinking = "\n".join(f"Step {i}: thinking" for i in range(1, 201))
    content = "\n".join(f'  "field_{i}": {i},' for i in range(200))
    for width, height in ((40, 8), (60, 12), (80, 24), (120, 50)):
        rows = _render_stream_panel(thinking, content, width, height)
        assert len(rows) <= height - 1, f"too tall at {width}x{height}"
        assert max(len(r) for r in rows) <= width, f"too wide at {width}x{height}"
        body = "\n".join(rows)
        assert "[Thinking]" in body and "[Response]" in body


def test_stream_panel_gives_spare_rows_to_the_longer_half():
    thinking = "\n".join(f"Step {i}: thinking" for i in range(1, 201))
    rows = _render_stream_panel(thinking, '{"action": "tap"}', height=24)

    assert len(rows) == 23, "a short response should not strand rows"
    assert '{"action": "tap"}' in "\n".join(rows)


def test_stream_panel_keeps_bracketed_model_text_verbatim():
    # Model text is not rich markup: '[/data/app]' used to raise MarkupError
    # from the refresh thread and kill the panel, and '[foo]' was swallowed.
    hostile = "checking [/data/app] and [foo] and [/dim] plus [bold red]x"
    body = "\n".join(_render_stream_panel(hostile, ""))

    for fragment in ("[/data/app]", "[foo]", "[/dim]", "[bold red]x"):
        assert fragment in body


def test_live_reporter_keeps_drawing_a_long_stream_on_a_terminal(monkeypatch):
    # The whole bug in one test: 400 chunks of reasoning, then one more after a
    # frame boundary, and that newest chunk has to reach the terminal.
    import io
    import sys
    import time

    from adbagent.cli import _STREAM_FPS, Out, _live_reporter

    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    buf = FakeTTY()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        reporter = _live_reporter(Out(), max_steps=20)
        reporter("llm_start", purpose="decide", model="deepseek-r1")
        for i in range(400):
            reporter("llm_stream", stream_type="thinking",
                     text=f"chunk {i} at [/data/app] node [foo] ... ")
        time.sleep(1.5 / _STREAM_FPS)  # let the frame throttle open
        reporter("llm_stream", stream_type="thinking", text="the newest thought")
        reporter("llm_stream", stream_type="content", text='{"action": "tap"}')
        reporter("llm_end", purpose="decide", elapsed=2.5)
    finally:
        monkeypatch.undo()

    written = buf.getvalue()
    assert "Traceback" not in written
    assert "the newest thought" in written
    assert "[/data/app]" in written


def test_prevent_sleep():
    from adbagent.cli import prevent_sleep
    with prevent_sleep():
        pass




