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
