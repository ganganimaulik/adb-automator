"""CLI surface: argument parsing, config precedence, and the commands that
work without a device.
"""

from __future__ import annotations

import json

import pytest

from adbagent.cli import build_config, build_parser, cmd_memory, cmd_report, main


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
        ["explore", "--app", "com.android.settings"],
        ["memory", "ls"],
        ["memory", "show", "12"],
        ["memory", "forget", "--state", "quarantined"],
        ["memory", "gc"],
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


def test_app_flag_pins_the_package_allowlist():
    cfg = build_config(parse(["explore", "--app", "com.android.settings"]))
    allowed = cfg.allowed_packages()
    assert "com.android.settings" in allowed
    # System chrome is always tolerated, or every permission dialog is an escape.
    assert "com.android.systemui" in allowed


def test_no_app_flag_means_unrestricted():
    assert build_config(parse(["run", "g"])).allowed_packages() == []


def test_no_cache_flag_inverts_to_memory_disabled():
    assert build_config(parse(["run", "g", "--no-cache"])).memory.enabled is False
    assert build_config(parse(["run", "g"])).memory.enabled is True


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

def test_memory_commands_on_an_empty_database(tmp_path, capsys):
    db = str(tmp_path / "m.db")
    assert cmd_memory(parse(["memory", "ls", "--db", db])) == 0
    assert "nothing learned yet" in capsys.readouterr().out

    assert cmd_memory(parse(["memory", "stats", "--db", db])) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 0

    assert cmd_memory(parse(["memory", "gc", "--db", db])) == 0
    assert cmd_memory(parse(["memory", "show", "999", "--db", db])) == 1


def test_report_reads_a_run(tmp_path, capsys):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"t": 1, "kind": "run_start", "goal": "turn on wifi", "model": "m"},
        {"t": 2, "kind": "decide", "step": 1, "source": "llm",
         "action": {"action": "tap", "target": {"index": 3}}},
        {"t": 3, "kind": "verify", "step": 1, "grade": "success", "reason": ""},
        {"t": 4, "kind": "decide", "step": 2, "source": "cache",
         "action": {"action": "done"}},
        {"t": 5, "kind": "run_end", "outcome": "success", "steps": 2,
         "llm_calls": 1, "cache_hits": 1, "usd": 0.0031},
    ]))
    assert cmd_report(parse(["report", str(run_dir)])) == 0
    out = capsys.readouterr().out
    assert "turn on wifi" in out
    assert "CACHE" in out and "LLM" in out
    assert "SUCCESS" in out and "0.0031" in out


def test_report_on_a_missing_run():
    assert cmd_report(parse(["report", "/nonexistent/run"])) == 1


def test_main_returns_an_exit_code():
    assert main(["memory", "stats", "--db", "/tmp/adbagent-cli-test.db"]) == 0


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

