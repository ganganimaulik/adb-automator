"""The README has to stay true.

It drifted a long way once: it documented an `explore` command that did not
exist, and an `--app` flag on `run` that had never existed, so the very first
example a reader would try errored out. Both survived several rewrites because
nothing checked them. These tests check them.

Only mechanically verifiable claims are asserted -- the commands, the flags and
the config defaults in the tuning table. Prose is not testable and is not tested.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import pytest

from adbagent.cli import build_parser
from adbagent.config import Config

README = Path(__file__).resolve().parent.parent / "README.md"


def documented_commands() -> list:
    """Every `adbagent ...` invocation in the README, comments stripped and
    backslash continuations joined."""
    out: list = []
    pending = ""
    for raw in README.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        # A markdown heading is not a shell comment.
        line = "" if stripped.startswith("#") else raw.split("#", 1)[0].strip()
        if pending:
            pending += " " + line.rstrip("\\").strip()
            if not line.endswith("\\"):
                out.append(pending)
                pending = ""
            continue
        if line.startswith("adbagent "):
            if line.endswith("\\"):
                pending = line.rstrip("\\").strip()
            else:
                out.append(line)
    assert pending == "", "unterminated line continuation in the README"
    return out


def test_the_readme_documents_some_commands():
    """A guard on the guard: a parsing bug that found nothing would pass every
    test below without checking anything."""
    assert len(documented_commands()) >= 15


@pytest.mark.parametrize("command", documented_commands())
def test_every_documented_command_parses(command):
    parser = build_parser()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(shlex.split(command)[1:])
    except SystemExit:  # argparse's way of rejecting an argument
        pytest.fail(f"the README documents a command that does not work: {command}")


def test_no_command_the_cli_does_not_have_is_mentioned():
    """`explore` was documented for months after the command was removed."""
    subcommands = set(build_parser()._subparsers._group_actions[0].choices)
    mentioned = {shlex.split(c)[1] for c in documented_commands()}
    assert mentioned <= subcommands, f"not real commands: {mentioned - subcommands}"


def test_every_command_is_documented():
    """The other direction: a command nobody can find is a command nobody uses."""
    subcommands = set(build_parser()._subparsers._group_actions[0].choices)
    mentioned = {shlex.split(c)[1] for c in documented_commands()}
    assert subcommands <= mentioned, f"undocumented: {subcommands - mentioned}"


# ---------------------------------------------------------------------------
# The tuning table
# ---------------------------------------------------------------------------

#: `| `section.key` | `default` | description |` rows.
_ROW = re.compile(r"^\|\s*`([a-z_]+\.[a-z_]+)`\s*\|\s*`([^`]+)`\s*\|")


def tuning_rows() -> list:
    return [(m.group(1), m.group(2))
            for m in (_ROW.match(line) for line
                      in README.read_text(encoding="utf-8").splitlines())
            if m]


def test_the_tuning_table_has_rows():
    assert len(tuning_rows()) >= 5


@pytest.mark.parametrize("dotted,shown", tuning_rows())
def test_documented_defaults_are_the_real_defaults(dotted, shown):
    section, _, key = dotted.partition(".")
    config = Config()
    assert hasattr(config, section), f"no config section {section!r}"
    group = getattr(config, section)
    assert hasattr(group, key), f"{dotted} is not a config setting"

    actual = getattr(group, key)
    # A table cell cannot render an empty string, so `""` stands in for one.
    if shown in ('""', "''"):
        shown = ""
    expected = {"true": True, "false": False}.get(shown.lower(), shown)
    if isinstance(actual, bool):
        assert actual is expected, f"{dotted} is {actual}, README says {shown}"
    else:
        assert actual == type(actual)(expected), (
            f"{dotted} is {actual}, README says {shown}")
