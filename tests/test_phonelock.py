"""One driver per phone, across processes: the lock every driver takes."""

from __future__ import annotations

import json

import pytest

from adbagent import phonelock
from adbagent.phonelock import PhoneBusy, PhoneLock


def test_a_phone_nobody_has_taken_is_free(tmp_path):
    assert phonelock.holder(tmp_path) == ""
    assert phonelock.holder(tmp_path / "never-created") == ""


def test_the_holder_is_named_while_it_holds_the_phone(tmp_path):
    with PhoneLock(tmp_path, "an MCP session (claude-code): read my messages"):
        held = phonelock.holder(tmp_path)
        assert "an MCP session (claude-code): read my messages" in held
        assert "pid" in held
    assert phonelock.holder(tmp_path) == ""


def test_a_second_driver_is_refused_with_the_first_ones_name(tmp_path):
    first = PhoneLock(tmp_path, "adbagent watch: reply to new chats").acquire()
    try:
        with pytest.raises(PhoneBusy, match="adbagent watch: reply to new chats"):
            PhoneLock(tmp_path, "adbagent run: turn on wifi").acquire()
    finally:
        first.release()
    # Free again, so the one that was refused can have it now.
    PhoneLock(tmp_path, "adbagent run: turn on wifi").acquire().release()


def test_a_note_left_behind_by_a_crash_does_not_hold_the_phone(tmp_path):
    """The kernel drops the lock with the process; the note is only a label."""
    (tmp_path / phonelock.LOCK_NAME).write_bytes(b"")
    (tmp_path / phonelock.NOTE_NAME).write_text(
        json.dumps({"holder": "a run that crashed", "pid": 999999}),
        encoding="utf-8")
    assert phonelock.holder(tmp_path) == ""
    PhoneLock(tmp_path, "adbagent run").acquire().release()


def test_releasing_twice_is_harmless(tmp_path):
    lock = PhoneLock(tmp_path, "x").acquire()
    lock.release()
    lock.release()
    assert not lock.held
