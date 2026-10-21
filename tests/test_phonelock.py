"""One driver per phone, across processes: the lock every driver takes."""

from __future__ import annotations

import json
import threading
import time

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


def test_the_note_is_read_only_while_the_phone_is_held(tmp_path):
    assert phonelock.note(tmp_path) == {}
    with PhoneLock(tmp_path, "adbagent run: turn on wifi"):
        assert phonelock.note(tmp_path)["holder"] == "adbagent run: turn on wifi"
    (tmp_path / phonelock.NOTE_NAME).write_text(
        json.dumps({"holder": "a run that crashed"}), encoding="utf-8")
    assert phonelock.note(tmp_path) == {}


def test_a_look_is_waited_out_rather_than_refused(tmp_path):
    """A watch that has let go of the phone still takes it for a moment, every
    interval, to read the screen. A driver starting in that moment gets the
    phone a moment later instead of being told it is busy."""
    looker = PhoneLock(tmp_path, "adbagent watch: reply to new chats").acquire(
        looking=True)
    assert "looking at the screen for a moment" in phonelock.holder(tmp_path)
    assert phonelock.note(tmp_path)["looking"] is True
    threading.Timer(0.5, looker.release).start()
    began = time.monotonic()
    with PhoneLock(tmp_path, "adbagent run: turn on wifi"):
        assert time.monotonic() - began >= 0.4
        note = phonelock.note(tmp_path)
        assert note["holder"] == "adbagent run: turn on wifi"
        assert "looking" not in note


def test_a_driver_is_still_refused_at_once(tmp_path, monkeypatch):
    """The patience is for a look only: a phone somebody is driving is said
    to be busy straight away, as it always was."""
    monkeypatch.setattr(phonelock, "PATIENCE_S", 30.0)
    with PhoneLock(tmp_path, "an MCP session (antigravity-client): Hinge"):
        began = time.monotonic()
        with pytest.raises(PhoneBusy, match="is driving the phone"):
            PhoneLock(tmp_path, "adbagent run: turn on wifi").acquire()
        assert time.monotonic() - began < 2


def test_a_look_that_outlasts_the_patience_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(phonelock, "PATIENCE_S", 0.5)
    with PhoneLock(tmp_path, "adbagent watch: x").acquire(looking=True):
        with pytest.raises(PhoneBusy, match="looking at the screen"):
            PhoneLock(tmp_path, "adbagent run: turn on wifi").acquire()
