"""One driver per phone, across processes.

The web UI already refuses to start a run while another of its own children
holds the phone (`web.server.phone_busy`). What it cannot see is a process it did
not start: a run typed into a terminal, or an MCP session, which a coding agent
launches on its own. Two drivers on one phone each read the other's taps as
their own, and opening a device session resets the animation scales, rotation
and keyboard underneath whatever else is driving it.

So every driver takes this lock for as long as it holds the phone, and anything
about to drive it asks first. An OS file lock rather than a pid file: the
kernel drops it when the holder exits, however it exits, so a crash cannot leave
the phone "busy" until somebody deletes a file. The note beside it only says who
the holder is; it is never what decides whether the phone is free.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import IO, Optional

LOCK_NAME = "phone.lock"
NOTE_NAME = "phone.lock.json"


class PhoneBusy(RuntimeError):
    """Another process is driving the phone."""


def _try_lock(handle: IO[bytes]) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[bytes]) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _describe(directory: Path) -> str:
    try:
        note = json.loads((directory / NOTE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "another adbagent process is driving the phone"
    who = note.get("holder") or "another adbagent process"
    since = note.get("since")
    when = ""
    if isinstance(since, (int, float)):
        when = f" (since {time.strftime('%H:%M', time.localtime(since))})"
    return f"{who} is driving the phone (pid {note.get('pid', '?')}){when}"


def holder(directory: str | Path) -> str:
    """Who holds the phone, or "" when nobody does.

    Answered by trying the lock, not by reading the note: a note outlives a
    holder that crashed, and the lock does not.
    """
    directory = Path(directory)
    path = directory / LOCK_NAME
    if not path.exists():
        return ""
    try:
        handle = open(path, "a+b")
    except OSError:
        return ""
    try:
        if _try_lock(handle):
            _unlock(handle)
            return ""
        return _describe(directory)
    finally:
        handle.close()


class PhoneLock:
    """Held for as long as one driver has the phone. A context manager."""

    def __init__(self, directory: str | Path, who: str):
        self.directory = Path(directory)
        self.who = who
        self._handle: Optional[IO[bytes]] = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "PhoneLock":
        if self._handle is not None:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(self.directory / LOCK_NAME, "a+b")
        # Twice, a beat apart: `holder` takes the lock for an instant to test it,
        # and a driver that started in that instant would read the probe as a
        # holder.
        if not _try_lock(handle):
            time.sleep(0.05)
            if not _try_lock(handle):
                handle.close()
                raise PhoneBusy(_describe(self.directory))
        self._handle = handle
        try:
            (self.directory / NOTE_NAME).write_text(json.dumps(
                {"holder": self.who, "pid": os.getpid(), "since": time.time()}),
                encoding="utf-8")
        except OSError:
            pass  # the note is a courtesy; the lock is what counts
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            (self.directory / NOTE_NAME).unlink()
        except OSError:
            pass
        _unlock(handle)
        handle.close()

    def __enter__(self) -> "PhoneLock":
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
