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

Holding it is not the same as using it. A watch spends most of its life with
nothing new on screen, and an MCP session waits on a driver that may be reading
or thinking -- or gone. Either one, once it has gone `device.let_go_after_s`
without acting, closes its device session and lets go of the lock, so that a run,
another session or another watch can have the phone, and takes it back when it
next has something to do (`Watch`, `remote.RemoteAgent`). A watch that has let go
still looks at the screen every interval, and takes the lock for that moment
with ``looking=True``; whoever finds the phone taken by a look waits for it
rather than being turned away (`PATIENCE_S`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import IO, Any, Dict, Optional

LOCK_NAME = "phone.lock"
NOTE_NAME = "phone.lock.json"

#: How long a driver about to start waits for a phone that is only being looked
#: at. A watch that has let go of the phone takes it back for a moment every
#: interval to read the screen -- one UI dump and one `dumpsys`, a few seconds,
#: longer when the uiautomator server has to be started first -- and a run or a
#: session refused because it happened to start in that moment would have been
#: refused for nothing. A phone that somebody is actually driving is said to be
#: busy at once, as it always was: the wait is only ever for a look.
PATIENCE_S = 20.0


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


def _read_note(directory: Path) -> Dict[str, Any]:
    try:
        note = json.loads((directory / NOTE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return note if isinstance(note, dict) else {}


def _describe(directory: Path) -> str:
    note = _read_note(directory)
    if not note:
        return "another adbagent process is driving the phone"
    who = note.get("holder") or "another adbagent process"
    pid = note.get("pid", "?")
    if note.get("looking"):
        return (f"{who} is looking at the screen for a moment (pid {pid}); the "
                f"phone is free again in a few seconds")
    since = note.get("since")
    when = ""
    if isinstance(since, (int, float)):
        when = f" (since {time.strftime('%H:%M', time.localtime(since))})"
    return f"{who} is driving the phone (pid {pid}){when}"


def note(directory: str | Path) -> Dict[str, Any]:
    """The note of whoever holds the phone, or {} when nobody does.

    The lock is tried first, as `holder` does, so a note left behind by a
    holder that crashed is never mistaken for one that is there.
    """
    directory = Path(directory)
    path = directory / LOCK_NAME
    if not path.exists():
        return {}
    try:
        handle = open(path, "a+b")
    except OSError:
        return {}
    try:
        if _try_lock(handle):
            _unlock(handle)
            return {}
        return _read_note(directory) or {"holder": ""}
    finally:
        handle.close()


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

    def acquire(self, *, looking: bool = False) -> "PhoneLock":
        """Take the phone, or raise `PhoneBusy` saying who has it.

        `looking` says it is wanted only for a moment, to read the screen: the
        note says so, and anyone else wanting the phone in that moment waits
        for it instead of being refused (`PATIENCE_S`).
        """
        if self._handle is not None:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = open(self.directory / LOCK_NAME, "a+b")
        deadline = time.monotonic() + PATIENCE_S
        # Twice, a beat apart: `holder` takes the lock for an instant to test it,
        # and a driver that started in that instant would read the probe as a
        # holder. Then again for as long as the holder says it is only looking.
        while not _try_lock(handle):
            time.sleep(0.05)
            if _try_lock(handle):
                break
            if (not _read_note(self.directory).get("looking")
                    or time.monotonic() >= deadline):
                handle.close()
                raise PhoneBusy(_describe(self.directory))
            time.sleep(0.2)
        self._handle = handle
        written = {"holder": self.who, "pid": os.getpid(), "since": time.time()}
        if looking:
            written["looking"] = True
        try:
            (self.directory / NOTE_NAME).write_text(json.dumps(written),
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
