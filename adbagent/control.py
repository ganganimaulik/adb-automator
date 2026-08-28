"""The one channel from outside into a run that is already going.

Everything a run says travels outward through files: `events.jsonl` for what it
decided, `stream.jsonl` for the model thinking, `screens.jsonl` for where the
elements were. Nothing travelled the other way. A run started from the browser
could be *stopped* -- a SIGINT, which is the whole reason runs are subprocesses
-- and that was the entire vocabulary. Watching an agent walk into a wall and
having no way to say "wait" is the gap this closes.

**Why a file.** The obvious channel is the child's stdin, and it is the wrong
one: `runner` spawns runs with `stdin=DEVNULL` on purpose, because the CLI's
destructive-action prompt is an `input()` and a web run must never be able to
block on it. Handing that pipe a second job puts the prompt back within reach of
a bug. A file also matches the direction that already works -- one directory per
run, both sides reading and writing files in it -- and needs no cleanup when
either end dies.

**Why in the run directory.** It scopes itself. A new run is a new directory, so
a command nobody consumed cannot leak into the next run the way a command parked
in a well-known location would. Under `--repeat` each iteration is its own
directory, so a pause does not survive an iteration boundary on its own; the
server re-applies it, because a pause is a mode somebody switched on and not an
instruction to one iteration.

**What is quiescent.** Commands are read at the top of a step, next to where the
checkpoint is written, and nowhere else. That is the only place the run is
between things rather than in the middle of one: the last step is complete, the
next has not begun, and the phone is where the last action left it. Pausing
anywhere else would stop the loop between an observation and the action it was
made for.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("adbagent.control")

#: The file, inside `runs/<id>/`.
NAME = "control.json"

#: What may be asked for.
#:
#: `pause` and `run` are a mode; `step` is a mode too -- "run one step, then be
#: paused" -- rather than an event, so that a `step` arriving while the loop is
#: between polls is not lost the way a queued one-shot would be.
COMMANDS = ("pause", "run", "step")

#: How often a paused loop looks for what to do next. Short enough that resuming
#: feels like a button and not a request, long enough that a run left paused
#: overnight is not spinning: at four reads a second of a file the OS has in
#: cache, this costs less than the `adb` call it is standing in for.
POLL_S = 0.25


@dataclass
class Command:
    """One instruction, and the count that says whether it is a new one."""

    seq: int = 0
    cmd: str = "run"

    @property
    def valid(self) -> bool:
        return self.cmd in COMMANDS


def _path(run_dir: Any) -> Path:
    return Path(run_dir).expanduser() / NAME


def read(run_dir: Any) -> Optional[Command]:
    """The command sitting in a run's directory, or None when there is none.

    Unreadable is the same as absent, and deliberately: this file is written by
    another process, so a read landing between the write and the rename sees a
    partial file, and the answer to that is to look again in a quarter of a
    second rather than to fail a run over it.
    """
    try:
        data = json.loads(_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    command = Command(seq=int(data.get("seq") or 0),
                      cmd=str(data.get("cmd") or ""))
    return command if command.valid else None


def send(run_dir: Any, cmd: str, seq: int) -> None:
    """Put a command where the run will find it.

    Written to a neighbouring file and renamed over the target. `os.replace` is
    atomic on both platforms this runs on, so the reader sees either the old
    command or the new one and never half of either -- which a plain rewrite in
    place does not promise, and which matters precisely because the reader is
    polling as fast as it can.
    """
    if cmd not in COMMANDS:
        raise ValueError(f"unknown command {cmd!r}")
    target = _path(run_dir)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"seq": seq, "cmd": cmd}), encoding="utf-8")
    os.replace(tmp, target)


def clear(run_dir: Any) -> None:
    """Drop a run's control file. Nothing is asking for anything."""
    try:
        _path(run_dir).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not remove the control file: %s", exc)


class Control:
    """A run's end of the channel: what it has been told, and the waiting.

    Held by the agent and polled once a step. A run whose directory never gets a
    control file pays one failed `open` per step for the privilege, which is
    nothing beside the `adb` round trip that follows it.
    """

    def __init__(self, run_dir: Any,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.run_dir = Path(run_dir)
        self.mode = "run"
        #: The highest `seq` acted on. A command is obeyed once: re-reading the
        #: same file every quarter second must not re-announce it, and a resumed
        #: run must not act on what the last sitting was already told.
        self.seen = 0
        #: The last thing said out loud. What is announced is what the loop is
        #: *doing*, which is not the same as what it was last told: a `step`
        #: spends itself and leaves the run held, so one command produces two
        #: states and the second of them is the one somebody is waiting to see.
        self.reported = "run"
        self._sleep = sleep

    def _poll(self) -> None:
        """Take on whatever the run has been told since the last look."""
        command = read(self.run_dir)
        if command is None or command.seq <= self.seen:
            return
        self.seen = command.seq
        self.mode = command.cmd

    def wait(self, state: Any, on_change: Optional[Callable[..., None]] = None) -> None:
        """Return at once unless paused; block here for as long as it is.

        Called at the top of a step. `step` means "this one, then stop", so it
        becomes `pause` on the way past -- the next call blocks, and the step in
        between runs exactly as any other does.

        The time spent here is given back to the run. `RunState.elapsed` is what
        the wall-clock budget is measured against, and a run held for five
        minutes by the person watching it has not spent five minutes of its
        budget: without this, resuming a paused run is how you discover it was
        killed while you were reading it.

        `KeyboardInterrupt` is not caught anywhere in here. Stopping is what
        restores the phone's keyboard, animations and rotation, and it has to
        stay able to interrupt a pause -- which, with a plain sleep loop and no
        handler, it does.
        """
        def say(mode: str) -> None:
            if mode == self.reported:
                return
            self.reported = mode
            log.info("control: %s", mode)
            if on_change is not None:
                on_change(mode)

        def spend_step() -> None:
            """Take the one step that was bought, and be held again after it."""
            self.mode = "pause"
            # Said every time, even back to back: two steps in a row are two
            # things that happened, and a state machine that deduplicated them
            # would show the second as nothing happening at all.
            self.reported = None
            say("step")

        self._poll()
        if self.mode == "step":
            spend_step()
            return
        if self.mode != "pause":
            say("run")
            return

        held_from = time.monotonic()
        try:
            say("pause")
            while self.mode == "pause":
                self._sleep(POLL_S)
                self._poll()
                if self.mode == "step":
                    spend_step()
                    return
            say(self.mode)          # let go
        finally:
            # In a `finally` so that a run interrupted mid-pause still hands the
            # time back before it unwinds -- `run_end` is written on the way out
            # and reports this, and a number lost here would leave the wall
            # clock in the trace disagreeing with the one on the page.
            state.paused_s += time.monotonic() - held_from
