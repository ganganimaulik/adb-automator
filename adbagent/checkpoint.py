"""Where a failed run keeps what it learned, so it can be continued.

A run that exhausts its step budget used to be a total loss: forty steps of
history, the scratchpad it filled, the gallery items it had already read, the
dead ends it mapped -- all of it thrown away, and the next invocation started
from the launcher knowing nothing. Worse, the events it left behind recorded
*that* it failed but not the state it failed *in*, so "pick up where it
stopped" was not even reconstructable after the fact.

This module writes that state to ``runs/<id>/checkpoint.json`` at the top of
every step, and ``adbagent run --resume`` loads it back. The resumed run keeps
its own directory -- events.jsonl is appended to, not started over -- so one
file still reads as one run, with a ``run_resume`` event where the sessions
join. A run that succeeds deletes its checkpoint: a checkpoint on disk means
"this run has unfinished business", and `--resume latest` can simply take the
newest directory that has one.

Only what survives the process belongs here. The live screen does not: the
phone has moved on by the time anyone resumes, so the loop re-observes and the
model re-decides from the restored history rather than from a frame that no
longer exists. The wall-clock budget likewise restarts -- it bounds one
sitting, not one goal.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from . import runlog
from .pager import ItemRecord
from .scratchpad import Entry, normalise_key

log = logging.getLogger("adbagent.checkpoint")

#: The file, inside `runs/<id>/`.
NAME = "checkpoint.json"

#: Written into every checkpoint so a future format change can tell what it is
#: looking at rather than guessing from the keys.
VERSION = 1


def _path(cfg: Any, run_id: str) -> Path:
    return runlog.run_dir(cfg, run_id) / NAME


def save(cfg: Any, state: Any) -> None:
    """Snapshot the run's state next to its events.

    Called at the top of every step, so the cost has to stay next to nothing
    next to an adb round trip -- a few kilobytes of JSON. A failure to write
    is logged, never raised: a run must not die of its own insurance.
    """
    data = {
        "version": VERSION,
        "saved_at": round(time.time(), 3),
        "goal": state.goal,
        "run_id": state.run_id,
        "intent_id": state.intent_id,
        "step": state.step,
        "llm_calls": state.llm_calls,
        "consecutive_failures": state.consecutive_failures,
        "history": state.history,
        "visits": state.visits,
        "loops": {
            "history": [list(pair) for pair in state.loops.history],
            "banned": {k: sorted(v) for k, v in state.loops.banned.items()},
            "element_actions": {
                k: [list(entry) for entry in v]
                for k, v in state.loops.element_actions.items()
            },
            "scroll_dir_log": state.loops.scroll_dir_log,
            "total_scroll_count": state.loops.total_scroll_count,
            "consecutive_backs": state.loops.consecutive_backs,
        },
        "want_screenshot": state.want_screenshot,
        "last_failure": state.last_failure,
        "scroll_warnings": state.scroll_warnings,
        "scratchpad": {
            "entries": [
                {"key": e.key, "value": e.value,
                 "first_step": e.first_step, "last_step": e.last_step,
                 "superseded": e.superseded}
                for e in state.scratchpad.entries.values()
            ],
            "evicted": state.scratchpad.evicted,
        },
        "progress_log": state.progress_log,
        "packages": sorted(state.packages),
        "package_steps": state.package_steps,
        "items": {
            # The dict key and the record's own `key` differ: the first is the
            # ledger key, which a `#2` suffix disambiguates when two items
            # share a caption; the second is the raw key both were minted from.
            "items": [
                {"ledger_key": ledger_key, "key": r.key, "label": r.label,
                 "first_step": r.first_step, "last_step": r.last_step,
                 "read": r.read, "detail": r.detail, "visits": r.visits}
                for ledger_key, r in state.items.items.items()
            ],
            "total": state.items.total,
            "set_id": state.items.set_id,
            "cursor": state.items.cursor,
            "edges": sorted(state.items.edges),
        },
    }
    try:
        path = _path(cfg, state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    except OSError as exc:
        log.warning("could not write the checkpoint: %s", exc)


def load(run_dir: Any) -> Optional[Dict[str, Any]]:
    """A run's checkpoint, or None when it has none (or none readable)."""
    path = Path(run_dir).expanduser() / NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if path.exists():
            log.warning("could not read the checkpoint at %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not data.get("goal"):
        return None
    return data


def clear(cfg: Any, run_id: str) -> None:
    """Drop a run's checkpoint. Success means there is nothing left to resume."""
    try:
        _path(cfg, run_id).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not remove the checkpoint: %s", exc)


def latest_resumable(runs_dir: Any) -> Optional[Path]:
    """The newest run directory that still has a checkpoint.

    `--resume` with no argument means "continue what I was doing", and what the
    user was doing is the run that did not finish -- not a later one that did.
    """
    runs_dir = Path(runs_dir).expanduser()
    if not runs_dir.is_dir():
        return None
    candidates = [d for d in runs_dir.iterdir() if (d / NAME).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / NAME).stat().st_mtime)


def restore(state: Any, data: Dict[str, Any]) -> None:
    """Fill `state` with what a checkpoint knows, leaving the rest fresh.

    Every field is read defensively: a checkpoint outlives the process, so it
    can outlive the code that wrote it, and resuming must never be harder than
    starting over. What is not here simply keeps its dataclass default, which
    is the same state a fresh run would have had.
    """
    state.step = int(data.get("step") or 0)
    state.llm_calls = int(data.get("llm_calls") or 0)
    state.consecutive_failures = int(data.get("consecutive_failures") or 0)
    state.history = [str(line) for line in data.get("history") or []]
    state.visits = {str(k): int(v) for k, v in (data.get("visits") or {}).items()}
    state.want_screenshot = bool(data.get("want_screenshot"))
    state.last_failure = str(data.get("last_failure") or "")
    state.scroll_warnings = int(data.get("scroll_warnings") or 0)
    state.progress_log = [str(line) for line in data.get("progress_log") or []]
    state.packages = set(data.get("packages") or [])
    state.package_steps = {str(k): int(v)
                           for k, v in (data.get("package_steps") or {}).items()}

    loops = data.get("loops") or {}
    state.loops.history = [tuple(pair) for pair in loops.get("history") or []]
    state.loops.banned = {str(k): set(v)
                          for k, v in (loops.get("banned") or {}).items()}
    state.loops.element_actions = {
        str(k): [tuple(entry) for entry in v]
        for k, v in (loops.get("element_actions") or {}).items()
    }
    state.loops.scroll_dir_log = [str(d) for d in loops.get("scroll_dir_log") or []]
    state.loops.total_scroll_count = int(loops.get("total_scroll_count") or 0)
    state.loops.consecutive_backs = int(loops.get("consecutive_backs") or 0)

    scratch = data.get("scratchpad") or {}
    for raw in scratch.get("entries") or []:
        key = str(raw.get("key") or "")
        if not key:
            continue
        state.scratchpad.entries[normalise_key(key)] = Entry(
            key=key, value=str(raw.get("value") or ""),
            first_step=int(raw.get("first_step") or 0),
            last_step=int(raw.get("last_step") or 0),
            superseded=[str(v) for v in raw.get("superseded") or []])
    state.scratchpad.evicted = int(scratch.get("evicted") or 0)

    items = data.get("items") or {}
    for raw in items.get("items") or []:
        ledger_key = str(raw.get("ledger_key") or raw.get("key") or "")
        if not ledger_key:
            continue
        state.items.items[ledger_key] = ItemRecord(
            key=str(raw.get("key") or ledger_key),
            label=str(raw.get("label") or ledger_key),
            first_step=int(raw.get("first_step") or 0),
            last_step=int(raw.get("last_step") or 0),
            read=bool(raw.get("read")),
            detail=str(raw.get("detail") or ""),
            visits=int(raw.get("visits") or 1))
    state.items.items = OrderedDict(state.items.items)
    state.items.total = int(items.get("total") or 0)
    state.items.set_id = str(items.get("set_id") or "")
    state.items.cursor = str(items.get("cursor") or "")
    state.items.edges = set(items.get("edges") or [])
