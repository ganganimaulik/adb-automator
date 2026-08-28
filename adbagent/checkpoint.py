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
from pathlib import Path
from typing import Any, Dict, Optional

from . import runlog
from .plan import DEFAULT_STATUS, Step, normalise_status
from .scratchpad import Entry, normalise_key

log = logging.getLogger("adbagent.checkpoint")

#: The file, inside `runs/<id>/`.
NAME = "checkpoint.json"

#: Written into every checkpoint so a future format change can tell what it is
#: looking at rather than guessing from the keys.
VERSION = 1

#: An answer to the run's `ask_user`, put here by whoever was asked.
#:
#: `save` never writes this key -- it is not part of `RunState`, and the run that
#: asked the question has already ended by the time there is anything to answer.
#: `set_answer` adds it to the file afterwards, and `restore` folds it into the
#: resumed run's history and leaves it behind. The next step's `save` rewrites
#: the whole file, so a consumed answer disappears on its own rather than being
#: re-injected into a second `ask_user` that meant something else.
#:
#: Which is also why VERSION does not move: what `save` produces is unchanged,
#: and a checkpoint carrying this key is one an outside hand has written on.
ANSWER = "user_answer"


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
            # Flattened because the key is a tuple and JSON keys are strings.
            # Kept rather than rebuilt from `history`, which is a twenty-entry
            # ring buffer: "have I tried this here before" is asked precisely
            # about the steps that have already fallen out of it.
            "attempts": [[sid, sig, n]
                         for (sid, sig), n in state.loops.attempts.items()],
            "element_actions": {
                k: [list(entry) for entry in v]
                for k, v in state.loops.element_actions.items()
            },
            "scroll_dir_log": state.loops.scroll_dir_log,
            "scroll_exhausted": state.loops.scroll_exhausted,
            "total_scroll_count": state.loops.total_scroll_count,
            "dead_scrolls": state.loops.dead_scrolls,
            "consecutive_backs": state.loops.consecutive_backs,
        },
        "want_screenshot": state.want_screenshot,
        "last_failure": state.last_failure,
        "scroll_warnings": state.scroll_warnings,
        # The stall ladder. A resumed run that dropped these would restart at
        # tier zero and have to rediscover, over another eight steps, the
        # stall it was already in the middle of.
        "steps_since_progress": state.steps_since_progress,
        "last_progress": state.last_progress,
        "strategy": state.strategy,
        "replanned_at": state.replanned_at,
        "scratchpad": {
            "entries": [
                {"key": e.key, "value": e.value,
                 "first_step": e.first_step, "last_step": e.last_step,
                 "superseded": e.superseded}
                for e in state.scratchpad.entries.values()
            ],
            "evicted": state.scratchpad.evicted,
        },
        # `credited` goes with the steps, and is the reason this is not just a
        # list of statuses. Without it a resumed run could be paid a second time
        # for every step it had already finished -- one free stall-ladder reset
        # per completed step, handed over at exactly the point a run is most
        # likely to be in trouble.
        "plan": {
            "steps": [
                {"id": s.id, "text": s.text, "status": s.status,
                 "first_step": s.first_step, "last_step": s.last_step}
                for s in state.plan.entries.values()
            ],
            "credited": sorted(state.plan.credited),
            "refused": state.plan.refused,
        },
        "packages": sorted(state.packages),
        "package_steps": state.package_steps,
        # A whole item ledger used to be persisted here -- per-item captions,
        # read flags, the set's size and which ends had been hit. None of it
        # survives, because none of it was knowable; see `pager.py`. What is
        # left is the readings of a sweep still in flight, which are lost
        # anyway if the process dies mid-sweep.
        "sweep": {
            "gesture": state.sweep.gesture,
            "readings": list(state.sweep.readings),
            "repeats": state.sweep.repeats,
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


def set_answer(run_dir: Any, text: str) -> bool:
    """Record the answer to a run's `ask_user`, for its resume to pick up.

    False when there is no checkpoint to write on, which is the whole of "this
    run cannot be answered": a run that succeeded has had its checkpoint cleared,
    and one that never wrote one has nothing to continue from either. An
    unreadable or malformed file is the same answer -- `load` has already
    logged why.

    Not guarded against a live run rewriting the file underneath: the only run
    that can be answered is one that has already stopped to ask, and it stopped
    by ending. A `save` racing this could only come from a *resumed* sitting, by
    which point the answer being overwritten has already been read.
    """
    path = Path(run_dir).expanduser() / NAME
    data = load(path.parent)
    if data is None:
        return False
    data[ANSWER] = " ".join(str(text).split())
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    return True


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
    state.steps_since_progress = int(data.get("steps_since_progress") or 0)
    state.last_progress = str(data.get("last_progress") or "the run was resumed")
    state.strategy = str(data.get("strategy") or "")
    state.replanned_at = int(data.get("replanned_at") or 0)
    plan = data.get("plan") or {}
    for raw in plan.get("steps") or []:
        sid = str(raw.get("id") or "")
        if not sid:
            continue
        state.plan.entries[normalise_key(sid)] = Step(
            id=sid, text=str(raw.get("text") or ""),
            status=normalise_status(raw.get("status")) or DEFAULT_STATUS,
            first_step=int(raw.get("first_step") or 0),
            last_step=int(raw.get("last_step") or 0))
    state.plan.credited = {normalise_key(c) for c in plan.get("credited") or []
                           if str(c).strip()}
    state.plan.refused = int(plan.get("refused") or 0)
    # A checkpoint written before the plan existed carries `progress_log`, a
    # one-element list of the free-text status. It restores the way a model
    # writing prose does today -- into the one entry that is never credited --
    # so resuming an older run keeps its working memory instead of dropping it.
    if not state.plan.entries:
        for line in data.get("progress_log") or []:
            state.plan.update(str(line), 0)
    state.packages = set(data.get("packages") or [])
    state.package_steps = {str(k): int(v)
                           for k, v in (data.get("package_steps") or {}).items()}

    loops = data.get("loops") or {}
    state.loops.history = [tuple(pair) for pair in loops.get("history") or []]
    state.loops.banned = {str(k): set(v)
                          for k, v in (loops.get("banned") or {}).items()}
    state.loops.attempts = {}
    for row in loops.get("attempts") or []:
        try:
            sid, sig, n = row
        except (TypeError, ValueError):
            continue
        state.loops.attempts[(str(sid), str(sig))] = int(n)
    state.loops.element_actions = {
        str(k): [tuple(entry) for entry in v]
        for k, v in (loops.get("element_actions") or {}).items()
    }
    state.loops.scroll_dir_log = [str(d) for d in loops.get("scroll_dir_log") or []]
    # Missing in checkpoints written before the flag existed: an older file
    # restores with nothing marked, which counts reversals the way it used to.
    state.loops.scroll_exhausted = [int(i) for i in
                                    loops.get("scroll_exhausted") or []]
    state.loops.total_scroll_count = int(loops.get("total_scroll_count") or 0)
    state.loops.dead_scrolls = {str(k): str(v)
                                for k, v in (loops.get("dead_scrolls") or {}).items()}
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

    sweep = data.get("sweep") or {}
    state.sweep.gesture = str(sweep.get("gesture") or "")
    state.sweep.readings = [str(r) for r in sweep.get("readings") or []]
    state.sweep.repeats = int(sweep.get("repeats") or 0)

    # The answer to whatever the run stopped to ask, as the last thing that
    # happened -- which is what it is. It enters as history rather than through
    # a prompt of its own because history is already the block the model reads
    # to know what has gone on, and an answer needs no special pleading to be
    # read there. Appended last so it lands after the step that asked.
    #
    # It is a *window*, not a permanent fact: `prompts.HISTORY_KEEP` shows the
    # last two dozen lines, so an answer goes out of view once the resumed run
    # has run that far past it. That suits what these answers are -- a code, a
    # choice, a confirmation, all of them wanted on the next step or not at all
    # -- and anything the run must not forget belongs in the scratchpad.
    answer = str(data.get(ANSWER) or "").strip()
    if answer:
        state.remember(f"{state.step}. the person was asked for something and "
                       f"answered: {answer}")
