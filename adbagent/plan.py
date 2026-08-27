"""The plan ledger: which sub-steps of the goal are done, maintained by code.

:mod:`scratchpad` fixed this problem once already, for collected data. The model
used to rewrite its whole ledger every turn, and in ``runs/af76720d05c4`` four
measured readings vanished in a single rewrite. The fix was to stop asking for a
rewrite: the model sends what is new or corrected, the harness keeps the union,
and a record it stops mentioning cannot be dropped because nothing replaces it.

The ``progress`` field was left behind by that change. It stayed one free-text
string rewritten from scratch every turn -- ``state.progress_log`` was assigned a
fresh one-element list each time -- so a sub-step the model stopped mentioning
was a sub-step nobody could get back, with no detector and nothing in the trace
to show it went. This module gives ``progress`` the treatment ``notes`` got:
``{id, text, status}`` records in, the union kept here.

**Why the harness may now credit it.** `RunState.note_progress` deliberately
stopped listening to ``progress``, and the measurements in its docstring say
why: the field was present on 76 of 103 turns and its text changed on 72, so the
model was resetting the stall ladder on 70% of all steps by rewording, and three
of the ladder's four tiers had never fired. Every signal that survived is about
the device -- a screen not seen before, content that moved, a setting that
flipped -- or about data collection. None is about the goal's own structure, so
a run working steadily through sub-step 3 of 5 in a familiar app reads as a
stall.

A status transition can carry that signal where the prose could not, because
here it can be made un-gameable rather than merely discouraged:

* **Only a step declared on an earlier turn is paid.** Declaring and completing
  five steps in one breath buys nothing, which closes off the rewording that
  made the old field worthless.
* **Each id is paid at most once, ever.** Reaching ``done`` *consumes* an id's
  one credit whether or not it was paid, so declaring a step already done and
  then oscillating it back to ``pending`` cannot mint a second chance.
* **Corrections stay honest.** A model that marked something done and finds it
  was not may say so, and say so again when it is really finished. It just does
  not get paid twice.

What remains uncredited is prose. A model that ignores the record shape and
writes a sentence -- and every action replayed from a recording made before this
change -- lands in one reserved entry (:data:`PROSE_ID`) that is overwritten
rather than accumulated, and that is never eligible to credit anything. Being
un-gameable is not conditional on compliance.

The block is bounded by :data:`MAX_STEPS` and :data:`MAX_TEXT_CHARS` rather than
by a character budget like the scratchpad's: a plan is a handful of titles, and
a plan that reached forty entries is being used as a log, which is the case the
cap exists to stop rather than to render tidily.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .scratchpad import normalise_key

log = logging.getLogger("adbagent.plan")

#: The id every prose ``progress`` value lands under. Reserved: one entry that
#: successive turns overwrite, so a model rewriting its sentence every turn
#: cannot flood the plan with near-duplicate steps.
PROSE_ID = "note"

#: The statuses a step may hold. ``blocked`` is not a failure -- it is "this one
#: cannot proceed and the run should route around it", which is exactly what the
#: replan call wants to be told.
STATUSES = ("pending", "active", "done", "blocked")
DEFAULT_STATUS = "pending"

#: Spellings a model reaches for that mean one of :data:`STATUSES`. Anything not
#: listed leaves an existing step's status alone rather than resetting it -- an
#: unreadable status is missing information, not a demotion to ``pending``.
_STATUS_ALIASES = {
    "complete": "done", "completed": "done", "finished": "done",
    "finish": "done", "ok": "done", "yes": "done", "true": "done",
    "in_progress": "active", "in progress": "active", "inprogress": "active",
    "doing": "active", "current": "active", "started": "active",
    "working": "active", "now": "active",
    "todo": "pending", "to do": "pending", "to_do": "pending",
    "not started": "pending", "not_started": "pending", "next": "pending",
    "remaining": "pending", "waiting": "pending", "open": "pending",
    "stuck": "blocked", "failed": "blocked", "fail": "blocked",
    "cannot": "blocked", "skipped": "blocked",
}

#: A step title. Long enough for "read today's messages from Priya", short
#: enough that forty of them cannot dominate the prompt.
MAX_TEXT_CHARS = 120
#: The prose fallback gets more room: it is a whole status line, not a title,
#: and truncating it mid-sentence loses the part that says what remains.
MAX_PROSE_CHARS = 400
#: Ceiling on the plan. Reached only by a run treating the field as a log; new
#: steps past it are refused and counted rather than evicting the plan the run
#: is actually following.
MAX_STEPS = 40

_MARK = {"pending": " ", "active": ">", "done": "x", "blocked": "!"}
_LEGEND = "[x] done, [>] doing now, [ ] still to do, [!] blocked"


def normalise_status(status: Any) -> str:
    """One of :data:`STATUSES`, or ``""`` when the value says nothing usable."""
    text = " ".join(str(status or "").strip().lower().split())
    if text in STATUSES:
        return text
    return _STATUS_ALIASES.get(text, "")


def as_steps(progress: Any) -> List[Tuple[str, str, str]]:
    """Whatever arrived in ``progress``, as ``(id, text, status)`` triples.

    Accepts the record list the schema asks for, the bare string every recording
    made before this change contains, and the shapes in between. A *top-level*
    string is the legacy prose format and goes to :data:`PROSE_ID`; a string
    inside a list is a step title being declared, which is a different thing and
    is keyed by its own text.
    """
    if progress is None or progress == "" or progress == []:
        return []

    if isinstance(progress, str):
        text = " ".join(progress.split())[:MAX_PROSE_CHARS]
        return [(PROSE_ID, text, "active")] if text else []

    if isinstance(progress, dict):
        if any(k in progress for k in ("id", "text", "status", "step")):
            return as_steps([progress])
        # A mapping of step -> status, e.g. {"open the app": "done"}. The value
        # is read as a status when it is one and as the title otherwise, so
        # {"1": "open the app"} works too.
        out: List[Tuple[str, str, str]] = []
        for raw_id, raw_value in progress.items():
            if not str(raw_id).strip():
                continue
            status = normalise_status(raw_value)
            text = "" if status else str(raw_value or "")
            out.append((str(raw_id), text, status))
        return out

    if isinstance(progress, Sequence):
        out = []
        for item in progress:
            if isinstance(item, str):
                title = " ".join(item.split())
                if title:
                    out.append((title, title, ""))
                continue
            if isinstance(item, dict):
                sid = item.get("id") or item.get("step") or item.get("text")
                text = item.get("text")
                status = item.get("status")
            else:
                sid = (getattr(item, "id", None) or getattr(item, "step", None)
                       or getattr(item, "text", None))
                text = getattr(item, "text", None)
                status = getattr(item, "status", None)
            if not str(sid or "").strip():
                continue
            out.append((str(sid), str(text or ""), normalise_status(status)))
        return out

    return as_steps(str(progress))


@dataclass
class Step:
    """One sub-step of the goal, and where the run has got to with it."""

    id: str
    text: str = ""
    status: str = DEFAULT_STATUS
    #: Run step it was first declared on. What makes "declared earlier" -- the
    #: condition for being paid -- a fact rather than a claim.
    first_step: int = 0
    last_step: int = 0

    @property
    def prose(self) -> bool:
        return normalise_key(self.id) == PROSE_ID

    def render(self) -> str:
        if self.prose:
            return self.text
        return f"[{_MARK.get(self.status, ' ')}] {self.text or self.id}"


@dataclass
class PlanUpdate:
    """What one turn's delta did to the plan."""

    added: List[str] = field(default_factory=list)
    #: Ids that transitioned to ``done`` *and* earned a stall-ladder reset. The
    #: only field `Agent._loop` acts on; see the module docstring for the two
    #: rules that keep it honest.
    completed: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    #: New steps turned away because the plan is already at :data:`MAX_STEPS`.
    refused: int = 0

    def __bool__(self) -> bool:
        return bool(self.changed or self.refused)


@dataclass
class TaskLedger:
    """The run's plan: every sub-step declared, in the order declared."""

    entries: "OrderedDict[str, Step]" = field(default_factory=OrderedDict)
    #: Ids that have already used up their one credit. One-way, and kept
    #: separately from the entries so that correcting a step -- or refusing it at
    #: the cap -- cannot hand its credit back.
    credited: Set[str] = field(default_factory=set)
    #: New steps refused at the cap. Reported rather than hidden, for the same
    #: reason `NoteLedger` reports evictions.
    refused: int = 0

    def __len__(self) -> int:
        return sum(1 for e in self.entries.values() if not e.prose)

    def __bool__(self) -> bool:
        return bool(self.entries)

    # -- writing -----------------------------------------------------------

    def update(self, progress: Any, step: int) -> PlanUpdate:
        """Merge this turn's delta in, and report what it changed."""
        out = PlanUpdate()
        for raw_id, raw_text, raw_status in as_steps(progress):
            key = normalise_key(raw_id)
            if not key:
                continue
            limit = MAX_PROSE_CHARS if key == PROSE_ID else MAX_TEXT_CHARS
            text = " ".join(str(raw_text).split())[:limit]
            status = normalise_status(raw_status)
            existing = self.entries.get(key)

            if existing is None:
                if len(self.entries) >= MAX_STEPS:
                    out.refused += 1
                    self.refused += 1
                    continue
                status = status or DEFAULT_STATUS
                self.entries[key] = Step(
                    id=str(raw_id).strip()[:MAX_TEXT_CHARS] or key,
                    text=text or str(raw_id).strip()[:limit],
                    status=status, first_step=step, last_step=step)
                # Arriving already done consumes the credit without paying it:
                # the step was never outstanding, and leaving the credit unspent
                # is what would let an oscillation collect it later.
                if status == "done":
                    self.credited.add(key)
                out.added.append(key)
                out.changed.append(key)
                continue

            existing.last_step = step
            touched = False
            if text and text != existing.text:
                existing.text = text
                touched = True
            if status and status != existing.status:
                was_done = existing.status == "done"
                existing.status = status
                touched = True
                if status == "done" and not was_done \
                        and key not in self.credited:
                    self.credited.add(key)
                    if existing.first_step < step:
                        out.completed.append(key)
            if touched:
                out.changed.append(key)

        if out.changed:
            log.info("plan: %d step(s) changed (%d of %d done)%s: %s",
                     len(out.changed), self.done_count, len(self),
                     f", {len(out.completed)} completed" if out.completed else "",
                     ", ".join(out.changed[:8]))
        if out.refused:
            log.warning("plan: refused %d new step(s) -- already at the cap of "
                        "%d", out.refused, MAX_STEPS)
        return out

    # -- reading -----------------------------------------------------------

    @property
    def steps(self) -> List[Step]:
        """The real sub-steps, prose fallback excluded."""
        return [e for e in self.entries.values() if not e.prose]

    @property
    def done_count(self) -> int:
        return sum(1 for e in self.steps if e.status == "done")

    def outstanding(self) -> List[str]:
        """What the plan still says is left, for the judge and the stall note."""
        return [e.text or e.id for e in self.steps if e.status != "done"]

    def records(self, ids: Sequence[str] = ()) -> List[Dict[str, Any]]:
        """The steps `ids` name -- all of them when empty -- as JSON.

        Written into the ``plan`` event, so a reader sees the plan the run was
        following rather than only the names of what changed. The normalised id
        rides along as ``id`` for the reason `NoteLedger.records` gives: this
        module owns how two spellings become one step, and the web feed keeping
        the union of these deltas must not re-derive that and drift.
        """
        wanted = list(ids) if ids else list(self.entries)
        out: List[Dict[str, Any]] = []
        for raw in wanted:
            key = normalise_key(raw)
            entry = self.entries.get(key)
            if entry is None:
                continue
            out.append({"id": key, "text": entry.text or entry.id,
                        "status": entry.status, "step": entry.last_step})
        return out

    def plain(self) -> str:
        """The plan alone, for a terminal, a report, the judge or the replan."""
        lines = [line for line in (e.render() for e in self.entries.values())
                 if line.strip()]
        if self.refused:
            lines.append(f"(... {self.refused} further step(s) refused -- the "
                         f"plan is at its cap of {MAX_STEPS})")
        return "\n".join(lines)

    def render(self) -> str:
        """The block handed to the model.

        States that the harness owns it, for the same reason
        `NoteLedger.render` does: the previous contract had the model restate
        everything, and a model that keeps doing so spends its output budget
        re-emitting a list it is already looking at.
        """
        body = self.plain()
        if not body:
            return ""
        total = len(self)
        if total:
            head = (f"YOUR PLAN ({self.done_count} of {total} step(s) done). "
                    f"This is kept for you and cannot be lost -- do NOT restate "
                    f"it. Send only steps that are new or whose status changed. "
                    f"{_LEGEND}:")
        else:
            # Prose only: there is no checklist to explain, and the delta
            # contract does not apply to a line that is overwritten anyway.
            head = "YOUR PROGRESS (your working memory of what is done and what "
            head += "remains):"
        return "\n".join([head] + [f"  {line}" for line in body.splitlines()])


def replay(events: Iterable[Any]) -> TaskLedger:
    """Rebuild a plan from recorded ``decide`` events.

    ``progress`` in an event is a delta now, so the last one is not the whole
    plan -- reading a finished run means replaying them all, exactly as
    `scratchpad.replay` does for the data ledger.
    """
    ledger = TaskLedger()
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or event.get("kind") != "decide":
            continue
        action = event.get("action")
        if isinstance(action, dict) and action.get("progress"):
            ledger.update(action["progress"], event.get("step", index))
    return ledger
