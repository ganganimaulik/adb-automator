"""The collected-data ledger, maintained by code rather than by the model.

The scratchpad used to be one free-text field the model rewrote from scratch
every turn: it re-emitted its *complete* ledger each time and the loop kept only
the latest value. That put the entire run's findings behind one instruction the
model had to obey perfectly, every single turn, with no way to notice when it did
not -- and it did not. From ``runs/af76720d05c4``, one turn apart:

    step 73  ... 9:45 chicken 425g (OK); 9:51 chicken 426g (+1g); 9:52
             [pending]; 9:59 potatoes 403g (+3g vs menu 400g); 10:03 tomatoes
             120g (matches no-carb tomato 120g).

    step 74  ... 9:45 [pending]; 9:51 [pending]; 9:52 [pending]; 9:59
             [pending]; 10:03 [pending].

Four measured readings gone in one rewrite, never restated across the remaining
59 turns, and the run's final report listed the 10:03 photo as unreadable -- a
question it had already answered and then forgotten. The prompt already told the
model each note must carry ALL items; the instruction was not the missing piece,
and a detector that compared consecutive rewrites and handed back what went
missing was treating the symptom.

So the model no longer writes the ledger. It emits only what is *new or
corrected* this turn -- ``{key, value}`` records -- and this module maintains the
union. A record the model stops mentioning cannot be dropped, because nothing
re-states it: it is simply still there. That removes the failure mode by
construction instead of detecting it after the fact, and shrinks per-turn output
from a whole ledger (487 chars median over the runs in ``runs/``) to a delta.

Two things survive from the detector, because both were load-bearing:

* **Superseding is visible, not silent.** The run above read 9:59 as 403g, lost
  it, and later re-read it as 413g. An upsert that quietly replaced the value
  would hide the disagreement, so the previous value is kept and rendered
  alongside the current one.
* **Prose still works.** A model that ignores the record shape and writes a
  sentence -- and every action replayed from a recording made before this
  change -- is split into records by :func:`as_records` and enters the same
  union. Being drop-free by construction is not conditional on compliance.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

log = logging.getLogger("adbagent.scratchpad")

#: Prose arrives as a list. The model writes it with newline or semicolon
#: separators; sentence-level splitting is deliberately not attempted, because
#: "9g+9g." and "no-carb." are not sentence ends.
_SPLIT = re.compile(r"[;\n]+")
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")
_TOKEN = re.compile(r"[\w:$£€¥₹%+/-]+")
_TRIM = ".,;:()[]{}<>'\"!?"
#: Separators between a leading key and the value it introduces.
_KEY_SEP = " \t:-–,="

#: Filler that would otherwise be mistaken for a record's identifier.
_STOPWORDS = frozenset((
    "with", "that", "this", "then", "than", "from", "have", "been", "were",
    "will", "into", "onto", "your", "there", "which", "about", "after",
    "before", "also", "some", "each", "both", "same", "other", "them", "they",
    "when", "what", "where", "while", "would", "could", "should", "still",
    "just", "only", "more", "most", "less", "over", "under", "here",
))

MAX_KEY_CHARS = 80
MAX_VALUE_CHARS = 500
#: Ceiling on the ledger itself. Reached only by a run collecting hundreds of
#: distinct records; the oldest entries are evicted and the count is reported.
MAX_KEYS = 300
#: How many superseded values one key keeps. The current value plus the reading
#: it replaced is the conflict worth seeing; ten revisions of it are not.
MAX_SUPERSEDED = 2


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    for raw in _TOKEN.findall(text.lower()):
        token = raw.strip(_TRIM)
        if token:
            out.append(token)
    return out


def distinctive(text: str) -> frozenset:
    """The tokens specific enough to identify this record's content.

    A token qualifies when it is long enough to mean something or contains a
    digit -- figures are the whole point of a collection scratchpad.
    """
    return frozenset(
        token for token in _tokens(text)
        if token not in _STOPWORDS
        and (len(token) >= 4 or any(c.isdigit() for c in token))
    )


def split_records(note: str) -> List[str]:
    out: List[str] = []
    for chunk in _SPLIT.split(note or ""):
        text = _BULLET.sub("", chunk).strip()
        if text:
            out.append(text)
    return out


def record_key(text: str) -> str:
    """The identifier a prose record hangs off -- its first distinctive token.

    For ``"10:03 tomatoes 120g"`` that is ``"10:03"``; for ``"Item B: $15"`` it
    is ``"item"``, which collides with ``"Item A: $10"`` -- :func:`as_records`
    resolves that rather than letting one overwrite the other.
    """
    for token in _tokens(text):
        if token not in _STOPWORDS and (len(token) >= 4
                                        or any(c.isdigit() for c in token)):
            return token
    return ""


def normalise_key(key: str) -> str:
    """Keys are matched case- and punctuation-insensitively.

    ``"9:45"``, ``"9:45 "`` and ``"9:45:"`` are the same record; a model that
    varies the incidental characters is not creating a second one.
    """
    return " ".join(str(key or "").strip().strip(_TRIM).lower().split())[:MAX_KEY_CHARS]


def _strip_leading_key(key: str, value: str) -> str:
    """Drop a key the value merely repeats, so rendering does not say it twice."""
    if key and value.lower().startswith(key.lower()):
        trimmed = value[len(key):].lstrip(_KEY_SEP)
        if trimmed:
            return trimmed
    return value


def as_records(notes: Any) -> List[Tuple[str, str]]:
    """Whatever arrived in ``notes``, as ``(key, value)`` pairs.

    Accepts the record list the schema now asks for, the plain string every
    recording made before this change contains, and the mixed bag in between.
    Prose is split into records and keyed by its leading identifier; when two
    records in the *same* note key alike -- ``"Item A: $10; Item B: $15"`` both
    lead with ``"item"`` -- the later one falls back to its own full text as the
    key, because a collision resolved by overwriting is exactly the data loss
    this module exists to prevent.
    """
    if notes is None or notes == "" or notes == []:
        return []

    if isinstance(notes, str):
        pairs: List[Tuple[str, str]] = []
        seen: Dict[str, str] = {}
        for text in split_records(notes):
            key = record_key(text)
            value = _strip_leading_key(key, text) if key else text
            if not key or (key in seen and seen[key] != value):
                key = normalise_key(text) or normalise_key(value)
                value = text
            if not key:
                continue
            seen.setdefault(key, value)
            pairs.append((key, value))
        return pairs

    if isinstance(notes, dict):
        # Either one record, or a mapping of key -> value.
        if "key" in notes:
            return as_records([notes])
        return [(str(k), str(v)) for k, v in notes.items() if str(k).strip()]

    if isinstance(notes, Sequence):
        pairs = []
        for item in notes:
            if isinstance(item, str):
                pairs.extend(as_records(item))
                continue
            key = getattr(item, "key", None)
            value = getattr(item, "value", None)
            if key is None and isinstance(item, dict):
                key, value = item.get("key"), item.get("value")
            if not str(key or "").strip():
                continue
            pairs.append((str(key), str(value if value is not None else "")))
        return pairs

    return as_records(str(notes))


@dataclass
class Entry:
    key: str
    value: str
    first_step: int
    last_step: int
    #: Values this key held before, most recent first. Kept so a re-read that
    #: disagrees with the first reading shows up as a disagreement.
    superseded: List[str] = field(default_factory=list)

    def render(self) -> str:
        line = f"{self.key}: {self.value}" if self.value else self.key
        if self.superseded:
            line += " [earlier: " + "; ".join(self.superseded) + "]"
        return line


@dataclass
class NoteLedger:
    """Every record the run has collected, keyed, in the order first written."""

    entries: "OrderedDict[str, Entry]" = field(default_factory=OrderedDict)
    #: Records evicted to stay under :data:`MAX_KEYS`. Reported rather than
    #: hidden: a silently truncated ledger reads as a complete one.
    evicted: int = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def update(self, notes: Any, step: int) -> List[str]:
        """Merge this turn's records in. Returns the keys that changed."""
        changed: List[str] = []
        for raw_key, raw_value in as_records(notes):
            key = normalise_key(raw_key)
            if not key:
                continue
            value = " ".join(str(raw_value).split())[:MAX_VALUE_CHARS]
            value = _strip_leading_key(raw_key.strip(), value)
            existing = self.entries.get(key)
            if existing is None:
                if len(self.entries) >= MAX_KEYS:
                    self.entries.popitem(last=False)
                    self.evicted += 1
                self.entries[key] = Entry(key=raw_key.strip()[:MAX_KEY_CHARS] or key,
                                          value=value, first_step=step,
                                          last_step=step)
                changed.append(key)
                continue
            existing.last_step = step
            if value and value != existing.value:
                if existing.value:
                    existing.superseded = (
                        [existing.value]
                        + [v for v in existing.superseded if v != existing.value]
                    )[:MAX_SUPERSEDED]
                existing.value = value
                changed.append(key)
        if changed:
            log.info("scratchpad: %d record(s) written (%d total): %s",
                     len(changed), len(self.entries), ", ".join(changed[:8]))
        return changed

    def _lines(self, max_chars: int = 0) -> Tuple[List[str], int]:
        """The rendered records that fit, and how many did not.

        Trimmed from the front when a budget is set, matching `pager.ItemLedger`:
        the oldest records are the ones most likely already acted on. What was
        trimmed is returned rather than swallowed -- a ledger that quietly shrank
        reads exactly like a complete one.
        """
        lines = [entry.render() for entry in self.entries.values()]
        if max_chars <= 0:
            return lines, 0
        kept: List[str] = []
        used = 0
        for line in reversed(lines):
            if used + len(line) + 1 > max_chars and kept:
                break
            kept.append(line)
            used += len(line) + 1
        kept.reverse()
        return kept, len(lines) - len(kept)

    def render(self, max_chars: int = 0) -> str:
        """The block handed to the model, and to the completion judge.

        States that the harness owns it, because the previous contract asked the
        model to restate everything and a model that keeps doing so wastes its
        whole output budget re-emitting a list it is looking at.
        """
        if not self.entries:
            return ""
        lines, trimmed = self._lines(max_chars)
        dropped = self.evicted + trimmed
        head = f"COLLECTED DATA ({len(self.entries)} record(s)"
        if dropped:
            head += f", {dropped} earlier record(s) dropped for space"
        head += ("). This is kept for you and cannot be lost -- do NOT restate "
                 "these. Send only new or corrected records:")
        return "\n".join([head] + [f"  - {line}" for line in lines])

    def plain(self, max_chars: int = 0) -> str:
        """The records alone, for a terminal, a report or the judge."""
        lines, trimmed = self._lines(max_chars)
        if trimmed:
            lines = [f"(... {trimmed} earlier record(s) omitted)"] + lines
        return "\n".join(lines)


def replay(events: Iterable[Any]) -> NoteLedger:
    """Rebuild a ledger from recorded ``decide`` events.

    ``notes`` in an event is a delta, so the last one is no longer the whole
    ledger -- reading a finished run means replaying them all.
    """
    ledger = NoteLedger()
    for step, event in enumerate(events, start=1):
        if not isinstance(event, dict) or event.get("kind") != "decide":
            continue
        action = event.get("action")
        if isinstance(action, dict) and action.get("notes"):
            ledger.update(action["notes"], event.get("step", step))
    return ledger
