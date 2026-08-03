"""Stopping the model's own scratchpad from losing what it collected.

The scratchpad is the model's memory for data-collection goals: it rewrites the
complete collected state into ``notes`` every turn and the loop keeps only the
latest value. That is a deliberate choice -- the model re-emits its whole ledger
each turn, so *appending* would archive a hundred near-identical copies -- but it
puts the entire run's findings behind one instruction the model must obey
perfectly, every single turn, with no way to notice when it does not.

It does not obey it perfectly. From ``runs/af76720d05c4``, one turn apart:

    step 73  ... 9:45 chicken 425g (OK); 9:51 chicken 426g (+1g); 9:52
             [pending]; 9:59 potatoes 403g (+3g vs menu 400g); 10:03 tomatoes
             120g (matches no-carb tomato 120g).

    step 74  ... 9:45 [pending]; 9:51 [pending]; 9:52 [pending]; 9:59
             [pending]; 10:03 [pending].

Four readings gone in one rewrite, never restated across the remaining 59 turns,
and the run's final report listed the 10:03 photo as unreadable -- a question it
had already answered and then forgotten. The prompt already tells the model each
note must carry ALL items; the instruction is not the missing piece.

So this module keeps an append-only archive of every record the model has
written, keyed by the identifier the record starts with, and each turn reports
which archived records the new note no longer covers. The model's latest note
stays authoritative and stays the only curated view; the archive just refuses to
let a figure disappear silently.

Comparison is deliberately per key rather than across the whole note. A note that
holds both a menu and a set of measured weights restates the same figures twice,
so a note-wide token test finds "120g" still present and concludes nothing was
lost -- which is exactly the case above.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger("adbagent.scratchpad")

#: A scratchpad is a list. The model writes it with newline or semicolon
#: separators; sentence-level splitting is deliberately not attempted, because
#: "9g+9g." and "no-carb." are not sentence ends.
_SPLIT = re.compile(r"[;\n]+")
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")
_TOKEN = re.compile(r"[\w:$£€¥₹%+/-]+")
_TRIM = ".,;:()[]{}<>'\"!?"

#: Filler that would otherwise make two unrelated records look related.
_STOPWORDS = frozenset((
    "with", "that", "this", "then", "than", "from", "have", "been", "were",
    "will", "into", "onto", "your", "there", "which", "about", "after",
    "before", "also", "some", "each", "both", "same", "other", "them", "they",
    "when", "what", "where", "while", "would", "could", "should", "still",
    "just", "only", "more", "most", "less", "over", "under", "here",
))

#: A record must have at least this many distinctive tokens to be tracked at
#: all. One-token fragments ("OK", "done") carry nothing worth guarding.
MIN_TOKENS = 2

#: A shortened record is only a loss when a figure went missing, or when most of
#: the record went missing. Rewording -- "mixed nuts ~5g" to "nuts 5g" -- must
#: not be reported, or the block becomes noise the model learns to skip.
MIN_LOST_FRACTION = 0.5

#: How many turns in a row one lost record is put back in front of the model.
#: It stays in the archive (and so reaches the judge) either way; this only
#: bounds the nagging.
MAX_REPORTS = 3
#: Ceiling on the rendered block, and on the archive itself.
MAX_REPORTED = 8
MAX_KEYS = 300


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
    """The identifier a record hangs off -- its first distinctive token.

    For ``"10:03 tomatoes 120g"`` that is ``"10:03"``; for ``"Item B: $15"`` it
    is ``"item"``, shared with ``"Item A: $10"``, whose own tokens then keep the
    two apart. Crude, and good enough: the key only has to localise the
    comparison, not be unique.
    """
    for token in _tokens(text):
        if token not in _STOPWORDS and (len(token) >= 4
                                        or any(c.isdigit() for c in token)):
            return token
    return ""


@dataclass
class _Live:
    tokens: frozenset
    text: str


def index(note: str) -> Dict[str, _Live]:
    """Group a note's records by key, unioning the tokens under each."""
    out: Dict[str, _Live] = {}
    for text in split_records(note):
        tokens = distinctive(text)
        if len(tokens) < MIN_TOKENS:
            continue
        key = record_key(text)
        if not key:
            continue
        existing = out.get(key)
        if existing is None:
            out[key] = _Live(tokens=tokens, text=text)
        else:
            out[key] = _Live(tokens=existing.tokens | tokens,
                             text=existing.text if len(existing.tokens) >= len(tokens)
                             else text)
    return out


def context_for(key: str, live: Dict[str, _Live]) -> frozenset:
    """The tokens now keeping `key` company, wherever in the note it ended up.

    Usually that is its own record. But records merge -- three separate
    ``"9:45 [pending]"``, ``"9:52 [pending]"``, ``"9:59 [pending]"`` entries
    become one ``"9:45, 9:52, 9:59 pending"`` line -- and the absorbed keys then
    stop being keys while remaining perfectly present. Looking the key up
    wherever it is mentioned is what keeps that from reading as a loss.
    """
    entry = live.get(key)
    if entry is not None:
        return entry.tokens
    found: frozenset = frozenset()
    for candidate in live.values():
        if key in candidate.tokens:
            found |= candidate.tokens
    return found


@dataclass
class Archived:
    key: str
    #: Union of every distinctive token ever written under this key. Never
    #: shrinks: a value the model has since dropped is precisely what must
    #: outlive the rewrite that dropped it.
    tokens: set
    #: The richest phrasing seen, which is what gets shown back.
    text: str
    first_step: int
    reports: int = 0


@dataclass
class Loss:
    key: str
    text: str
    lost: frozenset
    first_step: int


@dataclass
class ScratchpadGuard:
    keys: "Dict[str, Archived]" = field(default_factory=dict)

    def update(self, note: str, step: int) -> List[Loss]:
        """Archive `note`, then report which archived records it stopped covering."""
        live = index(note)

        for key, entry in live.items():
            archived = self.keys.get(key)
            if archived is None:
                if len(self.keys) >= MAX_KEYS:
                    self.keys.pop(next(iter(self.keys)))
                self.keys[key] = Archived(key=key, tokens=set(entry.tokens),
                                          text=entry.text, first_step=step)
                continue
            archived.tokens |= entry.tokens
            if len(distinctive(entry.text)) > len(distinctive(archived.text)):
                archived.text = entry.text

        losses: List[Loss] = []
        for key, archived in self.keys.items():
            lost = frozenset(archived.tokens) - context_for(key, live)
            if not lost:
                archived.reports = 0
                continue
            figure_lost = any(c.isdigit() for token in lost for c in token)
            mostly_lost = len(lost) / max(1, len(archived.tokens)) > MIN_LOST_FRACTION
            if figure_lost or mostly_lost:
                losses.append(Loss(key=key, text=archived.text, lost=lost,
                                   first_step=archived.first_step))
        if losses:
            log.info("scratchpad dropped %d record(s): %s",
                     len(losses), ", ".join(l.key for l in losses))
        return losses

    def report(self, losses: List[Loss]) -> str:
        """The block shown to the model, and the reporting budget it spends."""
        fresh = [loss for loss in losses
                 if self.keys[loss.key].reports < MAX_REPORTS]
        if not fresh:
            return ""
        shown = fresh[:MAX_REPORTED]
        for loss in shown:
            self.keys[loss.key].reports += 1
        lines = ["YOU DROPPED DATA YOU HAD ALREADY COLLECTED. Your latest `notes` "
                 "no longer contains these records, which you wrote earlier. Put "
                 "them back into `notes` verbatim, or restate the corrected "
                 "value if they were superseded — do not leave them out:"]
        for loss in shown:
            lines.append(f"  - (step {loss.first_step}) {loss.text}")
        if len(fresh) > len(shown):
            lines.append(f"  - ... and {len(fresh) - len(shown)} more")
        return "\n".join(lines)

    def preserved(self, note: str) -> str:
        """`note` plus any archived record it no longer covers.

        Handed to the completion judge so a verdict is reached on everything the
        run actually collected, not only on whatever survived the last rewrite.
        """
        live = index(note)
        orphaned = []
        for key, archived in self.keys.items():
            if frozenset(archived.tokens) - context_for(key, live):
                orphaned.append(archived.text)
        if not orphaned:
            return note
        return (note + "\n\nEARLIER RECORDS THE AGENT COLLECTED AND LATER "
                       "DROPPED FROM ITS NOTES:\n"
                + "\n".join(f"  - {text}" for text in orphaned[:MAX_KEYS]))
