"""Durable record of which conversations have already been replied to.

A watch loop that replies to messages has one failure mode that dwarfs every
other: sending the same reply twice. Everything else the harness gets wrong
costs a wasted step. A double reply is visible to another person and cannot be
taken back.

The screen cannot answer "did I already handle this". You can scroll away and
back, the app can re-render, a send can go out while the confirmation is still
animating, and an iteration can die between tapping Send and observing the
result. So the answer lives here instead: on disk, fsynced before the send is
allowed to proceed, and consulted by `safety.reply_gate` immediately before
every send.

**What identifies a conversation's state.** Not the whole message list -- that
moves under you as you scroll -- but the *tail* of it: the last `TAIL_MESSAGES`
message texts, masked through `fingerprint.mask_text` so a relative timestamp
ticking from "2m" to "3m" is not mistaken for a new message. A chat app opens
scrolled to the bottom, so the tail is the stable part. No incoming/outgoing
bubble detection is involved, because that can only be guessed from bounds.

**Why a set of digests and not one.** The tail is read twice per reply: once
before the send, and once after, when our own message has joined it. Both are
states this loop has already responded to, so both are remembered, and the gate
refuses when the tail it sees is either of them. Recording the *first* one
before the tap is what makes a crash mid-send safe -- the record is already on
disk when the gesture goes out.

**Why a cooldown as well.** One window survives that scheme: the process dies
after the send lands but before the confirming record. The pre-send digest is
stored, the post-send tail is not, and no digest comparison can tell that
history from "a new message arrived". Time can. An attempt that was never
confirmed puts the thread in doubt, and a thread in doubt is not replied to
again until `UNCONFIRMED_COOLDOWN_MULTIPLIER` times the ordinary cooldown has
passed -- loudly, so the operator sees it in the log.

**Storage.** Append-only JSONL, flushed and fsynced before the write is
acknowledged. Not SQLite: a few hundred bytes per reply, one query ("what has
this thread been through"), and an append that reached the platter cannot be
left half-applied by a kill -9. A truncated final line -- the one shape of
corruption a crash mid-write leaves -- is dropped on load rather than being
allowed to poison the ledger.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .fingerprint import mask_goal, mask_text

log = logging.getLogger("adbagent.ledger")

#: How many trailing messages form a thread's content digest. Six is enough to
#: change when anything new arrives and short enough that scrolling up by a
#: screenful does not rewrite it.
TAIL_MESSAGES = 6

#: Characters of masked text kept per message inside the digest input, to stop
#: one pasted wall of text from dominating the tail.
#:
#: Messages are masked with `mask_goal`, not `mask_text`, despite not being goals:
#: the two apply the same substitutions but `mask_text` also truncates to 32
#: characters, which would collide every pair of messages sharing a 32-character
#: prefix -- and a digest collision reads as "already replied to this", so a real
#: message would be silently skipped.
DIGEST_TEXT_LIMIT = 200

#: An attempt we never saw confirmed leaves the thread in doubt. Multiply its
#: cooldown by this before allowing another send into it.
UNCONFIRMED_COOLDOWN_MULTIPLIER = 4.0


def _sha(parts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def thread_key(title: str) -> str:
    """Stable id for a conversation, from whatever the app calls it.

    Masked like any other screen text: a title carrying an unread count
    ("khushi (2)") or a last-seen time must not mint a second identity for the
    same person. Empty when there is no usable title -- the gate treats that as
    "cannot identify" and refuses.
    """
    masked = mask_text(title or "")
    return _sha(["thread", masked]) if masked else ""


def content_digest(messages: Sequence[str]) -> str:
    """Digest of a conversation's tail. Empty when there is nothing to digest.

    Empty is a meaningful answer and every caller must read it as "I could not
    tell", never as "no messages": `safety.reply_gate` refuses to send on it.
    """
    tail = [mask_goal(m)[:DIGEST_TEXT_LIMIT] for m in messages[-TAIL_MESSAGES:]]
    tail = [t for t in tail if t]
    return _sha(["tail", *tail]) if tail else ""


@dataclass(frozen=True)
class ThreadState:
    """Everything the ledger knows about one conversation."""

    thread_key: str
    #: Every tail this loop has already acted on -- pre-send and post-send both.
    digests: FrozenSet[str] = frozenset()
    #: When a send was last attempted into this thread.
    last_attempt_at: float = 0.0
    #: False when the most recent attempt was never confirmed to have landed.
    #: The thread is in doubt and gets the long cooldown.
    confirmed: bool = True
    #: Attempts, not confirmations: what the rate limits should count.
    reply_count: int = 0
    #: Human-readable, for the log and the prompt. Never used for identity.
    preview: str = ""


@dataclass
class Verdict:
    """Why the ledger would or would not allow a reply into a thread."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class ReplyLedger:
    """Append-only store of per-thread reply state.

    Cheap to construct, and safe to construct repeatedly against the same file:
    state is folded from the file every time, so the fresh `Agent` each watch
    iteration builds reads back everything earlier iterations committed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._threads: Dict[str, ThreadState] = {}
        #: (timestamp, thread_key) per attempt, oldest first. Kept whole rather
        #: than counted, because the rate limits ask about a moving window.
        self._events: List[Tuple[float, str]] = []
        self._load()

    # -- reading -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        rows: List[dict] = []
        dropped = 0
        for line in self.path.read_text(encoding="utf-8",
                                        errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not isinstance(d, dict) or not d.get("thread_key"):
                    raise ValueError("missing thread_key")
                rows.append(d)
            except (ValueError, TypeError):
                # In practice only ever the final line, after a crash mid-append.
                dropped += 1
        for d in sorted(rows, key=lambda r: float(r.get("at") or 0.0)):
            key = str(d["thread_key"])
            digest = str(d.get("digest") or "")
            at = float(d.get("at") or 0.0)
            confirmed = bool(d.get("confirmed"))
            prior = self._threads.get(key)
            digests = set(prior.digests) if prior else set()
            if digest:
                digests.add(digest)
            attempts = prior.reply_count if prior else 0
            # A confirmation follows its own attempt and must not be counted as
            # a second one; only the attempt record increments.
            if not confirmed:
                attempts += 1
                self._events.append((at, key))
            self._threads[key] = ThreadState(
                thread_key=key,
                digests=frozenset(digests),
                last_attempt_at=max(at, prior.last_attempt_at if prior else 0.0),
                confirmed=confirmed,
                reply_count=attempts,
                preview=str(d.get("preview") or "")
                        or (prior.preview if prior else ""),
            )
        self._events.sort()
        if dropped:
            log.warning("ledger %s: dropped %d unreadable record(s)",
                        self.path, dropped)
        in_doubt = [k for k, s in self._threads.items() if not s.confirmed]
        if in_doubt:
            log.warning("ledger %s: %d thread(s) with an unconfirmed send -- "
                        "they get the long cooldown", self.path, len(in_doubt))
        log.debug("ledger %s: %d record(s), %d thread(s)",
                  self.path, len(rows), len(self._threads))

    def state(self, key: str) -> Optional[ThreadState]:
        return self._threads.get(key)

    def already_acted(self, key: str, digest: str) -> bool:
        """True when this exact tail is one we have already responded to.

        An empty `digest` is never "acted on". It is not "fresh" either -- the
        gate refuses on it -- so False here cannot become a licence to send.
        """
        if not key or not digest:
            return False
        st = self._threads.get(key)
        return st is not None and digest in st.digests

    def check(self, key: str, digest: str, *, cooldown_s: float,
              now: Optional[float] = None) -> Verdict:
        """Would the ledger allow a reply into this thread right now?

        The two rules that need durable state: this tail has been answered
        before, and this thread was written to too recently. Rate limits and
        screen-readability live in `safety.reply_gate`, which owns the config.
        """
        if not key:
            return Verdict(False, "the conversation could not be identified")
        if not digest:
            return Verdict(False, "the conversation's messages could not be read")
        st = self._threads.get(key)
        if st is None:
            return Verdict(True)
        if digest in st.digests:
            return Verdict(False, "this exact conversation state has already "
                                  "been replied to")
        now = time.time() if now is None else now
        window = cooldown_s * (1.0 if st.confirmed
                               else UNCONFIRMED_COOLDOWN_MULTIPLIER)
        waited = now - st.last_attempt_at
        if waited < window:
            doubt = "" if st.confirmed else " (previous send unconfirmed)"
            return Verdict(False, f"replied to this conversation "
                                  f"{waited:.0f}s ago{doubt}; the cooldown is "
                                  f"{window:.0f}s")
        return Verdict(True)

    def replies_since(self, since: float, key: str = "") -> int:
        """Attempts recorded at or after `since`, for one thread or for all."""
        return sum(1 for t, k in self._events
                   if t >= since and (not key or k == key))

    def recent(self, limit: int = 12) -> List[ThreadState]:
        """Most recently written-to threads, newest first."""
        return sorted(self._threads.values(),
                      key=lambda s: s.last_attempt_at, reverse=True)[:limit]

    def __len__(self) -> int:
        return len(self._events)

    # -- writing -----------------------------------------------------------

    def _append(self, key: str, digest: str, confirmed: bool,
                preview: str, at: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps({
            "thread_key": key,
            "digest": digest,
            "at": round(at, 3),
            "confirmed": confirmed,
            "preview": preview[:200],
        }, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(row + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def record_attempt(self, key: str, digest: str, preview: str = "",
                       at: Optional[float] = None) -> ThreadState:
        """Commit "we are about to reply to this tail", before the gesture.

        Called *before* the send, on purpose: a record written afterwards is a
        record a crash can lose, and a lost record is a second reply. Raises on
        an unwritable ledger rather than logging and carrying on -- failing the
        send is the safer half of that trade.
        """
        if not key:
            raise ValueError("refusing to record a reply with no thread key")
        now = time.time() if at is None else at
        self._append(key, digest, False, preview, now)
        prior = self._threads.get(key)
        digests = set(prior.digests) if prior else set()
        if digest:
            digests.add(digest)
        st = ThreadState(
            thread_key=key,
            digests=frozenset(digests),
            last_attempt_at=now,
            confirmed=False,
            reply_count=(prior.reply_count + 1) if prior else 1,
            preview=preview or (prior.preview if prior else ""),
        )
        self._threads[key] = st
        self._events.append((now, key))
        return st

    def record_confirmed(self, key: str, digest: str, preview: str = "",
                         at: Optional[float] = None) -> ThreadState:
        """Commit the post-send tail, once our own message is on screen.

        This is what lets the next iteration recognise its own reply instead of
        reading it as new incoming content, and it is what lifts the thread out
        of doubt.
        """
        if not key:
            raise ValueError("refusing to confirm a reply with no thread key")
        now = time.time() if at is None else at
        self._append(key, digest, True, preview, now)
        prior = self._threads.get(key)
        digests = set(prior.digests) if prior else set()
        if digest:
            digests.add(digest)
        st = ThreadState(
            thread_key=key,
            digests=frozenset(digests),
            last_attempt_at=max(now, prior.last_attempt_at if prior else 0.0),
            confirmed=True,
            reply_count=prior.reply_count if prior else 1,
            preview=preview or (prior.preview if prior else ""),
        )
        self._threads[key] = st
        return st
