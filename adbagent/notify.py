"""Telling the owner, on their phone: the `notify` action, sent through Telegram.

A watch works while nobody is looking at it, and some of what it finds is worth
knowing the moment it happens -- somebody wrote, a reply went out, a new match --
rather than the next time anybody opens History. `notify` is the model saying
so: an action like any other, chosen when the goal or the policy asks for one,
and delivered by a bot to the owner's own Telegram chat (`config.NotifyConfig`).

**A line reaches the phone once, ever.** Every pass is a fresh run that remembers
nothing of the one before, and a watch goes back to the same screens pass after
pass: a message nobody answered is still unanswered twenty minutes later, a new
match still new. So the model reports the same thing again and again, and only
the harness can know that it already has. Every line that goes out is written to
`notified.jsonl` in the runs directory -- one file for every run, watch and MCP
session that records there -- and a line found in it is not sent again. That is
what lets a policy say "notify each new message" without the model having to
know which ones are new. One `notify` may carry several lines; each is judged on
its own, and the new ones go out together, as one message.

Lines are compared on their words and their emoji (`same`): case, punctuation
and spacing are ignored, so "hi" and "Hi!" from one person are one line. Emoji
count, because a message can be nothing else -- "😂😂" and "😍" from the same
match are two messages, and compared on words alone they were one, and the
second never went out. But the element list draws every emoji outside the
Basic Multilingual Plane -- nearly all of them -- as two dots, so which one it
was is read off the screenshot, and that reading is a guess that changes from
pass to pass: one "😊" from P left unanswered went out as 😊, 🤗, 🥰, 😄, 😋
and 😉 on six passes. What the tree does say is how many there were, so those
emoji are compared by count, not by face -- "😂😂" and "😍" still differ, "😊"
and "😉" do not. Emoji the tree carries as they are (❤, ☺, ✨) are compared as
they are. A line holding the list's own `..` matches its words with any emoji.

**A line that could not be delivered is not dropped.** It is kept in the same
file as pending and goes out with the next notification that does get through:
the pass that sent a reply never comes back to that moment, so nothing else
would bring its notification back. A pending line older than `PENDING_TTL_S` is
let go rather than delivered a day late.

**What was notified can be read back** (`read_notified`, `Notifier.read_back`).
A pass learns that a message is old news only once it has opened the
conversation and notified it again -- and in the seven days to 2026-10-28 that
was 2,471 Hinge chats and 648 Bumble ones opened to find nothing new, against
28 that held something. A conversation list already shows each newest message,
and a policy that notifies every new message has written down every one a pass
read: so before opening a conversation, the model can look its row up here,
and leave alone one whose newest message an earlier pass has already seen. The
lookup is by words (`matches`), the same words `same` compares, and it decides
nothing -- the model compares what it finds with what the row shows.

Nothing here touches the phone, and nothing here is a send in the sense of
`conversation.send_label`: a line to the owner's own chat is not a message to
anybody the policy is about, so draft mode and `send_limits` leave it alone. A
dry run does not notify -- see `agent.Agent._notify`.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List,
                    Optional, Sequence, Set, Tuple)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config

log = logging.getLogger("adbagent.notify")

#: The ledger, in the runs directory beside every run's own.
LEDGER_NAME = "notified.jsonl"

API = "https://api.telegram.org"

#: Telegram refuses a message over 4096 characters, so lines that would make a
#: longer one go out as several.
MESSAGE_MAX_CHARS = 4000

#: One line is a notification, not a transcript.
LINE_MAX_CHARS = 1000

#: How long a line that could not be delivered is kept for another try.
PENDING_TTL_S = 24 * 3600

#: How many lines one `read_notified` shows at most, and how much text: its
#: answer is one line of the run's history. The newest are the ones kept, and
#: naming the people on screen narrows it to what matters.
READ_MAX_LINES = 50
READ_MAX_CHARS = 5000

#: One process can hold a session's run thread and serve a web request at once.
#: Reading the ledger, delivering and writing down what was delivered must not
#: interleave, or two callers both find a line unsent and both send it.
_LOCK = threading.Lock()


class NotifyError(RuntimeError):
    """Telegram did not take the message.

    `refused` when Telegram answered and said no -- a wrong token, an unknown
    chat -- which trying again will not change, as against not answering.
    """

    def __init__(self, message: str, *, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


#: How the element list draws an emoji it cannot render: two dots for every
#: character outside the Basic Multilingual Plane (see `screen.as_dumped`).
_DRAWN_AS_DOTS = re.compile(r"\.\.")


def words(line: str) -> str:
    """`line`'s words, in one case and single-spaced; punctuation and emoji go."""
    return " ".join(re.sub(r"[\W_]+", " ", line.casefold()).split())


#: Stands in for an emoji the element list cannot draw (see `emoji`).
_UNDRAWN = "￼"


def faces(line: str) -> str:
    """The emoji in `line`, in order, exactly as written.

    Skin tones, joiners and variation selectors are not symbols of their own and
    are left out, so one emoji drawn two ways is still one emoji.
    """
    return "".join(ch for ch in line if unicodedata.category(ch) == "So")


def emoji(line: str) -> Optional[str]:
    """The emoji in `line` as far as the screen can vouch for them -- or None.

    An emoji outside the Basic Multilingual Plane is two dots in the element
    list, so its face was read off the screenshot and is not to be trusted: it
    counts as one `_UNDRAWN`, whichever face it is. One the list carries as it
    is keeps its face. None when the line holds the list's `..` itself: how many
    emoji that was is unknown too, so it is compared as any of them.
    """
    if _DRAWN_AS_DOTS.search(line):
        return None
    return "".join(_UNDRAWN if ord(ch) > 0xFFFF else ch for ch in faces(line))


def same(a: str, b: str) -> bool:
    """Whether two lines say the same thing -- see the module docstring."""
    if words(a) != words(b):
        return False
    ea, eb = emoji(a), emoji(b)
    return ea is None or eb is None or ea == eb


def matches(line: str, wanted: str) -> bool:
    """Whether `line` says the words of `wanted`, whole and in order.

    Whole words, so that "new message from S" finds what S said and not what
    Swastika did. A `wanted` with no words in it -- only emoji, only
    punctuation -- matches nothing.
    """
    wanted = words(wanted)
    return bool(wanted) and f" {wanted} " in f" {words(line)} "


def lines_of(text: str) -> List[str]:
    """The notifications in one `notify`: its non-blank lines, spacing tidied.

    A line said twice in the same text is one notification -- said twice in
    words and emoji alike: "😂" and "😍" in one `notify` are two messages the
    model told apart while looking at both, not one message read two ways.
    """
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if len(line) > LINE_MAX_CHARS:
            line = line[:LINE_MAX_CHARS - 1] + "…"
        if any(words(line) == words(kept) and faces(line) == faces(kept)
               for kept in out):
            continue
        out.append(line)
    return out


def _chunks(lines: Sequence[str]) -> List[List[str]]:
    """`lines` grouped into messages Telegram will take, in order."""
    out: List[List[str]] = [[]]
    size = 0
    for line in lines:
        if out[-1] and size + len(line) + 1 > MESSAGE_MAX_CHARS:
            out.append([])
            size = 0
        out[-1].append(line)
        size += len(line) + 1
    return [chunk for chunk in out if chunk]


class Telegram:
    """The Bot API, as one bot writing to one chat."""

    def __init__(self, token: str, chat_id: str, *, timeout_s: float = 10.0,
                 attempts: int = 2,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout_s = timeout_s
        #: Two, and a short pause between: a blip in the connection is worth a
        #: second try, and anything longer is what the pending lines are for.
        #: Every try holds up the run that is waiting on it.
        self.attempts = max(1, attempts)
        self._sleep = sleep

    def send(self, text: str) -> None:
        """Deliver `text` as one message. Raises `NotifyError`."""
        self._call("sendMessage", {
            "chat_id": self.chat_id, "text": text,
            # A match's message can carry a link, and a preview of it is
            # somebody else's page in the owner's chat.
            "link_preview_options": json.dumps({"is_disabled": True})})

    def whoami(self) -> str:
        """The bot's @username: proof the token works, without sending anything."""
        result = self._call("getMe").get("result") or {}
        return str(result.get("username") or result.get("first_name") or "")

    def _call(self, method: str,
              data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One Bot API call, tried up to `attempts` times. Raises `NotifyError`.

        The token is in the URL, and an HTTP library's error can quote the URL,
        so every message raised from here is passed through `_redact` first:
        what fails a notification lands in `events.jsonl` and in the model's
        history, neither of which may carry a credential.
        """
        import httpx

        url = f"{API}/bot{self.token}/{method}"
        last = ""
        for attempt in range(self.attempts):
            wait = float(attempt + 1)
            try:
                reply = httpx.post(url, data=data or {}, timeout=self.timeout_s)
            except Exception as exc:  # noqa: BLE001 - reported, token removed
                last = self._redact(f"{type(exc).__name__}: {exc}")
            else:
                try:
                    body = reply.json()
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                if reply.status_code == 200 and body.get("ok"):
                    return body
                said = str(body.get("description") or f"HTTP {reply.status_code}")
                if reply.status_code == 429:
                    retry = (body.get("parameters") or {}).get("retry_after")
                    if isinstance(retry, (int, float)):
                        wait = min(float(retry), 10.0)
                elif 400 <= reply.status_code < 500:
                    # A wrong token, an unknown chat, a bot the owner blocked:
                    # asking again gets the same answer.
                    raise NotifyError(self._redact(f"Telegram refused it: {said}"),
                                      refused=True)
                last = self._redact(f"Telegram answered: {said}")
            if attempt + 1 < self.attempts:
                self._sleep(wait)
        raise NotifyError(last or "Telegram did not answer")

    def _redact(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text


@dataclass
class Outcome:
    """What one `notify` did with each of its lines."""

    #: Delivered now.
    sent: List[str] = field(default_factory=list)
    #: Delivered before -- on this pass or an earlier one -- so not again.
    repeats: List[str] = field(default_factory=list)
    #: Not delivered; kept, to go out with the next notification.
    kept: List[str] = field(default_factory=list)
    #: Lines kept from an earlier `notify` that went out along with these.
    caught_up: List[str] = field(default_factory=list)
    #: Why something could not be delivered.
    error: str = ""
    #: No bot or no chat is configured, so nothing was even tried.
    unconfigured: bool = False

    @property
    def grade(self) -> str:
        """One word for the step's history line."""
        if self.unconfigured:
            return "not sent"
        if self.sent:
            return "sent"
        if self.kept:
            return "not delivered"
        if self.repeats:
            return "already sent"
        return "nothing to send"

    @property
    def worked(self) -> bool:
        """Whether the owner has been told everything this `notify` said."""
        return not self.unconfigured and not self.kept

    def summary(self) -> str:
        """What the model is told this `notify` did."""
        if self.unconfigured:
            return ("notifications are not set up -- no notify.telegram_token or "
                    "notify.telegram_chat_id -- so nothing was sent; carry on "
                    "with the goal")
        parts = []
        if self.sent:
            parts.append(f"{len(self.sent)} sent to the owner's phone")
        if self.repeats:
            parts.append(f"{len(self.repeats)} already sent before, so not "
                         f"repeated")
        if self.kept:
            parts.append(f"{len(self.kept)} not delivered ({self.error}) and "
                         f"kept: it goes out with the next notification, so "
                         f"do not send it again")
        if self.caught_up:
            parts.append(f"{len(self.caught_up)} kept from earlier went out "
                         f"with it")
        return "; ".join(parts) or "the text held no line to send"


@dataclass
class Notified:
    """A line in the ledger, as `read_notified` shows it."""

    #: When it was first written down: the pass that read it.
    at: float
    text: str
    #: Whether it has reached the owner. A kept line has not, yet.
    sent: bool

    def shown(self) -> str:
        try:
            when = time.strftime("%d %b %H:%M", time.localtime(self.at)) \
                if self.at else "?"
        except (OverflowError, OSError, ValueError):
            when = "?"      # a time no clock can show: the file was edited
        return f"{when} {self.text}" + ("" if self.sent else " [not delivered yet]")


@dataclass
class ReadBack:
    """What one `read_notified` found in the ledger."""

    #: What it looked for, one per line; none for everything.
    filters: List[str] = field(default_factory=list)
    #: Every line that matched, oldest first.
    found: List[Notified] = field(default_factory=list)
    #: The newest of those, as many as one answer holds, oldest first.
    shown: List[Notified] = field(default_factory=list)
    #: Nothing was ever notified, and no bot is set up to notify with.
    unconfigured: bool = False

    @property
    def grade(self) -> str:
        """One word for the step's history line."""
        return "found" if self.found else "nothing found"

    def summary(self) -> str:
        """What the model is told: the lines themselves, on one line."""
        if not self.found:
            if self.unconfigured:
                return ("nothing has been notified: notifications are not set "
                        "up, so nothing records what earlier passes read")
            if self.filters:
                return "nothing notified before matches that"
            return "nothing has been notified yet"
        head = (f"{len(self.found)} notified before" if self.filters
                else f"{len(self.found)} notified so far")
        out = f"{head}, oldest first: " + " | ".join(
            line.shown() for line in self.shown)
        hidden = len(self.found) - len(self.shown)
        if hidden:
            out += (f" (+{hidden} older not shown: name the people you are "
                    f"looking for)")
        return out


class Notifier:
    """Delivers the lines of a `notify` that were never delivered, and remembers.

    `transport` is what does the delivering -- `Telegram`, or anything with its
    `send(text)` -- and None when no bot is configured.
    """

    def __init__(self, directory: "str | Path", transport: Optional[Any] = None,
                 *, now: Callable[[], float] = time.time) -> None:
        self.path = Path(directory).expanduser() / LEDGER_NAME
        self.transport = transport
        self._now = now

    @classmethod
    def from_config(cls, cfg: "Config") -> "Notifier":
        token, chat = cfg.telegram_token(), cfg.telegram_chat_id()
        transport = Telegram(token, chat) if token and chat else None
        return cls(cfg.run.artifacts_dir, transport)

    def notify(self, text: str) -> Outcome:
        """Send what is new in `text`, one line per notification. Never raises.

        A notification is never the reason a run stops: delivery failing is
        reported in the outcome, and the line is kept for the next one.
        """
        out = Outcome()
        if self.transport is None:
            out.unconfigured = True
            return out
        lines = lines_of(text)
        with _LOCK:
            ledger = self._read()
            delivered_before = _Index(e["text"] for e in ledger.values()
                                      if e.get("sent"))
            fresh: List[str] = []
            for line in lines:
                if delivered_before.holds(line):
                    out.repeats.append(line)
                else:
                    fresh.append(line)
            asked = _Index(fresh)
            now = self._now()
            # Kept lines still owed: not delivered since in any spelling, not
            # asked for again by this very notify, and not too old to matter.
            backlog = [e["text"] for e in ledger.values()
                       if not e.get("sent")
                       and now - _when(e) < PENDING_TTL_S
                       and not delivered_before.holds(e["text"])
                       and not asked.holds(e["text"])]
            if not fresh and not backlog:
                return out
            delivered, out.error = self._deliver(backlog + fresh)
            records: List[Dict[str, Any]] = []
            for line in backlog:
                if line in delivered:
                    out.caught_up.append(line)
                    records.append({"at": now, "text": line, "sent": True})
            known = _Index(ledger)
            for line in fresh:
                if line in delivered:
                    out.sent.append(line)
                    records.append({"at": now, "text": line, "sent": True})
                    continue
                out.kept.append(line)
                # Only the first failure is written down: its time is what the
                # line ages from, and a bot that keeps failing would otherwise
                # add a record per line per pass.
                if not known.holds(line):
                    records.append({"at": now, "text": line, "sent": False,
                                    "error": out.error})
            self._append(records)
        return out

    def read_back(self, text: str = "") -> ReadBack:
        """The lines notified before that `text` asks about. Never raises.

        Each line of `text` looks for one thing (`matches`), and a line in the
        ledger is found when it matches any of them; with nothing to look for
        -- no `text`, or none with a word in it -- every line is found.

        Not under `_LOCK`: `notify` holds it across a delivery, which can take
        seconds, and the file is only ever appended to. The worst a read can do
        is miss the line being written, and one cut short is skipped.
        """
        filters = lines_of(text)
        wanted = [line for line in filters if words(line)]
        history = self._history()
        found = [line for line in history
                 if not wanted or any(matches(line.text, w) for w in wanted)]
        shown: List[Notified] = []
        size = 0
        for line in reversed(found):
            size += len(line.shown()) + 3
            if shown and (len(shown) >= READ_MAX_LINES or size > READ_MAX_CHARS):
                break
            shown.append(line)
        shown.reverse()
        return ReadBack(filters=filters, found=found, shown=shown,
                        unconfigured=self.transport is None and not history)

    def _deliver(self, lines: Sequence[str]) -> Tuple[Set[str], str]:
        """Send `lines` in as few messages as will go; what went, and why not."""
        delivered: Set[str] = set()
        assert self.transport is not None
        for chunk in _chunks(lines):
            try:
                self.transport.send("\n".join(chunk))
            except Exception as exc:  # noqa: BLE001 - a notification never ends a run
                why = str(exc) if isinstance(exc, NotifyError) \
                    else f"{type(exc).__name__}: {exc}"
                log.warning("notification not delivered: %s", why)
                return delivered, why
            delivered.update(chunk)
        return delivered, ""

    def _read(self) -> Dict[str, Dict[str, Any]]:
        """The latest entry for each line ever notified, by its exact text.

        Exact, so that a line kept and later delivered is one line whose last
        word is "sent". Whether two different texts say the same thing is
        `same`'s question, asked of what this returns -- never written into the
        file, so a change to how lines are compared never makes an old one new.
        """
        return {entry["text"]: entry for entry in self._entries()}

    def _history(self) -> List[Notified]:
        """Every line ever notified, once each, in the order they first were.

        Dated by its first entry, which the pass that read it wrote; a line
        kept then and delivered since counts as delivered.
        """
        out: Dict[str, Notified] = {}
        for entry in self._entries():
            known = out.get(entry["text"])
            if known is None:
                out[entry["text"]] = Notified(at=_when(entry), text=entry["text"],
                                              sent=bool(entry.get("sent")))
            elif entry.get("sent"):
                known.sent = True
        return list(out.values())

    def _entries(self) -> Iterator[Dict[str, Any]]:
        """The ledger's entries, oldest first, each with its `text` a string."""
        try:
            with self.path.open(encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        entry = json.loads(raw)
                    except ValueError:
                        continue    # a line a crash cut short
                    if isinstance(entry, dict) and entry.get("text"):
                        entry["text"] = str(entry["text"])
                        yield entry
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("could not read %s: %s", self.path, exc)

    def _append(self, records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A crash can leave the last line cut short, without its newline.
            # Appended straight after it, the next record would share its line
            # and be skipped with it -- a line delivered and never written down.
            torn = False
            try:
                with self.path.open("rb") as tail:
                    tail.seek(-1, 2)
                    torn = tail.read(1) != b"\n"
            except OSError:
                pass    # no file yet, or an empty one
            with self.path.open("a", encoding="utf-8") as handle:
                if torn:
                    handle.write("\n")
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Delivered and not written down: the next pass may send these
            # again. Said loudly, because that is the one thing this file is for.
            log.error("could not record notifications in %s (%s); they may be "
                      "sent again", self.path, exc)


class _Index:
    """Lines, looked up by `same` rather than by spelling.

    Grouped by `words` first, so a lookup compares emoji only among the few
    lines that already say the same words -- the ledger grows by a line or two a
    pass, and is read whole on every `notify`.
    """

    def __init__(self, lines: "Any") -> None:
        self._by_words: Dict[str, List[Optional[str]]] = {}
        for line in lines:
            self._by_words.setdefault(words(line), []).append(emoji(line))

    def holds(self, line: str) -> bool:
        mine = emoji(line)
        return any(mine is None or theirs is None or mine == theirs
                   for theirs in self._by_words.get(words(line), ()))


def _when(entry: Dict[str, Any]) -> float:
    try:
        return float(entry.get("at") or 0)
    except (TypeError, ValueError):
        return 0.0
