"""Reading a chat screen, and the gate that stands in front of every send.

Two jobs, kept together because the second is worthless without the first.

**Reading.** Which conversation is this, and what are the last things said in it?
Answered geometrically and from id naming conventions, not from any one app's
layout: the message list is the biggest scroller, the messages are the text
inside it, and the title is the most title-shaped thing above it. Nothing here
knows what Instagram is. An app whose header the heuristics cannot read yields a
`Conversation` that says so, and the gate refuses on it rather than guessing.

**Gating.** `reply_gate` is the harness-level half of the never-double-reply
guarantee. The prompt also tells the model which threads are handled, but a
prompt is advice; this is the part that cannot be talked out of it. It runs
immediately before the gesture goes out, on the screen the gesture will land on,
and it says no when:

* the conversation cannot be identified or read at all (`watch.fail_closed`),
* this exact conversation tail has already been replied to,
* the thread is inside its cooldown, or was left in doubt by an unconfirmed send,
* a rolling reply ceiling has been reached,
* draft mode is on, in which case nothing is ever sent.

The send paths it covers are the Send control *and* the enter key, because in
most chat apps the keyboard's action key sends too -- gating only the button
would leave the other door open.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .actions import AgentAction, resolve_target
from .config import Config
from .fingerprint import CHAT_SEND_TEXT, rid_norm
from .ledger import ReplyLedger, Verdict, content_digest, thread_key
from .screen import SYSTEM_UI_PACKAGES, Element, Screen

log = logging.getLogger("adbagent.conversation")

#: Resource-id fragments that mark the element holding a conversation's name.
#: A naming convention across Android apps, not one app's ids.
_TITLE_RID = re.compile(r"title|name|username|thread|contact|header", re.I)

#: Header controls that are never the conversation's name.
_NOT_A_TITLE = re.compile(
    r"^\s*(?:back|up|close|navigate up|cancel|menu|more|options|search|"
    r"call|video call|audio call|info|details|profile picture|avatar)\s*$", re.I)

#: A header text longer than this is a message that has drifted up, not a name.
_TITLE_MAX_CHARS = 60

#: Fraction of screen height treated as header when the scroller cannot be used
#: to split one off -- no scroller at all, or one that contains the header.
_HEADER_BAND = 0.15

#: A conversation's message list is tall. Anything shorter than this fraction of
#: the screen is a reaction strip, a suggestion carousel or an emoji tray, and
#: picking one of those as the message list finds no messages.
_MIN_SCROLLER_HEIGHT = 0.25

#: Keys that send a message in a chat composer.
_SEND_KEYS = {"enter", "search", "send", "done", "go"}

#: A matching label longer than this is prose, not a button.
_SEND_LABEL_MAX_CHARS = 24


@dataclass
class Conversation:
    """What could be read off a chat screen."""

    title: str = ""
    messages: List[str] = field(default_factory=list)
    #: Empty when the screen was read successfully; otherwise why it was not.
    problem: str = ""

    @property
    def key(self) -> str:
        return thread_key(self.title)

    @property
    def digest(self) -> str:
        return content_digest(self.messages)

    @property
    def readable(self) -> bool:
        """Is this a conversation this loop can safely act in?

        `problem` is authoritative. It used to be ignored here, which made every
        check that only sets `problem` -- the missing-composer one especially --
        dead code, because `reply_gate` asks this and not that.
        """
        return not self.problem and bool(self.key and self.digest)

    def preview(self) -> str:
        last = self.messages[-1] if self.messages else ""
        return f"{self.title}: {last}"[:200]


def app_nodes(screen: Screen) -> List[Element]:
    """Every visible node the *app* drew, before pruning.

    `screen.elements` is unusable here, and the reason is worth stating because
    it is not obvious. It used to be the sharpest one: `_absorb_labels` folded a
    non-interactive subtree's text into any *interactive* ancestor, and a
    scroller counts as interactive, so the whole conversation arrived as one
    string on the message scroller (``label='hey you around? 2m'``) with the
    bubbles pruned away. That is fixed -- absorption is limited to actionable
    ancestors, and a thread now renders bubble by bubble.

    The rest of the reasoning stands, so this still reads raw nodes. `prune`
    legitimately drops a bubble whose text its *tappable* row already carries;
    `_collapse_identical_siblings` folds a repeated message into one entry with
    a count, which is right for a render and wrong for a digest; and
    `RENDER_LIMIT` truncates at 80 elements. A digest built on any of those
    would move when the render did, which is not what it is measuring.

    System chrome is dropped on the same reasoning as `Screen.content_elements`:
    the status bar clock is not part of the conversation. Unless the system UI
    *is* the screen, in which case nothing is excluded and the caller will find
    no conversation on it -- correctly.
    """
    if not screen.package or screen.package in SYSTEM_UI_PACKAGES:
        return [n for n in screen.nodes if n.visible]
    return [n for n in screen.nodes if n.visible and not n.is_system_chrome]


def _leaves(nodes: List[Element]) -> List[Element]:
    """Text-bearing leaves, in document order.

    Leaves only, so a bubble whose text is repeated on its container is counted
    once rather than twice.
    """
    return [n for n in nodes if not n.children and n.best_text.strip()]


def message_scroller(screen: Screen) -> Optional[Element]:
    """The scroller holding the messages: the *deepest* tall one.

    Deepest, not largest, and the difference is not academic. Observed live on
    `com.instagram.android`: the largest scrollable is a full-screen
    `swipeable_tab_view_pager` that contains the thread header as well as the
    message list. Choosing it reads the correspondent's name as one of the
    messages, finds no title above the list, and refuses every send -- safe, and
    useless. The message list is the innermost thing still tall enough to be a
    conversation.

    The height floor is what keeps "innermost" from picking a reaction strip or
    an emoji tray nested inside the thread.
    """
    scrollers = [e for e in app_nodes(screen) if e.scrollable and e.area > 0]
    if not scrollers:
        return None
    floor = (screen.height or 0) * _MIN_SCROLLER_HEIGHT
    tall = [e for e in scrollers if e.height >= floor]
    return max(tall or scrollers, key=lambda e: (e.depth, e.area))


def _in_scroller(el: Element, scroller: Element) -> bool:
    return any(anc is scroller for anc in el.ancestors())


def _title_from(candidates: List[Element]) -> str:
    """The most title-shaped text among a header's elements.

    Preference order: an id that says it is a title, then the widest remaining
    text. Icon buttons are narrow and square; a name is wide.
    """
    usable = [e for e in candidates
              if e.best_text.strip()
              and len(e.best_text.strip()) <= _TITLE_MAX_CHARS
              and not _NOT_A_TITLE.match(e.best_text)
              and not e.editable]
    if not usable:
        return ""
    named = [e for e in usable if _TITLE_RID.search(rid_norm(e.resource_id))]
    pool = named or usable
    best = max(pool, key=lambda e: (e.width, -e.node_index))
    return best.best_text.strip()


def read_conversation(screen: Screen) -> Conversation:
    """Read the conversation on screen. Never raises; says what it could not do.

    The composer is excluded from the messages on purpose: it holds our own
    half-typed draft, and letting that into the digest would make the tail change
    as the model types.
    """
    content = _leaves(app_nodes(screen))
    if not content:
        return Conversation(problem="the screen has no app content on it")

    band = int((screen.height or 0) * _HEADER_BAND)
    scroller = message_scroller(screen)
    if scroller is not None:
        top = scroller.bounds[1]
        body = [e for e in content if _in_scroller(e, scroller)]
        header = [e for e in content
                  if e.bounds[3] <= top and not _in_scroller(e, scroller)]
    else:
        # No scroller: a short thread in a plain container. Split by geometry.
        body = [e for e in content if e.bounds[3] > band]
        header = [e for e in content if e.bounds[3] <= band]

    if not header:
        # The chosen scroller starts at the top of the window, so it contains its
        # own header -- some apps really do draw it that way. Fall back to the
        # geometric band, and take those elements out of the messages so the
        # correspondent's name does not end up in the conversation digest.
        header = [e for e in body if e.bounds[3] <= band]
        chosen = {id(e) for e in header}
        body = [e for e in body if id(e) not in chosen]

    messages = [e.best_text.strip() for e in body if not e.editable]
    title = _title_from(header)

    problem = ""
    if not _has_composer(screen):
        # Observed live: the Instagram inbox in multi-select mode has a scroller,
        # rows, and a plausible title ("0 selected"), so every other test here
        # passed on a screen that is not a conversation at all. A thread always
        # has somewhere to type; a list of threads does not. Requiring the
        # composer is what makes `readable` mean "this is a conversation", which
        # is the thing the gate actually needs to know.
        problem = "this screen has no message composer, so it is not a conversation"
    elif not title:
        problem = "no conversation name could be found above the message list"
    elif not messages:
        problem = "no messages could be read in the conversation"
    convo = Conversation(title=title, messages=messages, problem=problem)
    log.debug("conversation: title=%r messages=%d digest=%s%s",
              convo.title, len(messages), convo.digest[:8],
              f" problem={problem!r}" if problem else "")
    return convo


def _has_composer(screen: Screen) -> bool:
    """Is there somewhere to type on this screen?

    The cheapest reliable evidence that a screen is a conversation rather than a
    list of them.
    """
    return any(e.editable for e in app_nodes(screen))


def _composer_focused(screen: Screen) -> bool:
    return any(e.editable and e.focused for e in app_nodes(screen))


def send_label(action: AgentAction, screen: Screen) -> str:
    """The label of the control this action would send with, or "".

    All three doors, because gating only the button leaves the others open:

    * a tap on something labelled send/post/share/publish,
    * the keyboard action key while a composer holds focus,
    * ``input_text`` with ``press_enter``, which types and sends in one step.

    The label must be short as well as matching: a message bubble that happens
    to contain the word "send" is not a send control, and refusing a tap on it
    would strand the loop on a screen it is allowed to read.
    """
    if action.action in ("tap", "long_press") and action.target is not None:
        element = resolve_target(action.target, screen)
        if element is None or not element.interactive:
            return ""
        label = (element.best_text or "").strip()
        if label and len(label) <= _SEND_LABEL_MAX_CHARS \
                and CHAT_SEND_TEXT.search(label):
            return label
        return ""
    if action.action == "press_key":
        key = (action.key or "").strip().lower()
        if key in _SEND_KEYS and _composer_focused(screen):
            return key
    if action.action == "input_text" and action.press_enter:
        return "input_text with press_enter"
    return ""


def reply_gate(action: AgentAction, screen: Screen, ledger: ReplyLedger,
               cfg: Config, *, now: Optional[float] = None) -> Verdict:
    """Whether this send may go out. Allowed by default for non-send actions.

    Returning `Verdict(False, reason)` is not an error: the reason is handed back
    to the model as the outcome of the step, so it can move on to another thread
    instead of retrying the one it is not allowed to answer.
    """
    label = send_label(action, screen)
    if not label:
        return Verdict(True)

    w = cfg.watch
    if w.draft:
        return Verdict(False, "draft mode is on -- the reply was composed and "
                              "recorded but not sent; move on to the next thread")

    convo = read_conversation(screen)
    if not convo.readable:
        why = convo.problem or "the conversation could not be read"
        if w.fail_closed:
            return Verdict(False, f"refusing to send because {why}, so a "
                                  f"duplicate reply could not be ruled out")
        log.warning("sending without an identifiable conversation (%s) -- "
                    "watch.fail_closed is off", why)
        return Verdict(True)

    verdict = ledger.check(convo.key, convo.digest,
                           cooldown_s=w.thread_cooldown_s, now=now)
    if not verdict:
        return verdict

    stamp = time.time() if now is None else now
    hour_ago = stamp - 3600.0
    if w.max_replies_per_thread_per_hour > 0:
        n = ledger.replies_since(hour_ago, convo.key)
        if n >= w.max_replies_per_thread_per_hour:
            return Verdict(False, f"this conversation has already had {n} "
                                  f"repl(ies) this hour, which is the limit")
    if w.max_replies_per_hour > 0:
        n = ledger.replies_since(hour_ago)
        if n >= w.max_replies_per_hour:
            return Verdict(False, f"{n} replies have gone out this hour, which "
                                  f"is the limit -- not sending any more")
    return Verdict(True)
