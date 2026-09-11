"""Reading a chat screen: who said what, and which side said it.

**Reading.** Which conversation is this, what are the last things said in it, and
which of them are ours? Answered geometrically and from id naming conventions,
not from any one app's layout: the message list is the biggest scroller, the
messages are the text inside it, the title is the most title-shaped thing above
it, and a bubble that hugs the right edge is one we sent. Nothing here knows
what Instagram is. An app whose header the heuristics cannot read yields a
`Conversation` that says so, and the prompt block renders nothing for it.

**Why sides are guessed from bounds now.** The reply ledger that used to stand
here refused to do this. Its reasoning was sound for what it was: it was a hard
gate, fsynced before every send, and a *guess* about which side a bubble is on
is not a thing to refuse a send on -- so it digested the tail instead and kept
the answer on disk, where "have I replied to this" was a lookup rather than an
inference.

That gate is gone, and with it the reason to avoid the guess. What the sides
feed now is `prompts.conversation_block`: the thread, newest last, each message
marked `us:` or `them:`, handed to the model on the turn it decides. The model
is the thing deciding whether a reply is owed, and it decides better seeing the
thread than not seeing it. A misread bubble costs a worse-informed decision,
which is the ordinary cost of every other heuristic in this file -- not a
duplicate message, which is what it would have cost behind a gate.

Two consequences worth being explicit about, because they are the trade the
ledger's removal actually made:

* **Nothing survives the process.** A reply this loop sent is knowable only
  because it is *on the screen*. Scroll it out of the tail, or let the app fail
  to render it, and the evidence is gone. The ledger's digests-and-cooldowns
  closed exactly that window; this does not.
* **A determined model can still double-reply.** `ITERATION_CONTRACT` and the
  policy say not to, and this block gives them the evidence to act on. None of
  the three can refuse.

`watch.draft` is the one thing here that still refuses, and it is not about
duplicates: it means "compose, never send", so it holds whatever the model
concluded. It covers the Send control *and* the keyboard's action key, because
in most chat apps the action key sends too and gating only the button would
leave the other door open.

Sides are read for left-to-right layouts: the sent side is the right one. An
RTL locale mirrors that, and nothing here detects it -- a thread read in one
would come back with the sides swapped, which is why `side` is advisory and the
block labels what it shows rather than asserting it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .actions import AgentAction, resolve_target
from .config import Config
from .fingerprint import CHAT_SEND_TEXT, rid_norm
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

#: How much closer to one edge than the other a bubble must sit before its side
#: is called, as a fraction of the message list's width.
#:
#: It is a *difference* between the two margins rather than a threshold on
#: either, because neither margin alone says anything: apps inset the whole
#: thread, draw avatars in the gutter on the incoming side only, and let a long
#: message run nearly edge to edge. What survives all three is that an outgoing
#: bubble's right margin is much smaller than its left.
#:
#: At 0.1 a date separator, a full-width system notice and an
#: "encrypted end-to-end" banner all come back `""` -- they are centred or fill
#: the width, so their margins differ by less than a tenth of it. That is the
#: right answer for them: they are in the thread and belong in the block, and
#: nobody sent them.
_SIDE_MARGIN_DIFF = 0.1

#: Keys that send a message in a chat composer.
_SEND_KEYS = {"enter", "search", "send", "done", "go"}

#: A matching label longer than this is prose, not a button.
_SEND_LABEL_MAX_CHARS = 24

#: Which side a message is on, and what the prompt block calls it.
OURS = "us"
THEIRS = "them"


@dataclass
class Message:
    """One thing said in a conversation, and which side said it.

    `side` is `OURS`, `THEIRS`, or `""` for the things a thread contains that
    nobody said -- date separators, unread markers, security notices. Empty is a
    real answer and never a synonym for `THEIRS`: a block that labelled the
    "Today" separator as incoming would be inventing a message.
    """

    text: str
    side: str = ""

    @property
    def ours(self) -> bool:
        return self.side == OURS


@dataclass
class Conversation:
    """What could be read off a chat screen."""

    title: str = ""
    messages: List[Message] = field(default_factory=list)
    #: Empty when the screen was read successfully; otherwise why it was not.
    problem: str = ""

    @property
    def readable(self) -> bool:
        """Is this a conversation at all?

        `problem` is authoritative -- every check in `read_conversation` that
        sets it is a reason the screen is not a thread this loop can read.
        """
        return not self.problem and bool(self.title and self.messages)

    @property
    def texts(self) -> List[str]:
        return [m.text for m in self.messages]

    @property
    def said(self) -> List[Message]:
        """The messages somebody actually sent, separators excluded."""
        return [m for m in self.messages if m.side]

    @property
    def last_is_ours(self) -> bool:
        """Is the newest thing *said* in this thread ours?

        The one derived fact the block states outright, because it is the whole
        question on a reply pass: our own message sitting at the bottom of the
        thread means the last incoming one has been answered.

        Read off `said` rather than `messages`, so a date separator or an
        "unread" divider drawn below the final bubble cannot hide the answer.
        """
        said = self.said
        return bool(said) and said[-1].ours

    def preview(self) -> str:
        last = self.messages[-1].text if self.messages else ""
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
    a count, which is right for a render and wrong for a thread being read line
    by line; and `RENDER_LIMIT` truncates at 80 elements.

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
    messages and finds no title above the list.

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


def side_of(el: Element, left: int, right: int) -> str:
    """Which side of the thread `el` sits on, or "" when it straddles.

    `left` and `right` are the message list's own edges, not the screen's: a
    thread inset from the window, or one beside a navigation rail on a tablet,
    is still read against the list it is drawn in.
    """
    width = right - left
    if width <= 0:
        return ""
    gap_left = el.bounds[0] - left
    gap_right = right - el.bounds[2]
    if abs(gap_left - gap_right) < width * _SIDE_MARGIN_DIFF:
        return ""
    return OURS if gap_right < gap_left else THEIRS


def read_conversation(screen: Screen) -> Conversation:
    """Read the conversation on screen. Never raises; says what it could not do.

    The composer is excluded from the messages on purpose: it holds our own
    half-typed draft, and a draft is not something anybody has said yet.
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
        edges = (scroller.bounds[0], scroller.bounds[2])
    else:
        # No scroller: a short thread in a plain container. Split by geometry,
        # and measure the sides against the window.
        body = [e for e in content if e.bounds[3] > band]
        header = [e for e in content if e.bounds[3] <= band]
        edges = (0, screen.width or 0)

    if not header:
        # The chosen scroller starts at the top of the window, so it contains its
        # own header -- some apps really do draw it that way. Fall back to the
        # geometric band, and take those elements out of the messages so the
        # correspondent's name does not end up read as something they said.
        header = [e for e in body if e.bounds[3] <= band]
        chosen = {id(e) for e in header}
        body = [e for e in body if id(e) not in chosen]

    messages = [Message(text=e.best_text.strip(), side=side_of(e, *edges))
                for e in body if not e.editable and e.best_text.strip()]
    title = _title_from(header)

    problem = ""
    if not _has_composer(screen):
        # Observed live: the Instagram inbox in multi-select mode has a scroller,
        # rows, and a plausible title ("0 selected"), so every other test here
        # passed on a screen that is not a conversation at all. A thread always
        # has somewhere to type; a list of threads does not. Requiring the
        # composer is what makes `readable` mean "this is a conversation".
        problem = "this screen has no message composer, so it is not a conversation"
    elif not title:
        problem = "no conversation name could be found above the message list"
    elif not messages:
        problem = "no messages could be read in the conversation"
    convo = Conversation(title=title, messages=messages, problem=problem)
    log.debug("conversation: title=%r messages=%d (%d ours)%s",
              convo.title, len(messages),
              sum(1 for m in messages if m.ours),
              f" problem={problem!r}" if problem else "")
    return convo


def _has_composer(screen: Screen) -> bool:
    """Is there somewhere to type on this screen?

    The cheapest reliable evidence that a screen is a conversation rather than a
    list of them, and the gate on doing any of the rest of this work.
    """
    return any(e.editable for e in app_nodes(screen))


def _composer_focused(screen: Screen) -> bool:
    return any(e.editable and e.focused for e in app_nodes(screen))


def on_a_conversation(screen: Screen) -> bool:
    """Worth trying to read a thread off this screen at all.

    Checked before `read_conversation` on every turn, so the common case -- a
    launcher, a settings page, a feed -- costs one pass over the nodes instead
    of the scroller search, the header split and the side arithmetic.
    """
    return _has_composer(screen)


def send_label(action: AgentAction, screen: Screen) -> str:
    """The label of the control this action would send with, or "".

    All three doors, because covering only the button leaves the others open:

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


def draft_refusal(action: AgentAction, screen: Screen, cfg: Config) -> str:
    """Why this send may not go out, or "" when it may.

    The only refusal left in front of a send, and the only one that never needed
    durable state: `watch.draft` means "compose, never send", so what it holds
    back is whatever the model decided, whether or not that decision was right.
    It is the first thing to run when a policy changes -- the failure mode
    becomes a wrong draft in the log instead of a wrong message in somebody's
    inbox.

    Never refuses on anything else. Whether a reply is *owed* is the model's
    call, made from `prompts.conversation_block`; there is no harness veto on it
    any more.
    """
    if not cfg.watch.draft:
        return ""
    if not send_label(action, screen):
        return ""
    return ("draft mode is on -- the reply was composed and recorded but not "
            "sent; move on to the next thread")
