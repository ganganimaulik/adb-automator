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
  policy say not to, and this block gives them the evidence to act on. What
  can refuse is the send check (`Agent._check_send`): a second, small model
  shown only the policy, this thread, the draft and `SendLog`, asked whether
  the send breaks a rule. It sees what the decider saw, so it closes the gap
  left by a decider that misread it -- ``runs/0fc8159ca26c`` was about to
  answer "Nvmm." with the reply the greeting rule reserves for greetings --
  and not the gap left by a thread nobody can see.

Three things here refuse a send without asking a model. `watch.draft` means
"compose, never send", so it holds whatever the model concluded; a policy's
`send_limits` cap how many sends of a kind one run may make (`SendLog`). Both
see every door a send can go out through -- `send_label` -- because in most
chat apps the keyboard's action key sends too, and gating only the button would
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
from typing import Dict, List, Mapping, Optional, Tuple

from .actions import AgentAction, element_at_point, resolve_target
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

#: Resource-id fragments marking something drawn *in* the thread that nobody
#: said: a timestamp, a date divider, an unread marker.
#:
#: Geometry alone cannot tell these from a short incoming message, and the
#: commonest of them breaks the one signal that matters. A relative timestamp
#: under the newest bubble is drawn hard against the left edge, so it classifies
#: as `THEIRS` -- and then a thread whose last real bubble is ours reads as
#: "they spoke last", which is the exact wrong answer on a reply pass. Width
#: does not separate them either: "2m" and "ok" are both short.
#:
#: So they are recognised the way the title is, by what the app calls them.
#: Anything matching keeps its place in the thread and carries no side.
_NOT_A_MESSAGE_RID = re.compile(r"time|stamp|date|divider|separator|unread",
                                re.I)

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

#: A matching label longer than this is prose, not a button...
_SEND_LABEL_MAX_CHARS = 24

#: ...unless it opens with the verb, which is how a control is phrased and how
#: a bubble that merely mentions sending usually is not. Hinge's like pill reads
#: "Send priority like with message" once a comment is on it -- 31 characters --
#: and under the 24-character rule alone not one like that went out with its
#: comment ever counted as a send, so draft mode would have let every one of
#: them through.
_SEND_COMMAND_MAX_CHARS = 48

#: A `tap_at` names its control in the model's own words ("the Send Priority
#: Like pill at the bottom of the sheet"), so the cap on what can count as naming
#: a send control is a description's length rather than a button's.
_SEND_DESCRIPTION_MAX_CHARS = 80

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
    once rather than twice. Document order is not reading order -- see the sort
    in `read_conversation`.
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
    """Which side of the thread `el` sits on, or "" when nobody said it.

    `left` and `right` are the message list's own edges, not the screen's: a
    thread inset from the window, or one beside a navigation rail on a tablet,
    is still read against the list it is drawn in.
    """
    if _NOT_A_MESSAGE_RID.search(rid_norm(el.resource_id)):
        return ""
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

    # Oldest first by where the bubbles are drawn, not by where they sit in the
    # tree. A chat list laid out from the bottom up -- `reverseLayout`, or an
    # adapter that holds the newest message at position 0 -- emits its children
    # newest-first, and Hinge's does: in ``runs/0fc8159ca26c`` step 3 the bubbles
    # come out at y=1355, then 1194, then 934. Read in that order, `said[-1]` is
    # the *oldest* message -- on Hinge the like comment that opened the match --
    # so a thread they answered last read as "yours, do not reply", and a thread
    # you had already answered, opened by *their* like, read as "theirs, no reply
    # under it yet". Top edge first, then left edge, and the sort is stable, so
    # two bubbles on one row keep the order the tree gave them.
    body = sorted(body, key=lambda e: (e.bounds[1], e.bounds[0]))

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


def _names_a_send(label: str) -> bool:
    """Does this control's label say it sends? See `_SEND_COMMAND_MAX_CHARS`."""
    label = label.strip()
    if not label or not CHAT_SEND_TEXT.search(label):
        return False
    if len(label) <= _SEND_LABEL_MAX_CHARS:
        return True
    return (len(label) <= _SEND_COMMAND_MAX_CHARS
            and CHAT_SEND_TEXT.match(label) is not None)


def send_label(action: AgentAction, screen: Screen) -> str:
    """The label of the control this action would send with, or "".

    All four doors, because covering only the button leaves the others open:

    * a tap on something labelled send/post/share/publish,
    * a `tap_at` that names one -- the escape hatch for a control the list does
      not show, and the way every Hinge like went out before the parser exposed
      the like pill. It used to be no door at all: draft mode, which checks only
      this function, would have let all of them through,
    * the keyboard action key while a composer holds focus,
    * ``input_text`` with ``press_enter``, which types and sends in one step.

    The label must look like a control's as well as matching: a message bubble
    that happens to contain the word "send" is not a send control, and refusing
    a tap on it would strand the loop on a screen it is allowed to read.
    """
    if action.action in ("tap", "long_press") and action.target is not None:
        element = resolve_target(action.target, screen)
        if element is None or not element.interactive:
            return ""
        label = (element.best_text or "").strip()
        return label if _names_a_send(label) else ""
    if action.action == "tap_at":
        # Named first: before the locate runs there is no point yet, and the
        # model's description is the only thing saying what is about to be hit.
        described = " ".join((action.text or "").split())
        if (described and len(described) <= _SEND_DESCRIPTION_MAX_CHARS
                and CHAT_SEND_TEXT.search(described)):
            return described
        if action.x is not None and action.y is not None:
            element = element_at_point(screen, action.x, action.y)
            label = (element.best_text or "").strip() if element is not None else ""
            return label if _names_a_send(label) else ""
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

    `watch.draft` means "compose, never send", so what it holds back is
    whatever the model decided, whether or not that decision was right. It is
    the first thing to run when a policy changes -- the failure mode becomes a
    wrong draft in the log instead of a wrong message in somebody's inbox.

    Never refuses on anything else. Whether a reply is *owed* is decided from
    `prompts.conversation_block` by the model, and then checked by
    `Agent._check_send`; how many sends one run may make is `SendLog`'s.
    """
    if not cfg.watch.draft:
        return ""
    if not send_label(action, screen):
        return ""
    return ("draft mode is on -- the reply was composed and recorded but not "
            "sent; move on to the next thread")


# ---------------------------------------------------------------------------
# what a send says, and what this run has already sent
# ---------------------------------------------------------------------------

def draft_in(action: AgentAction, screen: Screen) -> str:
    """What a send is about to say, or "" when the screen does not show it.

    The text `input_text` is about to type, when the send is that; otherwise
    whatever sits in a composer, the focused one first. A field's hint is not
    a draft -- an empty composer often dumps its placeholder as its text. ""
    is a real answer: Hinge's like sheet keeps its comment out of the tree, and
    the send check is told to read the recent steps for it instead.
    """
    if action.action == "input_text":
        return (action.text or "").strip()
    fields = sorted((e for e in app_nodes(screen) if e.editable),
                    key=lambda e: not e.focused)
    for f in fields:
        text = (f.text or "").strip()
        if text and text != (f.hint or "").strip():
            return text
    return ""


def thread_on(screen: Screen) -> Optional[Conversation]:
    """The conversation on screen, or None when there is no readable one."""
    if not on_a_conversation(screen):
        return None
    convo = read_conversation(screen)
    return convo if convo.readable else None


def _mentions(label: str, word: str) -> bool:
    """`word` appears in `label` as a word of its own, in any case."""
    return re.search(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", label,
                     re.I) is not None


def parse_send_limits(text: str) -> Dict[str, int]:
    """``like=5, rose=0`` as ``{"like": 5, "rose": 0}``. Raises ValueError.

    The policy's `send_limits` front matter key. Each word caps the sends whose
    control mentions it -- so ``like=5`` counts "Send priority like with
    message" and leaves "Send message" alone -- and 0 forbids them outright.
    Strict on purpose: a limit that is silently misread is a limit that is not
    there, and the operator is told so before the watch starts rather than
    after it has sent the sixth one.
    """
    limits: Dict[str, int] = {}
    for part in re.split(r"[,;]", text or ""):
        part = part.strip()
        if not part:
            continue
        matched = re.fullmatch(r"([^=:]+?)\s*[=:]\s*(\d+)", part)
        if not matched:
            raise ValueError(f"{part!r} is not a send limit -- write it as "
                             f"word=N, e.g. like=5")
        limits[matched.group(1).strip().lower()] = int(matched.group(2))
    return limits


@dataclass
class Sent:
    """One send that went through: the control it went out on, and where."""

    step: int
    label: str
    #: The conversation it went into, when it went into a readable one.
    thread: str = ""


@dataclass
class SendLog:
    """What this run has sent, counted by the harness rather than remembered.

    A model's own tally drifts. ``runs/115da462a220`` sent eight likes on a
    goal of seven and its `done` said "Sent 7"; ``runs/8de32967fc18`` had sent
    six against "at most 5" and was composing a seventh when it was stopped.
    The judge passed the first -- "at least 7" -- because nothing it was shown
    could tell the difference. This counts the sends `verify` saw land, and the
    count is what the model, the send check and the judge are all shown.

    Per run, which is per pass in a watch: nothing survives the process, the
    same trade `conversation_block` makes. A send that opens a confirmation
    instead of sending -- an upsell's "Send like anyway" -- counts twice for
    one like. That is the safe direction to be wrong in.
    """

    sent: List[Sent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sent)

    def record(self, step: int, label: str, thread: str = "") -> None:
        self.sent.append(Sent(step=step, label=label, thread=thread))

    def count(self, word: str) -> int:
        """Sends whose control mentions `word`."""
        return sum(1 for s in self.sent if _mentions(s.label, word))

    def over_limit(self, label: str, limits: Mapping[str, int]) -> str:
        """Why one more send on `label` would break a limit, or ""."""
        for word, most in limits.items():
            if not _mentions(label, word):
                continue
            used = self.count(word)
            if not most:
                return (f"the policy forbids \"{word}\" sends -- do not send "
                        f"this; finish whatever else the goal needs, then "
                        f"report done")
            if used >= most:
                return (f"the policy allows {most} \"{word}\" send(s) per run "
                        f"and {used} have already gone out -- send no more of "
                        f"them; finish whatever else the goal needs, then "
                        f"report done")
        return ""

    def render(self, limits: Optional[Mapping[str, int]] = None) -> str:
        """The block the decider, the send check and the judge are shown.

        "" when there is nothing sent and nothing limited, so a run that never
        sends carries no extra text into any prompt.
        """
        limits = limits or {}
        if not self.sent and not limits:
            return ""
        if self.sent:
            lines = [f"SENT THIS RUN, counted by the harness from the sends that "
                     f"went through -- when your own count disagrees, this one "
                     f"is right: {len(self.sent)}"]
            groups: Dict[Tuple[str, str], int] = {}
            for s in self.sent:
                groups[(s.label, s.thread)] = groups.get((s.label, s.thread), 0) + 1
            for (label, thread), n in groups.items():
                where = f" in the conversation with {thread}" if thread else ""
                lines.append(f"  - {n} x \"{label}\"{where}")
        else:
            lines = ["SENT THIS RUN: nothing yet."]
        for word, most in limits.items():
            used = self.count(word)
            if not most:
                lines.append(f"FORBIDDEN: \"{word}\" sends. Any will be "
                             f"refused.")
            elif used >= most:
                lines.append(f"LIMIT REACHED: {used} of {most} \"{word}\" "
                             f"send(s) used. Any more will be refused -- finish "
                             f"whatever else the goal needs, then report done.")
            else:
                lines.append(f"LIMIT: at most {most} \"{word}\" send(s) this "
                             f"run -- {used} used, {most - used} left.")
        return "\n".join(lines)
