"""The send side of a chat screen: which actions send, and what has been sent.

**Reading the thread is the model's job, not this module's.** Who said what, and
whether the newest message is already ours, used to be read here: the message
list was the deepest tall scroller, a bubble hugging the right edge was ours,
and `prompts.conversation_block` handed the model the thread marked `us:` and
`them:` with a verdict under it -- "the last message in this thread is YOURS
... Do NOT reply again". It grew a rule for every app that broke it: Instagram's
full-screen pager, Hinge's newest-first list, a relative timestamp drawn hard
against the left edge. ``runs/425beba3ca6c`` settled it. On a thread whose
composer row held a camera, a GIF and a voice control, the reader took all
three for messages -- the camera as theirs, the GIF and the voice control as
ours -- and the verdict came from those buttons instead of from the newest
bubble. The element list in the same prompt named every one of them beside the
text field, so the model had what it needed to read the screen correctly, and
the block told it otherwise in the imperative.

So nothing here reads the thread any more. A screen with a text field gets its
screenshot taken (`agent.needs_screenshot`), and whoever decides -- the decider,
then the send check -- takes who sent the newest message from the pixels, where
the sides and the order are drawn rather than inferred. The vision pass reports
it in a field of its own, `ScreenAnalysis.last_message`, for a decider that
cannot see.

Two consequences worth being explicit about:

* **Nothing survives the process.** A reply this loop sent is knowable only
  because it is *on the screen*. Scroll it out of the frame, or let the app fail
  to render it, and the evidence is gone.
* **A determined model can still double-reply.** `ITERATION_CONTRACT` and the
  policy say not to, and the screenshot gives them the evidence to act on. What
  can refuse is the send check (`Agent._check_send`): a second, small model
  shown only the policy, the screen, the draft and `SendLog`, asked whether the
  send breaks a rule. It closes the gap left by a decider that misread the
  thread -- ``runs/0fc8159ca26c`` was about to answer "Nvmm." with the reply the
  greeting rule reserves for greetings -- and not the gap left by a thread
  nobody can see: a screen the app keeps out of screenshots comes back blank,
  and then the element list is all either model has.

What is left here refuses sends without asking a model, and it is code because
a model must not be able to argue its way past it. `watch.draft` means
"compose, never send", so it holds whatever the model concluded; a policy's
`send_limits` cap how many sends of a kind one run may make (`SendLog`). Both
see every door a send can go out through -- `send_label` -- because in most
chat apps the keyboard's action key sends too, and gating only the button would
leave the other door open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .actions import (AgentAction, element_at_point, elements_named,
                      resolve_target)
from .config import Config
from .fingerprint import CHAT_SEND_TEXT
from .screen import SYSTEM_UI_PACKAGES, Element, Screen

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


def app_nodes(screen: Screen) -> List[Element]:
    """Every visible node the *app* drew, before pruning.

    Not `screen.elements`, which is the model's view of the screen and is shaped
    for that: `prune` drops a node whose text its *tappable* ancestor already
    carries, `_collapse_identical_siblings` folds a repeated line into one entry
    with a count, and `RENDER_LIMIT` truncates at 80 elements. All of that is
    right for a prompt and wrong for `watch.screen_digest`, which has to notice a
    second "ok" arriving under the first.

    System chrome is dropped on the same reasoning as `Screen.content_elements`:
    the status bar clock is not part of the app. Unless the system UI *is* the
    screen, in which case nothing is excluded.
    """
    if not screen.package or screen.package in SYSTEM_UI_PACKAGES:
        return [n for n in screen.nodes if n.visible]
    return [n for n in screen.nodes if n.visible and not n.is_system_chrome]


def has_text_field(screen: Screen) -> bool:
    """Is there somewhere to type on this screen?

    What `agent.needs_screenshot` takes a frame for. A conversation always has
    one, and who sent its newest message is read off the frame, not the tree.
    A search box or a form trips it too, and that costs a screenshot a screen did
    not need -- the cheap direction to be wrong in.
    """
    return any(e.editable for e in app_nodes(screen))


def _composer_focused(screen: Screen) -> bool:
    return any(e.editable and e.focused for e in app_nodes(screen))


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
    a tap on it would strand the loop on a screen it is allowed to read. The
    element must be a control, too -- pressable, which counts a button the dump
    does not flag clickable (`Element.pressable`) -- and never a text field,
    whatever its hint says. Hinge's message field is hinted "Send a message",
    and a tap that only focused it went down as a reply sent
    (``runs/f94def19db5d``); what sends from a field is its action key or
    ``press_enter``, both counted below.
    """
    if action.action in ("tap", "long_press") and action.target is not None:
        element = resolve_target(action.target, screen)
        if element is None or element.editable or not element.pressable:
            return ""
        label = (element.best_text or "").strip()
        return label if _names_a_send(label) else ""
    if action.action == "tap_at":
        # Named first: before the locate runs there is no point yet, and the
        # model's description is the only thing saying what is about to be hit.
        # Unless it names listed elements and every one is a text field: then
        # what is about to be hit is a field.
        described = " ".join((action.text or "").split())
        if (described and len(described) <= _SEND_DESCRIPTION_MAX_CHARS
                and CHAT_SEND_TEXT.search(described)):
            named = elements_named(screen, described)
            if named and all(e.editable for e in named):
                return ""
            return described
        if action.x is not None and action.y is not None:
            element = element_at_point(screen, action.x, action.y)
            if element is None or element.editable:
                return ""
            label = (element.best_text or "").strip()
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

    Never refuses on anything else. Whether a reply is *owed* is the model's
    call, made from the screen, and then checked by `Agent._check_send`; how
    many sends one run may make is `SendLog`'s.
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
    """One send that went through: the control it went out on, and when.

    It used to carry the conversation it went into as well, read off the
    thread's header by the same heuristics that read the thread. Those are gone
    (see the module docstring), and a name read by them was the one part of
    this record that could be wrong. Which conversation a send went into is in
    the step history, in the model's own words.
    """

    step: int
    label: str


@dataclass
class SendLog:
    """What this run has sent, counted by the harness rather than remembered.

    A model's own tally drifts. ``runs/115da462a220`` sent eight likes on a
    goal of seven and its `done` said "Sent 7"; ``runs/8de32967fc18`` had sent
    six against "at most 5" and was composing a seventh when it was stopped.
    The judge passed the first -- "at least 7" -- because nothing it was shown
    could tell the difference. This counts the sends `verify` saw land, and the
    count is what the model, the send check and the judge are all shown.

    Per run, which is per pass in a watch: nothing survives the process, and
    what earlier passes sent is known only from what is on the screen. A send
    that opens a confirmation instead of sending -- an upsell's "Send like
    anyway" -- counts twice for one like. That is the safe direction to be
    wrong in.
    """

    sent: List[Sent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.sent)

    def record(self, step: int, label: str) -> None:
        self.sent.append(Sent(step=step, label=label))

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
            groups: Dict[str, int] = {}
            for s in self.sent:
                groups[s.label] = groups.get(s.label, 0) + 1
            for label, n in groups.items():
                lines.append(f"  - {n} x \"{label}\"")
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
