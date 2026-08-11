"""Guardrails: scope, credentials, irreversible actions, loops, explore mode.

Ordered by how much damage the thing being prevented would do.

1. **Credentials.** The agent never types a password, PIN, one-time code or card
   number. It detects those screens, stops, and hands control to the person. The
   screen is not written to artifacts and not learned.
2. **Irreversible actions.** Sending, buying, deleting, posting. These need a
   human yes, and they can never be replayed from cache -- the token that names
   them is stored as forbidden, so a cached step that would tap one is refused
   before it runs.
3. **Scope.** The run can be pinned to one package; anything else is an escape
   to be corrected, not explored.
4. **Loops.** The dominant failure mode in every published mobile agent: the
   model revisits the same screen forever and burns the step budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .actions import AgentAction, resolve_target
from .config import Config
from .fingerprint import CHAT_SEND_TEXT, DESTRUCTIVE_TEXT
from .screen import Element, Screen

log = logging.getLogger("adbagent.safety")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

SENSITIVE_TEXT = re.compile(
    r"\b(password|passcode|passphrase|\bpin\b|cvv|cvc|security code|"
    r"card number|credit card|debit card|expiry|expiration date|"
    r"one[- ]time (?:code|password)|\botp\b|verification code|2fa|"
    r"two[- ]factor|authenticator|billing address|bank account|"
    r"sort code|routing number|iban|social security|ssn)\b",
    re.I,
)


@dataclass
class SensitiveFinding:
    reason: str
    element: Optional[Element] = None


def sensitive_screen(screen: Screen) -> Optional[SensitiveFinding]:
    """Detect a screen the agent must not touch.

    Only flags credential screens when sensitive keywords appear in editable
    input fields — not in read-only text views (e.g. chat message bodies that
    mention words like "pin" or "code").
    """
    for el in screen.elements:
        if el.password:
            return SensitiveFinding("a password field is focused on this screen", el)
    for el in screen.elements:
        if el.editable:
            context = " ".join(filter(None, (el.best_text, el.hint, el.resource_id)))
            if SENSITIVE_TEXT.search(context):
                return SensitiveFinding(
                    f"an input field asks for {context.strip()!r}", el)
    return None


# ---------------------------------------------------------------------------
# Irreversible actions
# ---------------------------------------------------------------------------

def irreversible(action: AgentAction, screen: Screen) -> Optional[str]:
    """The label of the control being pressed, when pressing it cannot be undone.

    Chat-send labels ("send", "post", "share", "publish") are NOT treated as
    irreversible — sending messages is a normal part of automation.  Only truly
    destructive actions (delete, wipe, uninstall, buy, etc.) trigger a
    confirmation.
    """
    if action.action not in ("tap", "long_press"):
        return None
    if action.target is None:
        return None
    element = resolve_target(action.target, screen)
    if element is None:
        return None
    label = element.best_text
    if not label:
        return None
    # Only flag truly destructive actions, not send/post/share/publish.
    if DESTRUCTIVE_TEXT.search(label) and not CHAT_SEND_TEXT.search(label):
        return label
    return None


# ---------------------------------------------------------------------------
# Interstitials
# ---------------------------------------------------------------------------

#: Buttons that only ever *decline* or *acknowledge* noise -- rating nags,
#: feature tours, "what's new" cards. Tapping one can turn an offer down; it
#: cannot advance a flow or throw work away.
#:
#: Deliberately excludes "Allow", "Accept" and "Agree": granting a permission or
#: accepting terms is a decision, not noise, so it goes to the model (and to the
#: user, if it is irreversible) rather than being auto-tapped.
#:
#: "Continue", "Next" and "Close" used to be here and are decisions by that same
#: test. They are the primary control of an onboarding wizard, a checkout step or
#: a permission rationale, so an app that opens on one had its CTA pressed on
#: every turn until the step budget ran out, without the model being consulted
#: once. "Close" was worse than wasted steps: a compose screen whose X is
#: described as "Close" offers exactly one dismiss-shaped candidate, so the
#: single-candidate rule below fired and discarded the draft -- and `irreversible`
#: never saw it, because that only grades actions the *model* chose. The model
#: can press any of the three itself when it wants them.
DISMISS_LABELS = re.compile(
    r"^\s*(?:no thanks|no, thanks|not now|maybe later|later|remind me later|"
    r"don't show again|skip|skip for now|dismiss|got it|ok, got it)\s*$",
    re.I,
)

INTERSTITIAL_PACKAGES = {
    "com.android.vending",          # Play Store rating nags
    "com.google.android.gms",
}


def find_interstitial(screen: Screen, target_package: str = "") -> Optional[Element]:
    """A dismiss button on a dialog that is in the way but not part of the task."""
    if not screen.has_system_dialog() and not any(
            p in INTERSTITIAL_PACKAGES for p in screen.packages):
        # Also allow in-app nags, but only when the label is unambiguous noise.
        candidates = [el for el in screen.actionable
                      if DISMISS_LABELS.match(el.best_text or "")]
        return candidates[0] if len(candidates) == 1 else None

    for el in screen.actionable:
        if el.package == target_package:
            continue
        if DISMISS_LABELS.match(el.best_text or ""):
            return el
    return None




# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

REPEAT_HINT_AT = 6
FORCE_BACK_AT = 9
WINDOW = 20

_SCROLL_OPPOSITES = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class LoopDetector:
    """Ring buffer of recent (screen, action) pairs, plus a per-screen ban list.

    Roughly four fifths of failures in the published baselines are navigation
    loops eating the step budget, so this is not an optional nicety.

    Screens are identified here by ``skeleton_id`` -- the content-free hash --
    and not by ``exact_id``. That was the bug this class existed to catch and
    could not. ``exact_id`` hashes every content element *including its bounds*,
    so a pixel of layout jitter, one new list row or an animation frame makes it
    a different screen. In ``runs/2521862d7a23`` the agent ran a textbook
    two-cycle -- ``tap #7``, ``back``, ``tap #7``, ``back`` -- for twenty steps
    between two screens whose ``skeleton_id``s were rock stable, and not one
    detector here fired, because every visit minted a fresh ``exact_id``. Every
    step was graded ``success`` too, so nothing else noticed either; the run was
    ended by the person watching it.

    The other half of that fix is *what* is compared. Both detectors below now
    key on the (screen, action) pair rather than on the screen alone, because
    the screen alone cannot tell "tapping #7 ten times" from "tapping #1 through
    #10" -- and on a stable id the latter is what walking a grid legitimately
    looks like.
    """

    history: List[Tuple[str, str]] = field(default_factory=list)
    banned: Dict[str, Set[str]] = field(default_factory=dict)
    #: Every (screen, action) pair the run has taken, and how often. Unlike
    #: `history` this is not a ring buffer: "have I done this here before" has to
    #: stay answerable after twenty other steps, which is exactly the span a
    #: slow two-cycle outlives.
    attempts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    #: Element-level history per screen identity (skeleton_id): list of (step, signature, action_desc)
    element_actions: Dict[str, List[Tuple[int, str, str]]] = field(default_factory=dict)
    #: Every scroll direction across the entire run, not cleared by taps.
    #: Used to detect direction reversals even when taps break the
    #: consecutive-scroll counter.
    scroll_dir_log: List[str] = field(default_factory=list)
    #: Positions in `scroll_dir_log` whose gesture had already run out of
    #: content when it was taken. Reversing away from one of those is the only
    #: move left, not a change of mind -- see `direction_reversals`.
    scroll_exhausted: List[int] = field(default_factory=list)
    total_scroll_count: int = 0
    #: Scroll gestures that revealed nothing, "{skeleton_id}/{direction}" ->
    #: the ``exact_id`` of the frame they failed on. Valued by ``exact_id``
    #: rather than flagged forever because end-of-list is a property of the
    #: content, not of the layout: a screen whose content has since changed --
    #: new items loaded, a refresh, a navigation away and back -- re-arms the
    #: gesture with no bookkeeping here.
    dead_scrolls: Dict[str, str] = field(default_factory=dict)
    #: Consecutive forced-back presses without an intervening real action.
    #: When this reaches the cap the agent stops pressing back and lets the
    #: LLM try a different approach.
    consecutive_backs: int = 0
    _BACK_LOOP_CAP: int = field(default=2, repr=False)

    def record(self, screen_id: str, signature: str) -> None:
        self.history.append((screen_id, signature))
        del self.history[:-WINDOW]
        key = (screen_id, signature)
        self.attempts[key] = self.attempts.get(key, 0) + 1

    def times_on(self, screen_id: str, signature: str) -> int:
        """How often this exact action has been taken on this screen, ever.

        Reads `attempts` rather than `history` on purpose: the ring buffer holds
        twenty entries, and the question "have I already tried this here" is
        asked precisely when the answer is further back than that.
        """
        return self.attempts.get((screen_id, signature), 0)

    def tried_on(self, screen_id: str) -> List[Tuple[str, int]]:
        """Distinct actions tried on this screen, most-repeated first."""
        rows = [(sig, n) for (sid, sig), n in self.attempts.items()
                if sid == screen_id and sig != "forced-back"]
        return sorted(rows, key=lambda row: (-row[1], row[0]))

    def record_element_action(self, skeleton_id: str, step: int, signature: str, action_desc: str) -> None:
        """Record an action performed on an element for a specific screen identity."""
        actions = self.element_actions.setdefault(skeleton_id, [])
        if not any(s == step for s, _, _ in actions):
            actions.append((step, signature, action_desc))
            del actions[:-10]

    def element_history_hint(self, skeleton_id: str,
                             repeatable: str = "") -> Optional[str]:
        """Contextual hint summarizing previous element actions performed on this screen identity.

        ``repeatable`` names an element that is *meant* to be acted on every
        turn -- the pager of a carousel. Without it this hint tells the agent not
        to reuse the one element that advances a gallery, which pushes it into
        flinging arbitrary indices instead.

        The exemption is granted only when the repetition it excuses is actually
        the pager gesture. In ``runs/2521862d7a23`` it was not: the last five
        actions were five ``press_key back`` presses, and this hint recited them
        and then said "repeating it is correct. Do not substitute a different
        index for it." The agent was in a hard two-cycle at the time, and the one
        piece of context that should have broken it instead licensed it for
        another 30 steps. An exemption for the pager must not cover an action
        that never touches the pager.
        """
        actions = self.element_actions.get(skeleton_id, [])
        if not actions:
            return None
        recent = actions[-5:]
        formatted = [f"step {step}: {desc}" for step, _, desc in recent]
        summary = "; ".join(formatted)
        # Split rather than substring-match: `signature` joins its parts with
        # "/", and a prefix of one key is not another key. `repeatable` is an
        # `Element.key`, matching the content-keyed form `signature` now emits --
        # it used to be an ordinal, which was both what the signature carried and
        # a thing that moves for the same control 47% of the time in ``runs/``.
        target = f"k={repeatable}"
        on_pager = any(target in sig.split("/") for _, sig, _ in recent)
        if repeatable and on_pager:
            return (
                f"PREVIOUS ACTIONS ON THIS SCREEN: {summary}. "
                f"The element k={repeatable} is the pager for this screen: "
                f"swiping it again is how you reach the next item, so repeating "
                f"it is correct. Do not substitute a different element for it."
            )
        return (
            f"PREVIOUS ACTIONS ON THIS SCREEN: {summary}. "
            f"If you are reviewing multiple items or photos in a list or grid, "
            f"do NOT act again on an element you have already opened -- match on "
            f"the k= value, not the #N, which is renumbered every turn. "
            f"Choose the next uninspected element or take a different action."
        )

    def repeats(self, screen_id: str) -> int:
        return sum(1 for seen, _ in self.history if seen == screen_id)

    def hint(self, screen_id: str) -> Optional[str]:
        n = self.repeats(screen_id)
        if n >= REPEAT_HINT_AT:
            return (f"You have now seen this exact screen {n} times. Whatever you "
                    f"have been trying is not working -- do something different, "
                    f"or go back.")
        return None

    @staticmethod
    def _counts_as_repetition(signature: str) -> bool:
        """Whether repeating this signature on one screen means anything.

        Repeated scrolls do not: a scroll stall means end-of-list, not that the
        agent is bouncing between screens. Horizontal swipes do not either --
        paging a carousel is browsing, and the remedy here (a back press) ejects
        the agent from the set it was working through. "forced-back" does not,
        because it is this class's own intervention, and counting it created a
        positive-feedback loop where each forced back made the next fire sooner.
        """
        if signature.startswith("scroll/"):
            return False
        if (signature.startswith("swipe/")
                and signature.rsplit("/", 1)[-1] in ("left", "right")):
            return False
        return signature != "forced-back"

    def should_force_back(self, screen_id: str) -> bool:
        """True when one action has been repeated on one screen to no effect.

        Counts *per signature* rather than totalling every action taken here.
        The total cannot tell a stuck agent from a working one: opening ten grid
        items in turn puts ten taps on the grid's ``skeleton_id``, and totalling
        them would press back in the middle of legitimate work. Repeating the
        *same* tap nine times is unambiguous.
        """
        counts: Dict[str, int] = {}
        for sid, sig in self.history:
            if sid != screen_id or not self._counts_as_repetition(sig):
                continue
            counts[sig] = counts.get(sig, 0) + 1
        return max(counts.values(), default=0) >= FORCE_BACK_AT

    def in_back_loop(self) -> bool:
        """True when pressing back is itself the problem.

        After ``_BACK_LOOP_CAP`` consecutive forced-backs, the caller
        should stop pressing back and let the LLM try something else.
        """
        return self.consecutive_backs >= self._BACK_LOOP_CAP

    def oscillating(self) -> bool:
        """A repeating 2- or 3-step cycle that has persisted for 5 full
        repetitions, e.g. [A,B,A,B,A,B,A,B,A,B].

        Compares whole (screen, action) pairs, not screen ids alone. Screen ids
        alone cannot separate the two things that look identical from the
        outside: ``tap #7 -> back -> tap #7 -> back`` bounces between two
        screens getting nowhere, while ``tap #1 -> back -> tap #2 -> back``
        bounces between the same two screens working through a list. The action
        is the only part that differs, so the action has to be in the key.

        Ignores forced-back entries, because those are a symptom of *this* guard
        firing repeatedly rather than of the agent misbehaving.
        """
        MIN_REPS = 5  # cycle must repeat this many times
        real = [(h, s) for h, s in self.history if s != "forced-back"]
        for period in (2, 3):
            needed = period * MIN_REPS
            if len(real) >= needed:
                tail = real[-needed:]
                cycle = tail[:period]
                # A cycle has to have somewhere to go and come back from. Doing
                # one thing over and over -- ``[A,A,A,A,...]`` -- satisfies the
                # period-2 and period-3 patterns trivially, and used to be
                # reported here as oscillation: paging an album is exactly that
                # shape, and the remedy this triggers is a back press that ejects
                # the agent from the album it was halfway through. Repetition of
                # a single action is `should_force_back`'s question, and that one
                # knows which gestures are browsing rather than thrashing.
                if len(set(cycle)) < 2:
                    continue
                if all(tail[i * period:(i + 1) * period] == cycle
                       for i in range(1, MIN_REPS)):
                    return True
        return False

    def _consecutive_scroll_dirs(self) -> List[str]:
        """Trailing run of scroll directions from the history buffer."""
        dirs: List[str] = []
        for _, sig in reversed(self.history):
            parts = sig.split("/")
            if parts[0] not in ("scroll", "swipe"):
                break
            dirs.append(parts[-1])  # direction is always the last segment
        dirs.reverse()
        return dirs

    def scroll_oscillating(self, *, threshold: int = 6) -> bool:
        """Alternating opposite scrolls, e.g. down/up/down/up.

        Regular ``oscillating()`` compares screen ``exact_id`` sequences, but
        scrolling changes visible content (and therefore ``exact_id``) on every
        step, so it never triggers.  This method looks at the trailing run of
        scroll action signatures instead.

        *threshold* controls how many alternating scrolls are required.  The
        default of 4 can be raised (e.g. to 6 in chat mode) to tolerate natural
        chat-browsing patterns that scroll up to read history and back down.
        """
        dirs = self._consecutive_scroll_dirs()
        if len(dirs) < threshold:
            return False
        tail = dirs[-threshold:]
        return all(
            tail[i] == _SCROLL_OPPOSITES.get(tail[i - 1], "")
            for i in range(1, len(tail))
        )

    def record_scroll(self, direction: str) -> None:
        """Track a scroll direction across the entire run.

        Unlike ``_consecutive_scroll_dirs()`` which breaks on any non-scroll
        action, this persists through taps and other actions so we can detect
        direction reversals even when taps (like "Go to most recent message")
        are interleaved.
        """
        self.scroll_dir_log.append(direction)
        self.total_scroll_count += 1

    def mark_scroll_exhausted(self) -> None:
        """Record that the last scroll logged had nothing left to reveal.

        Called when the harness learns the gesture stopped advancing -- either
        the action came back ``no_change``, or a mechanical sweep repeated it
        until the content stopped moving. `direction_reversals` reads this to
        tell "it changed its mind" from "it reached the end".
        """
        if not self.scroll_dir_log:
            return
        last = len(self.scroll_dir_log) - 1
        if last not in self.scroll_exhausted:
            self.scroll_exhausted.append(last)

    def _axis_pairs(self, axis: str = "") -> List[Tuple[str, bool]]:
        """The scroll log as (direction, was exhausted) pairs, narrowed to axis."""
        if axis == "horizontal":
            wanted = ("left", "right")
        elif axis == "vertical":
            wanted = ("up", "down")
        else:
            wanted = ()
        done = set(self.scroll_exhausted)
        return [(d, i in done)
                for i, d in enumerate(self.scroll_dir_log)
                if not wanted or d in wanted]

    def axis_log(self, axis: str = "") -> List[str]:
        """The scroll log, optionally narrowed to one axis.

        The two axes are different activities -- walking a chat history versus
        paging a carousel -- and mixing them produces nonsense: eight vertical
        reversals while browsing a gallery gets reported as "you have reversed
        direction 8 times" next to a history reading "left → left → left".
        """
        if axis == "horizontal":
            wanted = ("left", "right")
        elif axis == "vertical":
            wanted = ("up", "down")
        else:
            return list(self.scroll_dir_log)
        return [d for d in self.scroll_dir_log if d in wanted]

    def direction_reversals(self, axis: str = "") -> int:
        """Count how many times the scroll direction has flipped *pointlessly*.

        A reversal is any transition from one vertical direction to its
        opposite (up→down or down→up), or horizontal (left→right, right→left).
        Consecutive scrolls in the same direction count as one "run".
        Pass *axis* to count only reversals on that axis.

        A reversal away from a gesture that had already run out of content does
        not count. There is nothing left that way, so the opposite direction is
        the only move on that axis -- and the harness says as much itself, both
        in `last_failure` ("Try the opposite direction") and in the sweep, which
        is "browsing, not thrashing" and already exempt from the loop detector.

        Counting them was not free. In ``runs/a7ef4e0e45e9`` the policy's own
        loop is one scroll to the bottom of a profile and one back to the top,
        three times, and every one of those six gestures had been repeated until
        the content stopped moving. That scored 5 reversals, which put
        `scroll_direction_hint` into the prompt on 10 of 23 decide calls telling
        the model "You MUST stop reversing now" about the up-scroll its policy
        made mandatory -- and left it one reversal short of
        `agent`'s >=5 threshold for refusing the gesture outright.
        """
        pairs = self._axis_pairs(axis)
        if len(pairs) < 2:
            return 0
        reversals = 0
        prev, prev_spent = pairs[0]
        for direction, spent in pairs[1:]:
            if direction != prev:
                if (direction == _SCROLL_OPPOSITES.get(prev, "")
                        and not prev_spent):
                    reversals += 1
                prev = direction
            prev_spent = spent
        return reversals

    def scroll_direction_hint(self, axis: str = "") -> Optional[str]:
        """Warning when the agent keeps reversing scroll direction.

        Returns ``None`` when there are fewer than 3 reversals.  Otherwise
        returns a strongly-worded hint that tells the model to commit to one
        direction and stop undoing its progress.
        """
        reversals = self.direction_reversals(axis)
        if reversals < 3:
            return None

        recent = " → ".join(self.axis_log(axis)[-8:])
        parts = [
            f"WARNING: You have reversed your scroll direction {reversals} "
            f"times during this task. Your scroll history: {recent}.",
            "Each time you scroll in one direction and then reverse (or tap "
            "a button like 'Go to most recent message'), you UNDO all your "
            "scrolling progress and have to start over.",
        ]
        if axis == "horizontal":
            # Naming a vertical remedy on a carousel is worse than saying
            # nothing: there is no "older content" along this axis.
            parts.append(
                "Commit to one direction: keep swiping LEFT to move forward "
                "through the set until you reach the end, then stop. Do not "
                "alternate left and right — that re-shows items you have "
                "already looked at."
            )
        elif reversals >= 5:
            parts.append(
                "You MUST stop reversing now. Commit to one direction: if you "
                "are looking for OLDER content, keep scrolling UP consistently. "
                "If you are looking for RECENT content, keep scrolling DOWN. "
                "Do NOT tap any 'jump to bottom' or 'go to recent' buttons "
                "while searching upward. If you cannot find what you need "
                "after scrolling consistently in one direction, report done "
                "with what you have or report fail."
            )
        else:
            parts.append(
                "Commit to one direction. If searching for older content, "
                "scroll UP consistently. Do NOT tap 'Go to most recent "
                "message' or similar buttons — this undoes your progress."
            )
        return " ".join(parts)

    def scroll_context(self, axis: str = "") -> Optional[str]:
        """Rich situational context about scrolling patterns for the LLM.

        Returns ``None`` when there is nothing noteworthy.  Otherwise returns
        a multi-sentence description that tells the model *exactly* what it
        has been doing so it can course-correct on its own.  Pass *axis* to
        narrow the report to the axis actually in play on this screen.
        """
        # Start with direction reversal context (survives interleaved taps).
        reversal_hint = self.scroll_direction_hint(axis)

        dirs = self._consecutive_scroll_dirs()
        if axis == "horizontal":
            dirs = [d for d in dirs if d in ("left", "right")]
        elif axis == "vertical":
            dirs = [d for d in dirs if d in ("up", "down")]
        if len(dirs) < 6 and reversal_hint is None:
            return None

        parts: List[str] = []

        if len(dirs) >= 6:
            n = len(dirs)
            counts: Dict[str, int] = {}
            for d in dirs:
                counts[d] = counts.get(d, 0) + 1
            breakdown = ", ".join(f"{c}x {d}" for d, c in counts.items())

            # Determine axis for user-facing messages.
            h_dirs = counts.get("left", 0) + counts.get("right", 0)
            v_dirs = counts.get("up", 0) + counts.get("down", 0)
            if h_dirs > v_dirs:
                axis_label = "horizontally"
            elif v_dirs > h_dirs:
                axis_label = "vertically"
            else:
                axis_label = "in multiple directions"

            parts.append(
                f"You have scrolled {n} times consecutively {axis_label} ({breakdown}).")

            if self.scroll_oscillating():
                recent = " → ".join(dirs[-6:])
                parts.append(
                    f"Your recent scroll pattern is: {recent}. "
                    f"You are alternating between opposite directions, which "
                    f"means you are re-reading content you already saw."
                )
                parts.append(
                    "You must stop scrolling now. Either you have already seen "
                    "all the content and should report 'done' with a summary of "
                    "what you found, or the information is not here and you "
                    "should try a different approach (e.g. search, go back) or "
                    "report 'fail'."
                )
            elif n >= 12:
                parts.append(
                    "You have been scrolling for a while without finding what "
                    "you need. If you are confident the content is further in "
                    "this direction, keep going. Otherwise consider whether "
                    "you have covered enough."
                )

        # Append reversal hint (fires even when consecutive count is low).
        if reversal_hint:
            parts.append(reversal_hint)

        return " ".join(parts) if parts else None

    def ban(self, skeleton_id: str, signature: str) -> None:
        log.info("banning %s on this screen for the rest of the run", signature)
        self.banned.setdefault(skeleton_id, set()).add(signature)

    def bans_for(self, skeleton_id: str) -> Set[str]:
        return self.banned.get(skeleton_id, set())

    def mark_scroll_dead(self, skeleton_id: str, direction: str,
                         exact_id: str) -> None:
        """Remember that this gesture revealed no new content on this frame."""
        log.info("scroll %s revealed nothing on this screen; refusing it "
                 "while the screen stays identical", direction)
        self.dead_scrolls[f"{skeleton_id}/{direction}"] = exact_id

    def scroll_dead(self, skeleton_id: str, direction: str,
                    exact_id: str) -> bool:
        """True when this gesture already revealed nothing on this exact frame.

        Compared against the ``exact_id`` recorded at the failure, so a screen
        whose content has changed since -- new items loaded, a pull-to-refresh,
        a navigation away and back -- re-arms the gesture on its own.
        """
        return self.dead_scrolls.get(f"{skeleton_id}/{direction}") == exact_id


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

class Aborted(RuntimeError):
    """The user declined, or a confirmation was needed and none was possible."""


def confirm(prompt: str, cfg: Config) -> bool:
    """Ask the person. Unattended runs refuse rather than guess."""
    if cfg.safety.allow_destructive:
        log.warning("proceeding without confirmation (--allow-destructive): %s",
                    prompt)
        return True
    if cfg.safety.unattended:
        log.error("refusing in unattended mode: %s", prompt)
        return False
    try:
        answer = input(f"\n  {prompt}\n  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")
