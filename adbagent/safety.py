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

#: Buttons that only ever dismiss noise -- rating nags, feature tours, "what's
#: new" cards. Deliberately excludes "Allow", "Accept" and "Agree": granting a
#: permission or accepting terms is a decision, not noise, so it goes to the
#: model (and to the user, if it is irreversible) rather than being auto-tapped.
DISMISS_LABELS = re.compile(
    r"^\s*(?:no thanks|not now|maybe later|later|skip|skip for now|dismiss|"
    r"got it|ok, got it|close|no, thanks|remind me later|don't show again|"
    r"continue|next)\s*$",
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

REPEAT_HINT_AT = 3
FORCE_BACK_AT = 5
WINDOW = 8

_SCROLL_OPPOSITES = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class LoopDetector:
    """Ring buffer of recent screens, plus a per-screen ban list.

    Roughly four fifths of failures in the published baselines are navigation
    loops eating the step budget, so this is not an optional nicety.
    """

    history: List[Tuple[str, str]] = field(default_factory=list)
    banned: Dict[str, Set[str]] = field(default_factory=dict)
    #: Every scroll direction across the entire run, not cleared by taps.
    #: Used to detect direction reversals even when taps break the
    #: consecutive-scroll counter.
    scroll_dir_log: List[str] = field(default_factory=list)
    total_scroll_count: int = 0
    #: Consecutive forced-back presses without an intervening real action.
    #: When this reaches the cap the agent stops pressing back and lets the
    #: LLM try a different approach.
    consecutive_backs: int = 0
    _BACK_LOOP_CAP: int = field(default=2, repr=False)

    def record(self, exact_id: str, signature: str) -> None:
        self.history.append((exact_id, signature))
        del self.history[:-WINDOW]

    def repeats(self, exact_id: str) -> int:
        return sum(1 for seen, _ in self.history if seen == exact_id)

    def hint(self, exact_id: str) -> Optional[str]:
        n = self.repeats(exact_id)
        if n >= REPEAT_HINT_AT:
            return (f"You have now seen this exact screen {n} times. Whatever you "
                    f"have been trying is not working -- do something different, "
                    f"or go back.")
        return None

    def should_force_back(self, exact_id: str) -> bool:
        # Don't count repeated scrolls on the same screen as a navigation
        # loop.  Scroll stalls mean end-of-list, not that the agent is stuck
        # bouncing between screens.
        # Also exclude "forced-back" signatures: those are our own
        # interventions, not real agent actions.  Counting them created a
        # positive-feedback loop where each forced back made the next one
        # trigger sooner.
        non_scroll = sum(
            1 for eid, sig in self.history
            if eid == exact_id
            and not sig.startswith("scroll/")
            and sig != "forced-back"
        )
        return non_scroll >= FORCE_BACK_AT

    def in_back_loop(self) -> bool:
        """True when pressing back is itself the problem.

        After ``_BACK_LOOP_CAP`` consecutive forced-backs, the caller
        should stop pressing back and let the LLM try something else.
        """
        return self.consecutive_backs >= self._BACK_LOOP_CAP

    def oscillating(self) -> bool:
        """A repeating 2- or 3-step cycle, e.g. open -> back -> open -> back.

        Ignores patterns that consist entirely of forced-back entries,
        because those are a symptom of *this* guard firing repeatedly,
        not of the agent misbehaving.
        """
        # Filter out forced-back entries to check for real oscillation.
        real = [(h, s) for h, s in self.history if s != "forced-back"]
        ids = [h for h, _ in real]
        for period in (2, 3):
            if len(ids) >= period * 2:
                tail = ids[-period * 2:]
                if tail[:period] == tail[period:]:
                    return True
        return False

    def _consecutive_scroll_dirs(self) -> List[str]:
        """Trailing run of scroll directions from the history buffer."""
        dirs: List[str] = []
        for _, sig in reversed(self.history):
            parts = sig.split("/")
            if parts[0] != "scroll":
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

    def direction_reversals(self) -> int:
        """Count how many times the scroll direction has flipped.

        A reversal is any transition from one vertical direction to its
        opposite (up→down or down→up), or horizontal (left→right, right→left).
        Consecutive scrolls in the same direction count as one "run".
        """
        if len(self.scroll_dir_log) < 2:
            return 0
        reversals = 0
        prev = self.scroll_dir_log[0]
        for d in self.scroll_dir_log[1:]:
            if d != prev and d == _SCROLL_OPPOSITES.get(prev, ""):
                reversals += 1
            if d != prev:
                prev = d
        return reversals

    def scroll_direction_hint(self) -> Optional[str]:
        """Warning when the agent keeps reversing scroll direction.

        Returns ``None`` when there are fewer than 3 reversals.  Otherwise
        returns a strongly-worded hint that tells the model to commit to one
        direction and stop undoing its progress.
        """
        reversals = self.direction_reversals()
        if reversals < 3:
            return None

        recent = " → ".join(self.scroll_dir_log[-8:])
        parts = [
            f"WARNING: You have reversed your scroll direction {reversals} "
            f"times during this task. Your scroll history: {recent}.",
            "Each time you scroll in one direction and then reverse (or tap "
            "a button like 'Go to most recent message'), you UNDO all your "
            "scrolling progress and have to start over.",
        ]
        if reversals >= 5:
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

    def scroll_context(self) -> Optional[str]:
        """Rich situational context about scrolling patterns for the LLM.

        Returns ``None`` when there is nothing noteworthy.  Otherwise returns
        a multi-sentence description that tells the model *exactly* what it
        has been doing so it can course-correct on its own.
        """
        # Start with direction reversal context (survives interleaved taps).
        reversal_hint = self.scroll_direction_hint()

        dirs = self._consecutive_scroll_dirs()
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


# ---------------------------------------------------------------------------
# Explore mode
# ---------------------------------------------------------------------------

#: Explore may press these; anything else needs the user's blessing.
def is_read_only(action: AgentAction, screen: Screen) -> Tuple[bool, str]:
    """Would this action only navigate, or could it change something?

    Explore mode runs on a real, logged-in phone. Every published system that
    explored by tapping freely reported sending messages, deleting data or
    spending money by accident, so the default here is to refuse anything that
    is not plainly navigational.
    """
    if action.action in ("scroll", "wait", "done", "fail", "ask_user"):
        return True, ""
    if action.action == "press_key":
        if action.key in ("back", "home", "recent"):
            return True, ""
        return False, f"key {action.key} may commit something"
    if action.action == "input_text":
        return False, "typing can submit a form or send a message"
    if action.action == "open_app":
        return True, ""
    if action.action == "long_press":
        return False, "long press usually opens a destructive context menu"
    if action.action != "tap":
        return False, f"{action.action} is not a navigation action"

    if action.target is None:
        return False, "no target"
    element = resolve_target(action.target, screen)
    if element is None:
        return False, "target not on screen"

    label = element.best_text
    if label and DESTRUCTIVE_TEXT.search(label):
        return False, f"{label!r} names an irreversible action"
    if element.checkable:
        return False, "toggling a setting changes state"
    if element.editable:
        return False, "focusing a text field leads to typing"
    return True, ""


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
