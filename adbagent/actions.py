"""The action space, its execution, and the verification DSL.

This module is the contract in three directions at once: it is the JSON schema
the model must answer with, the thing the device layer executes, and the record
the cache stores and replays. Keeping all three in one place is what stops them
drifting apart.

Two deliberate schema choices:

* A flat ``Literal`` enum rather than a discriminated union. ``Field(
  discriminator=...)`` emits ``oneOf``, which OpenAI's strict mode rejects and
  which weaker open-weight models adhere to poorly.
* ``observation`` and ``reasoning`` come first. Constrained decoders emit
  properties in schema order, and Fireworks' ``response_format`` suppresses
  reasoning output on reasoning models -- putting them first recovers a short
  chain of thought inside the structured answer itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .screen import Element, Screen
from .fingerprint import DESTRUCTIVE_TEXT

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from .device import Device

log = logging.getLogger("adbagent.actions")

ActionName = Literal[
    "tap", "long_press", "input_text", "press_key", "scroll",
    "open_app", "wait", "ask_user", "done", "fail",
]

#: Only names the on-device server actually accepts.
KeyName = Literal["back", "home", "enter", "recent", "delete", "search", "menu",
                  "center", "up", "down", "left", "right",
                  "volume_up", "volume_down"]

ScrollDir = Literal["down", "up", "left", "right"]

PostKind = Literal["screen_changed", "element_state", "text_present",
                   "app_is", "noop_ok"]

TERMINAL_ACTIONS = frozenset({"done", "fail", "ask_user"})
#: Actions whose whole purpose is to move to a different screen.
NAVIGATIONAL = frozenset({"tap", "long_press", "press_key", "open_app"})


def is_navigation_action(action: AgentAction, element: Optional[Element] = None) -> bool:
    """Return True if the action represents pure navigation (no state mutation)."""
    if element is not None and element.checkable:
        return False
        
    texts_to_check = []
    if action.text:
        texts_to_check.append(action.text)
    if action.target and action.target.text:
        texts_to_check.append(action.target.text)
    if element and element.best_text:
        texts_to_check.append(element.best_text)
        
    for text in texts_to_check:
        if DESTRUCTIVE_TEXT.search(text):
            return False
            
    if action.action == "tap":
        return True
    if action.action == "press_key" and action.key in ("back", "home"):
        return True
    if action.action == "open_app":
        return True
        
    return False


class Target(BaseModel):
    """Which element to act on. ``index`` is the unambiguous form."""

    model_config = ConfigDict(extra="forbid")

    index: Optional[int] = Field(
        None, description="The #N of the element from the list. Prefer this.")
    resource_id: Optional[str] = Field(
        None, description="Short resource-id, e.g. 'switch_widget'.")
    text: Optional[str] = Field(
        None, description="Exact visible text. Only when no #N fits.")

    @model_validator(mode="after")
    def _needs_something(self) -> "Target":
        if self.index is None and not (self.resource_id or self.text):
            raise ValueError("target needs index, resource_id or text")
        return self

    def describe(self) -> str:
        if self.index is not None:
            return f"#{self.index}"
        return self.resource_id or f"{self.text!r}"


class Postcondition(BaseModel):
    """What must be true after the action for it to count as having worked."""

    model_config = ConfigDict(extra="forbid")

    kind: PostKind = "screen_changed"
    resource_id: Optional[str] = None
    field: Optional[Literal["checked", "selected", "text"]] = None
    value: Optional[str] = None
    text: Optional[str] = None
    package: Optional[str] = None


class AgentAction(BaseModel):
    """One step. The model replies with exactly this object and nothing else."""

    model_config = ConfigDict(extra="forbid")

    observation: str = Field(description="One sentence: what screen is this?")
    reasoning: str = Field(
        description="One sentence: why this action advances the goal.")
    action: ActionName
    target: Optional[Target] = Field(
        None, description="For tap, long_press, input_text and element scrolls.")
    text: Optional[str] = Field(
        None,
        description="input_text: what to type. open_app: the package name. "
                    "ask_user: the question. done/fail: a one-line summary.")
    key: Optional[KeyName] = Field(None, description="For press_key.")
    direction: Optional[ScrollDir] = Field(
        None, description="For scroll: which way the content should move.")
    scroll_amount: float = Field(
        1, description="For scroll: how many screenfuls to scroll (0.5 for half-page, "
                       "1 for one page, up to 5). Use higher values to move quickly "
                       "through long content like chat history.",
        ge=0.25, le=5)
    confidence: Literal["high", "low"] = Field(
        "high", description="Use 'low' when unsure; you will be shown a screenshot.")
    notes: Optional[str] = Field(
        None,
        description="Data collection scratchpad. See DATA COLLECTION above.")
    progress: Optional[str] = Field(
        None,
        description="Multi-step progress tracker. See PROGRESS TRACKING above.")

    @model_validator(mode="after")
    def _check_arguments(self) -> "AgentAction":
        need_target = {"tap", "long_press", "input_text"}
        if self.action in need_target and self.target is None:
            raise ValueError(f"{self.action} requires a target")
        if self.action == "input_text" and self.text is None:
            raise ValueError("input_text requires text")
        if self.action == "press_key" and self.key is None:
            raise ValueError("press_key requires key")
        if self.action == "scroll" and self.direction is None:
            raise ValueError("scroll requires direction")
        if self.action == "open_app" and not self.text:
            raise ValueError("open_app requires the package name in text")
        if self.action == "ask_user" and not self.text:
            raise ValueError("ask_user requires the question in text")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.action in TERMINAL_ACTIONS

    def signature(self) -> str:
        """Stable identity for loop detection and ban lists."""
        parts = [self.action]
        if self.target is not None:
            parts.append(self.target.describe())
        for extra in (self.key, self.direction):
            if extra:
                parts.append(str(extra))
        return "/".join(parts)

    def describe(self, element: Optional[Element] = None) -> str:
        bits = [self.action]
        if self.target is not None:
            bits.append(describe_target(self.target, element))
        if self.action == "input_text" and self.text is not None:
            bits.append(f"{self.text!r}")
        elif self.action in ("open_app", "done", "fail", "ask_user") and self.text:
            bits.append(self.text)
        if self.key:
            bits.append(self.key)
        if self.direction:
            bits.append(self.direction)
            if self.scroll_amount != 1.0:
                bits.append(f"amount={self.scroll_amount}")
        return " ".join(bits)


def describe_target(target: Target, element: Optional[Element] = None) -> str:
    base = target.describe()
    if element is not None:
        details = [element.kind()]
        if element.best_text:
            text_str = " ".join(element.best_text.split())
            if len(text_str) > 30:
                text_str = text_str[:27] + "..."
            details.append(f'"{text_str}"')
        if element.resource_id:
            details.append(f"id={element.resource_id}")
        if element.checkable:
            details.append(f"checked={'true' if element.checked else 'false'}")
        if element.selected:
            details.append("selected")
        return f"{base} [{' '.join(details)}]"
    return base


def format_history_entry(step: int, action: AgentAction,
                         screen: Optional[Screen] = None,
                         element: Optional[Element] = None,
                         grade: Optional[str] = None,
                         reason: str = "",
                         prefix: str = "") -> str:
    """Format an action history entry with rich target, screen, observation, and outcome context."""
    if element is None and screen is not None and action.target is not None:
        element = resolve_target(action.target, screen)

    parts = [f"{step}."]
    if prefix:
        parts.append(prefix)

    parts.append(action.describe(element=element))

    pkg = screen.package if screen and screen.package else ""
    if pkg:
        parts.append(f"in {pkg}")

    obs = (action.observation or "").strip()
    if obs:
        obs_clean = " ".join(obs.split())
        if len(obs_clean) > 60:
            obs_clean = obs_clean[:57] + "..."
        parts.append(f"(Obs: {obs_clean})")

    if grade:
        parts.append(f"-> {grade}")
        if reason:
            parts.append(f"({reason})")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def resolve_target(target: Target, screen: Screen) -> Optional[Element]:
    """Find the element a target refers to on this screen."""
    if target.index is not None:
        el = screen.by_index(target.index)
        if el is not None:
            return el
    if target.resource_id:
        matches = [e for e in screen.elements if e.resource_id == target.resource_id]
        if len(matches) == 1:
            return matches[0]
        if matches and target.text:
            for e in matches:
                if e.best_text.strip().lower() == target.text.strip().lower():
                    return e
        if matches:
            return matches[0]
    if target.text:
        wanted = target.text.strip().lower()
        exact = [e for e in screen.elements if e.best_text.strip().lower() == wanted]
        if exact:
            return exact[0]
        # Fuzzy, never `==`: label text drifts between app versions and locales.
        loose = [e for e in screen.elements if wanted in e.best_text.strip().lower()]
        if loose:
            return min(loose, key=lambda e: len(e.best_text))
    return None


class ActionError(RuntimeError):
    """The action could not be carried out on this screen."""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute(dev: "Device", action: AgentAction, screen: Screen) -> Optional[Element]:
    """Carry out `action`. Returns the element acted on, when there was one."""
    element: Optional[Element] = None
    if action.target is not None:
        element = resolve_target(action.target, screen)
        if element is None and action.action in ("tap", "long_press", "input_text"):
            raise ActionError(f"no element matches {action.target.describe()}")

    if action.action == "tap":
        assert element is not None
        dev.tap(*element.center)
    elif action.action == "long_press":
        assert element is not None
        dev.long_press(*element.center)
    elif action.action == "input_text":
        assert element is not None
        # Focus the field first; the IME broadcast path types into whatever has
        # focus, not into a selector.
        dev.tap(*element.center)
        dev.input_text(action.text or "", clear=True)
    elif action.action == "press_key":
        dev.press(action.key or "back")
    elif action.action == "scroll":
        box = None
        if element is not None and element.scrollable:
            box = element.bounds
            # Remap wrong-axis directions so the LLM doesn't waste steps.
            if element.is_horizontal and action.direction in ("up", "down"):
                remapped = "left" if action.direction == "up" else "right"
                log.warning("remapping scroll %s -> %s for horizontal scroller",
                            action.direction, remapped)
                action = action.model_copy(update={"direction": remapped})
            elif not element.is_horizontal and action.direction in ("left", "right"):
                remapped = "up" if action.direction == "left" else "down"
                log.warning("remapping scroll %s -> %s for vertical scroller",
                            action.direction, remapped)
                action = action.model_copy(update={"direction": remapped})
        amount = max(0.25, min(action.scroll_amount, 5.0))
        base_scale = 0.6
        full_scrolls = int(amount)
        remainder = amount - full_scrolls
        import time
        for i in range(full_scrolls):
            dev.scroll(action.direction or "down", scale=base_scale, box=box)
            if i < full_scrolls - 1 or remainder > 0:
                time.sleep(0.15)
        if remainder >= 0.1:
            dev.scroll(action.direction or "down",
                       scale=round(base_scale * remainder, 2), box=box)
    elif action.action == "open_app":
        dev.open_app((action.text or "").strip())
    elif action.action == "wait":
        import time
        time.sleep(1.0)
    else:
        raise ActionError(f"{action.action} is terminal and is not executed here")
    return element


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

Grade = Literal["success", "soft_fail", "hard_fail", "no_change"]


class VerifyOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: Grade
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.grade in ("success", "soft_fail")


def synthesise_postcondition(action: AgentAction,
                             element: Optional[Element]) -> Postcondition:
    """Derive what "it worked" means for this action.

    A universal "did the screen change" check is wrong for the two most common
    silent failures. Flipping a toggle changes one attribute in place, and typing
    into a field must NOT navigate anywhere -- both would be scored as failures
    by a naive screen-changed test.
    """
    if action.action == "input_text":
        return Postcondition(kind="element_state",
                             resource_id=element.resource_id if element else None,
                             field="text", value=action.text or "")
    if action.action in ("tap", "long_press") and element is not None and element.checkable:
        return Postcondition(kind="element_state", resource_id=element.resource_id,
                             field="checked",
                             value="false" if element.checked else "true")
    if action.action == "open_app":
        return Postcondition(kind="app_is", package=(action.text or "").strip())
    if action.action == "wait":
        return Postcondition(kind="noop_ok")
    return Postcondition(kind="screen_changed")


def _find(screen: Screen, resource_id: Optional[str]) -> Optional[Element]:
    if not resource_id:
        return None
    for el in screen.elements:
        if el.resource_id == resource_id:
            return el
    return None


def check_postcondition(post: Postcondition, before: Screen,
                        after: Screen) -> Tuple[bool, str]:
    if post.kind == "noop_ok":
        return True, ""

    if post.kind == "screen_changed":
        if after.exact_id != before.exact_id:
            return True, ""
        return False, "the screen did not change"

    if post.kind == "app_is":
        wanted = (post.package or "").strip()
        if not wanted:
            return True, ""
        if after.package == wanted:
            return True, ""
        return False, f"foreground app is {after.package or '?'}, wanted {wanted}"

    if post.kind == "text_present":
        wanted = (post.text or "").strip().lower()
        if not wanted:
            return True, ""
        for el in after.elements:
            if wanted in el.best_text.strip().lower():
                return True, ""
        return False, f"{post.text!r} is not on screen"

    if post.kind == "element_state":
        el = _find(after, post.resource_id)
        if el is None:
            # The element vanishing is usually navigation, not failure; the
            # successor check decides. Treat it as inconclusive-but-passing so a
            # legitimate screen transition is not scored as a hard failure.
            return True, ""
        if post.field == "checked":
            actual = "true" if el.checked else "false"
        elif post.field == "selected":
            actual = "true" if el.selected else "false"
        else:
            actual = el.best_text
        if post.field == "text":
            if (post.value or "") in actual:
                return True, ""
            return False, f"field reads {actual!r}, expected {post.value!r}"
        if actual == (post.value or ""):
            return True, ""
        return False, f"{post.field} is {actual}, expected {post.value}"

    return True, ""


# ---------------------------------------------------------------------------
# Multi-signal scroll-changed detection
# ---------------------------------------------------------------------------

def _scroller_texts(screen: Screen) -> frozenset:
    """Collect text from elements inside scrollable containers."""
    return frozenset(
        el.best_text.strip()
        for el in screen.elements
        if el.scroller() is not None and el.best_text.strip()
    )


def _scroll_changed(before: Screen, after: Screen) -> bool:
    """Multi-signal check for whether a scroll actually revealed new content.

    Three signals, cheapest first:

    1. **exact_id identity** -- the hierarchy hash is byte-identical, so nothing
       changed at all.
    2. **skeleton + simhash proximity + scroller content** -- the exact_id *did*
       change (e.g. a toggle flipped, an animation frame) but the skeleton is
       identical, the simhash moved by at most 2 bits, **and** the text inside
       scrollable containers is unchanged.  The screen is *effectively* the same.
    3. **Scroller-child text overlap** -- ≥ 90 % of the text-bearing elements
       inside scrollable containers are identical in both dumps.  This catches
       the case where the hierarchy drifted more but the user is seeing the
       same list content.
    """
    # Signal 1: byte-identical hierarchy.
    if after.exact_id == before.exact_id:
        return False

    before_texts = _scroller_texts(before)
    after_texts = _scroller_texts(after)

    # Signal 2: near-identical chrome change + scroller content unchanged.
    if after.skeleton_id == before.skeleton_id:
        from .fingerprint import hamming
        dist = hamming(after.simhash, before.simhash)
        if dist <= 2:
            # Chrome barely changed.  Did the scroller content actually move?
            if not before_texts and not after_texts:
                return False  # No scroller content; chrome-only change.
            if before_texts == after_texts:
                return False  # Scroller content is identical.

    # Signal 3: scroller content overlap.
    if before_texts and after_texts:
        overlap = before_texts & after_texts
        total = max(len(before_texts), len(after_texts))
        if total and len(overlap) / total >= 0.90:
            return False

    return True


def verify(action: AgentAction, before: Screen, after: Screen,
           post: Optional[Postcondition] = None,
           expected_skeleton: Optional[str] = None) -> VerifyOutcome:
    """Grade what actually happened."""
    if action.action == "wait":
        return VerifyOutcome(grade="success")

    condition = post or synthesise_postcondition(action, None)

    # Scroll that didn't move = end of list, not a hard failure.  Grading it
    # ``no_change`` causes the direction to be banned on this screen and gives
    # the LLM a clear "end of list" signal instead of a confusing hard_fail.
    if action.action == "scroll" and not _scroll_changed(before, after):
        return VerifyOutcome(grade="no_change",
                             reason="scrolling did not reveal new content")

    # Checked before the generic postcondition, because "nothing happened at
    # all" is both the most common silent failure and a more actionable
    # diagnosis than a bare condition failure -- it is what feeds the per-run
    # ban list, so the same dud tap is not retried forever.
    if action.action in NAVIGATIONAL and after.exact_id == before.exact_id:
        return VerifyOutcome(grade="no_change", reason="nothing on screen changed")

    passed, why = check_postcondition(condition, before, after)
    if not passed:
        return VerifyOutcome(grade="hard_fail", reason=why)

    if expected_skeleton and after.skeleton_id != expected_skeleton:
        # Screens legitimately fan out -- the same tap can land somewhere new
        # depending on state. Note it, do not punish it.
        return VerifyOutcome(grade="soft_fail",
                             reason="landed on an unexpected screen")

    return VerifyOutcome(grade="success")
