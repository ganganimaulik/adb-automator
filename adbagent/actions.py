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
    confidence: Literal["high", "low"] = Field(
        "high", description="Use 'low' when unsure; you will be shown a screenshot.")

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

    def describe(self) -> str:
        bits = [self.action]
        if self.target is not None:
            bits.append(self.target.describe())
        if self.action == "input_text" and self.text is not None:
            bits.append(f"{self.text!r}")
        elif self.action in ("open_app", "done", "fail", "ask_user") and self.text:
            bits.append(self.text[:60])
        if self.key:
            bits.append(self.key)
        if self.direction:
            bits.append(self.direction)
        return " ".join(bits)


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
        dev.scroll(action.direction or "down", box=box)
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


def verify(action: AgentAction, before: Screen, after: Screen,
           post: Optional[Postcondition] = None,
           expected_skeleton: Optional[str] = None) -> VerifyOutcome:
    """Grade what actually happened."""
    if action.action == "wait":
        return VerifyOutcome(grade="success")

    condition = post or synthesise_postcondition(action, None)

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
