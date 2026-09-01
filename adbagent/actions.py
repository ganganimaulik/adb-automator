"""The action space, its execution, and the verification DSL.

This module is the contract in two directions at once: it is the JSON schema
the model must answer with and the thing the device layer executes. Keeping
both in one place is what stops them drifting apart.

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
import re
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal,
                    Optional, Tuple)

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from .screen import Element, Screen

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from .device import Device

log = logging.getLogger("adbagent.actions")

ActionName = Literal[
    "tap", "tap_at", "long_press", "long_press_at", "double_tap", "drag",
    "input_text", "press_key", "scroll", "scroll_to_edge", "swipe",
    "open_app", "open_url", "restart_app", "list_apps",
    "get_clipboard", "set_clipboard", "wait", "sleep", "ask_user", "done", "fail",
]

#: Only names the on-device server actually accepts, plus the two panels
#: `Device.press` opens directly because they have no keycode.
#:
#: `power` is absent on purpose. The server accepts it, and a run that pressed
#: it would blank the screen it is driving -- on a phone with a PIN,
#: unrecoverably, since `Device.wake` can swipe a lock screen away but cannot
#: answer it. `camera` is absent for want of a use.
KeyName = Literal["back", "home", "enter", "recent", "delete", "search", "menu",
                  "center", "up", "down", "left", "right",
                  "volume_up", "volume_down", "volume_mute",
                  "notifications", "quick_settings"]

ScrollDir = Literal["down", "up", "left", "right"]

PostKind = Literal["screen_changed", "element_state", "text_present",
                   "app_is", "noop_ok"]

#: Where a sub-step of the goal has got to. Mirrors :data:`plan.STATUSES`; the
#: aliases a model reaches for instead ("completed", "in progress") are folded
#: into these by `plan.normalise_status` rather than being listed here, which
#: would put four spellings of "done" in front of a constrained decoder.
PlanStatus = Literal["pending", "active", "done", "blocked"]

TERMINAL_ACTIONS = frozenset({"done", "fail", "ask_user"})
#: Actions whose whole purpose is to move to a different screen.
NAVIGATIONAL = frozenset({"tap", "tap_at", "long_press", "long_press_at",
                          "double_tap", "press_key", "open_app", "open_url",
                          "restart_app"})
#: Actions that take a point rather than an element, and so are quantised into
#: their loop-detection signature the same way.
POINT_ACTIONS = frozenset({"tap_at", "long_press_at", "double_tap"})
#: Actions that name an element from the list and are meaningless without one.
#: A module constant rather than a set built inside the validator because
#: `_salvage_target` has to agree with `_check_arguments` about exactly which
#: actions it is rescuing, and two literals drift.
NEEDS_TARGET = frozenset({"tap", "long_press", "input_text"})
#: Actions aimed at one specific control, whether it is named from the list or
#: placed at a point. What they have in common is the failure mode the tap_at
#: and stuck hatches are an answer to: the run knows what it wants to press and
#: cannot press it. A scroll is not one of these -- it moves a surface, and its
#: failing says nothing about whether controls can be hit.
CONTROL_ACTIONS = NEEDS_TARGET | POINT_ACTIONS

#: How a model refers to a listed control in prose, in the three shapes seen in
#: ``runs/``: a bare ``#2``, ``#2 'Like photo'``, and the element list's own
#: ``#3 [Button "Darshana Oh 1 day ago. Reply?"]`` echoed back. The role and the
#: label are both optional, so the ordinal alone still matches. See
#: `AgentAction._salvage_target`.
_PROSE_TARGET = re.compile(
    r"#(\d+)"                                   # the ordinal
    r"(?:\s*\[[A-Za-z]+\]?)?"                   # an optional "[Button"
    r"(?:\s*[\"']([^\"'\n]{1,80})[\"'])?")      # an optional quoted label


class Target(BaseModel):
    """Which element to act on. ``index`` is the unambiguous form."""

    model_config = ConfigDict(extra="forbid")

    index: Optional[int] = Field(
        None, description="The #N of the element from the list. Prefer this.")
    key: Optional[str] = Field(
        None, description="The k=XXXX printed beside that element. Always send "
                          "it with the index: it identifies the element itself, "
                          "so the harness can tell if the list shifted.")
    resource_id: Optional[str] = Field(
        None, description="Short resource-id, e.g. 'switch_widget'.")
    text: Optional[str] = Field(
        None, description="Exact visible text. Only when no #N fits.")

    @model_validator(mode="after")
    def _needs_something(self) -> "Target":
        if self.index is None and not (self.resource_id or self.text or self.key):
            raise ValueError("target needs index, key, resource_id or text")
        return self

    def describe(self) -> str:
        if self.index is not None:
            return f"#{self.index}"
        if self.key:
            return f"k={self.key}"
        return self.resource_id or f"{self.text!r}"

    def identity(self) -> str:
        """The most durable name for this target, for a ban list or a ledger.

        Prefers the content key over the ordinal. `describe()` still leads with
        `#N` because that is the handle the model and the history speak in; this
        is what anything *remembering* the target should use, because an ordinal
        is a position in one dump and 47% of controls in ``runs/`` took more than
        one within a single run -- so a ban earned by `tap/#4` missed the same
        control at #1 and hit whatever else landed on #4.
        """
        if self.key:
            return f"k={self.key}"
        if self.resource_id:
            return self.resource_id
        if self.text:
            return repr(self.text)
        return f"#{self.index}"


class Postcondition(BaseModel):
    """What must be true after the action for it to count as having worked."""

    model_config = ConfigDict(extra="forbid")

    kind: PostKind = "screen_changed"
    resource_id: Optional[str] = None
    field: Optional[Literal["checked", "selected", "text"]] = None
    value: Optional[str] = None
    text: Optional[str] = None
    package: Optional[str] = None


class Note(BaseModel):
    """One collected fact.

    The model sends only the records that are new or corrected this turn; the
    harness maintains the union across turns (see :mod:`scratchpad`). Reusing a
    key corrects that record; a key never sent again keeps its value.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        description="Short stable identifier this fact hangs off: a timestamp, "
                    "item name or label (e.g. \"9:45\", \"Item B\", \"total\"). "
                    "Reuse the SAME key to correct a value you already sent.")
    value: str = Field(
        description="The fact itself, e.g. \"chicken 425g (+1g vs menu 424g)\".")


class PlanStep(BaseModel):
    """One sub-step of the goal, and where the run has got to with it.

    The same delta contract as `Note`, for the same reason: the model sends only
    the steps that are new or whose status changed, and the harness maintains the
    plan across turns (see :mod:`plan`). Reusing an id updates that step; an id
    never sent again keeps the status it had.

    Unlike `Note`, this one is *credited*: a step declared on an earlier turn and
    later marked ``done`` resets the stall ladder, once. The rules that keep that
    un-gameable live in :mod:`plan`, not here.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Short stable identifier for this sub-step (e.g. \"1\", "
                    "\"open app\", \"send\"). Reuse the SAME id to update a "
                    "step you already declared.")
    text: str = Field(
        "",
        description="What the sub-step is, in a few words. Send it once, when "
                    "you first declare the step.")
    status: Optional[PlanStatus] = Field(
        None,
        description="pending (still to do), active (doing it now), done "
                    "(finished), blocked (cannot proceed). Omit to leave a "
                    "step's status as it was.")


class AgentAction(BaseModel):
    """One step. The model replies with exactly this object and nothing else."""

    model_config = ConfigDict(extra="forbid")

    #: Properties `harden_schema` names in `required` even though pydantic
    #: defaults them. "tap needs a target" is a rule between two fields, and a
    #: flat schema cannot say it -- the alternative that could, a discriminated
    #: union keyed on `action`, is the `oneOf` this module's docstring rules
    #: out. So the schema says the weaker thing it *can* say: the key is always
    #: present, and the model has to write something in it rather than skipping
    #: past it. Skipping past it is not hypothetical: on 2026-09-01 five
    #: consecutive `glm-5p3-flash` runs died on a `tap` whose target key was
    #: never emitted at all, four of them on the run's first tap, while the
    #: reasoning beside it named the control. Only `target` is listed. Forcing
    #: the rest would put `x`/`y` on every action, which is the guessed
    #: coordinate `LOCATE_SYSTEM` exists to prevent.
    always_required: ClassVar[Tuple[str, ...]] = ("target",)

    observation: str = Field(description="One sentence: what screen is this?")
    reasoning: str = Field(
        description="One sentence: why this action advances the goal.")
    action: ActionName
    target: Optional[Target] = Field(
        None, description="Which element to act on, as {index, key}. REQUIRED "
                          "for tap, long_press and input_text -- those actions "
                          "are rejected without it, so never send null for "
                          "them. Also takes element scrolls. Send null only "
                          "when the action does not name an element.")
    x: Optional[float] = Field(
        None, description="For tap_at, long_press_at, double_tap and the start "
                          "of a drag: horizontal position as a fraction of "
                          "screen width, 0.0 (left edge) to 1.0 (right edge). "
                          "Omit -- and name the control in `text` instead -- "
                          "when no screenshot is attached.",
        ge=0, le=1)
    y: Optional[float] = Field(
        None, description="For tap_at, long_press_at, double_tap and the start "
                          "of a drag: vertical position as a fraction of "
                          "screen height, 0.0 (top edge) to 1.0 (bottom edge).",
        ge=0, le=1)
    to_x: Optional[float] = Field(
        None, description="For drag: horizontal position to release at, as a "
                          "fraction of screen width.", ge=0, le=1)
    to_y: Optional[float] = Field(
        None, description="For drag: vertical position to release at, as a "
                          "fraction of screen height.", ge=0, le=1)
    text: Optional[str] = Field(
        None,
        description="input_text: what to type. open_app/restart_app: package name or app search query. "
                    "open_url: the link (https:, tel:, mailto:, geo:, sms:, market:) "
                    "or a Settings screen such as android.settings.WIFI_SETTINGS. "
                    "list_apps: optional package name or keyword filter. "
                    "set_clipboard: text to put in clipboard. "
                    "tap_at/long_press_at/double_tap: the control to locate, when x and y are omitted. "
                    "ask_user: the question. done/fail: a one-line summary.")
    clear: Optional[bool] = Field(
        None, description="For input_text: clear field before typing (default True). Set False to append.")
    press_enter: Optional[bool] = Field(
        None, description="For input_text: press enter/search key after typing (default False).")
    key: Optional[KeyName] = Field(None, description="For press_key.")
    direction: Optional[ScrollDir] = Field(
        None, description="For scroll and swipe: which direction to move content or gesture ('down', 'up', 'left', 'right').")
    scroll_amount: float = Field(
        1, description="For scroll/swipe: distance or step multiplier (0.25 for small scroll, 1 for single page, 2-5 for fast multi-step scroll when searching long feeds/history).",
        ge=0.25, le=5)
    base_scale: Optional[float] = Field(
        None, description="For scroll: base drag scale per step (default 0.6, range 0.1 to 1.0; use 0.8 for larger page coverage per swipe).",
        ge=0.1, le=1.0)
    read_each: Optional[bool] = Field(
        None, description="For scroll/swipe: when the harness repeats this gesture for you, whether to "
                          "analyse each new screen with the vision model (default true). Set false to keep "
                          "paging without reading the screens in between -- e.g. skipping through a long "
                          "feed to reach something, when the in-between content does not matter.")
    duration: Optional[float] = Field(
        None, description="For swipe/scroll/wait/sleep: duration in seconds (e.g. 0.15 for fast flick, 0.3 for scroll, 1.0 for wait/sleep). "
                          "For long_press/long_press_at: how long to hold (default 0.6; use 1.5 or more for press-and-hold controls such as a voice-note button). "
                          "For drag: how long the move takes (default 0.5).",
        ge=0.05, le=30.0)
    expect_text: Optional[str] = Field(
        None,
        description="Optional. Text that must be on screen afterwards for this "
                    "action to count as having worked, e.g. \"Sent\". Use it "
                    "when the action has a silent failure mode; the harness "
                    "checks it and tells you if it did not appear.")
    wait_for_text: Optional[str] = Field(
        None, description="For wait/sleep: text to wait for on screen before returning.")
    timeout: Optional[float] = Field(
        None, description="For wait/sleep: max seconds to wait (0.5 to 30.0).", ge=0.5, le=30.0)
    confidence: Literal["high", "low"] = Field(
        "high", description="Use 'low' when unsure; you will be shown a screenshot.")
    notes: Optional[List[Note]] = Field(
        None,
        description="NEW or CORRECTED data records only, as {key, value}. The "
                    "harness keeps every record you have ever sent, so never "
                    "restate ones already listed under COLLECTED DATA. See DATA "
                    "COLLECTION above.")
    progress: Optional[List[PlanStep]] = Field(
        None,
        description="NEW or CHANGED plan steps only, as {id, text, status}. The "
                    "harness keeps the whole plan for you, so never restate a "
                    "step whose status has not changed. See PROGRESS TRACKING "
                    "above.")

    @field_validator("notes", mode="before")
    @classmethod
    def _accept_prose_notes(cls, value: object) -> object:
        """Take a bare string where records were asked for.

        Two callers need this. A model that ignores the record shape and writes
        ``"9:45 chicken 425g; 9:51 chicken 426g"`` should still have its readings
        kept rather than rejected into a repair round-trip; and every action in a
        recording made before the schema changed has ``notes`` as a string, which
        ``replay`` must still be able to load.
        """
        if isinstance(value, (str, dict)):
            from .scratchpad import as_records
            return [{"key": key, "value": text}
                    for key, text in as_records(value)] or None
        return value

    @field_validator("progress", mode="before")
    @classmethod
    def _accept_prose_progress(cls, value: object) -> object:
        """Take a bare string where plan steps were asked for.

        The same two callers `_accept_prose_notes` covers, and one extra
        consequence. A model that writes ``"Done: opened app. Next: send"`` is
        not rejected into a repair round-trip, and every action in a recording
        made before the schema changed still loads.

        The prose is *not* split into steps. "Done: opened app, found contact"
        has real structure and no reliable delimiter -- the comma separates two
        finished steps here and nothing at all in "chicken 425g, rice 290g" --
        and a step list guessed wrong is worse than none, because the harness
        credits step completions. So it lands whole in the one reserved entry
        `plan.PROSE_ID`, which is overwritten each turn and can never be marked
        done, and therefore never buys the stall ladder anything. That is
        exactly the behaviour the field had before this change.

        Every shape goes through `plan.as_steps`, not just the prose, so that a
        model writing ``"completed"`` or ``"in progress"`` where the schema names
        four exact strings is folded into one of them instead of failing
        validation and buying a repair round-trip. A status that means nothing at
        all becomes `None`, which the ledger reads as "no status change" -- the
        safe reading, since demoting a step to ``pending`` because its status was
        misspelt would undo real progress.
        """
        if value is None:
            return None
        from .plan import as_steps
        return [{"id": sid, "text": text, "status": status or None}
                for sid, text, status in as_steps(value)] or None

    @model_validator(mode="before")
    @classmethod
    def _salvage_target(cls, data: object) -> object:
        """Rebuild a dropped `target` from the ``#N`` the reasoning already names.

        A model that writes ``"action": "tap"`` and then, one field later,
        "I tap the topmost like button (#2 'Like photo')" has decided which
        element it wants and merely failed to fill the key. Rejecting that costs
        a repair round-trip at best and the whole run at worst, and the answer
        was sitting in the same object.

        Read out of `reasoning` only. `observation` describes the screen rather
        than the intent, so the ordinals in it are as often the things being
        ruled out as the thing being chosen -- at the bottom of a Hinge profile
        it names both hearts, and the one the run wanted was neither.

        Conservative on purpose: it wants exactly one distinct ordinal in that
        sentence. Two means the model was comparing candidates and the right
        answer is a repair, not a coin toss. A quoted label next to the ordinal
        is carried along when there is one, because `resolve_target` cross-checks
        it against the element actually at that index and falls back to a text
        search when they disagree -- so a stale ordinal lands on the named
        control instead of on whatever inherited its position.
        """
        if not isinstance(data, dict):
            return data
        if data.get("action") not in NEEDS_TARGET or data.get("target"):
            return data
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, str):
            return data
        found = _PROSE_TARGET.findall(reasoning)
        if len({index for index, _label in found}) != 1:
            return data
        index, label = found[0]
        target: Dict[str, Any] = {"index": int(index)}
        if label.strip():
            target["text"] = label.strip()
        log.warning("%s arrived with no target; recovered %s from the "
                    "reasoning %r", data.get("action"),
                    Target(**target).describe(), reasoning)
        return {**data, "target": target}

    @model_validator(mode="after")
    def _check_arguments(self) -> "AgentAction":
        if self.action in NEEDS_TARGET and self.target is None:
            raise ValueError(f"{self.action} requires a target")
        if self.action in POINT_ACTIONS and (self.x is None or self.y is None) \
                and not (self.text or "").strip():
            raise ValueError(f"{self.action} requires x and y, or the "
                             f"control's name in text")
        if self.action == "drag":
            if self.x is None or self.y is None:
                raise ValueError("drag requires x and y to start from")
            if self.to_x is None or self.to_y is None:
                raise ValueError("drag requires to_x and to_y to release at")
        if self.action == "input_text" and self.text is None:
            raise ValueError("input_text requires text")
        if self.action == "press_key" and self.key is None:
            raise ValueError("press_key requires key")
        if self.action in ("scroll", "swipe", "scroll_to_edge") \
                and self.direction is None:
            raise ValueError(f"{self.action} requires direction")
        if self.action in ("open_app", "restart_app") and not self.text:
            raise ValueError(f"{self.action} requires the package name in text")
        if self.action == "open_url" and not self.text:
            raise ValueError("open_url requires the link or Settings action "
                             "in text")
        if self.action == "ask_user" and not self.text:
            raise ValueError("ask_user requires the question in text")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.action in TERMINAL_ACTIONS

    def signature(self) -> str:
        """Stable identity for loop detection and ban lists.

        Built from `Target.identity()` and not `Target.describe()`: this string
        is the primary key of `LoopDetector.attempts`, the per-screen ban set,
        the stall-tier refusal set, the pager exemption and the 24-hour cross-run
        `dead_end` rows, and it used to resolve to the bare ordinal. Measured
        across ``runs/``, 47% of the resource-ids seen more than once in a run
        appeared under more than one `#N` -- `id=back` took thirteen.
        """
        parts = [self.action]
        if self.target is not None:
            parts.append(self.target.identity())
        if self.action in POINT_ACTIONS:
            if self.x is not None and self.y is not None:
                # Quantised to a ~1% grid: a blind tap retried a few pixels off
                # is the same action for loop-detection purposes.
                parts.append(f"{self.x:.2f},{self.y:.2f}")
            elif self.text:
                # A named control is grounded by the vision locate at act time;
                # until then its name is the identity.
                parts.append(" ".join(self.text.lower().split()))
        if self.action == "drag" and None not in (self.x, self.y,
                                                  self.to_x, self.to_y):
            parts.append(f"{self.x:.2f},{self.y:.2f}->"
                         f"{self.to_x:.2f},{self.to_y:.2f}")
        if self.action in ("open_app", "open_url", "restart_app") and self.text:
            # The whole identity of these: two `open_url`s to different links
            # are different actions, and repeating one is what the ban list is
            # for.
            parts.append(" ".join(self.text.lower().split()))
        for extra in (self.key, self.direction):
            if extra:
                parts.append(str(extra))
        return "/".join(parts)

    def describe(self, element: Optional[Element] = None) -> str:
        bits = [self.action]
        if self.target is not None:
            bits.append(describe_target(self.target, element))
        if self.action in POINT_ACTIONS and self.x is not None \
                and self.y is not None:
            bits.append(f"({self.x:.2f},{self.y:.2f})")
        if self.action == "drag" and None not in (self.x, self.y,
                                                  self.to_x, self.to_y):
            bits.append(f"({self.x:.2f},{self.y:.2f})->"
                        f"({self.to_x:.2f},{self.to_y:.2f})")
        if self.action == "input_text" and self.text is not None:
            bits.append(f"{self.text!r}")
        elif self.action in ("open_app", "open_url", "restart_app", "list_apps",
                             "done", "fail", "ask_user") and self.text:
            bits.append(self.text)
        elif self.action in POINT_ACTIONS and self.text:
            bits.append(self.text)
        if self.key:
            bits.append(self.key)
        if self.direction:
            bits.append(self.direction)
            if self.action != "scroll_to_edge":
                bits.append(f"amount={self.scroll_amount}")
            if self.base_scale is not None:
                bits.append(f"base_scale={self.base_scale}")
            if self.read_each is False:
                bits.append("read_each=false")
        if self.action in ("long_press", "long_press_at") \
                and self.duration is not None:
            bits.append(f"hold={self.duration:.2f}s")
        return " ".join(bits)


def element_summary(element: Optional[Element]) -> Optional[Dict[str, Any]]:
    """What a target resolved to, flattened for `events.jsonl` and the web UI.

    The same facts `describe_target` puts in a string for the prompt history and
    the terminal, kept apart instead: `#3` is a position in a list the reader of
    a run cannot see, and a UI that wants to lead with the label and demote the
    id and the coordinates cannot get them back out of the rendered form.

    `None` for an unresolved target is not the same as no target at all -- for a
    tap it is the reason the step is about to fail -- so the caller writes the
    field only when there was something to resolve.
    """
    if element is None:
        return None
    return {
        "index": element.index,
        "kind": element.kind(),
        "text": " ".join(element.best_text.split()),
        "resource_id": element.resource_id,
        "center": list(element.center),
        # The rectangle, next to the point inside it. `center` is where the tap
        # lands and is the whole of what the run needs; this is what a reader
        # needs, because a point cannot be drawn over a screenshot as the thing
        # that was tapped -- it can only be drawn as a dot near it. Device
        # pixels, like `center`, against the `screen_w`/`screen_h` the `decide`
        # event carries.
        "bounds": list(element.bounds),
        "checkable": element.checkable,
        "checked": element.checked,
        "selected": element.selected,
        "enabled": element.enabled,
    }


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
        if element.center:
            details.append(f"at ({element.center[0]},{element.center[1]})")
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
        parts.append(f"(Obs: {obs_clean})")

    if grade:
        parts.append(f"-> {grade}")
        if reason:
            parts.append(f"({reason})")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Folding repeated history
# ---------------------------------------------------------------------------
#
# Paging an album produces the same line over and over. From
# `runs/af76720d05c4`, nine consecutive entries in one prompt:
#
#     34. swipe #4 [Scroller "Image" at (360,800)] left amount=1.0 in
#         com.whatsapp (Obs: ...) -> success
#
# identical but for the step number and the observation -- and the observation is
# the only part carrying information, because it holds what the model read off
# that item. So a run is folded into one line that keeps the count and the
# readings, which is what `screen._collapse_identical_siblings` already does for
# repeated elements.
#
# Folding happens at append time, and only against the immediately preceding
# entry, so the rendered block still only ever differs from last turn's at its
# final line. That is the same append-only property `prompts.history_only_block`
# needs for the prompt prefix to stay cacheable -- rewriting the tail is free,
# rewriting the middle is not.

_HIST_STEP = re.compile(r"^(\d+)(?:-(\d+))?\.\s+")
#: The Obs clause may itself contain brackets, so it ends where the next known
#: field begins rather than at the first closing paren.
_HIST_OBS = re.compile(r"\s*\(Obs: (.*?)\)(?=\s*(?:->|\[x\d+\]|$))")
_HIST_COUNT = re.compile(r"\s*\[x(\d+)\]")
#: How a fold records readings it had to drop, so re-folding keeps the tally
#: instead of forgetting it a second time.
_HIST_DROPPED = re.compile(r"^\.\.\.\s*\+(\d+)\s+more$")

#: How many distinct readings a folded line keeps, and how much room they get.
#: Enough to see what a sweep collected; not so much that the fold costs more
#: than the lines it replaced. Both bounds are enforced in one place, and what
#: they drop is stated rather than quietly disappearing.
MAX_FOLDED_READINGS = 6
MAX_FOLDED_READING_CHARS = 400


@dataclass
class _Folded:
    first: int
    last: int
    count: int
    body: str
    readings: List[str]
    dropped: int = 0


def _split_history(line: str) -> Optional[_Folded]:
    """Take a history line back apart, folded or not."""
    match = _HIST_STEP.match(line)
    if not match:
        return None
    first = int(match.group(1))
    last = int(match.group(2) or match.group(1))
    rest = line[match.end():]

    readings: List[str] = []
    dropped = 0
    obs = _HIST_OBS.search(rest)
    if obs:
        for part in obs.group(1).split(" | "):
            part = part.strip()
            if not part:
                continue
            already = _HIST_DROPPED.match(part)
            if already:
                dropped += int(already.group(1))
            else:
                readings.append(part)
        rest = rest[:obs.start()] + rest[obs.end():]

    count = 1
    repeat = _HIST_COUNT.search(rest)
    if repeat:
        count = int(repeat.group(1))
        rest = rest[:repeat.start()] + rest[repeat.end():]

    return _Folded(first, last, count, " ".join(rest.split()), readings, dropped)


def _join_history(folded: _Folded) -> str:
    steps = (f"{folded.first}." if folded.first == folded.last
             else f"{folded.first}-{folded.last}.")
    out = f"{steps} {folded.body}"
    if folded.count > 1:
        out += f" [x{folded.count}]"

    shown, used = [], 0
    for reading in folded.readings:
        if (len(shown) >= MAX_FOLDED_READINGS
                or used + len(reading) > MAX_FOLDED_READING_CHARS):
            break
        shown.append(reading)
        used += len(reading)
    dropped = folded.dropped + len(folded.readings) - len(shown)

    if shown or dropped:
        joined = " | ".join(shown)
        if dropped > 0:
            joined += f"{' | ' if shown else ''}... +{dropped} more"
        out += f" (Obs: {joined})"
    return out


def append_history(history: List[str], entry: str) -> List[str]:
    """Append `entry`, folding it into the previous line if it repeats it.

    Two entries repeat when everything except the step number and the
    observation matches: the same action on the same screen with the same
    outcome. Distinct observations are kept, deduplicated and in order, because
    on a pager they *are* the collected readings.
    """
    if not history:
        history.append(entry)
        return history

    new = _split_history(entry)
    prev = _split_history(history[-1])
    if new is None or prev is None or new.body != prev.body:
        history.append(entry)
        return history

    for reading in new.readings:
        if reading not in prev.readings:
            prev.readings.append(reading)
    prev.last = new.last
    prev.count += new.count
    prev.dropped += new.dropped
    history[-1] = _join_history(prev)
    return history


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def _best_app_match(pkgs: List[str], query: str) -> str:
    q = query.lower()
    def score(p: str) -> Tuple[int, int, int, str]:
        pl = p.lower()
        sub_variant = 1 if any(v in pl for v in (".lite", ".w4b", ".work", ".beta", ".debug")) else 0
        exact_seg = 0 if q in pl.split(".") else 1
        return (sub_variant, exact_seg, len(p), p)
    return min(pkgs, key=score)


def resolve_target(target: Target, screen: Screen) -> Optional[Element]:
    """Find the element a target refers to on this screen, with fallback verification."""
    if target.index is not None:
        el = screen.by_index(target.index)
        if el is not None:
            match = True
            # The content key outranks the ordinal. When both are given and they
            # disagree, the list moved between the dump the model was shown and
            # the one being acted on -- which is exactly what an ordinal cannot
            # survive and what this field exists to catch.
            if target.key and el.key and el.key != target.key:
                match = False
            if target.resource_id and el.resource_id != target.resource_id:
                match = False
            if target.text and target.text.strip().lower() not in el.best_text.strip().lower():
                match = False
            if match:
                return el
            log.warning("target index #%d mismatched (text=%r, id=%r, key=%r); "
                        "attempting fallback search",
                        target.index, el.best_text, el.resource_id, el.key)

    if target.key:
        keyed = [e for e in screen.elements if e.key == target.key]
        if len(keyed) == 1:
            if target.index is not None:
                log.info("target #%d moved to #%d; resolved by key %s",
                         target.index, keyed[0].index, target.key)
            return keyed[0]

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
        loose = [e for e in screen.elements if wanted in e.best_text.strip().lower()]
        if loose:
            return min(loose, key=lambda e: len(e.best_text))

    return None


#: A point that lands on a listed control no bigger than this (as a fraction of
#: the screen) is a tap an element target would have done better, and `tap_at`
#: is refused for it. Bigger containers -- a map, a video surface, a scroll
#: region -- are legitimate tap_at territory: the thing wanted inside them has
#: no element of its own.
_POINT_GUARD_MAX_AREA = 0.25

#: An input_text target bigger than this (as a fraction of the screen) is not
#: the field: it is a container the field was folded into. Tapping its centre
#: focuses nothing, and the IME broadcast types into whatever has focus --
#: i.e. into nothing. The agent loop answers with the vision locate instead.
_INPUT_CONTAINER_MIN_AREA = 0.5


def input_target_is_container(element: Element, screen: Screen) -> bool:
    """True when an input_text target is a container, not the field itself.

    A chat composer or a search page whose real field the tree does not list
    -- folded into the scroller's aggregated label, or simply never focusable
    -- leaves the model aiming `input_text` at the biggest thing that mentions
    it. Tapping that thing's centre hits bare canvas or a list item, never the
    field, so the keys go nowhere. Editable elements are exempt however large
    they are: a full-screen notes editor's centre takes focus fine.
    """
    if element.editable:
        return False
    if element.kind() == "Scroller":
        return True
    return (screen.width > 0 and screen.height > 0
            and element.area
            > screen.width * screen.height * _INPUT_CONTAINER_MIN_AREA)


def element_at_point(screen: Screen, x: float, y: float) -> Optional[Element]:
    """The listed control under a fractional point, when there is a small one.

    The point half of the tap_at refusal guard (the text half is
    `resolve_target`): the agent loop refuses a tap_at that lands on a
    button-sized listed element, with the #N it should have used. Returns None
    when the point hits nothing interactive, or only a container too big to be
    the thing meant.
    """
    if screen.width <= 0 or screen.height <= 0:
        return None
    px, py = x * screen.width, y * screen.height
    hits = [e for e in screen.elements if e.interactive
            and e.bounds[0] <= px <= e.bounds[2]
            and e.bounds[1] <= py <= e.bounds[3]]
    if not hits:
        return None
    innermost = min(hits, key=lambda e: e.area)
    if innermost.area > screen.width * screen.height * _POINT_GUARD_MAX_AREA:
        return None
    return innermost


class ActionError(RuntimeError):
    """The action could not be carried out on this screen."""


def _point(action: "AgentAction", screen: Screen,
           fx: Optional[float], fy: Optional[float]) -> Tuple[int, int]:
    """A fractional point as device pixels, clamped inside the frame."""
    if fx is None or fy is None:
        # A text-mode point action reaches here only when nothing grounded it
        # -- the agent loop's vision locate answers before execute.
        raise ActionError(f"{action.action} has no point: the control was "
                          f"never located")
    if screen.width <= 0 or screen.height <= 0:
        raise ActionError(f"{action.action} needs the screen dimensions, "
                          f"which are unknown for this screen")
    return (min(max(1, int(fx * screen.width)), screen.width - 1),
            min(max(1, int(fy * screen.height)), screen.height - 1))


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
    elif action.action in POINT_ACTIONS:
        px, py = _point(action, screen, action.x, action.y)
        if action.action == "tap_at":
            dev.tap(px, py)
        elif action.action == "long_press_at":
            dev.long_press(px, py, duration=action.duration or 0.6)
        else:
            dev.double_tap(px, py)
    elif action.action == "drag":
        fx, fy = _point(action, screen, action.x, action.y)
        tx, ty = _point(action, screen, action.to_x, action.to_y)
        dev.drag(fx, fy, tx, ty, duration=action.duration or 0.5)
    elif action.action == "long_press":
        assert element is not None
        # `duration` used to be dropped here, which made every long press
        # exactly the device default. A press-and-hold control -- a voice-note
        # button, a shutter, a reorder handle -- wants a hold measured in
        # seconds, and there was no way to ask for one.
        dev.long_press(*element.center, duration=action.duration or 0.6)
    elif action.action == "input_text":
        assert element is not None
        # Focus the field first; the IME broadcast path types into whatever has
        # focus, not into a selector. `_focus_point` is the harness's override:
        # set when the agent loop's vision locate placed the field because the
        # target was a container whose centre focuses nothing.
        focus = getattr(action, "_focus_point", None)
        if focus is not None and screen.width > 0 and screen.height > 0:
            px = min(max(1, int(focus[0] * screen.width)), screen.width - 1)
            py = min(max(1, int(focus[1] * screen.height)), screen.height - 1)
            dev.tap(px, py)
        else:
            dev.tap(*element.center)
        should_clear = action.clear if action.clear is not None else True
        should_enter = bool(action.press_enter)
        dev.input_text(action.text or "", clear=should_clear, press_enter=should_enter)
    elif action.action == "press_key":
        dev.press(action.key or "back")
    elif action.action in ("scroll", "swipe"):
        box = None
        # A horizontal swipe used to be silently retargeted onto "the pager" --
        # the largest full-bleed horizontal scroller on screen. That was meant
        # for a media viewer whose image is listed as #1, #4 or #11 depending on
        # which overlay chrome is showing. On an app that nests tabs above a
        # feed it picked the *tab strip*, so "next photo" became "next tab", and
        # the model could not tell because its stated target was overridden.
        # The model's target is now honoured; when it is the wrong one the swipe
        # moves nothing and `verify` reports that, which is a signal the model
        # can act on rather than one the harness hides.
        if element is not None:
            box = element.bounds
            # A "swipe up" on a container believed to be horizontal used to be
            # rewritten as "swipe left". `Element.is_horizontal` answers from the
            # class name when the geometry is inconclusive, and `ViewPager2` --
            # which it treats as horizontal -- is exactly what a vertical video
            # feed is built from. So on Reels, Shorts and TikTok every "next
            # video" swipe was silently turned ninety degrees, did nothing, and
            # came back as an unexplained failure the model could not diagnose
            # because it was never told its gesture had been changed.
            #
            # The direction the model asked for is now the direction that goes
            # out. A gesture aimed the wrong way moves nothing and `verify` says
            # so, which is a fact the model can act on.
        amount = max(0.25, min(action.scroll_amount, 5.0))
        if action.action == "swipe":
            # Swipe gesture: default duration 0.3s for reliable ViewPager/gallery transitions, single gesture with scale 0.8
            duration = action.duration if action.duration is not None else 0.3
            scale = min(0.95, max(0.2, 0.8 * amount))
            dev.scroll(action.direction or "left", scale=scale, box=box, duration=duration)
        else:
            # Scroll action: default duration 0.3s for steady list scrolling
            duration = action.duration if action.duration is not None else 0.3
            base_scale = action.base_scale if action.base_scale is not None else 0.6
            full_scrolls = int(amount)
            remainder = amount - full_scrolls
            import time
            for i in range(full_scrolls):
                dev.scroll(action.direction or "down", scale=base_scale, box=box, duration=duration)
                if i < full_scrolls - 1 or remainder > 0:
                    time.sleep(0.15)
            if remainder >= 0.1:
                dev.scroll(action.direction or "down",
                           scale=round(base_scale * remainder, 2), box=box, duration=duration)
    elif action.action == "scroll_to_edge":
        direction = action.direction or "up"
        moved = dev.fling_to_edge(direction)
        edge = {"up": "top", "down": "bottom",
                "left": "start", "right": "end"}.get(direction, direction)
        setattr(action, "_result_summary",
                f"flung to the {edge}" if moved
                else f"already at the {edge}; nothing moved")
    elif action.action == "open_url":
        summary = dev.open_url((action.text or "").strip())
        setattr(action, "_result_summary", summary)
    elif action.action == "restart_app":
        raw_pkg = (action.text or "").strip()
        target_pkg = raw_pkg
        if "." not in raw_pkg:
            pkgs = dev.list_apps(query=raw_pkg)
            if pkgs:
                target_pkg = _best_app_match(pkgs, raw_pkg)
        setattr(action, "_resolved_package", target_pkg)
        landed = dev.restart_app(target_pkg)
        summary = f"force-stopped and relaunched {target_pkg}"
        if landed is False:
            summary += "; it was still not in front when the wait ran out"
        setattr(action, "_result_summary", summary)
    elif action.action == "open_app":
        raw_pkg = (action.text or "").strip()
        target_pkg = raw_pkg
        if "." not in raw_pkg:
            pkgs = dev.list_apps(query=raw_pkg)
            if pkgs:
                target_pkg = _best_app_match(pkgs, raw_pkg)
                log.info("resolved app %r -> %r", raw_pkg, target_pkg)
        setattr(action, "_resolved_package", target_pkg)
        landed = dev.open_app(target_pkg)
        summary = f"opened {target_pkg}" + (f" (resolved from {raw_pkg!r})" if target_pkg != raw_pkg else "")
        # `is False` and not `not landed`: only the device layer reports on the
        # foreground, and saying "never arrived" about a launch nobody waited for
        # would be worse than saying nothing.
        if landed is False:
            summary += "; it was still not in front when the wait ran out"
        setattr(action, "_result_summary", summary)
    elif action.action == "list_apps":
        query = (action.text or "").strip()
        pkgs = dev.list_apps(query=query)
        if not pkgs:
            action_summary = f"no apps found matching {query!r}" if query else "no apps found"
        else:
            limit = 20
            shown = pkgs[:limit]
            overflow = len(pkgs) - limit
            action_summary = f"found {len(pkgs)} app(s): {', '.join(shown)}"
            if overflow > 0:
                action_summary += f" ... (+{overflow} more)"
        log.info("list_apps query=%r -> %s", query, action_summary)
        setattr(action, "_result_summary", action_summary)
    elif action.action == "get_clipboard":
        clip = dev.get_clipboard()
        summary = f"clipboard content: {clip!r}"
        log.info("get_clipboard -> %s", summary)
        setattr(action, "_result_summary", summary)
    elif action.action == "set_clipboard":
        val = action.text or ""
        dev.set_clipboard(val)
        summary = f"set clipboard to {val!r}"
        log.info("set_clipboard -> %s", summary)
        setattr(action, "_result_summary", summary)
    elif action.action in ("wait", "sleep"):
        import time
        past_verb = "slept" if action.action == "sleep" else "waited"
        timeout = action.timeout if action.timeout is not None else (action.duration or (5.0 if action.wait_for_text else 1.0))
        if action.wait_for_text:
            # Asked of the device, not of a poll loop here. This used to call
            # `dev.observe()` every 0.1s, and each of those is a whole
            # `dump_hierarchy` -- 0.55s on the phone in `device.current_app`'s
            # notes and far more over wireless adb -- so a long wait could
            # spend its entire budget on two or three polls and then report a
            # timeout for a screen that had already arrived.
            found = dev.wait_for_text(action.wait_for_text, timeout)
            summary = f"{past_verb} for {action.wait_for_text!r} -> {'found' if found else 'timed out'}"
            setattr(action, "_result_summary", summary)
        else:
            time.sleep(min(timeout, 30.0))
            summary = f"{past_verb} for {timeout:.1f}s"
            setattr(action, "_result_summary", summary)
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
    # The model's own expectation outranks anything derived here: it knows what
    # the action was for, and the synthesised conditions are a guess from the
    # action's shape. Only `text_present` is offered -- the rest of the DSL
    # needs a resource-id the model does not reliably have, and one string is
    # what the silent-failure cases ("did the message actually send?") need.
    if (action.expect_text or "").strip():
        return Postcondition(kind="text_present", text=action.expect_text.strip())
    if action.action == "input_text":
        return Postcondition(kind="element_state",
                             resource_id=element.resource_id if element else None,
                             field="text", value=action.text or "")
    if action.action in ("tap", "long_press", "long_press_at", "double_tap") \
            and element is not None and element.checkable:
        return Postcondition(kind="element_state", resource_id=element.resource_id,
                             field="checked",
                             value="false" if element.checked else "true")
    if action.action in ("open_app", "restart_app"):
        pkg = getattr(action, "_resolved_package", (action.text or "").strip())
        return Postcondition(kind="app_is", package=pkg)
    if action.action in ("wait", "sleep", "list_apps", "get_clipboard",
                         "set_clipboard", "scroll", "swipe", "scroll_to_edge"):
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
        # The tree is byte-identical, which is not the same thing as the screen
        # being unchanged. A feed whose content is a bitmap the accessibility
        # tree does not describe -- a meme card, a photo, a card stack -- swaps
        # one item for the next without moving a single node, so the hash cannot
        # tell "the tap advanced the feed" from "the tap hit nothing". The
        # pixels can, and they are what `_scroll_changed` has always asked for a
        # scroll; this asks the same question for a tap.
        if content_moved_in_pixels(before, after):
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


def content_moved_in_pixels(before: Screen, after: Screen) -> Optional[bool]:
    """Did the app's own content change, judged on the bitmap alone?

    ``None`` when there is no evidence either way -- one of the frames has no
    screenshot, or the hash could not be computed. Callers must read ``None``
    as "unknown", never as "no": it is the answer on every step that did not
    take a picture, which is most of them.

    Cropped to `pager.content_box`, so a toolbar fading in over the content is
    not mistaken for the content changing, and vetoed by
    `pager.video_only_drift`, so a video repainting inside an otherwise static
    frame is not either.
    """
    from .pager import content_moved, video_only_drift
    moved = content_moved(before, after)
    if moved is None:
        return None
    if moved and video_only_drift(before, after):
        return False
    return moved


def _scroll_changed(before: Screen, after: Screen,
                    action: Optional[AgentAction] = None) -> bool:
    """Multi-signal check for whether a scroll actually revealed new content.

    Four signals, cheapest first:

    0b. **Pager item identity** -- on a gallery or carousel this is the only
       trustworthy signal and it outranks everything below, including the
       perceptual hash: a media viewer fades its toolbar in and out over the
       image, which moves the whole-screen dhash by far more than four bits
       while the item underneath is unchanged. Asked the other way round, the
       hierarchy is *identical* between two different photos because
       ``mask_text`` rewrites the caption's timestamp -- so without this signal
       an advancing swipe and a dropped swipe are indistinguishable.

    Both pixel signals answer "did the bitmap move", though, and a video
    playing inside the frame moves the bitmap on its own -- no gesture needed.
    So each is vetoed by `pager.video_only_drift`: a byte-identical tree with
    a video on it means the motion is the video, not the scroll.

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
    # Signal 0a: the app's own content, hashed with the overlay bands cropped
    # off. Authoritative when both frames have a screenshot. This replaced
    # `pager.same_item`, which preferred the app's caption over the pixels and
    # so inherited every way a caption could be wrong -- including reading the
    # status-bar clock, which made two frames "different items" once a minute.
    moved = content_moved_in_pixels(before, after)
    if moved is not None:
        return moved

    # Signal 0b: Perceptual Image Fingerprinting
    if before.dhash is not None and after.dhash is not None:
        from .fingerprint import dhash_distance
        from .pager import video_only_drift
        dist = dhash_distance(before.dhash, after.dhash)
        if dist is not None:
            if dist >= 4:
                if video_only_drift(before, after):
                    return False
                return True
            if dist < 4 and after.exact_id == before.exact_id:
                return False

    # Last resort for a swipe or a horizontal scroll with no identity signal at
    # all: galleries, carousels and card stacks page between bitmaps that the
    # accessibility tree does not describe, so an unchanged hierarchy is not
    # evidence the gesture failed. Answer "changed" rather than punish a swipe
    # that probably worked -- but note this branch is now only reached when the
    # screen is not a recognised pager and no screenshot was taken.
    if action is not None:
        if action.action == "swipe" or action.direction in ("left", "right"):
            return True

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
            if before_texts and after_texts and before_texts == after_texts:
                return False  # Scroller content is present and identical.

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
    if action.action in ("wait", "sleep"):
        result_text = getattr(action, "_result_summary", "")
        default = "slept" if action.action == "sleep" else "waited"
        # Graded on what the wait actually produced, not on the fact that it ran.
        #
        # This used to be an unconditional `success`, which made a wait the one
        # action that could never be wrong -- and `_loop` reads `outcome.ok` to
        # zero `consecutive_failures` and clear `last_failure`, so a wait
        # laundered the failure before it. A fail/wait alternation could never
        # reach `max_consecutive_failures`, and it also dropped out of the
        # deeper-thinking and take-a-screenshot triggers, both of which key on
        # the same counter. 13 of 103 turns across ``runs/`` were waits.
        #
        # `no_change` rather than a failure: waiting for a screen that turned out
        # to be already finished is a reasonable thing to have tried once. It is
        # the *second* one on the same screen that is the problem, and
        # `no_change` is exactly what the existing ban machinery in `_loop` acts
        # on.
        if after.exact_id == before.exact_id:
            return VerifyOutcome(
                grade="no_change",
                reason=f"{result_text or default} and nothing on screen changed")
        return VerifyOutcome(grade="success", reason=result_text or default)
    if action.action in ("get_clipboard", "set_clipboard"):
        result_text = getattr(action, "_result_summary", "")
        return VerifyOutcome(grade="success", reason=result_text)
    if action.action == "list_apps":
        result_text = getattr(action, "_result_summary", "")
        reason = f"listed apps ({result_text})" if result_text else "listed apps"
        return VerifyOutcome(grade="success", reason=reason)
    if action.action == "open_url":
        # Graded on the screen and not on `am`'s exit status: `am start` exits
        # zero whether or not anything handled the intent, and the refusals it
        # does report are raised in `Device.open_url` before this is reached.
        result_text = getattr(action, "_result_summary", "")
        if after.exact_id == before.exact_id:
            return VerifyOutcome(
                grade="no_change",
                reason=f"{result_text or 'opened the link'}, but nothing on "
                       f"screen changed -- no app handled it")
        return VerifyOutcome(grade="success", reason=result_text)

    condition = post or synthesise_postcondition(action, None)

    # Scroll/swipe that didn't move = end of list or edge of gallery, not a hard failure.
    if action.action in ("scroll", "swipe", "scroll_to_edge") \
            and not _scroll_changed(before, after, action=action):
        # A fling that moves nothing has told the model something a plain
        # scroll's silence does not: it is already at that edge, and the
        # harness said so rather than leaving it to be inferred.
        if action.action == "scroll_to_edge":
            return VerifyOutcome(
                grade="no_change",
                reason=getattr(action, "_result_summary", "")
                       or "already at that edge; nothing moved")
        return VerifyOutcome(grade="no_change",
                             reason=f"{action.action}ing did not reveal new content")

    # Checked before the generic postcondition, because "nothing happened at
    # all" is both the most common silent failure and a more actionable
    # diagnosis than a bare condition failure -- it is what feeds the per-run
    # ban list, so the same dud tap is not retried forever.
    #
    # Which is also why an identical tree is not on its own enough to say it.
    # `exact_id` describes the accessibility tree, and a surface whose content is
    # a bitmap the tree does not describe -- a meme feed, a gallery, a card stack
    # -- replaces one item with the next without moving a node. On such a screen
    # the *working* tap and the tap that hit nothing are byte-identical here, and
    # this branch condemned both: `no_change` is the one grade that bans the
    # action for the rest of the run AND writes it to the 24-hour cross-run
    # dead-end store, so a correct control was struck off and every later pass
    # started with it already forbidden.
    #
    # Measured on ``runs/7640105057d7`` (Schmooze, a bitmap meme feed with no
    # accessibility node for its like button, so `tap_at` was the only way in):
    # step 11's tap missed the thumbs-up by ~50px and step 12's landed on it and
    # advanced the feed. Both were graded "nothing on screen changed" and both
    # were banned. Cropped-content dhash distance between the frames: 0 for the
    # miss, 30 for the hit, against a threshold of 6 -- so the pixels separate
    # the two cleanly, and they were never consulted. The run then spent its
    # remaining steps bouncing off the bottom navigation under escalating
    # "NO PROGRESS" pressure. The pass before it succeeded in 25 steps for $0.10
    # by choosing `swipe` over `tap` on the same screen -- and a swipe is graded
    # by `_scroll_changed`, which does look at the pixels. That asymmetry, not
    # the app and not the model, is what decided the two outcomes.
    if action.action in NAVIGATIONAL and after.exact_id == before.exact_id:
        # `None` -- no screenshot on one side -- leaves the tree as the only
        # evidence there is, which is the behaviour this always had.
        if not content_moved_in_pixels(before, after):
            return VerifyOutcome(grade="no_change",
                                 reason="nothing on screen changed")
        if condition.kind == "screen_changed":
            return VerifyOutcome(
                grade="success",
                reason="the app's content changed, though no element did")
        # A condition of the model's own (`expect_text`) or one about a named
        # element judges this better than a whole-screen hash can, so fall
        # through to it. It can still fail -- as a `hard_fail`, which states
        # something about that condition and does not ban the action.

    # The typing half of the check above. A dump that is byte-identical after
    # an input_text means the tap that was meant to focus the field landed on
    # nothing that takes focus -- the centre of a scroller, say -- and the
    # keys went nowhere. It must be caught here, before the postcondition,
    # because the element_state condition cannot: it passes vacuously whenever
    # the field has no resource-id to find (`_find` answers None and a missing
    # element is inconclusive-but-passing), which graded "nothing was typed" a
    # success and let the run proceed as if the text were in.
    if action.action == "input_text" and after.exact_id == before.exact_id:
        return VerifyOutcome(
            grade="no_change",
            reason="nothing on screen changed -- the field never took focus, "
                   "so nothing was typed")

    passed, why = check_postcondition(condition, before, after)
    if not passed:
        return VerifyOutcome(grade="hard_fail", reason=why)

    if expected_skeleton and after.skeleton_id != expected_skeleton:
        # Screens legitimately fan out -- the same tap can land somewhere new
        # depending on state. Note it, do not punish it.
        return VerifyOutcome(grade="soft_fail",
                             reason="landed on an unexpected screen")

    return VerifyOutcome(grade="success")
