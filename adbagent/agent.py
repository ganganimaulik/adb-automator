"""The control loop.

Read this file to understand the whole system. The shape is:

    perceive -> ask the LLM -> guard -> act -> verify -> learn -> repeat

The LLM appears in exactly two places, both marked `### LLM ###`. Everything
else -- recognising the screen, resolving an anchor, dismissing a nag, deciding
whether an action worked, noticing a loop, ending the run when a programmatic
assertion passes -- is ordinary code. That is the entire point: the model is
consulted when the agent is genuinely uncertain, and not otherwise.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import __version__, checkpoint, conversation, prompts, runlog, safety
from .actions import (ActionError, AgentAction, append_history, execute,
                      format_history_entry, synthesise_postcondition, verify)
from .config import Config
from .device import Device, DeviceTimeout, DeviceLost
from .fingerprint import crop_frac
from .ledger import ReplyLedger
from .llm import (BudgetExceeded, LLMClient, LLMError, Prefetch, ScreenAnalysis)
from .memory import Memory, intent_key
from .pager import (SweepLog, can_repeat, content_box,
                    content_moved as pager_content_moved,
                    stop_repeating, sweep_summary)
from .safety import Aborted, LoopDetector
from .scratchpad import NoteLedger
from .screen import Screen, render
from .skills import Skill, SkillRegistry, goal_app_candidates

log = logging.getLogger("adbagent.agent")

Outcome = str  # "success" | "failed" | "aborted" | "needs_user"

#: How many times the harness will tap one dismiss control on one screen before
#: accepting that it does not dismiss anything. Two rather than one, because the
#: turn after a dismissal re-observes without settling, so a dialog still
#: animating out reads as an unchanged screen.
MAX_DISMISS_TRIES = 2


@dataclass
class Oracle:
    """A machine-checkable definition of done. Costs nothing and never lies."""

    shell: str = ""
    equals: str = ""
    text: str = ""

    @property
    def defined(self) -> bool:
        return bool(self.shell or self.text)

    def describe(self) -> str:
        """The condition, in the words it was given in. A run the oracle ends
        has no model summary to report, so this is what the ending says."""
        parts = []
        if self.text:
            parts.append(f"{self.text!r} is on screen")
        if self.shell:
            parts.append(f"`{self.shell}` is {self.equals.strip()!r}")
        return " or ".join(parts)

    def satisfied(self, dev: Device, screen: Screen) -> bool:
        if self.text:
            wanted = self.text.strip().lower()
            if any(wanted in el.best_text.strip().lower() for el in screen.elements):
                return True
        if self.shell:
            try:
                got = dev.shell(self.shell, timeout=15).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("assertion command failed: %s", exc)
                return False
            return got == self.equals.strip()
        return False


@dataclass
class RunState:
    goal: str
    run_id: str
    intent_id: str
    step: int = 0
    llm_calls: int = 0
    consecutive_failures: int = 0
    #: Every step, oldest first. Kept in full rather than truncated: the prompt
    #: is bounded by how much of this `history_only_block` *renders*, and the one
    #: call that wants the long view -- the completion judge -- used to be handed
    #: whatever the last 12 steps happened to be, which is how a run that spent
    #: 130 steps collecting data got graded on the tail of it.
    history: List[str] = field(default_factory=list)
    visits: Dict[str, int] = field(default_factory=dict)
    loops: LoopDetector = field(default_factory=LoopDetector)
    want_screenshot: bool = False
    last_failure: str = ""
    scroll_warnings: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished: Optional[Outcome] = None
    #: What the run answered: the text of the terminal action that ended it --
    #: `done`'s summary, `fail`'s reason, `ask_user`'s question. The one thing a
    #: person actually asked for on a "read X and tell me" goal, and until this
    #: it left the agent nowhere: it was written into the step feed as part of
    #: the last line and never survived `run()`, so the caller had the outcome
    #: word and the cost and no answer. Set only when the action really
    #: terminates -- a rejected `done` is not an answer.
    result: str = ""
    #: Why the outcome is the outcome: the judge's evidence, the assertion that
    #: settled it, or the reason a completion was rejected. Separate from
    #: `result` because it is the harness talking, not the model.
    evidence: str = ""
    #: Everything the run has collected. The model sends only what is new or
    #: corrected each turn (the ``notes`` field) and this keeps the union, so a
    #: record it stops mentioning cannot go missing -- see `scratchpad.py`.
    scratchpad: NoteLedger = field(default_factory=NoteLedger)
    #: Progress tracker for multi-step goals.  The LLM writes into the
    #: ``progress`` field; we keep the latest entry and feed it back.
    progress_log: List[str] = field(default_factory=list)
    #: Distinct packages this run has been in. Two or more means it is a
    #: multi-app run, which is when the app-switching advice earns its tokens.
    packages: set = field(default_factory=set)
    #: Steps spent in each. The set alone cannot say which app a run was *about*
    #: -- glancing at the launcher for one step reads the same as forty steps of
    #: work -- and `history` needs that to find the recorded runs for one app.
    package_steps: Dict[str, int] = field(default_factory=dict)
    #: Readings collected by the last mechanical sweep, in the order they were
    #: read. Not a ledger of a set -- see `pager.SweepLog`.
    sweep: SweepLog = field(default_factory=SweepLog)
    #: What verification concluded about the last gesture: True the app's content
    #: moved, False it did not, None not observable (no screenshot).
    content_moved: Optional[bool] = None
    #: True once a directional gesture has been *seen* to move content without
    #: leaving the screen -- i.e. this screen pages. A property of the gesture
    #: that was tried, not of the screen, which is why it lives here and not on
    #: `Screen`.
    paging: bool = False
    #: Skeleton the `paging` evidence was gathered on; leaving it clears them.
    last_skeleton: str = ""
    #: ``#N`` of the target the paging gesture was aimed at, so the loop detector
    #: can tell "repeating the thing that works" from "stuck".
    repeatable_index: int = 0
    #: The gesture observed to page, e.g. ``("swipe", "left")``.
    paging_gesture: Tuple[str, str] = ("", "")
    #: ``exact_id/label`` of the control the harness last auto-dismissed, and how
    #: many times it has tried it on that screen. A dismissal that changes
    #: nothing means the control is part of the screen rather than a popup over
    #: it, and repeating it is how a whole step budget goes on one button.
    last_dismiss: str = ""
    dismiss_tries: int = 0
    #: Steps since the run last learned anything -- see `Agent._note_outcome` for
    #: what counts. This is the counter `consecutive_failures` cannot be: that
    #: one counts actions that *failed*, and the dominant way a run is lost is
    #: every action succeeding while the run goes nowhere. In
    #: ``runs/2521862d7a23`` a two-cycle ran for twenty steps with
    #: `consecutive_failures` pinned at zero and every step graded ``success``.
    steps_since_progress: int = 0
    #: What reset it last, for the log, the events and the stall note.
    last_progress: str = "the run started"
    #: The approach a `replan` call handed back. Carried in the prompt until
    #: progress resumes, because a plan outlives the turn that asked for it --
    #: which is the whole reason the replan asks for a plan and not an action.
    strategy: str = ""
    #: `steps_since_progress` when the last replan ran, so one stall episode
    #: buys one replan rather than one every turn it persists.
    replanned_at: int = 0
    #: Screens already re-read at full resolution. One sharper look per screen:
    #: see `Agent._reread_sharper` for why a second cannot help.
    rereads: set = field(default_factory=set)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def note_progress(self, reason: str) -> None:
        """Record that the run just learned something; reset the stall ladder.

        The strategy goes with it. It was bought to break a stall, and a run
        that is moving again should not keep being told to abandon the approach
        that started working -- if it stalls a second time the next replan sees
        the newer situation anyway.
        """
        self.steps_since_progress = 0
        self.last_progress = reason
        self.strategy = ""
        self.replanned_at = 0

    def remember(self, entry: str) -> None:
        """Add a history line, folding it into the previous one if it repeats it.

        Paging an album produces the same line dozens of times over; see
        `actions.append_history` for why the fold happens here, at append time,
        rather than when the block is rendered.
        """
        append_history(self.history, entry)


class Recorder:
    """Per-run artifacts: one JSONL of events, the run log, blobs alongside."""

    def __init__(self, cfg: Config, run_id: str):
        self.dir = runlog.run_dir(cfg, run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        # The live LLM stream, for the web UI to tail. Append mode for the same
        # reason as events.jsonl: a resumed run continues its own directory.
        self.stream = (self.dir / runlog.STREAM_NAME).open("a", encoding="utf-8")
        # Opened here rather than by the caller, so that every entry point which
        # runs an agent -- `run`, `skills generate`, an embedding program -- gets
        # a debuggable log without having to remember to ask for one.
        self.log = runlog.attach(self.dir)

    def event(self, kind: str, **fields: Any) -> None:
        record = {"t": round(time.time(), 3), "kind": kind, **fields}
        self.events.write(json.dumps(record, default=str) + "\n")
        self.events.flush()
        # And into the run log, so that one file reads in order: the decision
        # next to the adb traffic, the retries and the warnings around it.
        runlog.event(kind, fields)

    def stream_event(self, kind: str, **fields: Any) -> None:
        """One line of the raw LLM stream. A best-effort side channel: a disk
        that cannot take it must not kill the run over the live view."""
        if self.stream is None:
            return
        record = {"t": round(time.time(), 3), "kind": kind, **fields}
        try:
            self.stream.write(json.dumps(record, default=str) + "\n")
            self.stream.flush()
        except (OSError, ValueError) as exc:
            log.warning("stream log failed (%s); dropping it", exc)
            try:
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None

    def blob(self, name: str, data: bytes) -> str:
        path = self.dir / name
        path.write_bytes(data)
        return str(path)

    def screenshot(self, step: int, jpeg: bytes, purpose: str) -> str:
        """Keep the frame a model was shown, and return its file name.

        Best-effort, for the same reason as `stream_event`: this exists so the
        live view and the history can show the screenshot next to the call that
        saw it, and a disk that will not take a 40 KB JPEG is not a reason to end
        the run. The empty string means "there is nothing to show", which is what
        every caller passes on to the UI.
        """
        if not jpeg:
            return ""
        name = runlog.shot_name(step, purpose, jpeg)
        try:
            self.blob(name, jpeg)
        except OSError as exc:
            log.warning("could not keep the %s screenshot (%s)", purpose, exc)
            return ""
        return name

    def dump_messages(self, step: int, messages: List[Dict[str, Any]], purpose: str = "decide") -> str:
        """Dump step prompt messages to a formatted JSON file in the run directory."""
        filename = f"step_{step:03d}_{purpose}_messages.json"
        cleaned_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            if isinstance(msg_copy.get("content"), list):
                new_content = []
                for item in msg_copy["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url_str = item.get("image_url", {}).get("url", "")
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"[base64 image payload: {len(url_str)} chars]"}
                        })
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
            cleaned_messages.append(msg_copy)

        content_bytes = json.dumps(cleaned_messages, indent=2, ensure_ascii=False).encode("utf-8")
        return self.blob(filename, content_bytes)

    def close(self) -> None:
        try:
            self.events.close()
        except Exception:  # noqa: BLE001
            pass
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        if self.log is not None:
            self.log.close()


def _stream_tap(rec: Recorder, inner: Any) -> Any:
    """Fan the live LLM stream out to the run's ``stream.jsonl``.

    The terminal reporter renders these events and throws them away; the web UI
    can only follow a file, so every `llm_start`/`llm_stream`/`llm_end` is
    mirrored here as it happens. Fields are whitelisted rather than passed
    through: `llm_end` also carries the `Call` and sometimes a verdict object,
    and the stream file stays flat JSON.
    """
    def tap(kind: str, **kw: Any) -> None:
        if kind == "llm_start":
            rec.stream_event(kind, step=kw.get("step") or 0,
                             purpose=kw.get("purpose") or "decide",
                             model=kw.get("model") or "",
                             screenshot=bool(kw.get("screenshot")),
                             # The kept frame, when this call is the one that was
                             # shown it. `screenshot` says a screenshot was taken
                             # this turn; this says which model looked at it.
                             shot=kw.get("shot") or "",
                             effort=kw.get("effort") or "")
        elif kind == "llm_stream":
            rec.stream_event(kind, stream_type=kw.get("stream_type") or "content",
                             purpose=kw.get("purpose") or "",
                             text=kw.get("text") or "")
        elif kind == "llm_end":
            call = kw.get("call")
            rec.stream_event(kind, step=kw.get("step") or 0,
                             purpose=kw.get("purpose") or "decide",
                             elapsed=round(kw.get("elapsed") or 0.0, 3),
                             prompt_tokens=getattr(call, "prompt_tokens", 0) or 0,
                             completion_tokens=getattr(call, "completion_tokens", 0) or 0,
                             reasoning_tokens=getattr(call, "reasoning_tokens", 0) or 0)
        inner(kind, **kw)
    return tap


# ---------------------------------------------------------------------------
# Screenshot policy
# ---------------------------------------------------------------------------

def step_metrics(calls: List[Any], detail: bool = True) -> Dict[str, Any]:
    """Roll up the LLM calls one step made, for `events.jsonl`.

    A step is not one call: a screenshot turn is an image analysis *and* a
    decision, and a repaired reply is two decisions. Recording the total next to
    the per-call breakdown is what makes "where did the 26 seconds go" a query
    rather than an argument.

    `detail=False` drops the per-call list, for the run-level rollup where every
    call already has its own event.
    """
    per_call = [c.metrics() for c in calls]
    return {
        **({"calls": per_call} if detail else {}),
        "n_calls": len(per_call),
        "prompt_tokens": sum(c["prompt_tokens"] for c in per_call),
        "cached_tokens": sum(c["cached_tokens"] for c in per_call),
        "completion_tokens": sum(c["completion_tokens"] for c in per_call),
        "reasoning_tokens": sum(c["reasoning_tokens"] for c in per_call),
        "reasoning_chars": sum(c["reasoning_chars"] for c in per_call),
        "latency_s": round(sum(c["latency_s"] for c in per_call), 3),
        "usd": round(sum(c["usd"] for c in per_call), 6),
    }


def needs_reasoning(state: RunState, cfg: Config, *, visit: int,
                    blocked: bool, hint: str) -> Tuple[str, str]:
    """How hard to think about this turn, and why.

    Reasoning tokens are the run's wall clock: ~4,200 of every ~4,400 output
    tokens, at 26s median per step and 96s at the ninetieth percentile. Most of
    those turns do not need it. "Swipe left again on the pager the note block
    names" is not a problem, and thinking for a minute about it buys nothing.

    But *some* turns are genuinely hard, and the cost of getting those wrong is a
    wasted run. The loop already knows which: it knows the last action failed, it
    knows this screen is new, it knows a loop was detected, and it knows the model
    said it was guessing. So the depth follows the evidence rather than a global
    setting -- shallow by default, deep the moment anything is off.

    Returns ("", "") when the feature is switched off, which is the default:
    `llm.reasoning_effort` has to be set before any of this applies.
    """
    if not cfg.llm.reasoning_effort:
        return "", ""

    reason = ""
    if state.consecutive_failures:
        reason = (f"the last {state.consecutive_failures} action(s) did not work")
    elif (cfg.run.stall_nudge_at
            and state.steps_since_progress >= cfg.run.stall_nudge_at):
        reason = (f"nothing new has been learned for "
                  f"{state.steps_since_progress} steps")
    elif state.last_failure:
        reason = "the last action was rejected"
    elif state.want_screenshot:
        reason = "the model said it was unsure last turn"
    elif visit == 0:
        reason = "this screen has not been seen before in this run"
    elif blocked:
        reason = "actions here are already known to lead nowhere"
    elif hint:
        reason = "the loop detector has something to say"

    if reason:
        return cfg.llm.effort_for("decide", hard=True), reason
    return cfg.llm.effort_for("decide"), ""


#: Long edge for the one sharper capture a screen is allowed. Chosen as the
#: largest frame that still costs a single-digit fraction of a cent on the vision
#: models here: a 1080x2400 phone reaches 1080x2400 untouched at 2400, so this is
#: "the pixels the app actually drew" rather than an arbitrary bigger number.
SHARP_LONG_EDGE = 2400

#: How the prompts ask the model to report a value it can see but cannot make
#: out. `IMAGE_ANALYSIS_SYSTEM` and `ITEM_READING_SYSTEM` both name the word, so
#: matching on it is reading the contract rather than guessing at prose.
_UNREADABLE_HINTS = ("unreadable", "cannot make out", "can't make out",
                     "too blurry", "too small to read", "illegible")


def _unreadable(text: str) -> bool:
    """True when a vision answer says the value is there but it cannot read it.

    The distinction that matters is against "not applicable": an empty `reading`
    means the goal asked for nothing this screen holds, and re-reading that at
    four times the pixels buys nothing. Only a model that has told us it is
    looking at something it cannot resolve is worth a second, sharper look.
    """
    return any(hint in (text or "").lower() for hint in _UNREADABLE_HINTS)


def needs_screenshot(state: RunState, screen: Screen, cfg: Config) -> Tuple[bool, str]:
    """XML-first: pay for vision only when the tree cannot answer the question."""
    if cfg.run.never_screenshot:
        return False, ""
    if cfg.run.always_screenshot:
        return True, "always"
    if screen.degenerate:
        return True, ("the accessibility tree is nearly empty -- this is a WebView, "
                      "canvas or game, so rely on the screenshot")
    # A gallery item is a bitmap: whatever the goal wants read off it -- a weight
    # on a scale, a price, a name -- exists only in pixels. Guaranteeing one
    # screenshot per *unread item* is what stops the agent swiping blind through
    # an album and inferring, wrongly, which photo it is looking at. It is also
    # self-limiting: revisiting an item already read costs nothing.
    if state.paging and state.content_moved is not False:
        return True, ("a gesture just moved this screen's content, which lives in "
                      "the image rather than the tree -- read what is now shown "
                      "from the screenshot before moving on")
    if state.consecutive_failures >= 1:
        return True, "the last action did not work; look carefully at the screen"
    if (cfg.run.stall_nudge_at
            and state.steps_since_progress >= cfg.run.stall_nudge_at):
        # Being stuck for several steps while every action reports success is
        # usually the tree describing something the pixels contradict -- a
        # control that is drawn disabled, an overlay the dump does not carry.
        return True, ("nothing new has been learned for several steps; look at "
                      "the screen itself rather than at the element list")
    if state.want_screenshot:
        return True, "you said you were unsure last time"
    if screen.ambiguous:
        return True, ("several elements look identical in the list; use the image "
                      "to tell them apart")
    return False, ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, dev: Device, mem: Memory, llm: Optional[LLMClient],
                 cfg: Config, *, oracle: Optional[Oracle] = None,
                 on_event=None, ledger: Optional[ReplyLedger] = None,
                 policy: str = ""):
        self.dev = dev
        self.mem = mem
        self.llm = llm
        self.cfg = cfg
        self.oracle = oracle or Oracle()
        self.on_event = on_event or (lambda *a, **k: None)
        self.skills = SkillRegistry(cfg.skills.skills_dir)
        #: Set by `watch`, left None by `run`. When None no send is gated and no
        #: reply is recorded, so an ordinary run behaves exactly as it always did.
        self.ledger = ledger
        #: The operator's reply instructions, verbatim. Empty for a run.
        self.policy = policy

    # -- perception helpers ------------------------------------------------

    def _ensure_screenshot(self, screen: Screen) -> bytes:
        """The screenshot for `screen`, captured at most once.

        Verification already grabs a screenshot of the screen it lands on, and
        that screen becomes the next turn's `screen`. Capturing again here would
        pay a second device round trip for a frame nothing has touched since --
        and would leave the pager's `dhash` computed from different pixels than
        the ones the model is shown, which is the signal that decides whether a
        swipe advanced the item.

        Every path that acts on the device between the two points -- dismissing
        an interstitial, breaking a loop with `back`, recovering from a device
        error after the action already went out -- clears `screen` to None and
        re-observes, so a screen carrying a screenshot is always current.
        """
        if screen.screenshot is None:
            screen.screenshot = self.dev.screenshot()
        if screen.dhash is None:
            from .fingerprint import compute_dhash
            screen.dhash = compute_dhash(screen.screenshot)
        return screen.screenshot

    def _reread_sharper(self, state: RunState, screen: Screen, rec: Recorder, *,
                        rendered: str = "") -> Optional[ScreenAnalysis]:
        """Read the screen once more at full capture resolution.

        Called only when the vision model has said, in the words the prompt asks
        for, that a value is present but it cannot make it out. The everyday
        capture is downscaled to a 1280 long edge, so on a 1080x2400 phone the
        model is shown 576x1280 -- and a price in 24sp type, a timestamp, a weight
        on a scale reach it at fewer pixels than the app drew them with.

        Bounded on purpose. Once per screen: the second look is answering "is the
        blur mine or the app's", and asking that twice of the same pixels cannot
        change the answer. Nothing is cached back onto `screen.screenshot`, which
        must stay the frame `dhash` was computed from -- the pager compares those
        hashes to decide whether a swipe advanced the item, and swapping in a
        bigger frame would make every comparison read as movement.
        """
        if self.llm is None or self.cfg.run.never_screenshot:
            return None
        key = screen.exact_id or screen.skeleton_id
        if key and key in state.rereads:
            return None
        if key:
            state.rereads.add(key)
        try:
            sharper = self.dev.screenshot(max_long_edge=SHARP_LONG_EDGE, quality=92)
        except (DeviceTimeout, DeviceLost) as exc:
            # A re-read is an improvement, never a reason to lose the step. The
            # main loop's own device calls will hit the same fault and recover.
            log.warning("sharper re-read could not be captured: %s", exc)
            return None
        log.info("step %d: value reported unreadable; re-reading at %dpx "
                 "(%d bytes)", state.step, SHARP_LONG_EDGE, len(sharper))
        rec.event("vision_reread", step=state.step, long_edge=SHARP_LONG_EDGE,
                  bytes=len(sharper))
        self.on_event("vision_reread", step=state.step, long_edge=SHARP_LONG_EDGE)
        analysis = self.llm.analyze_image(
            sharper, goal=state.goal, rendered=rendered, step=state.step,
            recorder=rec, on_event=self.on_event)
        # A second look is only worth taking if it is allowed to fail. Against a
        # real phone the full-resolution frame came back with all four fields
        # empty where the downscaled one had at least said what it could not read
        # -- more pixels are not monotonically more answer. So the re-read has to
        # clear the bar it was bought to clear: an actual value, not another
        # "unreadable" and not silence. Anything else and the first answer stands,
        # because "unreadable, glare on the display" tells the run more than
        # nothing does.
        if analysis.unavailable or not analysis.reading.strip() \
                or _unreadable(analysis.reading):
            rec.event("vision_reread_no_better", step=state.step,
                      reading=analysis.reading)
            log.info("step %d: the sharper frame read no better; keeping the "
                     "first answer", state.step)
            return None
        return analysis

    def _skill_for_run(self, package: str, goal: str) -> Optional[Skill]:
        """The app skill for this step, picked by what the task needs.

        Which apps the goal names is settled once per run -- one
        `pm list packages` -- rather than per step: the answer cannot change
        while the run is on, and a per-step lookup would pay an adb round trip
        for it every turn.
        """
        if not self.skills.skills:
            return None
        if self._goal_apps is None:
            self._goal_apps = goal_app_candidates(self.dev, goal)
        return self.skills.find_for_run(package, goal,
                                        goal_names_app=bool(self._goal_apps))

    # -- public ------------------------------------------------------------

    def run(self, goal: str, run_id: str = "",
            resume: Optional[Dict[str, Any]] = None) -> Tuple[Outcome, RunState]:
        run_id = run_id or uuid.uuid4().hex[:12]
        state = RunState(goal=goal, run_id=run_id, intent_id=intent_key(goal))
        if resume:
            checkpoint.restore(state, resume)
        self._goal_apps: Optional[List[str]] = None
        # Read once per run, not per turn: the string is identical for the whole
        # run, and it sits in the prompt above the goal where a value that
        # changed would evict everything after it from the cache. A run that
        # crosses midnight keeps the date it started on -- which is also how the
        # person who typed "today" meant it. "" if the phone would not say.
        self._today: str = self.dev.today()
        recorder = Recorder(self.cfg, run_id)
        self._log_header(goal, recorder, resumed_from=state.step if resume else 0)
        self.mem.begin_run(run_id, goal, state.intent_id)
        if resume:
            # One directory per run still holds: the events of this sitting are
            # appended to the failed one's, with this event where they join.
            recorder.event("run_resume", goal=goal,
                           model=getattr(self.llm, "model", ""),
                           resumed_at_step=state.step)
        else:
            recorder.event("run_start", goal=goal,
                           model=getattr(self.llm, "model", ""))

        # Mirror the live LLM stream into the run directory as it happens, so
        # the web UI can show the model thinking rather than just its verdicts.
        on_event = self.on_event
        self.on_event = _stream_tap(recorder, on_event)

        try:
            self._loop(state, recorder)
        except (BudgetExceeded, LLMError) as exc:
            log.error("%s", exc)
            # The message on the console, the stack in the run log: which of the
            # dozen `_ask` paths raised is the whole question a day later, and
            # printing a traceback for an expected abort is noise on the day.
            log.debug("aborting on %s", type(exc).__name__, exc_info=True)
            state.finished = "aborted"
            recorder.event("error", error=str(exc))
        except (DeviceLost, DeviceTimeout) as exc:
            log.error("device: %s", exc)
            log.debug("aborting on %s", type(exc).__name__, exc_info=True)
            state.finished = "aborted"
            recorder.event("error", error=str(exc))
        except Aborted as exc:
            log.warning("aborted: %s", exc)
            state.finished = "aborted"
        except KeyboardInterrupt:
            log.warning("interrupted")
            state.finished = "aborted"
        except Exception as exc:  # noqa: BLE001 -- re-raised immediately
            # Not handling it, only recording it. `finally` closes the run log
            # below, so a crash that reached the top of the loop has this one
            # chance to leave the traceback where the run's own files are; the
            # console gets Python's copy on the way out.
            log.debug("unhandled %s", type(exc).__name__, exc_info=True)
            recorder.event("error", error=str(exc),
                           traceback=traceback.format_exc())
            raise
        finally:
            self.on_event = on_event
            outcome = state.finished or "failed"
            # A checkpoint is unfinished business: keep it current on every way
            # out but success, so `--resume` can put the run back where it
            # stopped. Success deletes it -- there is nothing left to continue.
            if outcome == "success":
                checkpoint.clear(self.cfg, run_id)
            else:
                checkpoint.save(self.cfg, state)
            usd = self.llm.ledger.total_usd if self.llm else 0.0
            self.mem.end_run(run_id, outcome, state.step, state.llm_calls, usd)
            recorder.event("run_end", outcome=outcome, steps=state.step,
                           llm_calls=state.llm_calls,
                           usd=round(usd, 6),
                           # The answer, in the one event a reader of this file
                           # is guaranteed to look at. `report` and the web UI
                           # both reconstruct the run from here, and neither
                           # could show what it concluded without it.
                           result=state.result, evidence=state.evidence,
                           # Which apps this run was in and how long it spent in
                           # each, so a later run can find the recorded runs for
                           # *its* app. Nothing else in the file says, and the
                           # signals `history.packages_in` falls back to for runs
                           # recorded before this are much weaker.
                           packages=sorted(state.packages),
                           package_steps=state.package_steps,
                           llm=step_metrics(self.llm.ledger.calls if self.llm else [],
                                            detail=False))
            recorder.close()

        return state.finished or "failed", state

    # -- internals ---------------------------------------------------------

    def _log_header(self, goal: str, rec: Recorder, *,
                    resumed_from: int = 0) -> None:
        """Write who, what and with which settings at the top of the run log.

        All of it is knowable the moment a run starts and none of it is
        recoverable a week later. "Was this the run with `never_screenshot` set?"
        decides whether the rest of the file is surprising or expected, and a
        shell history that has since scrolled away cannot answer it.
        """
        cfg = self.cfg
        runlog.preamble(
            run=rec.dir.name,
            goal=goal,
            # A resumed run's log continues the failed sitting's; this is the
            # only line that says why the first step is not step 1.
            resumed=(f"from step {resumed_from}" if resumed_from else ""),
            artifacts=str(rec.dir),
            # `events.jsonl` timestamps are epoch seconds and these lines are
            # wall clock, so one run needs one number that converts between them.
            epoch=f"{time.time():.3f}",
            adbagent=f"{__version__} (python {sys.version.split()[0]} on "
                     f"{platform.platform()})",
            device=getattr(self.dev, "serial", "") or "(only attached device)",
            # What the run thinks "today" means, which is the phone's date and
            # not this host's. Empty if the phone would not say -- and a goal
            # bounded in time behaves very differently then, so the file has to
            # record which of the two runs this was.
            clock=self._today or "(the phone did not say)",
            model=cfg.llm.model,
            model_image=cfg.llm.image(),
            model_small=cfg.llm.small(),
            model_skill=cfg.llm.skill(),
            model_skill_image=cfg.llm.skill_image(),
            reasoning=f"effort={cfg.llm.reasoning_effort or '(model default)'} "
                      f"hard={cfg.llm.reasoning_effort_hard} "
                      f"style={cfg.llm.reasoning_style} "
                      f"vision_in_decider={cfg.llm.decider_sees()}",
            limits=f"max_steps={cfg.run.max_steps} "
                   f"max_wall_clock_s={cfg.run.max_wall_clock_s:g} "
                   f"budget_usd={cfg.safety.budget_usd:g}",
            # A run that stopped at tier 4 reads as an ordinary failure unless
            # the file says which ladder it was climbing.
            stall=f"nudge={cfg.run.stall_nudge_at} "
                  f"block={cfg.run.stall_block_at} "
                  f"replan={cfg.run.stall_replan_at} "
                  f"give_up={cfg.run.stall_give_up_at} "
                  f"max_consecutive_failures={cfg.run.max_consecutive_failures}",
            vision=f"always={cfg.run.always_screenshot} "
                   f"never={cfg.run.never_screenshot}",
            pager=f"sweep={cfg.run.pager_sweep} max={cfg.run.pager_sweep_max}",
            flags=f"dry_run={cfg.run.dry_run} "
                  f"unattended={cfg.safety.unattended} "
                  f"allow_destructive={cfg.safety.allow_destructive} "
                  f"learn_after_run={cfg.skills.learn_after_run}",
            # An assertion ends the run without a judge call, so a reader wonders
            # why the last step has no verdict unless the file says one was set.
            oracle=(f"shell={self.oracle.shell!r} equals={self.oracle.equals!r} "
                    f"text={self.oracle.text!r}" if self.oracle.defined else ""),
        )

    def _loop(self, state: RunState, rec: Recorder) -> None:
        cfg = self.cfg
        screen: Optional[Screen] = None
        while state.finished is None:
            if state.step >= cfg.run.max_steps:
                log.error("step budget (%d) exhausted", cfg.run.max_steps)
                state.finished = "failed"
                return
            if state.elapsed > cfg.run.max_wall_clock_s:
                log.error("wall-clock budget exhausted")
                state.finished = "failed"
                return
            # The resume point: everything the run knows, through the last
            # *completed* step. Written here rather than only on the way out so
            # a run that dies mid-step -- a hang, a kill, a crash -- still
            # leaves something `--resume` can pick up.
            checkpoint.save(cfg, state)
            state.step += 1
            # Counted up here, at the top, and reset wherever progress is found
            # below. Every `continue` in this loop is a step that learned nothing
            # -- a rejected action, a failed one, a dismissed nag -- and counting
            # at the bottom instead would let all of them through for free.
            state.steps_since_progress += 1

            # ---- 1. perceive (no LLM) -----------------------------------
            if screen is None:
                t0_perceive = time.monotonic()
                try:
                    screen = self.dev.observe()
                except (DeviceTimeout, DeviceLost) as exc:
                    if not self._recover_device(state, exc):
                        state.finished = "aborted"
                        return
                    continue
                self.on_event("perceive", step=state.step, elapsed=time.monotonic() - t0_perceive)
            if screen.package:
                if screen.package not in state.packages:
                    state.note_progress(f"it reached {screen.package}")
                state.packages.add(screen.package)
                state.package_steps[screen.package] = \
                    state.package_steps.get(screen.package, 0) + 1

            # A programmatic assertion is the cheapest and most reliable way to
            # know we are done, so it is checked before anything else happens.
            if self.oracle.defined and self.oracle.satisfied(self.dev, screen):
                log.info("assertion satisfied -- goal reached")
                # It ends the run before the model is asked anything, so there
                # is no `done` text and never will be. Saying which check passed
                # is the whole answer this path has, and without it the summary
                # would report a success with nothing at all to show for it.
                state.evidence = f"the success check passed: {self.oracle.describe()}"
                state.finished = "success"
                return

            # ---- 2. guards that need no model ---------------------------
            finding = safety.sensitive_screen(screen)
            if finding is not None:
                rec.event("sensitive", reason=finding.reason)
                self._hand_over(state, finding.reason)
                return

            interstitial = safety.find_interstitial(screen, screen.package)
            if interstitial is not None:
                label = interstitial.best_text
                # Keyed on `exact_id` rather than `skeleton_id`: the skeleton is
                # content-free, so a second, *different* nag card hashes the same
                # as the one just dismissed and would be refused.
                sig = f"{screen.exact_id}/{label}"
                if sig != state.last_dismiss:
                    state.last_dismiss, state.dismiss_tries = sig, 0
                state.dismiss_tries += 1
                if state.dismiss_tries <= MAX_DISMISS_TRIES:
                    log.info("step %d: dismissing %r", state.step, label)
                    rec.event("dismiss", label=label, attempt=state.dismiss_tries)
                    # In the history too, not just the artifact: the model is
                    # about to be handed a screen transition it did not cause,
                    # and nothing else would tell it why.
                    state.remember(f"{state.step}. the harness dismissed {label!r}")
                    self.dev.tap(*interstitial.center)
                    screen = None
                    continue
                if state.dismiss_tries == MAX_DISMISS_TRIES + 1:
                    # Said once, on the way in. Repeating it every turn would
                    # overwrite whatever the model's own last action had to say.
                    log.warning("step %d: %r dismisses nothing; handing the "
                                "screen to the model", state.step, label)
                    rec.event("dismiss_failed", label=label,
                              tries=MAX_DISMISS_TRIES)
                    state.last_failure = (
                        f"the harness tapped {label!r} to dismiss it and the "
                        f"screen did not change, so it is part of this screen "
                        f"rather than a popup over it. Decide what to do with "
                        f"it yourself.")

            # ---- 2b. is this screen paging? -----------------------------
            # Not asked of the screen -- answered by what the last gesture did.
            # A screen stops counting as paging the moment the agent navigates
            # away from it, because the evidence was about *that* screen.
            if screen.skeleton_id != state.last_skeleton:
                state.paging = False
                state.content_moved = None
            state.last_skeleton = screen.skeleton_id

            hint = state.loops.hint(screen.skeleton_id)

            # Scroll awareness: give the LLM full context about its
            # scrolling pattern so it can course-correct on its own. Suppressed
            # while the screen is known to page, where repeating one gesture is
            # the way forward and "you keep scrolling the same way" is noise.
            scroll_ctx = "" if state.paging else state.loops.scroll_context()
            if scroll_ctx:
                hint = scroll_ctx
                if (state.loops.scroll_oscillating()
                        or state.loops.direction_reversals() >= 5):
                    # Only the gestures recorded on *this* screen. `history`
                    # spans every screen of the last twenty steps, so banning
                    # everything in it pinned a scroll this screen had never seen
                    # -- and swipes were skipped entirely, though
                    # `scroll_oscillating` counts them.
                    for sid, sig in state.loops.history:
                        if sid != screen.skeleton_id:
                            continue
                        if not sig.startswith(("scroll/", "swipe/")):
                            continue
                        state.loops.ban(screen.skeleton_id, sig)

            if state.loops.should_force_back(screen.skeleton_id) or state.loops.oscillating():
                if state.loops.in_back_loop():
                    # Pressing back repeatedly is not helping; let the LLM
                    # try a different approach this turn.
                    log.warning("step %d: back-loop detected (%d consecutive "
                                "backs); falling through to LLM",
                                state.step, state.loops.consecutive_backs)
                    rec.event("back_loop_escape", exact_id=screen.exact_id,
                              consecutive_backs=state.loops.consecutive_backs)
                    extra = ("Pressing back repeatedly has not helped. You MUST "
                             "try a completely different approach — tap a "
                             "different element, scroll, use search, or report "
                             "done/fail.")
                    hint = f"{hint} {extra}" if hint else extra
                    state.loops.consecutive_backs = 0
                    # No ban here. This used to ban "forced-back", which is not a
                    # signature any action can produce -- `AgentAction.signature`
                    # cannot emit it -- so it blocked nothing and put one
                    # meaningless entry in the list of banned actions the model
                    # is shown.
                else:
                    log.warning("step %d: stuck in a loop; going back",
                                state.step)
                    self.on_event("loop_warning", message=f"step {state.step}: stuck in a loop; going back")
                    rec.event("loop_break", exact_id=screen.exact_id)
                    self.dev.press("back")
                    state.loops.record(screen.skeleton_id, "forced-back")
                    state.loops.consecutive_backs += 1
                    screen = None
                    continue

            # ---- 3. visit tracking --------------------------------------
            visit = state.visits.get(screen.skeleton_id, 0)
            state.visits[screen.skeleton_id] = visit + 1
            if visit == 0:
                state.note_progress("it reached a screen it had not seen before")

            # ---- 3b. the stall ladder -----------------------------------
            # Everything above this point can reset the counter, so this is the
            # first place that knows whether the run is actually getting
            # anywhere. The tiers run cheap to expensive: say something, refuse
            # something, spend a call rethinking, stop. See `config.RunConfig`.
            stalled = state.steps_since_progress
            limits = cfg.run

            if limits.stall_give_up_at and stalled >= limits.stall_give_up_at:
                log.error("no progress for %d steps (last: %s); giving up",
                          stalled, state.last_progress)
                rec.event("stalled_out", step=state.step, stalled=stalled,
                          last_progress=state.last_progress)
                self.on_event("loop_warning",
                              message=f"step {state.step}: nothing new for "
                                      f"{stalled} steps; stopping")
                state.finished = "failed"
                return

            # Actions already tried on this screen more than once. Twice with
            # nothing learned is evidence; once is not, so a single previous
            # attempt stays legal and the model keeps somewhere to go.
            refused: set = set()
            if limits.stall_block_at and stalled >= limits.stall_block_at:
                refused = {sig for sig, n in state.loops.tried_on(screen.skeleton_id)
                           if n >= 2}

            if (limits.stall_replan_at and self.llm is not None
                    and stalled >= limits.stall_replan_at
                    and stalled - state.replanned_at >= limits.stall_replan_at):
                state.replanned_at = stalled
                if self._replan(state, rec, screen, stalled) is False:
                    return

            # ---- 4. ask the model ---------------------------------------
            screenshot: Optional[bytes] = None
            want, note = needs_screenshot(state, screen, cfg)
            if want:
                screenshot = self._ensure_screenshot(screen)

            # The vision pass runs here rather than inside `decide` so its
            # structured fields reach the ledgers below: a `reading` the image
            # model took off this frame is the fact the run is collecting, and
            # routing it through prose for the decider to re-extract loses it the
            # moment the decider paraphrases.
            #
            # The step's cost mark and clock both start *before* it, because a
            # screenshot turn is an analysis and then a decision, and charging the
            # step for only the second one is how a two-call turn reads as a
            # one-call turn -- and how `report`'s latency/step quietly loses the
            # vision time on the ~22% of turns that take a screenshot.
            ledger_mark = self.llm.ledger.mark() if self.llm else 0
            t0_step_llm = time.monotonic()
            analysis: Optional[ScreenAnalysis] = None
            vision_note = ""
            if (screenshot and self.llm is not None
                    and self.llm.needs_vision_pass):
                analysis = self.llm.analyze_image(
                    screenshot, goal=state.goal, rendered=render(screen),
                    step=state.step, recorder=rec, on_event=self.on_event)
                # `note` was written on the assumption the image would arrive --
                # "rely on the screenshot", "look at the screen itself". When the
                # vision call failed, that instruction is now a lie, and a decider
                # told to consult evidence it does not have will describe pixels
                # it never saw. Withdraw it in the same breath.
                if analysis.unavailable:
                    vision_note = (
                        "The screenshot could NOT be read this turn -- the vision "
                        "model did not answer. Ignore any instruction above to "
                        "rely on the image: decide from the element list alone, "
                        "and say your confidence is low if the list cannot "
                        "settle it.")
                    rec.event("vision_unavailable", step=state.step,
                              model=self.llm.model_image)
                elif analysis.reading and _unreadable(analysis.reading):
                    # The prompts ask for "unreadable" by name and nothing used to
                    # act on it, so the run carried on as if the value were simply
                    # absent. A downscaled capture is the most likely reason: a
                    # 1080x2400 frame reaches the model as 576x1280, which is 47%
                    # of the linear resolution small print was rendered at.
                    sharper = self._reread_sharper(
                        state, screen, rec, rendered=render(screen))
                    if sharper is not None:
                        analysis = sharper

            # Whatever the last mechanical sweep read, handed back once and then
            # dropped. It is a record of what *this* run saw, in order, with no
            # claim about what else exists -- the model is told to copy anything
            # it needs into `notes`, which is the memory that actually persists.
            pager_note = state.sweep.render()
            if pager_note:
                state.sweep.start(state.sweep.gesture)  # handed over; clear it

            # Two sources, one note. `loops` remembers what failed in this run;
            # `mem.dead_ends` remembers what failed in *earlier* runs on this
            # screen for this goal, which is the only knowledge here that outlives
            # the process. Recording those and never reading them back meant
            # rediscovering the same dud control on every run.
            banned_actions = set(state.loops.bans_for(screen.skeleton_id))
            remembered = self.mem.dead_ends(screen, state.intent_id)
            ban_note = ""
            if banned_actions or remembered:
                lines = []
                if banned_actions:
                    lines.append(
                        "BANNED ACTIONS on this screen (these produced NO change "
                        f"- DO NOT REPEAT): {', '.join(sorted(banned_actions))}.")
                fresh = {sig: why for sig, why in remembered.items()
                         if sig not in banned_actions}
                if fresh:
                    lines.append(
                        "KNOWN DEAD ENDS here from earlier runs (do not repeat "
                        "them): " + "; ".join(
                            f"{sig} ({why})" for sig, why in sorted(fresh.items()))
                        + ".")
                ban_note = "\n".join(lines)
                rec.event("dead_ends", step=state.step,
                          this_run=sorted(banned_actions),
                          remembered=sorted(remembered))
            # Check for active app skill guidance
            skill_note = ""
            if cfg.skills.enabled:
                active_skill = self._skill_for_run(screen.package, state.goal)
                if active_skill:
                    skill_note = active_skill.to_prompt_text()
                    # The skill's own package, not the screen's: a goal-named
                    # skill loads before the run reaches its app, and reporting
                    # the foreground here reads as "this skill is about the app
                    # on screen" -- the misattribution history.packages_in
                    # warns about.
                    skill_pkg = (active_skill.packages[0] if active_skill.packages
                                 else screen.package)
                    rec.event("active_skill", name=active_skill.name, package=skill_pkg)
                    if getattr(self, "_active_skill_name", None) != active_skill.name:
                        self._active_skill_name = active_skill.name
                        self.on_event("skill_loaded", name=active_skill.name, package=skill_pkg)

            # `repeatable` names a target whose repetition is legitimate. It
            # used to be "the pager element" -- a classification. It is now the
            # target of the gesture that was *observed* to move content, which is
            # the same claim without the guess.
            elem_hint = state.loops.element_history_hint(
                screen.skeleton_id,
                repeatable=state.repeatable_index if state.paging else 0)
            # Advice that only applies sometimes lives here rather than in the
            # system prompt: this block is rebuilt every turn anyway, so varying
            # it is free, whereas varying the system message evicts the whole
            # prompt prefix from the provider's cache.
            situational = prompts.situational_notes(
                scrolls=state.loops.total_scroll_count,
                packages_seen=len(state.packages))
            # Placed ahead of the older hints on purpose: when the run has
            # stopped getting anywhere, that is the most important thing on the
            # turn, and it is the only block that names the actions the harness
            # has begun refusing outright.
            stall_text = ""
            if limits.stall_nudge_at and stalled >= limits.stall_nudge_at:
                stall_text = prompts.stall_note(
                    stalled, tried=state.loops.tried_on(screen.skeleton_id),
                    refused=sorted(refused), strategy=state.strategy)
            # `skill_note` is deliberately not in here: it goes to its own
            # message above the history instead, because it changes per app
            # rather than per turn. See `prompts.skill_block`.
            # Which conversations are already answered. Advisory only -- the
            # guarantee is `conversation.reply_gate` -- and it rides in `notes`
            # rather than beside the policy because it changes as replies go out,
            # and this block is rebuilt every turn anyway.
            handled_note = ""
            if self.ledger is not None:
                handled_note = prompts.handled_block(
                    [st.preview for st in self.ledger.recent() if st.preview])
            notes = "\n\n".join(filter(None, (note, vision_note, stall_text,
                                             pager_note, hint, elem_hint,
                                             ban_note, state.last_failure,
                                             handled_note, situational)))
            effort, hard_because = needs_reasoning(
                state, cfg, visit=visit,
                blocked=bool(banned_actions or remembered), hint=hint)
            if hard_because:
                log.info("step %d: thinking harder (%s) because %s",
                         state.step, effort, hard_because)
            model_name = self.llm.model if self.llm else ""
            # The frame reaches the decider itself only when the decider is the
            # model doing the looking. Otherwise the vision pass above is the call
            # that was shown it, and its own panel already carries the image.
            shot = rec.screenshot(state.step, screenshot, "decide") \
                if (screenshot and cfg.llm.decider_sees()) else ""
            self.on_event("llm_start", step=state.step, purpose="decide", model=model_name,
                          screenshot=bool(screenshot), shot=shot, effort=effort,
                          hard_because=hard_because)
            t0_llm = time.monotonic()
            action = self.llm.decide(                      ### LLM ###
                goal=state.goal, rendered=render(screen), history=state.history,
                width=screen.width, height=screen.height, package=screen.package,
                today=self._today,
                screenshot=screenshot, note=notes,
                scratchpad=state.scratchpad.render(cfg.run.scratchpad_max_chars),
                progress="\n".join(state.progress_log), skill=skill_note,
                policy=self.policy,
                # `is not None`, not truthiness: a pydantic model is always truthy
                # so this happened to be right, but the distinction it relies on
                # is between "the pass ran" and "it did not". `decide` reads "" as
                # the former and would buy a second look for the latter.
                image_analysis=analysis.render() if analysis is not None else None,
                step=state.step, recorder=rec, effort=effort,
                on_event=self.on_event)
            t_llm = time.monotonic() - t0_llm            # the decision alone
            t_step_llm = time.monotonic() - t0_step_llm  # + any vision pass
            step_calls = self.llm.ledger.since(ledger_mark) if self.llm else []
            last_call = step_calls[-1] if step_calls else None
            self.on_event("llm_end", step=state.step, purpose="decide", elapsed=t_llm, call=last_call)
            state.llm_calls += 1
            state.want_screenshot = action.confidence == "low"
            source = "llm"

            # Track scroll direction globally (survives interleaved taps).
            if action.action in ("scroll", "swipe") and action.direction:
                state.loops.record_scroll(action.direction)

            # Last-resort guard: if the LLM was given full scroll context
            # multiple times and still insists on scrolling, reject it.
            # Now also triggers on direction reversals (>=5), not just
            # strict consecutive oscillation.
            scroll_blocked = False
            if action.action in ("scroll", "swipe"):
                # Judged per axis. A run that scrolled a chat up and down
                # repeatedly has not earned the right to refuse the "next photo"
                # swipe that comes later, and refusing it leaves the agent with
                # no legal way to finish the album.
                axis = ("horizontal" if action.direction in ("left", "right")
                        else "vertical")
                axis_log = state.loops.axis_log(axis)
                if state.loops.scroll_oscillating():
                    scroll_blocked = True
                elif (state.loops.direction_reversals(axis) >= 5
                      and action.direction
                      and len(axis_log) >= 2):
                    # Check if this scroll would be yet another reversal.
                    from .safety import _SCROLL_OPPOSITES
                    # `record_scroll` above already appended this action, so the
                    # previous direction on this axis is the second-to-last entry.
                    if action.direction == _SCROLL_OPPOSITES.get(axis_log[-2], ""):
                        scroll_blocked = True

            if scroll_blocked:
                state.scroll_warnings += 1
                if state.scroll_warnings >= 3:
                    log.warning("step %d: rejecting scroll after %d warnings",
                                state.step, state.scroll_warnings)
                    self.on_event("loop_warning", message=f"step {state.step}: scroller stuck; rejecting scroll action")
                    rec.event("scroll_rejected", step=state.step,
                              action=action.describe())
                    state.last_failure = (
                        "scrolling was blocked because you have been "
                        "alternating directions or reversing despite being "
                        "told to stop. Commit to one direction, do something "
                        "else, or report done/fail.")
                    state.consecutive_failures += 1
                    state.remember(
                        format_history_entry(
                            state.step, action, screen=screen,
                            grade="rejected",
                            reason=f"scroll reversal, warning #{state.scroll_warnings}"
                        )
                    )
                    self._maybe_give_up(state)
                    continue

            # -- accumulate scratchpad notes --------------------------------
            # The model sends only what is new or corrected; the union is kept
            # here. Nothing it stops mentioning can be lost, because nothing is
            # being replaced -- which is what the previous contract, where the
            # model rewrote the whole ledger every turn, could not promise.
            if getattr(action, "notes", None):
                written = state.scratchpad.update(action.notes, state.step)
                if written:
                    rec.event("scratchpad", step=state.step, keys=written,
                              total=len(state.scratchpad))
                    # The clearest progress signal there is on a collection goal:
                    # a record the run did not have a moment ago. `update`
                    # returns only the keys that were new or corrected, so a
                    # model restating what it already sent cannot buy time here.
                    state.note_progress(f"it recorded {len(written)} new "
                                        f"data record(s)")

            # -- attach what the model read off this item --------------------
            # The observation describes the screen the model was just shown, so
            # it belongs to the item that was on it. Stored per item, it survives
            # the scratchpad being rewritten and cannot be dropped by omission.
            #
            # Only recorded on a turn that had a screenshot. Without one the model
            # cannot have seen the item's content, so its observation is about the
            # chrome ("the caption is hidden") -- and letting that overwrite the
            # reading taken a turn earlier would throw away the one thing worth
            # keeping.
            #
            # A `reading` from the vision pass already went in above and wins: the
            # decider's observation is its restatement of that same reading, and a
            # restatement is where a figure gets rounded or paraphrased away. The
            # observation is what there is when the decider has its own eyes and
            # there was no separate pass.
            # -- accumulate progress ----------------------------------------
            if getattr(action, "progress", None):
                prog_text = action.progress.strip()
                if prog_text:
                    if state.progress_log[:1] != [prog_text]:
                        state.note_progress("it updated its own progress note")
                    state.progress_log = [prog_text]

            rec.event("decide", step=state.step, source=source,
                      skeleton=screen.skeleton_id,
                      # The hash the loop detector no longer runs on, kept
                      # because change detection still does and because a trace
                      # that records only one of the two cannot answer "did the
                      # screen really repeat" after the fact.
                      exact=screen.exact_id,
                      stalled=stalled, action=action.model_dump(),
                      screenshot=bool(screenshot),
                      effort=effort, hard_because=hard_because,
                      wall_s=round(t_step_llm, 3), llm=step_metrics(step_calls))
            self.on_event("step", state=state, screen=screen, action=action,
                          source=source, screenshot=bool(screenshot))

            # ---- 5. guard the chosen action -----------------------------
            label = safety.irreversible(action, screen)
            if label is not None:
                self.on_event("safety_warning", message=f"step {state.step}: irreversible action {label!r} in {screen.package}")
                if not safety.confirm(
                        f"Step {state.step}: the agent wants to press {label!r} "
                        f"in {screen.package}. This cannot be undone.", cfg):
                    state.remember(
                        f"{state.step}. refused to press {label!r}"
                        + (f" in {screen.package}" if screen.package else "")
                    )
                    state.last_failure = (f"pressing {label!r} was refused; find "
                                          f"another way or stop")
                    rec.event("refused", label=label)
                    continue

            # Tier 2 of the stall ladder. Telling the model to stop repeating
            # itself is known not to be enough: in `runs/2521862d7a23` the
            # element-history hint said exactly that on ten consecutive turns and
            # the model tapped the same index ten more times. So past a point the
            # harness stops asking and refuses, the way `scroll_blocked` above
            # already does for gestures.
            #
            # Terminal actions are never refused. `done`, `fail` and `ask_user`
            # are the ways out of a stall, and a guard that blocked them would be
            # sealing the exit it is trying to push the agent through.
            if refused and not action.is_terminal \
                    and action.signature() in refused:
                sig = action.signature()
                tries = state.loops.times_on(screen.skeleton_id, sig)
                log.warning("step %d: refusing %s -- tried %d times here and the "
                            "run has learned nothing for %d steps",
                            state.step, sig, tries, stalled)
                self.on_event("loop_warning",
                              message=f"step {state.step}: refusing {sig} "
                                      f"(tried {tries}x here, no progress)")
                rec.event("stall_block", step=state.step, action=sig,
                          tries=tries, stalled=stalled)
                state.last_failure = (
                    f"{sig} was refused: you have already done it {tries} times "
                    f"on this screen and the run has learned nothing in "
                    f"{stalled} steps. It will keep being refused. Choose "
                    f"something you have not tried, or report done/fail.")
                state.loops.ban(screen.skeleton_id, sig)
                state.remember(format_history_entry(
                    state.step, action, screen=screen, grade="refused",
                    reason=f"already tried {tries}x here with no progress"))
                # Deliberately not counted as a `consecutive_failure`. That
                # counter has its own terminator, and letting a stall feed it
                # would end the run at `max_consecutive_failures` -- four blocked
                # turns -- before the replan tier below ever got a chance to run.
                continue

            if action.is_terminal:
                state.finished = self._terminal(state, screen, action, rec)
                if state.finished is None:
                    screen = None
                    continue
                return

            if cfg.run.dry_run:
                log.info("dry run: would %s", action.describe())
                state.remember(
                    format_history_entry(state.step, action, screen=screen, prefix="(dry run)")
                )
                continue

            # ---- 5b. the never-double-reply gate ------------------------
            # The harness half of the guarantee. The prompt also lists what has
            # been handled, but a prompt is advice; this runs on the very screen
            # the gesture is about to land on, and it is what a model that has
            # talked itself into answering the same message twice runs into.
            #
            # Placed here rather than beside the other guards so that the two
            # things between -- the stall block and the dry-run short circuit --
            # cannot leave an attempt recorded for a gesture that never went out.
            # `self.ledger` is None for an ordinary run, which is what leaves
            # `adbagent run` behaving exactly as it did.
            pending_reply: Optional[conversation.Conversation] = None
            if self.ledger is not None:
                verdict = conversation.reply_gate(action, screen, self.ledger, cfg)
                if not verdict:
                    log.warning("step %d: not sending -- %s",
                                state.step, verdict.reason)
                    self.on_event("safety_warning",
                                  message=f"step {state.step}: send refused -- "
                                          f"{verdict.reason}")
                    rec.event("send_refused", step=state.step,
                              reason=verdict.reason)
                    state.last_failure = (
                        f"the reply was not sent: {verdict.reason}. Do not try to "
                        f"send it again -- leave this conversation and deal with "
                        f"another one, or report done.")
                    state.remember(format_history_entry(
                        state.step, action, screen=screen, grade="refused",
                        reason=verdict.reason))
                    continue
                if conversation.send_label(action, screen):
                    convo = conversation.read_conversation(screen)
                    if convo.readable:
                        # Written *before* the gesture, on purpose: a record made
                        # afterwards is one a crash between the tap and the write
                        # can lose, and a lost record is a second reply. The price
                        # of this ordering is that a send which never lands leaves
                        # the thread in doubt, which is what the ledger's long
                        # cooldown exists to absorb.
                        self.ledger.record_attempt(convo.key, convo.digest,
                                                   convo.preview())
                        pending_reply = convo
                        rec.event("reply_attempt", step=state.step,
                                  thread=convo.key, digest=convo.digest,
                                  preview=convo.preview())

            # ---- 6. act -------------------------------------------------
            t0_act = time.monotonic()
            # "Did this gesture move the content" is answered by comparing two
            # frames, so the *before* frame has to exist before the gesture goes
            # out. A capture is a device round trip, not an LLM call, and without
            # it a directional gesture can never accumulate the evidence that
            # authorises repeating it -- the sweep would simply never start.
            if (action.action in ("scroll", "swipe") and action.direction
                    and cfg.run.pager_sweep and not cfg.run.never_screenshot):
                self._ensure_screenshot(screen)
            try:
                element = execute(self.dev, action, screen)
            except (ActionError, ValueError) as exc:
                log.warning("step %d: %s", state.step, exc)
                state.last_failure = str(exc)
                state.consecutive_failures += 1
                state.remember(
                    format_history_entry(
                        state.step, action, screen=screen,
                        grade="failed", reason=str(exc)
                    )
                )
                self._maybe_give_up(state)
                continue
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
                # The gesture went out before the device went quiet, and
                # `recover` may have restarted the uiautomator server on top of
                # that, so the frame in hand is at least one action out of date.
                # Dropping it is what sends the next turn back through
                # `observe` -- deciding from it taps coordinates that moved.
                screen = None
                continue
            self.on_event("act_end", step=state.step, action=action, elapsed=time.monotonic() - t0_act)

            # ---- 7. verify (no LLM) -------------------------------------
            self.on_event("settle_start", step=state.step, budget=cfg.device.settle_budget_s)
            t0_verify = time.monotonic()
            try:
                after = self.dev.observe(settle=True)
                if want or action.action in ("scroll", "swipe") or state.want_screenshot:
                    # Also the screenshot the *next* turn will show the model, if
                    # it wants one -- `_ensure_screenshot` will not re-take it.
                    self._ensure_screenshot(after)
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
                screen = None       # the action landed; re-read the phone
                continue
            t_settle = time.monotonic() - t0_verify
            post = synthesise_postcondition(action, element)
            outcome = verify(action, screen, after, post, None)

            # A ViewPager quietly drops a fling it judges too slow or too short,
            # and the item does not move. One stronger attempt here is far
            # cheaper than an LLM turn spent rediscovering that -- and before the
            # item-identity signal existed, the drop was not even detectable.
            if (outcome.grade == "no_change" and state.paging
                    and action.action in ("scroll", "swipe")
                    and action.direction):
                retry = action.model_copy(update={
                    "action": "swipe", "duration": 0.12,
                    "scroll_amount": min(5.0, max(2.0, action.scroll_amount * 2)),
                })
                log.info("step %d: content did not move; retrying harder",
                         state.step)
                rec.event("pager_retry", step=state.step, action=retry.describe())
                try:
                    execute(self.dev, retry, screen)
                    after = self.dev.observe(settle=True)
                    self._ensure_screenshot(after)
                    outcome = verify(retry, screen, after, post, None)
                except (ActionError, ValueError) as exc:
                    log.warning("step %d: pager retry failed: %s", state.step, exc)
                except (DeviceTimeout, DeviceLost) as exc:
                    if not self._recover_device(state, exc):
                        state.finished = "aborted"
                        return
                    screen = None   # the retry landed; re-read the phone
                    continue

            # Did this gesture page the screen? Answered by observation, and it
            # is the only thing that makes the sweep below legal. Note the
            # direction is not restricted: a vertical feed pages exactly like a
            # horizontal album, and refusing to notice that was what kept the
            # sweep off every short-video surface it would have helped on.
            if action.action in ("scroll", "swipe") and action.direction:
                # Asked of the pixels directly rather than read off `grade`.
                # `verify` falls back to "a swipe probably worked" when it has no
                # image to check, which is a fair default for grading an action
                # but not evidence, and `can_repeat` below turns this into the
                # authority to act thirty more times without asking.
                state.content_moved = pager_content_moved(screen, after)
                if state.content_moved and after.skeleton_id == screen.skeleton_id:
                    state.paging = True
                    state.repeatable_index = (element.index if element is not None
                                              else 0)
                    state.paging_gesture = (action.action, action.direction)

            self.on_event("verify_end", step=state.step, elapsed=t_settle, grade=outcome.grade, reason=outcome.reason)

            rec.event("verify", step=state.step, grade=outcome.grade,
                      reason=outcome.reason, after=after.skeleton_id)

            # The post-send tail, now that our own message has joined it. Two
            # jobs: it is what stops the next poll from reading our own reply as
            # new incoming content, and it lifts the thread out of the doubt that
            # `record_attempt` deliberately left it in.
            #
            # A reply that cannot be confirmed on the screen it landed on is left
            # in doubt rather than assumed sent -- the long cooldown then keeps
            # anything else out of that conversation until a human has looked.
            if pending_reply is not None:
                landed = conversation.read_conversation(after)
                digest = (landed.digest
                          if landed.readable and landed.key == pending_reply.key
                          else "")
                if digest:
                    self.ledger.record_confirmed(pending_reply.key, digest,
                                                 landed.preview())
                    rec.event("reply_confirmed", step=state.step,
                              thread=pending_reply.key, digest=digest)
                else:
                    log.warning("step %d: the reply to %r could not be confirmed "
                                "on the screen after it -- that conversation now "
                                "gets the long cooldown",
                                state.step, pending_reply.title)
                    self.on_event("safety_warning",
                                  message=f"step {state.step}: reply to "
                                          f"{pending_reply.title!r} unconfirmed")
                    rec.event("reply_unconfirmed", step=state.step,
                              thread=pending_reply.key)

            # ---- 8. learn (no LLM) --------------------------------------
            # Two more ways a step can count as progress, both about the device
            # rather than about what the model said. A gesture that moved content
            # revealed something that was not on screen before -- that is what
            # keeps a long feed search from reading as a stall. And an action
            # that changed device state did the thing it was for, whether or not
            # it navigated anywhere: typing into a field and flipping a toggle
            # both leave the screen looking much as it did.
            if action.action in ("scroll", "swipe"):
                # Not read off `grade`. `verify` answers "probably" for a swipe
                # it has no image to check -- a fair default for grading one
                # action, but as a progress signal it is a hole: under
                # `never_screenshot` every swipe would buy another step and a
                # run flinging at a wall could never stall. Pixels when there
                # are pixels, the tree when there are not.
                moved = state.content_moved
                if moved is None:
                    moved = after.exact_id != screen.exact_id
                if moved:
                    state.note_progress("a gesture revealed new content")
            elif outcome.ok and (action.action == "input_text"
                                 or (element is not None and element.checkable)):
                state.note_progress("it changed something on the device")

            if not outcome.ok:
                state.consecutive_failures += 1
                state.last_failure = f"{action.describe()} failed: {outcome.reason}"
                state.want_screenshot = True
                self.mem.record_dead_end(screen, state.intent_id,
                                         action.signature(), outcome.reason)

            if outcome.ok:
                state.consecutive_failures = 0
                state.last_failure = ""
                state.loops.consecutive_backs = 0
            if outcome.grade == "no_change":
                if action.action not in ("scroll", "swipe"):
                    state.loops.ban(screen.skeleton_id, action.signature())
                if action.action in ("scroll", "swipe"):
                    h_dir = action.direction in ("left", "right")
                    axis = "horizontal" if h_dir else "vertical"
                    act_name = "Swiping" if action.action == "swipe" else "Scrolling"
                    if state.paging:
                        # Say only what was seen. This used to add a verdict from
                        # the ledger -- "every item has been read", "4 items are
                        # still unread" -- which was a claim about a set nothing
                        # could actually count.
                        state.last_failure = (
                            f"{act_name} {action.direction} twice did not change "
                            f"the content, so that gesture no longer advances "
                            f"here. Try the opposite direction, or leave this "
                            f"screen.")
                    else:
                        state.last_failure = (
                            f"{act_name} {action.direction} did not reveal new "
                            f"content \u2014 you have reached the end of the "
                            f"{axis} scrollable area. Do not {action.action} "
                            f"{action.direction} again here.")

            state.loops.record(screen.skeleton_id, action.signature())
            state.loops.record_element_action(
                screen.skeleton_id, state.step, action.signature(), action.describe(element=element)
            )
            state.remember(
                format_history_entry(
                    state.step, action, screen=screen, element=element,
                    grade=outcome.grade, reason=outcome.reason
                )
            )

            self._maybe_give_up(state)
            screen = after

            # ---- 9. keep going, if the rest is mechanical ----------------
            # The model has just chosen to page through a set and the item moved.
            # Repeating that decision in code costs a vision read per item
            # instead of a reasoning turn per item.
            if (cfg.run.pager_sweep and state.finished is None
                    and not cfg.run.never_screenshot  # nothing to read with
                    and can_repeat(action=action.action,
                                   direction=action.direction or "",
                                   moved=state.content_moved)):
                swept = self._sweep_pager(state, rec, screen, action,
                                          element)
                if swept is not None:
                    screen = swept

    # -- repeating a gesture that works -------------------------------------

    def _sweep_pager(self, state: RunState, rec: Recorder, screen: Screen,
                     action: AgentAction, element) -> Optional[Screen]:
        """Repeat the gesture the model just made, for as long as it keeps working.

        Entered only from `can_repeat`, so the model has already chosen this
        gesture on this screen and it has already been *seen* to move the app's
        content. Each iteration reads what is currently shown, repeats the
        gesture, and checks whether anything moved. The read is started before
        the gesture and collected after it, so a ~1.5s vision call overlaps the
        swipe and the settle rather than following them.

        There is no set being walked here and no ledger being filled. The loop
        runs while the gesture keeps changing the content and stops when it does
        not -- which is the same rule for a photo album, a horizontal card stack
        and a vertical video feed, none of which it needs to tell apart.

        Returns the screen the sweep ended on, or None if it did nothing.
        """
        cfg = self.cfg
        first_step = state.step + 1
        direction = action.direction or ""
        gesture_name = f"{action.action} {direction}".strip()
        swept = 0
        read = 0
        reason = ""
        package = screen.package
        state.sweep.start(gesture_name)

        while True:
            reason = stop_repeating(screen, package=package,
                                    moved=state.content_moved)
            if reason:
                break
            if swept >= cfg.run.pager_sweep_max:
                reason = f"the {cfg.run.pager_sweep_max}-repeat limit was reached"
                break
            if state.step + 1 >= cfg.run.max_steps:
                reason = "the step budget is nearly exhausted"
                break
            if state.elapsed > cfg.run.max_wall_clock_s:
                reason = "the wall-clock budget is exhausted"
                break

            # A screen the guards have not cleared is not one to keep swiping on.
            finding = safety.sensitive_screen(screen)
            if finding is not None:
                rec.event("sensitive", reason=finding.reason)
                self._hand_over(state, finding.reason)
                return screen
            if safety.find_interstitial(screen, screen.package) is not None:
                reason = "a dialog appeared that needs handling"
                break

            state.step += 1

            # -- read what is on screen now, in the background --------------
            reading: Optional[Prefetch] = None
            shot_name = ""
            ledger_mark = self.llm.ledger.mark() if self.llm else 0
            if self.llm is not None and not cfg.run.never_screenshot:
                shot = self._ensure_screenshot(screen)
                # Kept here rather than inside `read_item`: the read runs on
                # another thread, and the name has to be in hand on this one to
                # travel with the reading when it is filed below. A sweep is most
                # of a run's vision calls, and "what did it read off item 7" is
                # not answerable from the reading alone.
                shot_name = rec.screenshot(state.step, shot, "read_item")
                # The item, not the frame it sits in. `read_item` is asking one
                # question about one bitmap, and the status bar and the nav bar
                # are neither -- ITEM_READING_SYSTEM spends two of its rules
                # telling the model to ignore chrome that need not be sent at
                # all. It has been read as the answer before: a clock read as an
                # item caption renamed every item once a minute (see
                # `screen.SYSTEM_UI_PACKAGES`), which was fixed on the tree side
                # while the pixels kept carrying it.
                #
                # `pager.content_box` and not a box of our own: the sweep is a
                # pager sweep, and that function is already this module's answer
                # to where an item lives -- the same crop `content_moved` judges
                # movement inside. The full frame is what gets kept on disk and
                # what `dhash` is computed from; only the copy the model reads is
                # cropped.
                item = crop_frac(shot, content_box(screen)) or shot
                reading = Prefetch(lambda s=item: self.llm.read_item(
                    s, goal=state.goal, step=state.step))

            # -- repeat the gesture ----------------------------------------
            # A copy of what the model issued, so the sweep can take no action
            # the model did not already authorise -- including its target, which
            # is why this is a copy rather than a fresh full-screen fling.
            repeat = action.model_copy(update={
                "observation": f"repeating `{gesture_name}`",
                "reasoning": "continuing the gesture the model chose",
            })
            try:
                execute(self.dev, repeat, screen)
                after = self.dev.observe(settle=True)
                self._ensure_screenshot(after)
            except (ActionError, ValueError) as exc:
                reason = f"the gesture could not be carried out ({exc})"
                if reading is not None and self._file_reading(state, screen,
                                                              reading, rec,
                                                              shot=shot_name):
                    read += 1
                break
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                return self._screen_after_recovery(state, screen)

            moved = verify(repeat, screen, after,
                           synthesise_postcondition(repeat, None),
                           None).grade != "no_change"

            if not moved:
                # A ViewPager silently drops a fling it judges too short or too
                # slow. The main loop retries harder before believing it, and a
                # sweep that skipped that step would read one dropped gesture as
                # the end and hand back several items early.
                harder = repeat.model_copy(update={"scroll_amount": 2.0,
                                                   "duration": 0.12})
                rec.event("pager_retry", step=state.step, during="sweep",
                          action=harder.describe())
                try:
                    execute(self.dev, harder, screen)
                    after = self.dev.observe(settle=True)
                    self._ensure_screenshot(after)
                    moved = verify(harder, screen, after,
                                   synthesise_postcondition(harder, None),
                                   None).grade != "no_change"
                except (ActionError, ValueError) as exc:
                    log.warning("sweep: harder fling failed: %s", exc)
                except (DeviceTimeout, DeviceLost) as exc:
                    if not self._recover_device(state, exc):
                        state.finished = "aborted"
                    return self._screen_after_recovery(state, screen)

            # -- collect the reading, against the frame it was taken of -----
            if reading is not None and self._file_reading(state, screen,
                                                          reading, rec,
                                                          shot=shot_name):
                read += 1
            swept += 1

            # Costed like any other step: a sweep step is a real LLM call, and a
            # report that only totalled `decide` events would show the sweep as
            # free and quietly understate the run.
            rec.event("sweep_step", step=state.step, gesture=gesture_name,
                      moved=moved, read_count=state.sweep.read_count,
                      llm=step_metrics(self.llm.ledger.since(ledger_mark)
                                       if self.llm else []))
            self.on_event("sweep_step", step=state.step, gesture=gesture_name,
                          moved=moved, swept=swept,
                          read_count=state.sweep.read_count)

            state.content_moved = moved
            state.sweep.repeats = swept
            if not moved:
                reason = ("the content stopped changing, so the gesture no "
                          "longer advances")
                screen = after
                break
            screen = after

        if not swept:
            return None

        log.info("sweep: repeated `%s` %d time(s), %d read (%s)", gesture_name,
                 swept, read, reason or "stopped")
        rec.event("sweep", first_step=first_step, last_step=state.step,
                  gesture=gesture_name, swept=swept, read=read, reason=reason,
                  read_count=state.sweep.read_count)
        self.on_event("sweep_end", first_step=first_step, last_step=state.step,
                      gesture=gesture_name, swept=swept, read=read,
                      reason=reason)
        state.sweep.reason = reason or "it stopped"
        state.remember(sweep_summary(first_step, state.step, gesture_name,
                                     swept, read, state.sweep.reason))
        # The sweep is browsing, not thrashing, so it is deliberately kept out of
        # the loop detector: twelve flings on one `skeleton_id` is exactly the
        # shape that makes `should_force_back` press back and eject the agent.
        state.last_failure = ""
        state.consecutive_failures = 0
        if read:
            state.note_progress(f"a sweep read {read} item(s)")
        self._maybe_give_up(state)
        return screen

    def _file_reading(self, state: RunState, screen: Screen,
                      reading: "Prefetch", rec: Recorder, shot: str = "") -> bool:
        """File a prefetched reading against the sweep that took it.

        Appended in order rather than filed under an item key: the key used to be
        the app's caption for the item, which is exactly the identity this module
        stopped claiming to know.

        `shot` is the frame the reading was taken from, so the record carries
        both halves of the call: a sweep read has no live panel -- it runs on
        another thread, and streaming two of those into one terminal interleaves
        them into nonsense -- so this event is the whole of it.
        """
        # Bounded, because `Prefetch.result()` defaults to joining forever. A
        # vision read that hangs gets `llm.read_timeout` (300s by default) times
        # `llm.max_retries` before the client itself gives up, and the loop's
        # wall-clock guard is only consulted between steps -- so one stuck call
        # could hold the run open for twenty minutes past its budget with no
        # step advancing and nothing in the log to say why. Losing the reading is
        # the cheaper failure: it degrades a sweep, it does not end a run.
        text = reading.result(default="", timeout=self.cfg.llm.read_timeout)
        if not text:
            return False
        state.sweep.add(text)
        rec.event("item_reading", step=state.step,
                  position=state.sweep.read_count, reading=text, shot=shot)
        self.on_event("item_reading", step=state.step,
                      position=state.sweep.read_count, reading=text, shot=shot)
        return True

    # -- terminal actions --------------------------------------------------

    def _terminal(self, state: RunState, screen: Screen, action: AgentAction,
                  rec: Recorder) -> Optional[Outcome]:
        # What the model is answering with, held aside until the run really
        # ends. A `done` that gets rejected below is not an answer, and letting
        # it stand would have the run report a summary it was told was wrong.
        answer = " ".join((action.text or "").split())

        if action.action == "ask_user":
            state.result = answer or "the agent needs your help"
            self._hand_over(state, state.result)
            return "needs_user"

        if action.action == "fail":
            log.warning("the agent gave up: %s", action.text)
            rec.event("gave_up", reason=action.text)
            state.result = answer
            return "failed"

        # `done` is the weakest evidence there is. Published agents claim
        # completion prematurely often enough that it must never stand alone.
        if self.oracle.defined:
            if self.oracle.satisfied(self.dev, screen):
                state.result = answer
                state.evidence = (f"the success check passed: "
                                  f"{self.oracle.describe()}")
                return "success"
            log.warning("the agent said done but the assertion disagrees")
            state.remember(
                f"{state.step}. claimed done, but the success check failed")
            return self._reject_done(state, "the success condition is still "
                                            "not met")

        if self.llm is None:
            state.result = answer
            return "success"

        shot = self._ensure_screenshot(screen)
        model_name = (self.llm.model_image if shot else self.llm.model_small) if self.llm else ""
        self.on_event("llm_start", step=state.step, purpose="judge", model=model_name, screenshot=bool(shot))
        t0_judge = time.monotonic()
        ledger_mark = self.llm.ledger.mark()
        # The judge is shown every record the run collected, not the last thing
        # the model happened to write. Without that it grades the goal on
        # whatever survived the model's last edit -- which is how a run that had
        # read a value ends up reporting it as unavailable.
        collected = state.scratchpad.plain(self.cfg.run.scratchpad_max_chars)
        verdict = self.llm.judge(goal=state.goal, rendered=render(screen),  ### LLM ###
                                 history=state.history, screenshot=shot,
                                 scratchpad=collected,
                                 progress="\n".join(state.progress_log),
                                 done_text=action.text or "",
                                 step=state.step, recorder=rec,
                                 on_event=self.on_event)
        t_judge = time.monotonic() - t0_judge
        judge_calls = self.llm.ledger.since(ledger_mark)
        last_call = judge_calls[-1] if judge_calls else None
        self.on_event("llm_end", step=state.step, purpose="judge", elapsed=t_judge, call=last_call, verdict=verdict)
        state.llm_calls += 1
        rec.event("judge", step=state.step, satisfied=verdict.satisfied,
                  evidence=verdict.evidence, wall_s=round(t_judge, 3),
                  llm=step_metrics(judge_calls))
        if verdict.satisfied:
            log.info("verified: %s", verdict.evidence)
            state.result = answer
            state.evidence = verdict.evidence
            return "success"
        log.warning("premature 'done': %s", verdict.evidence)
        state.remember(f"{state.step}. claimed done; rejected: {verdict.evidence}")
        return self._reject_done(state, verdict.evidence)

    def _reject_done(self, state: RunState, why: str) -> Optional[Outcome]:
        """Send a `done` back to the model. Returns the run's outcome, or None.

        A rejected completion is a failed step and has to be counted as one. It
        was not: `consecutive_failures` stayed at zero through every rejection,
        so `max_consecutive_failures` never fired and a model that answered
        `done` on every turn ran to the step budget -- paying for a screenshot,
        a vision pass and a high-effort judge call each time round.

        Counted here rather than in `_loop` because `_maybe_give_up` writes
        `state.finished`, which the terminal path is about to overwrite with
        this function's return value.
        """
        state.consecutive_failures += 1
        state.last_failure = f"your 'done' was rejected: {why}"
        # The rejection is the closest thing to an explanation this run will
        # have if it never recovers, so it is kept where the summary can find
        # it -- replaced by the next rejection, or by a completion that stands.
        state.evidence = f"the last claim of completion was rejected: {why}"
        if state.consecutive_failures >= self.cfg.run.max_consecutive_failures:
            log.error("giving up after %d rejected completion(s)",
                      state.consecutive_failures)
            return "failed"
        return None

    def _replan(self, state: RunState, rec: Recorder, screen: Screen,
                stalled: int) -> bool:
        """Buy one different approach. False when the run should stop.

        Tier 3 of the stall ladder, and the only tier that costs a call. It is
        worth one because the two cheap tiers have a shared blind spot: both
        speak to the decider, and the decider is looking at a history of the
        approach that is failing. `LLMClient.replan` is not shown that history.

        A failure here is swallowed. The run is already in trouble; losing it to
        an exception raised by the thing sent to rescue it would be worse than
        carrying on stuck, and the give-up tier is still ahead.
        """
        shot: Optional[bytes] = None
        if not self.cfg.run.never_screenshot:
            try:
                shot = self._ensure_screenshot(screen)
            except (DeviceTimeout, DeviceLost) as exc:
                log.warning("replan could not take a screenshot: %s", exc)

        self.on_event("llm_start", step=state.step, purpose="replan",
                      model=getattr(self.llm, "model", ""), screenshot=bool(shot))
        t0 = time.monotonic()
        ledger_mark = self.llm.ledger.mark()
        try:
            plan = self.llm.replan(                        ### LLM ###
                goal=state.goal, rendered=render(screen),
                tried=state.loops.tried_on(screen.skeleton_id),
                stalled=stalled,
                scratchpad=state.scratchpad.plain(self.cfg.run.scratchpad_max_chars),
                progress="\n".join(state.progress_log),
                packages=sorted(state.packages), screenshot=shot,
                step=state.step, recorder=rec, on_event=self.on_event)
        except LLMError as exc:
            log.warning("replan produced nothing usable (%s); carrying on", exc)
            rec.event("replan_failed", step=state.step, error=str(exc))
            return True
        elapsed = time.monotonic() - t0
        calls = self.llm.ledger.since(ledger_mark)
        self.on_event("llm_end", step=state.step, purpose="replan",
                      elapsed=elapsed, call=calls[-1] if calls else None)
        state.llm_calls += 1
        rec.event("replan", step=state.step, stalled=stalled,
                  assessment=plan.assessment, strategy=plan.strategy,
                  abandon=plan.abandon, wall_s=round(elapsed, 3),
                  llm=step_metrics(calls))

        if plan.abandon:
            log.error("replan says the goal is not reachable from here: %s",
                      plan.assessment or plan.strategy)
            state.remember(f"{state.step}. gave up after a replan: "
                           f"{plan.assessment or plan.strategy}")
            state.finished = "failed"
            return False

        state.strategy = " ".join((plan.strategy or "").split())
        if state.strategy:
            log.info("step %d: new approach -- %s", state.step, state.strategy)
            self.on_event("replan", step=state.step,
                          assessment=plan.assessment, strategy=state.strategy)
        return True

    def _hand_over(self, state: RunState, reason: str) -> None:
        """Stop and give the phone back to the person."""
        log.warning("handing over: %s", reason)
        print(f"\n  The agent has stopped and needs you:\n    {reason}\n"
              f"  Do it on the device yourself, then re-run the goal.\n")
        state.finished = "needs_user"


    def _recover_device(self, state: RunState, exc: Exception) -> bool:
        log.warning("device trouble (%s); recovering", exc)
        for tier in (1, 2, 3):
            if self.dev.recover(tier):
                return True
        return False

    def _screen_after_recovery(self, state: RunState, fallback: Screen) -> Screen:
        """What is on the phone once recovery has finished.

        `_loop` answers this by dropping its screen and letting the next turn
        re-observe. A sweep cannot: it has to hand a screen back. Returning the
        one it is holding returns a frame from before a fling that already went
        out, so it is re-read instead. `fallback` is only for when the run is
        ending anyway and the value is about to be discarded.
        """
        if state.finished is not None:
            return fallback
        try:
            return self.dev.observe()
        except (DeviceTimeout, DeviceLost) as exc:
            log.error("device still unusable after recovery: %s", exc)
            state.finished = "aborted"
            return fallback

    def _maybe_give_up(self, state: RunState) -> None:
        if state.consecutive_failures >= self.cfg.run.max_consecutive_failures:
            log.error("giving up after %d consecutive failures",
                      state.consecutive_failures)
            state.finished = "failed"
