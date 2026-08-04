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

from . import __version__, checkpoint, prompts, runlog, safety
from .actions import (ActionError, AgentAction, append_history, execute,
                      format_history_entry, synthesise_postcondition, verify)
from .config import Config
from .device import Device, DeviceTimeout, DeviceLost
from .llm import (BudgetExceeded, LLMClient, LLMError, Prefetch, ScreenAnalysis)
from .memory import Memory, intent_key
from .pager import (ItemLedger, attach_item, can_sweep, loop_id, pager_element,
                    set_id as pager_set_id, stop_sweeping, sweep_summary)
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
    #: Which items of a gallery / carousel have actually been looked at. Kept by
    #: code rather than by the model, because a ledger the model rewrites by hand
    #: every turn silently loses an entry the moment it forgets to repeat one.
    items: ItemLedger = field(default_factory=ItemLedger)
    #: What verification concluded about the last gesture on a pager: True the
    #: item advanced, False it did not, None not applicable.
    item_moved: Optional[bool] = None
    #: Key of the item on screen this turn, once resolved.
    item_key: str = ""
    #: ``exact_id/label`` of the control the harness last auto-dismissed, and how
    #: many times it has tried it on that screen. A dismissal that changes
    #: nothing means the control is part of the screen rather than a popup over
    #: it, and repeating it is how a whole step budget goes on one button.
    last_dismiss: str = ""
    dismiss_tries: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

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
    if screen.is_pager and not state.items.was_read(state.item_key):
        return True, ("this screen shows one item of a gallery and its content is "
                      "only in the image -- read the item from the screenshot and "
                      "record what it shows before moving on")
    if state.consecutive_failures >= 1:
        return True, "the last action did not work; look carefully at the screen"
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
                 on_event=None):
        self.dev = dev
        self.mem = mem
        self.llm = llm
        self.cfg = cfg
        self.oracle = oracle or Oracle()
        self.on_event = on_event or (lambda *a, **k: None)
        self.skills = SkillRegistry(cfg.skills.skills_dir)

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
            model=cfg.llm.model,
            model_image=cfg.llm.image(),
            model_small=cfg.llm.small(),
            model_skill=cfg.llm.skill(),
            model_skill_image=cfg.llm.skill_image(),
            reasoning=f"effort={cfg.llm.reasoning_effort or '(model default)'} "
                      f"hard={cfg.llm.reasoning_effort_hard} "
                      f"style={cfg.llm.reasoning_style} "
                      f"vision_in_decider={cfg.llm.vision_in_decider}",
            limits=f"max_steps={cfg.run.max_steps} "
                   f"max_wall_clock_s={cfg.run.max_wall_clock_s:g} "
                   f"budget_usd={cfg.safety.budget_usd:g}",
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
                state.packages.add(screen.package)
                state.package_steps[screen.package] = \
                    state.package_steps.get(screen.package, 0) + 1

            # A programmatic assertion is the cheapest and most reliable way to
            # know we are done, so it is checked before anything else happens.
            if self.oracle.defined and self.oracle.satisfied(self.dev, screen):
                log.info("assertion satisfied -- goal reached")
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

            # ---- 2b. where are we in a gallery? -------------------------
            # Resolved before anything else needs it: the screenshot policy, the
            # loop detector and the prompt all key off the item, not the screen.
            pager_el = pager_element(screen) if screen.is_pager else None
            if screen.is_pager:
                state.items.rebase(pager_set_id(screen))
                state.item_key = state.items.resolve(screen, moved=state.item_moved)
            else:
                # Left the set: a confirmed move inside it means nothing now.
                state.item_key = ""
                state.item_moved = None

            hint = state.loops.hint(loop_id(screen))

            # Scroll awareness: give the LLM full context about its
            # scrolling pattern so it can course-correct on its own.
            # Paging through a gallery is horizontal and has its own guidance, so
            # the vertical-scrolling advice ("scroll UP for older content") is
            # suppressed there -- it is not merely unhelpful on a carousel, it
            # points the agent at the wrong axis.
            scroll_ctx = state.loops.scroll_context(
                axis="horizontal" if screen.is_pager else "")
            if scroll_ctx:
                hint = scroll_ctx
                if (state.loops.scroll_oscillating()
                        or state.loops.direction_reversals() >= 5):
                    for _, sig in state.loops.history:
                        if not sig.startswith("scroll/"):
                            continue
                        # Never ban a horizontal gesture. On a carousel the same
                        # swipe on the same element is the only way forward, so a
                        # ban strands the agent mid-album -- and in the run that
                        # motivated this, it did exactly that.
                        if sig.rsplit("/", 1)[-1] in ("left", "right"):
                            continue
                        state.loops.ban(screen.skeleton_id, sig)

            if state.loops.should_force_back(loop_id(screen)) or state.loops.oscillating():
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
                    state.loops.ban(screen.skeleton_id, "forced-back")
                else:
                    log.warning("step %d: stuck in a loop; going back",
                                state.step)
                    self.on_event("loop_warning", message=f"step {state.step}: stuck in a loop; going back")
                    rec.event("loop_break", exact_id=screen.exact_id,
                              item=state.item_key)
                    self.dev.press("back")
                    state.loops.record(loop_id(screen), "forced-back")
                    state.loops.consecutive_backs += 1
                    screen = None
                    continue

            # ---- 3. visit tracking --------------------------------------
            visit = state.visits.get(screen.skeleton_id, 0)
            state.visits[screen.skeleton_id] = visit + 1

            # ---- 4. ask the model ---------------------------------------
            screenshot: Optional[bytes] = None
            want, note = needs_screenshot(state, screen, cfg)
            if want:
                screenshot = self._ensure_screenshot(screen)
                if screen.is_pager and not state.item_key:
                    # The caption was hidden, so the item had no key until now;
                    # the screenshot gives it a pixel-derived one.
                    state.item_key = state.items.resolve(
                        attach_item(screen), moved=state.item_moved)

            # The vision pass runs here rather than inside `decide` so its
            # structured fields reach the ledgers below: a `reading` the image
            # model took off this item is the same fact the pager ledger keeps per
            # item, and routing it through prose for the decider to re-extract
            # loses it the moment the decider paraphrases.
            #
            # The step's cost mark and clock both start *before* it, because a
            # screenshot turn is an analysis and then a decision, and charging the
            # step for only the second one is how a two-call turn reads as a
            # one-call turn -- and how `report`'s latency/step quietly loses the
            # vision time on the ~22% of turns that take a screenshot.
            ledger_mark = self.llm.ledger.mark() if self.llm else 0
            t0_step_llm = time.monotonic()
            analysis: Optional[ScreenAnalysis] = None
            if (screenshot and self.llm is not None
                    and self.llm.needs_vision_pass):
                analysis = self.llm.analyze_image(
                    screenshot, goal=state.goal, rendered=render(screen),
                    step=state.step, recorder=rec, on_event=self.on_event)

            # A pager item is ledgered here, once its identity and whether we
            # have vision on it are both settled. `read` is deliberately keyed to
            # the screenshot and not to the sighting: having seen an item's
            # caption is not the same as having looked at the item.
            pager_note = ""
            if screen.is_pager:
                state.items.note(state.item_key, screen, state.step,
                                 read=bool(screenshot),
                                 detail=analysis.reading if analysis else "",
                                 label=analysis.item_label if analysis else "")
                if screen.item_label:
                    # `item_moved` is a latch, not a per-turn flag: it has to
                    # survive the turns where the caption is hidden, because the
                    # tap that reveals the caption again is not itself a move.
                    # Here a caption was available, so the latch has been spent.
                    state.item_moved = None
                pager_note = state.items.render(state.item_key,
                                                screen.item_label)
                rec.event("pager_item", step=state.step, key=state.item_key,
                          label=screen.item_label, read=bool(screenshot),
                          read_count=state.items.read_count,
                          total=state.items.total)

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

            elem_hint = state.loops.element_history_hint(
                screen.skeleton_id,
                repeatable=pager_el.index if pager_el is not None else 0)
            # Advice that only applies sometimes lives here rather than in the
            # system prompt: this block is rebuilt every turn anyway, so varying
            # it is free, whereas varying the system message evicts the whole
            # prompt prefix from the provider's cache.
            situational = prompts.situational_notes(
                goal=state.goal,
                scrolls=state.loops.total_scroll_count,
                has_scroller=any(el.scrollable and not el.is_horizontal
                                 for el in screen.elements),
                packages_seen=len(state.packages))
            notes = "\n\n".join(filter(None, (note, pager_note,
                                             hint, elem_hint, ban_note,
                                             state.last_failure, skill_note,
                                             situational)))
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
                if (screenshot and cfg.llm.vision_in_decider) else ""
            self.on_event("llm_start", step=state.step, purpose="decide", model=model_name,
                          screenshot=bool(screenshot), shot=shot, effort=effort,
                          hard_because=hard_because)
            t0_llm = time.monotonic()
            action = self.llm.decide(                      ### LLM ###
                goal=state.goal, rendered=render(screen), history=state.history,
                width=screen.width, height=screen.height, package=screen.package,
                screenshot=screenshot, note=notes,
                scratchpad=state.scratchpad.render(cfg.run.scratchpad_max_chars),
                progress="\n".join(state.progress_log),
                image_analysis=analysis.render() if analysis else None,
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
            if (screen.is_pager and state.item_key and action.observation
                    and screenshot and not (analysis and analysis.reading)):
                state.items.note(state.item_key, screen, state.step,
                                 detail=action.observation)

            # -- accumulate progress ----------------------------------------
            if getattr(action, "progress", None):
                prog_text = action.progress.strip()
                if prog_text:
                    state.progress_log = [prog_text]

            rec.event("decide", step=state.step, source=source,
                      skeleton=screen.skeleton_id, action=action.model_dump(),
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

            # ---- 6. act -------------------------------------------------
            t0_act = time.monotonic()
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
            if (outcome.grade == "no_change" and screen.is_pager
                    and action.action in ("scroll", "swipe")
                    and action.direction in ("left", "right")):
                retry = action.model_copy(update={
                    "action": "swipe", "duration": 0.12,
                    "scroll_amount": min(5.0, max(2.0, action.scroll_amount * 2)),
                })
                log.info("step %d: pager did not advance; retrying harder",
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

            # Whether the item advanced is what the next turn's ledger keys off.
            if action.action in ("scroll", "swipe") and (screen.is_pager
                                                         or after.is_pager):
                state.item_moved = outcome.grade != "no_change"
                if (not state.item_moved
                        and action.direction in ("left", "right")):
                    # Two gestures in a row moved nothing, so this is an end of
                    # the set. Most apps never say how many items a set holds, so
                    # this is the only signal that the album is finished.
                    state.items.edges.add(action.direction)

            self.on_event("verify_end", step=state.step, elapsed=t_settle, grade=outcome.grade, reason=outcome.reason)

            rec.event("verify", step=state.step, grade=outcome.grade,
                      reason=outcome.reason, after=after.skeleton_id)

            # ---- 8. learn (no LLM) --------------------------------------
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
                    if screen.is_pager and h_dir:
                        # "You have reached the end" is a claim, and on a pager it
                        # is often the wrong one: the harder retry above has just
                        # failed too, so say what is actually known and let the
                        # ledger decide whether there is anywhere left to go.
                        state.last_failure = (
                            f"The item did not change after swiping "
                            f"{action.direction} twice, so you are at the "
                            f"{'start' if action.direction == 'right' else 'end'} "
                            f"of this set.")
                        unread = [r.label for r in state.items.items.values()
                                  if not r.read]
                        if state.items.complete:
                            state.last_failure += (
                                " Every item has been read \u2014 stop browsing "
                                "and report what you found.")
                        elif unread:
                            state.last_failure += (
                                f" {len(unread)} item(s) of this set are still "
                                f"unread ({', '.join(unread[:6])}); swipe the "
                                f"other way to reach them, or go back to the "
                                f"list this set came from.")
                        else:
                            state.last_failure += (
                                " Swipe the other way if you have not covered "
                                "the whole set, otherwise report what you found.")
                    else:
                        state.last_failure = (
                            f"{act_name} {action.direction} did not reveal new "
                            f"content \u2014 you have reached the end of the "
                            f"{axis} scrollable area. Do not {action.action} "
                            f"{action.direction} again here.")

            state.loops.record(loop_id(screen), action.signature())
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
                    and can_sweep(screen, state.items, action=action.action,
                                  direction=action.direction or "",
                                  moved=state.item_moved is True)):
                swept = self._sweep_pager(state, rec, screen,
                                          action.direction or "left")
                if swept is not None:
                    screen = swept

    # -- sweeping a carousel -----------------------------------------------

    def _sweep_pager(self, state: RunState, rec: Recorder, screen: Screen,
                     direction: str) -> Optional[Screen]:
        """Page through the rest of a set without asking the model each time.

        Entered only from `can_sweep`, so the model has already chosen this
        gesture on this screen and it has already been shown to move the item.
        Each iteration reads the item it is standing on, flings once in the same
        direction, and verifies. The read is started *before* the fling and
        collected after it, so a ~1.5s vision call overlaps the swipe and the
        settle rather than following them.

        Returns the screen the sweep ended on, or None if it did nothing.
        """
        cfg = self.cfg
        first_step = state.step + 1
        swept = 0
        read = 0
        reason = ""
        package = screen.package

        while True:
            reason = stop_sweeping(screen, state.items, direction=direction,
                                   package=package)
            if reason:
                break
            if swept >= cfg.run.pager_sweep_max:
                reason = f"the {cfg.run.pager_sweep_max}-item sweep limit was reached"
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

            # -- where are we? ---------------------------------------------
            # The main loop resolves the item at the *top* of a turn, so on entry
            # `state.item_key` still names the item we swiped away from. Resolving
            # here, once per iteration, is what keeps the ledger keyed to the item
            # actually on screen -- and `item_moved` is the latch that separates
            # two items sharing a caption.
            state.items.rebase(pager_set_id(screen))
            here = state.items.resolve(screen, moved=state.item_moved)
            state.item_key = here
            state.items.note(here, screen, state.step, read=False)
            if screen.item_label:
                state.item_moved = None
            label = screen.item_label

            # -- read the item we are on, in the background ----------------
            reading: Optional[Prefetch] = None
            shot_name = ""
            ledger_mark = self.llm.ledger.mark() if self.llm else 0
            if (self.llm is not None and not cfg.run.never_screenshot
                    and not state.items.was_read(here)):
                shot = self._ensure_screenshot(screen)
                # Kept here rather than inside `read_item`: the read runs on
                # another thread, and the name has to be in hand on this one to
                # travel with the reading when it is filed below. A sweep is most
                # of a run's vision calls, and "what did it read off item 7" is
                # not answerable from the reading alone.
                shot_name = rec.screenshot(state.step, shot, "read_item")
                reading = Prefetch(lambda s=shot, l=label: self.llm.read_item(
                    s, goal=state.goal, label=l, step=state.step))

            # -- fling to the next one -------------------------------------
            gesture = AgentAction(
                observation=f"sweeping {direction} through the set",
                reasoning="continuing the paging the model chose",
                action="swipe", direction=direction, duration=0.15)
            try:
                execute(self.dev, gesture, screen)
                after = self.dev.observe(settle=True)
                self._ensure_screenshot(after)
            except (ActionError, ValueError) as exc:
                reason = f"the swipe could not be carried out ({exc})"
                if reading is not None and self._file_reading(state, screen, here,
                                                              reading, rec,
                                                              shot=shot_name):
                    read += 1
                break
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                return self._screen_after_recovery(state, screen)

            moved = verify(gesture, screen, after,
                           synthesise_postcondition(gesture, None),
                           None).grade != "no_change"

            if not moved:
                # A ViewPager silently drops a fling it judges too short or too
                # slow. The main loop retries harder before believing it, and a
                # sweep that skipped that step would read one dropped gesture as
                # the end of the album and hand back four photos early.
                harder = gesture.model_copy(update={"scroll_amount": 2.0,
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

            # -- collect the reading, against the item it was taken of -----
            # Filed before `screen` moves on, because `note` reads the label and
            # the total off the screen the reading was taken of.
            if reading is not None and self._file_reading(state, screen, here,
                                                          reading, rec,
                                                          shot=shot_name):
                read += 1
            swept += 1

            # Costed like any other step: a sweep step is a real LLM call, and a
            # report that only totalled `decide` events would show the sweep as
            # free and quietly understate the run.
            rec.event("sweep_step", step=state.step, direction=direction,
                      item=here, label=label, moved=moved,
                      read_count=state.items.read_count,
                      llm=step_metrics(self.llm.ledger.since(ledger_mark)
                                       if self.llm else []))
            self.on_event("sweep_step", step=state.step, direction=direction,
                          label=label, moved=moved, swept=swept,
                          read_count=state.items.read_count,
                          total=state.items.total)

            state.item_moved = moved
            if not moved:
                # The same evidence the main loop uses: a gesture that changed
                # nothing on a pager is the edge of the set.
                state.items.edges.add(direction)
                reason = f"the item stopped moving, so this is the {direction} end"
                screen = after
                break
            screen = after

        if not swept:
            return None

        log.info("sweep: %d item(s) %s, %d read (%s)", swept, direction, read,
                 reason or "stopped")
        rec.event("sweep", first_step=first_step, last_step=state.step,
                  direction=direction, swept=swept, read=read, reason=reason,
                  read_count=state.items.read_count)
        self.on_event("sweep_end", first_step=first_step, last_step=state.step,
                      direction=direction, swept=swept, read=read, reason=reason)
        state.remember(sweep_summary(first_step, state.step, direction,
                                           swept, read, reason or "it stopped"))
        # The sweep is browsing, not thrashing, so it is deliberately kept out of
        # the loop detector: twelve flings on one `skeleton_id` is exactly the
        # shape that makes `should_force_back` press back and eject the agent.
        state.last_failure = ""
        state.consecutive_failures = 0
        self._maybe_give_up(state)
        return screen

    def _file_reading(self, state: RunState, screen: Screen, key: str,
                      reading: "Prefetch", rec: Recorder, shot: str = "") -> bool:
        """Attach a prefetched item reading to the item it was taken of.

        `shot` is the frame the reading was taken from, so the record carries
        both halves of the call: a sweep read has no live panel -- it runs on
        another thread, and streaming two of those into one terminal interleaves
        them into nonsense -- so this event is the whole of it.
        """
        text = reading.result(default="")
        if not text:
            return False
        state.items.note(key, screen, state.step, detail=text, read=True)
        rec.event("item_reading", step=state.step, item=key, reading=text,
                  shot=shot)
        self.on_event("item_reading", step=state.step, label=screen.item_label,
                      reading=text, shot=shot)
        return True

    # -- terminal actions --------------------------------------------------

    def _terminal(self, state: RunState, screen: Screen, action: AgentAction,
                  rec: Recorder) -> Optional[Outcome]:
        if action.action == "ask_user":
            self._hand_over(state, action.text or "the agent needs your help")
            return "needs_user"

        if action.action == "fail":
            log.warning("the agent gave up: %s", action.text)
            rec.event("gave_up", reason=action.text)
            return "failed"

        # `done` is the weakest evidence there is. Published agents claim
        # completion prematurely often enough that it must never stand alone.
        if self.oracle.defined:
            if self.oracle.satisfied(self.dev, screen):
                return "success"
            log.warning("the agent said done but the assertion disagrees")
            state.remember(
                f"{state.step}. claimed done, but the success check failed")
            return self._reject_done(state, "the success condition is still "
                                            "not met")

        if self.llm is None:
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
        if state.consecutive_failures >= self.cfg.run.max_consecutive_failures:
            log.error("giving up after %d rejected completion(s)",
                      state.consecutive_failures)
            return "failed"
        return None

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
