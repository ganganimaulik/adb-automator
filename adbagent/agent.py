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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import safety
from .actions import (ActionError, AgentAction, execute, format_history_entry,
                      synthesise_postcondition, verify)
from .config import Config
from .device import Device, DeviceTimeout, DeviceLost
from .llm import BudgetExceeded, LLMClient, LLMError
from .memory import Memory, intent_key
from .safety import Aborted, LoopDetector
from .screen import Screen, render

log = logging.getLogger("adbagent.agent")

Outcome = str  # "success" | "failed" | "aborted" | "needs_user"


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
    history: List[str] = field(default_factory=list)
    visits: Dict[str, int] = field(default_factory=dict)
    loops: LoopDetector = field(default_factory=LoopDetector)
    want_screenshot: bool = False
    last_failure: str = ""
    scroll_warnings: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished: Optional[Outcome] = None
    #: Running scratchpad for data-collection goals.  The LLM writes into the
    #: ``notes`` field of its action; we append here and feed it back each turn
    #: so it knows what it has already captured.
    scratchpad: List[str] = field(default_factory=list)
    scratchpad_chars: int = 0
    #: Progress tracker for multi-step goals.  The LLM writes into the
    #: ``progress`` field; we keep the last few entries and feed them back.
    progress_log: List[str] = field(default_factory=list)
    progress_chars: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


class Recorder:
    """Per-run artifacts: one JSONL of events, blobs alongside."""

    def __init__(self, cfg: Config, run_id: str):
        self.dir = Path(cfg.run.artifacts_dir).expanduser() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = (self.dir / "events.jsonl").open("a", encoding="utf-8")

    def event(self, kind: str, **fields: Any) -> None:
        record = {"t": round(time.time(), 3), "kind": kind, **fields}
        self.events.write(json.dumps(record, default=str) + "\n")
        self.events.flush()

    def blob(self, name: str, data: bytes) -> str:
        path = self.dir / name
        path.write_bytes(data)
        return str(path)

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


# ---------------------------------------------------------------------------
# Screenshot policy
# ---------------------------------------------------------------------------

def needs_screenshot(state: RunState, screen: Screen, cfg: Config) -> Tuple[bool, str]:
    """XML-first: pay for vision only when the tree cannot answer the question."""
    if cfg.run.never_screenshot:
        return False, ""
    if cfg.run.always_screenshot:
        return True, "always"
    if screen.degenerate:
        return True, ("the accessibility tree is nearly empty -- this is a WebView, "
                      "canvas or game, so rely on the screenshot")
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

    # -- public ------------------------------------------------------------

    def run(self, goal: str, run_id: str = "") -> Tuple[Outcome, RunState]:
        run_id = run_id or uuid.uuid4().hex[:12]
        state = RunState(goal=goal, run_id=run_id, intent_id=intent_key(goal))
        recorder = Recorder(self.cfg, run_id)
        self.mem.begin_run(run_id, goal, state.intent_id)
        recorder.event("run_start", goal=goal, model=getattr(self.llm, "model", ""))

        try:
            self._loop(state, recorder)
        except (BudgetExceeded, LLMError) as exc:
            log.error("%s", exc)
            state.finished = "aborted"
            recorder.event("error", error=str(exc))
        except (DeviceLost, DeviceTimeout) as exc:
            log.error("device: %s", exc)
            state.finished = "aborted"
            recorder.event("error", error=str(exc))
        except Aborted as exc:
            log.warning("aborted: %s", exc)
            state.finished = "aborted"
        except KeyboardInterrupt:
            log.warning("interrupted")
            state.finished = "aborted"
        finally:
            outcome = state.finished or "failed"
            usd = self.llm.ledger.total_usd if self.llm else 0.0
            self.mem.end_run(run_id, outcome, state.step, state.llm_calls, usd)
            recorder.event("run_end", outcome=outcome, steps=state.step,
                           llm_calls=state.llm_calls,
                           usd=round(usd, 6))
            recorder.close()

        return state.finished or "failed", state

    # -- internals ---------------------------------------------------------

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
            self.mem.note_screen(screen)
            self._last_package = screen.package

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
                log.info("step %d: dismissing %r", state.step,
                         interstitial.best_text)
                rec.event("dismiss", label=interstitial.best_text)
                self.dev.tap(*interstitial.center)
                screen = None
                continue

            hint = state.loops.hint(screen.exact_id)

            # Scroll awareness: give the LLM full context about its
            # scrolling pattern so it can course-correct on its own.
            scroll_ctx = state.loops.scroll_context()
            if scroll_ctx:
                hint = scroll_ctx
                if (state.loops.scroll_oscillating()
                        or state.loops.direction_reversals() >= 5):
                    for _, sig in state.loops.history:
                        if sig.startswith("scroll/"):
                            state.loops.ban(screen.skeleton_id, sig)

            if state.loops.should_force_back(screen.exact_id) or state.loops.oscillating():
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
                    rec.event("loop_break", exact_id=screen.exact_id)
                    self.dev.press("back")
                    state.loops.record(screen.exact_id, "forced-back")
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
                screenshot = self.dev.screenshot()
            banned_actions = state.loops.bans_for(screen.skeleton_id)
            ban_note = ""
            if banned_actions:
                ban_note = (f"BANNED ACTIONS on this screen (these produced NO change - DO NOT REPEAT): "
                            f"{', '.join(sorted(banned_actions))}.")
            elem_hint = state.loops.element_history_hint(screen.skeleton_id)
            notes = " ".join(filter(None, (note, hint, elem_hint, ban_note, state.last_failure)))
            model_name = self.llm.model if self.llm else ""
            self.on_event("llm_start", step=state.step, purpose="decide", model=model_name, screenshot=bool(screenshot))
            t0_llm = time.monotonic()
            action = self.llm.decide(                      ### LLM ###
                goal=state.goal, rendered=render(screen), history=state.history,
                width=screen.width, height=screen.height, package=screen.package,
                screenshot=screenshot, note=notes,
                scratchpad="\n".join(state.scratchpad),
                progress="\n".join(state.progress_log),
                step=state.step, recorder=rec)
            t_llm = time.monotonic() - t0_llm
            last_call = self.llm.ledger.calls[-1] if (self.llm and self.llm.ledger.calls) else None
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
                if state.loops.scroll_oscillating():
                    scroll_blocked = True
                elif (state.loops.direction_reversals() >= 5
                      and action.direction
                      and len(state.loops.scroll_dir_log) >= 2):
                    # Check if this scroll would be yet another reversal.
                    prev_dir = state.loops.scroll_dir_log[-2] if len(
                        state.loops.scroll_dir_log) >= 2 else ""
                    from .safety import _SCROLL_OPPOSITES
                    if action.direction == _SCROLL_OPPOSITES.get(prev_dir, ""):
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
                    state.history.append(
                        format_history_entry(
                            state.step, action, screen=screen,
                            grade="rejected",
                            reason=f"scroll reversal, warning #{state.scroll_warnings}"
                        )
                    )
                    del state.history[:-12]
                    self._maybe_give_up(state)
                    continue

            # -- accumulate scratchpad notes --------------------------------
            if getattr(action, "notes", None):
                cap = cfg.run.scratchpad_max_chars
                note_text = action.notes.strip()[:cap]
                if note_text:
                    # Append scroll position context so the LLM has spatial
                    # memory even if it forgets to include it in its notes.
                    if state.loops.total_scroll_count > 0:
                        scroll_info = (f"\n[Scroll stats: {state.loops.total_scroll_count} "
                                       f"scroll(s) so far, {state.loops.direction_reversals()} "
                                       f"direction reversal(s)]")
                        note_text = note_text + scroll_info
                    state.scratchpad = [note_text]
                    state.scratchpad_chars = len(note_text)

            # -- accumulate progress ----------------------------------------
            if getattr(action, "progress", None):
                prog_text = action.progress.strip()
                if prog_text:
                    state.progress_log = [prog_text]
                    state.progress_chars = len(prog_text)

            rec.event("decide", step=state.step, source=source,
                      skeleton=screen.skeleton_id, action=action.model_dump(),
                      screenshot=bool(screenshot))
            self.on_event("step", state=state, screen=screen, action=action,
                          source=source, screenshot=bool(screenshot))

            # ---- 5. guard the chosen action -----------------------------
            label = safety.irreversible(action, screen)
            if label is not None:
                self.on_event("safety_warning", message=f"step {state.step}: irreversible action {label!r} in {screen.package}")
                if not safety.confirm(
                        f"Step {state.step}: the agent wants to press {label!r} "
                        f"in {screen.package}. This cannot be undone.", cfg):
                    state.history.append(
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
                state.history.append(
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
                state.history.append(
                    format_history_entry(
                        state.step, action, screen=screen,
                        grade="failed", reason=str(exc)
                    )
                )
                del state.history[:-12]
                self._maybe_give_up(state)
                continue
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
                continue
            self.on_event("act_end", step=state.step, action=action, elapsed=time.monotonic() - t0_act)

            # ---- 7. verify (no LLM) -------------------------------------
            self.on_event("settle_start", step=state.step, budget=cfg.device.settle_budget_s)
            t0_verify = time.monotonic()
            try:
                after = self.dev.observe(settle=True)
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
                continue
            t_settle = time.monotonic() - t0_verify
            post = synthesise_postcondition(action, element)
            expected = ""
            outcome = verify(action, screen, after, post, None)
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
                state.loops.ban(screen.skeleton_id, action.signature())
                if action.action in ("scroll", "swipe"):
                    h_dir = action.direction in ("left", "right")
                    axis = "horizontal" if h_dir else "vertical"
                    act_name = "Swiping" if action.action == "swipe" else "Scrolling"
                    extra_tip = ""
                    if h_dir and ("MediaView" in screen.activity or "gallery" in screen.activity.lower()):
                        extra_tip = " Press back (#1) to return to the thumbnail grid or chat list and select photos directly."
                    state.last_failure = (
                        f"{act_name} {action.direction} did not reveal new "
                        f"content \u2014 you have reached the end of the "
                        f"{axis} scrollable area. Do not {action.action} "
                        f"{action.direction} again here.{extra_tip}")

            state.loops.record(screen.exact_id, action.signature())
            state.loops.record_element_action(
                screen.skeleton_id, state.step, action.signature(), action.describe(element=element)
            )
            state.history.append(
                format_history_entry(
                    state.step, action, screen=screen, element=element,
                    grade=outcome.grade, reason=outcome.reason
                )
            )
            del state.history[:-12]  # keep the prompt bounded

            self._maybe_give_up(state)
            screen = after

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
            state.history.append(
                f"{state.step}. claimed done, but the success check failed")
            state.last_failure = ("your 'done' was rejected: the success condition "
                                 "is still not met")
            return None

        if self.llm is None:
            return "success"

        shot = self.dev.screenshot()
        model_name = (self.llm.model_image if shot else self.llm.model_small) if self.llm else ""
        self.on_event("llm_start", step=state.step, purpose="judge", model=model_name, screenshot=bool(shot))
        t0_judge = time.monotonic()
        verdict = self.llm.judge(goal=state.goal, rendered=render(screen),  ### LLM ###
                                 history=state.history, screenshot=shot,
                                 scratchpad="\n".join(state.scratchpad),
                                 progress="\n".join(state.progress_log),
                                 step=state.step, recorder=rec)
        t_judge = time.monotonic() - t0_judge
        last_call = self.llm.ledger.calls[-1] if (self.llm and self.llm.ledger.calls) else None
        self.on_event("llm_end", step=state.step, purpose="judge", elapsed=t_judge, call=last_call, verdict=verdict)
        state.llm_calls += 1
        rec.event("judge", satisfied=verdict.satisfied, evidence=verdict.evidence)
        if verdict.satisfied:
            log.info("verified: %s", verdict.evidence)
            return "success"
        log.warning("premature 'done': %s", verdict.evidence)
        state.history.append(f"{state.step}. claimed done; rejected: {verdict.evidence}")
        state.last_failure = f"your 'done' was rejected: {verdict.evidence}"
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

    def _maybe_give_up(self, state: RunState) -> None:
        if state.consecutive_failures >= self.cfg.run.max_consecutive_failures:
            log.error("giving up after %d consecutive failures",
                      state.consecutive_failures)
            state.finished = "failed"
