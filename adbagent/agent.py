"""The control loop.

Read this file to understand the whole system. The shape is:

    perceive -> fingerprint -> cache lookup -> (replay | ask the LLM)
             -> guard -> act -> verify -> learn -> repeat

The LLM appears in exactly two places, both marked `### LLM ###`. Everything
else -- recognising the screen, resolving an anchor, dismissing a nag, deciding
whether an action worked, noticing a loop, ending the run when a programmatic
assertion passes -- is ordinary code. That is the entire point: the model is
consulted when the agent is genuinely uncertain, and not otherwise.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import safety, trust
from .actions import (ActionError, AgentAction, execute,
                      synthesise_postcondition, verify)
from .config import Config
from .device import Device, DeviceTimeout, DeviceLost
from .llm import BudgetExceeded, LLMClient, LLMError
from .memory import CachedStep, Memory, intent_key
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
    cache_hits: int = 0
    consecutive_failures: int = 0
    history: List[str] = field(default_factory=list)
    visits: Dict[str, int] = field(default_factory=dict)
    loops: LoopDetector = field(default_factory=LoopDetector)
    want_screenshot: bool = False
    last_failure: str = ""
    scroll_warnings: int = 0
    audits: int = 0
    audits_agreed: int = 0
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

    def cache_rate(self) -> float:
        return self.cache_hits / self.step if self.step else 0.0

    def audit_agreement(self) -> Optional[float]:
        """Measured cache precision, or None when nothing was audited."""
        return self.audits_agreed / self.audits if self.audits else None


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
                 on_event=None, sampler=None):
        self.dev = dev
        self.mem = mem
        self.llm = llm
        self.cfg = cfg
        self.oracle = oracle or Oracle()
        self.on_event = on_event or (lambda *a, **k: None)
        #: Injectable so tests can force or suppress shadow audits.
        self.sampler = sampler or random.random

    def _shadow_audit(self, state: RunState, screen: Screen,
                      entry: CachedStep, cached: AgentAction,
                      rec: Recorder) -> None:
        """Occasionally ask the model anyway, and record whether it agreed.

        We still execute the cache's answer -- this measures the cache, it does
        not second-guess it. Without this the trust states are an assumption;
        with it, disagreement over recent audits is the cache's measured
        precision, and a persistently high rate means the fingerprint is too
        loose.
        """
        if self.llm is None:
            return
        rate = trust.shadow_audit_rate(
            entry.state, self.cfg.memory.shadow_audit_probation,
            self.cfg.memory.shadow_audit_active,
            self.cfg.memory.shadow_audit_trusted)
        if rate <= 0 or self.sampler() >= rate:
            return
        try:
            proposed = self.llm.decide(                        ### LLM (audit only)
                goal=state.goal, rendered=render(screen), history=state.history,
                width=screen.width, height=screen.height, package=screen.package)
        except (LLMError, BudgetExceeded) as exc:
            log.debug("shadow audit skipped: %s", exc)
            return
        state.llm_calls += 1
        agreed = proposed.signature() == cached.signature()
        state.audits += 1
        if agreed:
            state.audits_agreed += 1
        else:
            log.warning("shadow audit DISAGREES on %s: cache says %s, model says %s",
                        entry.describe(), cached.describe(), proposed.describe())
        rec.event("shadow_audit", entry_id=entry.id, agreed=agreed,
                  cached=cached.describe(), proposed=proposed.describe())

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
            self.mem.end_run(run_id, outcome, state.step, state.llm_calls,
                             state.cache_hits, usd)
            recorder.event("run_end", outcome=outcome, steps=state.step,
                           llm_calls=state.llm_calls, cache_hits=state.cache_hits,
                           audits=state.audits, audit_agreement=state.audit_agreement(),
                           usd=round(usd, 6))
            recorder.close()

        return state.finished or "failed", state

    # -- internals ---------------------------------------------------------

    def _loop(self, state: RunState, rec: Recorder) -> None:
        cfg = self.cfg
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
            screen = self.dev.observe()
            self.mem.note_screen(screen)

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
                continue

            hint = state.loops.hint(screen.exact_id)

            # Scroll awareness: give the LLM full context about its
            # scrolling pattern so it can course-correct on its own.
            scroll_ctx = state.loops.scroll_context()
            if scroll_ctx:
                hint = scroll_ctx
                # Also ban cached scroll replays so only the LLM decides.
                if state.loops.scroll_oscillating():
                    for _, sig in state.loops.history:
                        if sig.startswith("scroll/"):
                            state.loops.ban(screen.skeleton_id, sig)

            if state.loops.should_force_back(screen.exact_id) or state.loops.oscillating():
                log.warning("step %d: stuck in a loop; going back", state.step)
                rec.event("loop_break", exact_id=screen.exact_id)
                self.dev.press("back")
                state.loops.record(screen.exact_id, "forced-back")
                continue

            # ---- 3. cache lookup (no LLM) -------------------------------
            visit = state.visits.get(screen.skeleton_id, 0)
            state.visits[screen.skeleton_id] = visit + 1

            entry, action, source = self._from_cache(state, screen)
            if entry is not None and action is not None:
                self._shadow_audit(state, screen, entry, action, rec)

            # ---- 4. ask the model, but only if we must ------------------
            screenshot: Optional[bytes] = None
            if action is None:
                if self.llm is None:
                    log.error("cache miss and no LLM configured")
                    state.finished = "failed"
                    return
                want, note = needs_screenshot(state, screen, cfg)
                if want:
                    screenshot = self.dev.screenshot()
                notes = " ".join(filter(None, (note, hint, state.last_failure)))
                action = self.llm.decide(                      ### LLM ###
                    goal=state.goal, rendered=render(screen), history=state.history,
                    width=screen.width, height=screen.height, package=screen.package,
                    screenshot=screenshot, note=notes,
                    scratchpad="\n".join(state.scratchpad),
                    progress="\n".join(state.progress_log))
                state.llm_calls += 1
                state.want_screenshot = action.confidence == "low"
                source = "llm"

            # Last-resort guard: if the LLM was given full scroll context
            # multiple times and still insists on scrolling, reject it.
            if action.action == "scroll" and state.loops.scroll_oscillating():
                state.scroll_warnings += 1
                if state.scroll_warnings >= 3:
                    log.warning("step %d: rejecting scroll after %d warnings",
                                state.step, state.scroll_warnings)
                    rec.event("scroll_rejected", step=state.step,
                              action=action.describe())
                    state.last_failure = (
                        "scrolling was blocked because you have been "
                        "alternating up and down despite being told to stop. "
                        "Do something else or report done/fail.")
                    state.consecutive_failures += 1
                    state.history.append(
                        f"{state.step}. {action.describe()} -> rejected "
                        f"(scroll oscillation, warning #{state.scroll_warnings})")
                    del state.history[:-12]
                    self._maybe_give_up(state)
                    continue

            # -- accumulate scratchpad notes --------------------------------
            if getattr(action, "notes", None):
                cap = cfg.run.scratchpad_max_chars
                note_text = action.notes.strip()
                if state.scratchpad_chars + len(note_text) > cap:
                    # Trim to fit within the budget.
                    room = max(0, cap - state.scratchpad_chars)
                    if room > 0:
                        note_text = note_text[:room]
                    else:
                        note_text = ""
                if note_text:
                    # Detect cumulative updates: if the new note contains
                    # the last entry, the LLM rewrote a full summary.
                    # Replace instead of append to avoid repetition.
                    if (state.scratchpad and
                            state.scratchpad[-1] in note_text):
                        old = state.scratchpad.pop()
                        state.scratchpad_chars -= len(old)
                    state.scratchpad.append(note_text)
                    state.scratchpad_chars += len(note_text)

            # -- accumulate progress ----------------------------------------
            if getattr(action, "progress", None):
                prog_text = action.progress.strip()
                if prog_text:
                    state.progress_log.append(prog_text)
                    state.progress_chars += len(prog_text)
                    # Keep only the most recent entries to stay bounded.
                    while len(state.progress_log) > 5:
                        removed = state.progress_log.pop(0)
                        state.progress_chars -= len(removed)

            rec.event("decide", step=state.step, source=source,
                      skeleton=screen.skeleton_id, action=action.model_dump(),
                      entry_id=getattr(entry, "id", None), screenshot=bool(screenshot))
            self.on_event("step", state=state, screen=screen, action=action,
                          source=source, screenshot=bool(screenshot))

            # ---- 5. guard the chosen action -----------------------------
            label = safety.irreversible(action, screen)
            if label is not None:
                if not safety.confirm(
                        f"Step {state.step}: the agent wants to press {label!r} "
                        f"in {screen.package}. This cannot be undone.", cfg):
                    state.history.append(f"{state.step}. refused to press {label!r}")
                    state.last_failure = (f"pressing {label!r} was refused; find "
                                          f"another way or stop")
                    rec.event("refused", label=label)
                    continue

            if action.is_terminal:
                state.finished = self._terminal(state, screen, action, rec)
                if state.finished is None:
                    continue
                return

            if cfg.run.dry_run:
                log.info("dry run: would %s", action.describe())
                state.history.append(f"{state.step}. (dry run) {action.describe()}")
                continue

            # ---- 6. act -------------------------------------------------
            try:
                element = execute(self.dev, action, screen)
            except (ActionError, ValueError) as exc:
                log.warning("step %d: %s", state.step, exc)
                state.last_failure = str(exc)
                state.consecutive_failures += 1
                if entry is not None:
                    self.mem.mark(entry, "hard_fail", state.run_id, str(exc))
                self._maybe_give_up(state)
                continue
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
                continue

            # ---- 7. verify (no LLM) -------------------------------------
            after = self.dev.observe(settle=True)
            post = (entry.postcondition if entry is not None
                    else synthesise_postcondition(action, element))
            expected = entry.next_skeleton_id if entry is not None else ""
            if entry is not None and after.skeleton_id in entry.alt_successors:
                expected = after.skeleton_id
            outcome = verify(action, screen, after, post, expected or None)

            rec.event("verify", step=state.step, grade=outcome.grade,
                      reason=outcome.reason, after=after.skeleton_id)

            # ---- 8. learn (no LLM) --------------------------------------
            if entry is not None:
                self.mem.mark(entry, outcome.grade, state.run_id, outcome.reason,
                              observed_successor=after.skeleton_id)
                if not outcome.ok:
                    # Rewind so the next pass looks the entry up again, misses
                    # (it has just been demoted), and hands over to the model.
                    state.visits[screen.skeleton_id] = visit
                    state.last_failure = (f"the remembered action "
                                          f"{action.describe()} failed: {outcome.reason}")
                    state.consecutive_failures += 1
            elif outcome.ok:
                self.mem.record(screen=screen, intent_id=state.intent_id, visit=visit,
                                action=action, element=element, postcondition=post,
                                after=after, run_id=state.run_id)
            else:
                state.consecutive_failures += 1
                state.last_failure = f"{action.describe()} failed: {outcome.reason}"
                state.want_screenshot = True

            if outcome.ok:
                state.consecutive_failures = 0
                state.last_failure = ""
            if outcome.grade == "no_change":
                state.loops.ban(screen.skeleton_id, action.signature())
                if action.action == "scroll":
                    h_dir = action.direction in ("left", "right")
                    axis = "horizontal" if h_dir else "vertical"
                    state.last_failure = (
                        f"Scrolling {action.direction} did not reveal new "
                        f"content \u2014 you have reached the end of the "
                        f"{axis} scrollable area. Do not scroll "
                        f"{action.direction} again here.")

            state.loops.record(screen.exact_id, action.signature())
            state.history.append(
                f"{state.step}. {action.describe()} -> {outcome.grade}"
                + (f" ({outcome.reason})" if outcome.reason else ""))
            del state.history[:-12]  # keep the prompt bounded

            self._maybe_give_up(state)

    # -- cache -------------------------------------------------------------

    def _from_cache(self, state: RunState, screen: Screen
                    ) -> Tuple[Optional[CachedStep], Optional[AgentAction], str]:
        from .fingerprint import destructive_tokens

        entry = self.mem.lookup(
            screen, state.intent_id, state.visits[screen.skeleton_id] - 1,
            forbidden_now=destructive_tokens(screen),
            banned_signatures=list(state.loops.bans_for(screen.skeleton_id)))
        if entry is None:
            return None, None, "llm"

        action = self.mem.rehydrate(entry, screen)
        if action is None and entry.anchor is not None and entry.anchor.scroller_rid:
            action = self._scroll_to_find(entry, screen)
        if action is None:
            self.mem.mark(entry, "hard_fail", state.run_id, "anchor did not bind")
            return None, None, "llm"

        state.cache_hits += 1
        log.info("step %d: cache hit %s", state.step, entry.describe())
        return entry, action, "cache"

    def _scroll_to_find(self, entry: CachedStep, screen: Screen
                        ) -> Optional[AgentAction]:
        """The element is in a list that has moved. Look for it before giving up."""
        # Determine scroll direction from the scroller's orientation.
        direction = "down"
        if entry.anchor is not None and entry.anchor.scroller_rid:
            from .fingerprint import rid_norm
            for el in screen.elements:
                if (el.scrollable
                        and rid_norm(el.resource_id) == entry.anchor.scroller_rid
                        and el.is_horizontal):
                    direction = "right"
                    break
        previous = screen.exact_id
        for attempt in range(4):
            self.dev.scroll(direction)
            current = self.dev.observe(settle=True)
            if current.exact_id == previous:
                return None  # the list did not move; it is not there
            previous = current.exact_id
            if current.skeleton_id != entry.skeleton_id:
                return None  # we scrolled off the screen entirely
            action = self.mem.rehydrate(entry, current, minor_deviation=True)
            if action is not None:
                log.info("found it after %d scroll(s)", attempt + 1)
                return action
        return None

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
        verdict = self.llm.judge(goal=state.goal, rendered=render(screen),  ### LLM ###
                                 history=state.history, screenshot=shot,
                                 scratchpad="\n".join(state.scratchpad),
                                 progress="\n".join(state.progress_log))
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


# ---------------------------------------------------------------------------
# Explore mode
# ---------------------------------------------------------------------------

EXPLORE_GOAL = (
    "Explore this app to learn how it is laid out. Visit as many DIFFERENT "
    "screens as you can. Open a section, look at it, then press back and open "
    "the next one. Do not change any setting, do not type anything, and do not "
    "press any button that sends, buys, deletes or posts. When you have seen "
    "everything reachable, reply done."
)


def explore(dev: Device, mem: Memory, llm: LLMClient, cfg: Config,
            package: str = "", max_screens: int = 40) -> Dict[str, Any]:
    """Wander an app to warm the cache, refusing anything that changes state.

    Read-only by construction: every action the model proposes is classified,
    and anything that could mutate is either skipped or -- when it is the only
    way forward -- put to the user.
    """
    seen: set = set()
    state = RunState(goal=EXPLORE_GOAL, run_id=uuid.uuid4().hex[:12],
                     intent_id=intent_key("explore:" + (package or "any")))
    mem.begin_run(state.run_id, EXPLORE_GOAL, state.intent_id)

    if package:
        dev.open_app(package)
        time.sleep(1.0)

    blocked: List[str] = []
    while len(seen) < max_screens and state.step < cfg.run.max_steps:
        state.step += 1
        screen = dev.observe()
        mem.note_screen(screen)
        seen.add(screen.skeleton_id)

        if safety.sensitive_screen(screen) is not None:
            log.info("skipping a credential screen")
            dev.press("back")
            continue
        if package and screen.package and screen.package != package:
            dev.press("back")
            continue

        want, note = needs_screenshot(state, screen, cfg)
        action = llm.decide(
            goal=EXPLORE_GOAL, rendered=render(screen), history=state.history,
            width=screen.width, height=screen.height, package=screen.package,
            screenshot=dev.screenshot() if want else None,
            note=" ".join(filter(None, (note, state.loops.hint(screen.exact_id)))))
        state.llm_calls += 1

        if action.action in ("done", "fail"):
            break

        ok, why = safety.is_read_only(action, screen)
        if not ok:
            message = (f"Explore wants to {action.describe()} in {screen.package}, "
                       f"which is not read-only ({why}).")
            if not safety.confirm(message, cfg):
                blocked.append(f"{action.describe()}: {why}")
                state.history.append(
                    f"{state.step}. skipped {action.describe()} -- not read-only. "
                    f"Go back and try a different part of the app.")
                del state.history[:-12]
                dev.press("back")
                continue

        try:
            execute(dev, action, screen)
        except (ActionError, ValueError) as exc:
            state.history.append(f"{state.step}. {action.describe()} failed: {exc}")
            continue

        after = dev.observe(settle=True)
        mem.note_transition(screen, after, action)
        state.loops.record(screen.exact_id, action.signature())
        state.history.append(f"{state.step}. {action.describe()} -> "
                             f"{'new screen' if after.skeleton_id not in seen else 'seen before'}")
        del state.history[:-12]

    mem.end_run(state.run_id, "success", state.step, state.llm_calls, 0,
                llm.ledger.total_usd)
    return {"screens": len(seen), "steps": state.step, "llm_calls": state.llm_calls,
            "blocked": blocked, "usd": llm.ledger.total_usd}
