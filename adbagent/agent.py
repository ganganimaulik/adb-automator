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
from .pager import (ItemLedger, attach_item, browsing_note, loop_id,
                    pager_element, set_id as pager_set_id)
from .safety import Aborted, LoopDetector
from .scratchpad import ScratchpadGuard
from .screen import Screen, render
from .skills import SkillRegistry

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
    #: Running scratchpad for data-collection goals.  The LLM writes the COMPLETE
    #: collected state into the ``notes`` field of its action every turn and we
    #: keep only that latest value, because the model re-emits its whole ledger
    #: each turn and appending would archive a hundred near-identical copies.
    #: ``guard`` is what makes overwriting safe -- see `scratchpad.py`.
    scratchpad: List[str] = field(default_factory=list)
    #: Progress tracker for multi-step goals.  The LLM writes into the
    #: ``progress`` field; we keep the latest entry and feed it back.
    progress_log: List[str] = field(default_factory=list)
    #: Append-only archive of every record the model has written, so a rewrite
    #: that quietly drops a figure is caught and handed back rather than lost.
    guard: ScratchpadGuard = field(default_factory=ScratchpadGuard)
    #: Rendered "you dropped these" block, carried to the next turn's prompt.
    dropped_note: str = ""
    #: Which items of a gallery / carousel have actually been looked at. Kept by
    #: code rather than by the model, because a ledger the model rewrites by hand
    #: every turn silently loses an entry the moment it forgets to repeat one.
    items: ItemLedger = field(default_factory=ItemLedger)
    #: What verification concluded about the last gesture on a pager: True the
    #: item advanced, False it did not, None not applicable.
    item_moved: Optional[bool] = None
    #: Key of the item on screen this turn, once resolved.
    item_key: str = ""

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
        an interstitial, breaking a loop with `back` -- clears `screen` to None
        and re-observes, so a screen carrying a screenshot is always current.
        """
        if screen.screenshot is None:
            screen.screenshot = self.dev.screenshot()
        if screen.dhash is None:
            from .fingerprint import compute_dhash
            screen.dhash = compute_dhash(screen.screenshot)
        return screen.screenshot

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
                           usd=round(usd, 6),
                           llm=step_metrics(self.llm.ledger.calls if self.llm else [],
                                            detail=False))
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

            # A pager item is ledgered here, once its identity and whether we
            # have vision on it are both settled. `read` is deliberately keyed to
            # the screenshot and not to the sighting: having seen an item's
            # caption is not the same as having looked at the item.
            pager_note = ""
            if screen.is_pager:
                state.items.note(state.item_key, screen, state.step,
                                 read=bool(screenshot))
                if screen.item_label:
                    # `item_moved` is a latch, not a per-turn flag: it has to
                    # survive the turns where the caption is hidden, because the
                    # tap that reveals the caption again is not itself a move.
                    # Here a caption was available, so the latch has been spent.
                    state.item_moved = None
                pager_note = "\n".join(filter(None, (
                    browsing_note(screen, state.items,
                                  swipe_failed=state.item_moved is False),
                    state.items.render(state.item_key, screen.item_label))))
                rec.event("pager_item", step=state.step, key=state.item_key,
                          label=screen.item_label, read=bool(screenshot),
                          read_count=state.items.read_count,
                          total=state.items.total)

            banned_actions = state.loops.bans_for(screen.skeleton_id)
            ban_note = ""
            if banned_actions:
                ban_note = (f"BANNED ACTIONS on this screen (these produced NO change - DO NOT REPEAT): "
                            f"{', '.join(sorted(banned_actions))}.")
            # Check for active app skill guidance
            skill_note = ""
            if cfg.skills.enabled:
                active_skill = self.skills.find_for_run(screen.package, state.goal)
                if active_skill:
                    skill_note = active_skill.to_prompt_text()
                    rec.event("active_skill", name=active_skill.name, package=screen.package)
                    if getattr(self, "_active_skill_name", None) != active_skill.name:
                        self._active_skill_name = active_skill.name
                        self.on_event("skill_loaded", name=active_skill.name, package=screen.package)

            elem_hint = state.loops.element_history_hint(
                screen.skeleton_id,
                repeatable=pager_el.index if pager_el is not None else 0)
            notes = "\n\n".join(filter(None, (note, state.dropped_note, pager_note,
                                             hint, elem_hint, ban_note,
                                             state.last_failure, skill_note)))
            model_name = self.llm.model if self.llm else ""
            self.on_event("llm_start", step=state.step, purpose="decide", model=model_name, screenshot=bool(screenshot))
            t0_llm = time.monotonic()
            ledger_mark = self.llm.ledger.mark() if self.llm else 0
            action = self.llm.decide(                      ### LLM ###
                goal=state.goal, rendered=render(screen), history=state.history,
                width=screen.width, height=screen.height, package=screen.package,
                screenshot=screenshot, note=notes,
                scratchpad="\n".join(state.scratchpad),
                progress="\n".join(state.progress_log),
                step=state.step, recorder=rec,
                on_event=self.on_event)
            t_llm = time.monotonic() - t0_llm
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
            # The latest note replaces the previous one; the guard is what keeps
            # that from being lossy. It archives every record the model writes and
            # tells us which ones this rewrite stopped covering, so a dropped
            # figure comes back in the next prompt instead of vanishing for the
            # rest of the run.
            if getattr(action, "notes", None):
                cap = cfg.run.scratchpad_max_chars
                note_text = action.notes.strip()[:cap]
                if note_text:
                    losses = state.guard.update(note_text, state.step)
                    state.dropped_note = state.guard.report(losses)
                    if losses:
                        rec.event("scratchpad_dropped", step=state.step,
                                  keys=[loss.key for loss in losses],
                                  reported=bool(state.dropped_note))
                    # Append scroll position context so the LLM has spatial
                    # memory even if it forgets to include it in its notes.
                    if state.loops.total_scroll_count > 0:
                        scroll_info = (f"\n[Scroll stats: {state.loops.total_scroll_count} "
                                       f"scroll(s) so far, {state.loops.direction_reversals()} "
                                       f"direction reversal(s)]")
                        note_text = note_text + scroll_info
                    state.scratchpad = [note_text]

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
            if (screen.is_pager and state.item_key and action.observation
                    and screenshot):
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
                      wall_s=round(t_llm, 3), llm=step_metrics(step_calls))
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
                if want or action.action in ("scroll", "swipe") or state.want_screenshot:
                    # Also the screenshot the *next* turn will show the model, if
                    # it wants one -- `_ensure_screenshot` will not re-take it.
                    self._ensure_screenshot(after)
            except (DeviceTimeout, DeviceLost) as exc:
                if not self._recover_device(state, exc):
                    state.finished = "aborted"
                    return
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

        shot = self._ensure_screenshot(screen)
        model_name = (self.llm.model_image if shot else self.llm.model_small) if self.llm else ""
        self.on_event("llm_start", step=state.step, purpose="judge", model=model_name, screenshot=bool(shot))
        t0_judge = time.monotonic()
        ledger_mark = self.llm.ledger.mark()
        # The judge is shown everything the run collected, including records a
        # later rewrite of the notes dropped. Without that it grades the goal on
        # whatever survived the model's last edit -- which is how a run that had
        # read a value ends up reporting it as unavailable.
        collected = state.guard.preserved("\n".join(state.scratchpad))
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
