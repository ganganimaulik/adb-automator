"""A run whose decisions come from outside this process.

`Agent` asks its model for an action once per step, in `_decide`, and that one
call is all `RemoteAgent` changes: it hands the turn to whoever is driving the
session -- a coding agent connected over MCP (`mcp_server.py`) -- and waits for
the action to come back. Everything around that call still runs here: the
interstitial dismissals, the sensitive-screen hand-over, the guards, the
verification, the stall ladder, and the trace in `runs/<id>/` that the web UI
and `adbagent report` read. Nothing about a run depends on who decided it, so
nothing had to be written twice.

The run is an ordinary `Agent.run` on a thread of its own. `Session` talks to it
through two queues in lock step -- one message in, one `Turn` out -- so the run
thread is the only thread that ever touches the phone, and a client call is
answered with the turn the loop is actually waiting on.

What the remote decider is shown is what the model would have been shown, less
what it already has. It keeps its own transcript, so the history comes as the
lines added since its last call rather than the whole of it, and the standing
advice blocks the loop repeats every turn are sent once per session.

The approvals that ask the person at a terminal -- an irreversible press, and a
send (`Agent._check_send`, a second model's job when the model decides) -- are
put to the person through the client instead: the turn comes back with a
`pending` approval, and the client asks them (`mcp_server.approve`).
"""

from __future__ import annotations

import copy
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import conversation, prompts, runlog
from .actions import POINT_ACTIONS, AgentAction
from .agent import Agent, Recorder, RunState
from .config import Config
from .phonelock import PhoneLock
from .safety import Aborted
from .screen import Screen
from .watch import screen_digest

log = logging.getLogger("adbagent.remote")

#: Advice the loop repeats on every turn its condition holds -- see
#: `prompts.situational_notes`. A model that is handed a fresh prompt each turn
#: needs them each turn; a client keeping its own transcript needs each once.
STANDING_ADVICE = (prompts.SCROLLING_ADVICE, prompts.COLLECTION_ADVICE,
                   prompts.MULTI_APP_ADVICE, prompts.TAP_AT_ADVICE,
                   prompts.TAP_AT_ADVICE_BLIND, prompts.STUCK_ADVICE)

#: How long a session waits for its next instruction before it gives the phone
#: back. The session holds the phone exclusively and with its settings changed
#: (no animations, the agent's keyboard, rotation locked), so a client that has
#: gone away must not keep it that way indefinitely.
IDLE_TIMEOUT_S = 15 * 60

#: How long a client call waits for the run thread to answer. Every device call
#: under it has its own watchdog, so this only fires if the thread is gone.
ANSWER_TIMEOUT_S = 10 * 60

_HIST_STEPS = re.compile(r"^(\d+)(?:-(\d+))?\.")
_WARNING_STEP = re.compile(r"^step \d+:\s*")


class SessionError(RuntimeError):
    """A call the session cannot answer as asked."""


@dataclass
class Approval:
    """Something the person has to agree to before the run may do it."""

    id: int
    kind: str   # "irreversible" | "send"
    prompt: str


@dataclass
class Turn:
    """What the driver is told after each call."""

    step: int = 0
    package: str = ""
    skeleton: str = ""
    screen: str = ""
    notes: str = ""
    budget: str = ""
    #: The goal and the operator's policy, on the first turn only: the driver
    #: did not necessarily write either -- a policy carries its own goal.
    goal: str = ""
    policy: str = ""
    #: The app's skill, when it is not the one already sent.
    skill: str = ""
    #: History lines added since the driver's last call, and anything the
    #: harness did on its own initiative that left no line of its own.
    happened: List[str] = field(default_factory=list)
    screenshot: Optional[bytes] = None
    pending: Optional[Approval] = None
    #: Why the last step did not work, "" when it did. Not rendered on its own:
    #: the loop already puts it in `notes`. It is what a batch stops on.
    failure: str = ""
    #: Set once the run has ended, with what it concluded.
    outcome: str = ""
    result: str = ""
    evidence: str = ""
    run_dir: str = ""

    def render(self) -> str:
        """The turn as text, in the order the loop's own prompt puts it."""
        parts: List[str] = []
        if self.outcome:
            ended = f"SESSION ENDED: {self.outcome}"
            if self.result:
                ended += f"\n  {self.result}"
            if self.evidence:
                ended += f"\n  ({self.evidence})"
            if self.run_dir:
                ended += f"\n  trace: {self.run_dir}"
            parts.append(ended)
        if self.happened:
            parts.append("WHAT HAPPENED:\n" + "\n".join(
                f"  {line}" for line in self.happened))
        if self.pending is not None:
            parts.append(
                f"APPROVAL NEEDED (request_id {self.pending.id}): "
                f"{self.pending.prompt}\n  Call `approve` with this request_id "
                f"to ask the person. Any other call declines it.")
        if self.outcome:
            return "\n\n".join(parts)
        if self.goal:
            parts.append(f"GOAL:\n{self.goal}")
        if self.policy:
            parts.append(self.policy)
        if self.skill:
            parts.append(f"APP SKILL (what earlier runs learned about this "
                         f"app):\n{self.skill}")
        if self.screen:
            parts.append(f"CURRENT SCREEN (step {self.step}):\n{self.screen}")
        if self.notes:
            parts.append(f"NOTE: {self.notes}")
        if self.budget:
            parts.append(self.budget)
        return "\n\n".join(parts)


def remote_config(cfg: Config) -> Config:
    """`cfg` as a session driven from outside should run it."""
    cfg = copy.deepcopy(cfg)
    # The driver is shown the frame itself whenever the loop wants one looked
    # at, so the notes offer the x/y form of `tap_at` and the frame is kept
    # beside the step that saw it, as for a model that sees.
    cfg.llm.vision_in_decider = True
    # The sweep repeats a gesture in code and has a vision model read each item
    # it lands on. There is no model here to read them.
    cfg.run.pager_sweep = False
    return cfg


def _first_step(line: str) -> Optional[int]:
    m = _HIST_STEPS.match(line)
    return int(m.group(1)) if m else None


def _merge(lines: List[str], more: Sequence[str]) -> None:
    """Append `more` to `lines`, a fold replacing the line it folds.

    `append_history` folds a repeat into the previous line ("3. back" becomes
    "3-4. back [x2]"), and a batch hears about step 3 and then about the fold;
    the driver needs the fold, not both.
    """
    for line in more:
        step = _first_step(line)
        if lines and step is not None and step == _first_step(lines[-1]):
            lines[-1] = line
        else:
            lines.append(line)


def _covered_steps(lines: Sequence[str]) -> set:
    steps: set = set()
    for line in lines:
        m = _HIST_STEPS.match(line)
        if m:
            first = int(m.group(1))
            steps.update(range(first, int(m.group(2) or first) + 1))
    return steps


class RemoteAgent(Agent):
    """An `Agent` whose decisions and approvals come through two queues."""

    def __init__(self, dev: Any, cfg: Config, *, inbox: "queue.Queue",
                 outbox: "queue.Queue", driver: str = "",
                 idle_timeout_s: float = IDLE_TIMEOUT_S, policy: str = "",
                 send_limits: Optional[Dict[str, int]] = None):
        super().__init__(dev, None, cfg, on_event=self._heard, policy=policy,
                         send_limits=send_limits)
        self._inbox = inbox
        self._outbox = outbox
        self.driver = driver
        self.idle_timeout_s = idle_timeout_s
        self._state: Optional[RunState] = None
        self._last_turn = Turn()
        self._seen = 0
        self._last_line = ""
        self._warnings: List[Tuple[int, str]] = []
        self._advised: set = set()
        self._skill_sent = ""
        self._briefed = False
        self._asked = 0
        #: The last thing the harness said on its own, for a run that ends on it.
        self.last_warning = ""
        #: Why the session stopped the run, when it was the session that did.
        self.abort_reason = ""
        #: Set when a `look` found the screen changed and sent the loop back to
        #: read it: the turn that follows owes the driver a screenshot.
        self._looked = False

    # -- the seams -----------------------------------------------------------

    def _decider_name(self) -> str:
        return f"{self.driver or 'client'} (mcp)"

    def _loop(self, state: RunState, rec: Recorder) -> None:
        self._state = state
        super()._loop(state, rec)

    def _decide(self, state: RunState, rec: Recorder, screen: Screen,
                **kwargs: Any) -> Optional[AgentAction]:
        turn = self._turn(state, screen, kwargs)
        self._post(turn)
        while True:
            message = self._next(state)
            kind = message[0]
            if kind == "act":
                return message[1]
            if kind == "look":
                if self._moved(screen):
                    # A fresh picture beside the list this turn was read from
                    # would put `#N` on a screen that is gone -- a dialog that
                    # came up, a page that finished loading. Read it again: one
                    # step, spent only when the answer would otherwise be wrong.
                    self._looked = True
                    return None
                self._post(replace(turn, happened=[], skill="",
                                   screenshot=self._shot()))
            elif kind == "release":
                self._release_phone(state, rec)
                self._post(replace(turn, happened=[
                    "the phone is the person's until it is taken back"],
                    skill="", screenshot=None))
            elif kind == "reclaim":
                self._reclaim_phone(state, rec)
                return None
            elif kind == "abort":
                raise self._abort(message[1])
            else:
                self._post(replace(turn, happened=[], skill=""))

    def _confirm(self, prompt: str) -> bool:
        if self.cfg.safety.allow_destructive:
            return super()._confirm(prompt)
        return self._ask("irreversible", prompt)

    @property
    def _checks_sends(self) -> bool:
        return True

    def _check_send(self, state: RunState, rec: Recorder, screen: Screen,
                    action: AgentAction, control: str) -> str:
        """The person, not a second model, says whether a send goes out.

        Fails closed like the model check it replaces: no answer is a no. A no
        is remembered against the frame, so the same send asked for again on
        the same screen is refused without asking the person twice.
        """
        key = f"{screen.exact_id}/{action.signature()}"
        if key in state.refused_sends:
            return state.refused_sends[key]
        thread = conversation.thread_on(screen)
        draft = conversation.draft_in(action, screen)
        prompt = f"The agent wants to send with {control!r}"
        if thread is not None and thread.title:
            prompt += f" in the conversation with {thread.title}"
        prompt += f" ({screen.package})." if screen.package else "."
        if draft:
            prompt += f" It says: {draft!r}"
        allowed = self._ask("send", prompt)
        rec.event("send_check", step=state.step, control=control, send=allowed,
                  reason=("the person allowed it" if allowed
                          else "the person did not allow it"),
                  asked="person")
        if allowed:
            return ""
        refusal = ("the person did not allow this send -- leave it and move on "
                   "to whatever else the goal needs, or end the session")
        state.refused_sends[key] = refusal
        return refusal

    def _hand_over(self, state: RunState, reason: str) -> None:
        # No print: stdout is the protocol channel of a stdio MCP server.
        log.warning("handing over: %s", reason)
        if not state.evidence:
            state.evidence = reason
        state.finished = "needs_user"

    # -- talking to the session ---------------------------------------------

    def _post(self, turn: Turn) -> None:
        self._last_turn = turn
        self._outbox.put(turn)

    def _next(self, state: RunState) -> tuple:
        """The driver's next message. Time spent waiting for it is not work.

        Given back through `paused_s`, as a hold from the web UI is: the
        wall-clock budget bounds how long the run spends *working*, and a driver
        reading, thinking or asking its person is not the run working.
        """
        waited_from = time.monotonic()
        try:
            return self._inbox.get(timeout=self.idle_timeout_s)
        except queue.Empty:
            raise self._abort(
                f"no instruction for {self.idle_timeout_s / 60:.0f} minutes; "
                f"the session ended and the phone was handed back") from None
        finally:
            state.paused_s += time.monotonic() - waited_from

    def _abort(self, reason: str) -> Aborted:
        self.abort_reason = reason
        return Aborted(reason)

    def _ask(self, kind: str, prompt: str) -> bool:
        state = self._state
        assert state is not None
        self._asked += 1
        approval = Approval(id=self._asked, kind=kind, prompt=prompt)
        self._post(replace(self._last_turn, happened=self._delta(state),
                           pending=approval, skill="", screenshot=None))
        while True:
            message = self._next(state)
            kind_ = message[0]
            if kind_ == "answer" and message[1] == approval.id:
                return bool(message[2])
            if kind_ == "look":
                self._post(replace(self._last_turn, happened=[],
                                   screenshot=self._shot()))
            elif kind_ == "abort":
                raise self._abort(message[1])
            else:
                return False

    def _heard(self, kind: str, **fields: Any) -> None:
        if kind not in ("loop_warning", "safety_warning"):
            return
        message = str(fields.get("message") or "")
        step = self._state.step if self._state is not None else 0
        self._warnings.append((step, message))
        self.last_warning = _WARNING_STEP.sub("", message)

    # -- building the turn ---------------------------------------------------

    def _shot(self) -> Optional[bytes]:
        try:
            return self.dev.screenshot()
        except Exception as exc:  # noqa: BLE001 - a missing frame is not fatal
            log.warning("screenshot for the driver failed: %s", exc)
            return None

    def _moved(self, screen: Screen) -> bool:
        """Whether the phone shows something other than `screen` now.

        Compared on what the app has written (masked, so a ticking clock is not
        a change), the app in front, and the layout -- one UI dump, no model.
        """
        try:
            now = self.dev.observe()
        except Exception as exc:  # noqa: BLE001 - then the frame is all there is
            log.warning("could not re-read the screen for a look: %s", exc)
            return False
        return (now.package != screen.package
                or now.skeleton_id != screen.skeleton_id
                or screen_digest(now) != screen_digest(screen))

    def _delta(self, state: RunState) -> List[str]:
        """History lines added since the last turn, and harness acts beside them."""
        history = state.history
        start = min(self._seen, len(history))
        # `append_history` folds a repeat into the line before it, so the last
        # line already sent may have been rewritten since.
        if start and history[start - 1] != self._last_line:
            start -= 1
        lines = list(history[start:])
        covered = _covered_steps(lines)
        for step, message in self._warnings:
            if step not in covered:
                lines.append(f"{step}. the harness: "
                             f"{_WARNING_STEP.sub('', message)}")
        self._warnings.clear()
        self._seen = len(history)
        self._last_line = history[-1] if history else ""

        def order(line: str) -> int:
            m = _HIST_STEPS.match(line)
            return int(m.group(1)) if m else 0
        return sorted(lines, key=order)

    def _fresh_notes(self, note: str) -> str:
        for block in STANDING_ADVICE:
            if block in note:
                if block in self._advised:
                    note = note.replace(block, "")
                else:
                    self._advised.add(block)
        return re.sub(r"\n{3,}", "\n\n", note).strip()

    def _turn(self, state: RunState, screen: Screen,
              kwargs: dict) -> Turn:
        skill = kwargs.get("skill") or ""
        fresh_skill = skill if skill != self._skill_sent else ""
        self._skill_sent = skill
        happened = self._delta(state)
        screenshot = kwargs.get("screenshot")
        if self._looked:
            self._looked = False
            happened.append("the screen changed since the last turn; this is "
                            "it now")
            if screenshot is None:
                screenshot = self._shot()
        goal = policy = ""
        if not self._briefed:
            self._briefed = True
            goal = kwargs.get("goal") or ""
            policy = prompts.policy_block(kwargs.get("policy") or "")
        return Turn(step=state.step, package=screen.package,
                    skeleton=screen.skeleton_id,
                    screen=kwargs.get("rendered") or "",
                    notes=self._fresh_notes(kwargs.get("note") or ""),
                    budget=kwargs.get("budget") or "",
                    goal=goal, policy=policy,
                    skill=fresh_skill,
                    happened=happened,
                    screenshot=screenshot,
                    failure=state.last_failure)


def open_phone(cfg: Config) -> Any:
    """The real phone, found the way `adbagent run` finds it but never asking.

    A stdio MCP server has no terminal to show a pairing code on, so a phone
    that cannot be found is an error for the session to report, not a prompt.
    """
    from .device import Device, auto_connect_devices
    try:
        auto_connect_devices(cfg.device.serial)
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        log.debug("adb auto-connect failed: %s", exc)
    return Device(cfg, cfg.device.serial).open()


class Session:
    """One goal on one phone for one driver, from `start` to the end of the run.

    Every method is a client call: it sends the run thread one message and
    returns the turn that comes back. Calls are serialised -- the protocol is
    lock step, and a client that issues two at once gets them one after the
    other rather than interleaved.
    """

    def __init__(self, cfg: Config, *,
                 open_device: Callable[[Config], Any] = open_phone,
                 driver: str = "", idle_timeout_s: float = IDLE_TIMEOUT_S,
                 policy: str = "",
                 send_limits: Optional[Dict[str, int]] = None):
        self.cfg = remote_config(cfg)
        self.driver = driver
        #: The operator's instructions, verbatim, and the caps on what one
        #: session may send -- carried the way a watch pass carries them.
        self.policy = policy
        self.send_limits = dict(send_limits or {})
        self.idle_timeout_s = idle_timeout_s
        self._open_device = open_device
        self._inbox: "queue.Queue" = queue.Queue()
        self._outbox: "queue.Queue" = queue.Queue()
        self._calls = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._last: Optional[Turn] = None
        self._released = False
        self.goal = ""
        self.lock = PhoneLock(self.cfg.run.artifacts_dir,
                              f"an MCP session ({driver or 'client'})")

    # -- state ---------------------------------------------------------------

    @property
    def live(self) -> bool:
        """Started, and its run has not ended -- whether or not anyone has read
        the ending yet."""
        if self._thread is None or (self._last and self._last.outcome):
            return False
        return self._thread.is_alive()

    @property
    def pending(self) -> Optional[Approval]:
        return self._last.pending if self._last is not None else None

    @property
    def last(self) -> Optional[Turn]:
        """The most recent turn, the final one once the run has ended."""
        return self._last

    # -- client calls ----------------------------------------------------------

    def start(self, goal: str) -> Turn:
        """Take the phone and run until the first decision. Raises `PhoneBusy`."""
        with self._calls:
            if self._thread is not None:
                raise SessionError("this session has already started")
            self.goal = goal
            self.lock.who = f"an MCP session ({self.driver or 'client'}): {goal}"
            self.lock.acquire()
            self._thread = threading.Thread(target=self._run, args=(goal,),
                                             name="adbagent-session",
                                             daemon=True)
            self._thread.start()
            return self._receive()

    def act(self, actions: Sequence[AgentAction]) -> Turn:
        """Do `actions` in order, stopping at the first that does not work.

        Also stops before an action that names an element or a point once the
        screen has changed under the batch: its `#N` or its x/y were read off a
        screen that is no longer there.
        """
        if not actions:
            raise SessionError("no actions given")
        with self._calls:
            if self._released:
                raise SessionError("the person has the phone; it comes back "
                                   "when they say they are done")
            happened: List[str] = []
            turn = self._decline_pending(happened)
            if turn.outcome:
                return turn
            planned_on = turn.skeleton
            for i, action in enumerate(actions):
                names_something = (action.target is not None
                                   or action.action in POINT_ACTIONS)
                # A key names the element itself, so a target given by one that
                # is still on screen still means what it meant -- the profile
                # scroller a down-fling and an up-fling are both aimed at.
                still_there = (action.target is not None
                               and bool(action.target.key)
                               and f"k={action.target.key}" in turn.screen)
                if (i and names_something and not still_there
                        and turn.skeleton != planned_on):
                    happened.append(
                        f"stopped before the remaining {len(actions) - i} "
                        f"action(s): the screen changed, so their targets may "
                        f"no longer be what they were")
                    break
                turn = self._exchange(("act", action))
                _merge(happened, turn.happened)
                if turn.outcome or turn.pending is not None or turn.failure:
                    break
            return replace(turn, happened=happened)

    def look(self) -> Turn:
        """The screen as it is now, with a screenshot.

        The turn the run is deciding on, when the phone still shows it; read
        again first when it does not.
        """
        with self._calls:
            self._require_live()
            return self._exchange(("look",))

    def answer(self, request_id: int, allow: bool) -> Turn:
        """Answer a pending approval."""
        with self._calls:
            pending = self.pending
            if pending is None or pending.id != request_id:
                raise SessionError(f"no approval {request_id} is waiting")
            return self._exchange(("answer", request_id, allow))

    def release(self) -> Turn:
        """Give the phone to the person: the device session closes, restored."""
        with self._calls:
            happened: List[str] = []
            turn = self._decline_pending(happened)
            if turn.outcome or self._released:
                return turn
            turn = self._exchange(("release",))
            self._released = True
            return replace(turn, happened=happened + turn.happened)

    def reclaim(self) -> Turn:
        """Take the phone back and read the screen from scratch."""
        with self._calls:
            if not self._released:
                raise SessionError("the phone was not handed over")
            self._released = False
            return self._exchange(("reclaim",))

    def end(self, summary: str, failed: bool = False) -> Turn:
        """Finish the run with `done` (or `fail`) and hand the phone back."""
        action = AgentAction(observation="", reasoning="",
                             action="fail" if failed else "done",
                             text=summary or None)
        turn = self.act([action])
        if not turn.outcome:
            # A `done` the harness would not take -- an assertion that still
            # fails -- leaves the run going. Ending is what was asked for.
            self.close("the driver ended the session")
            turn = self._last or turn
        return turn

    def close(self, reason: str = "the session was closed") -> None:
        """Stop the run if it is still going, and wait for the phone back.

        Afterwards `last` is the run's final turn, whether the run ended here
        or had already ended with nobody there to read it.
        """
        thread = self._thread
        if thread is None:
            self.lock.release()
            return
        if self.live:
            self._inbox.put(("abort", reason))
        thread.join(timeout=ANSWER_TIMEOUT_S)
        if thread.is_alive():
            log.error("the session did not stop within %ds", ANSWER_TIMEOUT_S)
        while True:
            try:
                self._last = self._outbox.get_nowait()
            except queue.Empty:
                break

    # -- internals -------------------------------------------------------------

    def _require_live(self) -> None:
        if self._thread is None:
            raise SessionError("no session has started")
        if self._last is not None and self._last.outcome:
            raise SessionError(f"the session has ended ({self._last.outcome}); "
                               f"start a new one")

    def _exchange(self, message: tuple) -> Turn:
        self._require_live()
        self._inbox.put(message)
        return self._receive()

    def _receive(self) -> Turn:
        try:
            turn = self._outbox.get(timeout=ANSWER_TIMEOUT_S)
        except queue.Empty:
            raise SessionError("the phone did not answer; the session is "
                               "stuck") from None
        self._last = turn
        return turn

    def _decline_pending(self, happened: List[str]) -> Turn:
        """Turn down a waiting approval: anything but `approve` declines it."""
        self._require_live()
        turn = self._last
        assert turn is not None
        while turn.pending is not None:
            happened.append(f"approval {turn.pending.id} was declined: "
                            f"{turn.pending.prompt}")
            turn = self._exchange(("answer", turn.pending.id, False))
            _merge(happened, turn.happened)
            if turn.outcome:
                break
        return turn

    def _why_it_ended(self, outcome: str, state: RunState,
                      agent: RemoteAgent) -> str:
        """The reason, for an ending the driver did not ask for.

        Most endings the loop reaches on its own say why only in the log --
        `_maybe_give_up` and the budgets write nothing a caller can read -- and a
        driver told "failed" and nothing else can only guess at what to change.
        """
        if state.evidence or outcome == "success":
            return state.evidence
        if agent.abort_reason:
            return agent.abort_reason
        limits = self.cfg.run
        if state.consecutive_failures >= limits.max_consecutive_failures:
            return (f"{state.consecutive_failures} actions in a row did not "
                    f"work; the last: {state.last_failure}")
        if state.step >= limits.max_steps:
            return f"the step budget ({limits.max_steps}) ran out"
        if state.elapsed > limits.max_wall_clock_s:
            return (f"the time budget ({limits.max_wall_clock_s:.0f}s of work) "
                    f"ran out")
        return agent.last_warning or state.last_failure

    def _run(self, goal: str) -> None:
        final = Turn(outcome="aborted")
        dev = None
        try:
            dev = self._open_device(self.cfg)
            agent = RemoteAgent(dev, self.cfg, inbox=self._inbox,
                                outbox=self._outbox, driver=self.driver,
                                idle_timeout_s=self.idle_timeout_s,
                                policy=self.policy,
                                send_limits=self.send_limits)
            outcome, state = agent.run(goal)
            final = Turn(outcome=outcome, step=state.step, result=state.result,
                         evidence=self._why_it_ended(outcome, state, agent),
                         happened=agent._delta(state),
                         run_dir=str(runlog.run_dir(self.cfg, state.run_id)))
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            log.exception("the session's run failed")
            final = Turn(outcome="aborted",
                         evidence=f"{type(exc).__name__}: {exc}")
        finally:
            if dev is not None:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001 - teardown logs its own
                    log.exception("closing the device failed")
            self.lock.release()
            self._outbox.put(final)
