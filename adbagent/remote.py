"""A run whose decisions come from outside this process.

`Agent` asks its model for an action once per step, in `_decide`, and that one
call is all `RemoteAgent` changes: it hands the turn to whoever is driving the
session -- a coding agent connected over MCP (`mcp_server.py`) -- and waits for
the action to come back. Everything around that call still runs here: the
sensitive-screen hand-over, the guards, the verification, the stall ladder,
and the trace in `runs/<id>/` that the web UI and `adbagent report` read. Nothing about a run depends on who decided it, so
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

A session can work through several policies, one pass of each in turn (`Leg`),
the way a watch of several goes round them. Each pass is a run of its own, in
its own `runs/<id>/`, under its own goal, instructions and caps; the phone and
its lock are held across all of them. The driver ends each with `end`, and the
answer to that call is the next one's first screen.

A driver that goes quiet does not keep the phone from everyone else. After
`device.let_go_after_s` without a call the session closes its device session
and lets go of the lock -- a driver reading code, thinking, or gone, is not using
the phone, and a run, a watch or another session may want it -- and its next
call takes the phone back, if nobody else has it by then. Only
`idle_timeout_s` without any call at all ends the session.
"""

from __future__ import annotations

import copy
import logging
import queue
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import conversation, prompts, runlog
from .actions import POINT_ACTIONS, AgentAction
from .agent import Agent, Recorder, RunState
from .config import Config
from .device import DeviceLost, DeviceTimeout
from .llm import ScreenAnalysis
from .phonelock import PhoneBusy, PhoneLock
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
    #: In a session of several policies: which one this turn belongs to
    #: (1-based), of how many, and what it is called. 0 in a session of one.
    leg: int = 0
    legs: int = 0
    leg_name: str = ""
    #: The session's id, on the first turn only: every later call names it.
    session: str = ""
    #: Why the call this answers could not be served at all: the session had
    #: let go of the phone, and somebody else has it now. Nothing was done, so
    #: it never stands as the session's last turn -- `Session` raises it instead.
    refused: str = ""

    def render(self) -> str:
        """The turn as text, in the order the loop's own prompt puts it."""
        parts: List[str] = []
        if self.session and not self.outcome:
            parts.append(f"SESSION {self.session} -- the phone is yours until "
                         f"it ends. Give session_id=\"{self.session}\" to act, "
                         f"look, approve, hand_over and end_session.")
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
        if self.goal and self.legs > 1:
            parts.append(f"POLICY {self.leg} of {self.legs}: "
                         f"{self.leg_name or f'policy {self.leg}'}")
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


@dataclass
class Leg:
    """One policy's pass in a session that works through several.

    `goal` is the whole of what the run pursues -- the policy's goal with the
    watch's pass contract under it, as `mcp_server.Phone` builds it.
    """

    goal: str
    #: The instructions, front matter stripped.
    policy: str = ""
    send_limits: Dict[str, int] = field(default_factory=dict)
    #: What to call it to the driver: the policy's title, else its name.
    name: str = ""


def remote_config(cfg: Config) -> Config:
    """`cfg` as a session driven from outside should run it."""
    cfg = copy.deepcopy(cfg)
    # The driver is shown the frame itself whenever the loop wants one looked
    # at, so the notes offer the x/y form of `tap_at` and the frame is kept
    # beside the step that saw it, as for a model that sees.
    cfg.llm.vision_in_decider = True
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


def _needs_phone(message: tuple) -> bool:
    """Whether the driver's message can only be acted on with the phone held.

    Ending needs none: `done`, `fail` and `ask_user` are answers, not acts on
    the phone, and a session being closed is the opposite of wanting it.
    """
    if message[0] == "abort":
        return False
    if message[0] == "act":
        return not message[1].is_terminal
    return True


def _minutes(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return f"{seconds:.0f} seconds"
    return f"{minutes} minute" + ("s" if minutes != 1 else "")


class RemoteAgent(Agent):
    """An `Agent` whose decisions and approvals come through two queues."""

    def __init__(self, dev: Any, cfg: Config, *, inbox: "queue.Queue",
                 outbox: "queue.Queue", driver: str = "",
                 idle_timeout_s: float = IDLE_TIMEOUT_S, policy: str = "",
                 send_limits: Optional[Dict[str, int]] = None,
                 leg: int = 0, legs: int = 0, leg_name: str = "",
                 intro: Sequence[str] = (),
                 lock: Optional[PhoneLock] = None,
                 let_go_after_s: float = 0.0):
        super().__init__(dev, None, cfg, on_event=self._heard, policy=policy,
                         send_limits=send_limits)
        self._inbox = inbox
        self._outbox = outbox
        self.driver = driver
        #: Which policy of a session of several this run is the pass of.
        self.leg, self.legs, self.leg_name = leg, legs, leg_name
        #: What the driver has to hear before this run's first screen: how the
        #: previous policy's pass ended, when this run follows one.
        self._intro = list(intro)
        self.idle_timeout_s = idle_timeout_s
        #: The session's lock on the phone, and how long the driver may say
        #: nothing before it is let go of. Without a lock the phone is kept.
        self.lock = lock
        self.let_go_after_s = let_go_after_s
        #: Set while the phone is let go of: device session closed, lock
        #: released, until the driver's next call takes them back.
        self._let_go = False
        #: How long the driver had been quiet when the phone was let go of.
        self._let_go_idle = 0.0
        #: Set when a call took the phone back, until that call has checked the
        #: screen it was made against is still there.
        self._retaken = False
        #: The screen the turn waiting on the driver was built from.
        self._on_screen: Optional[Screen] = None
        #: Lines owed to the driver at the head of the next turn.
        self._news: List[str] = []
        self._rec: Optional[Recorder] = None
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
        self._rec = rec
        super()._loop(state, rec)

    def _decide(self, state: RunState, rec: Recorder, screen: Screen,
                **kwargs: Any) -> Optional[AgentAction]:
        turn = self._turn(state, screen, kwargs)
        self._on_screen = screen
        self._post(turn)
        while True:
            message = self._next(state)
            kind = message[0]
            retaken, self._retaken = self._retaken, False
            if kind == "act":
                if retaken and self._moved(screen):
                    # Its target was read off a screen that has gone since:
                    # whoever had the phone meanwhile left it somewhere else.
                    self._news.append("the screen changed while the session "
                                      "did not have the phone, so that was "
                                      "not done; this is the screen now")
                    return None
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
                    action: AgentAction, control: str,
                    analysis: Optional[ScreenAnalysis] = None) -> str:
        """The person, not a second model, says whether a send goes out.

        Fails closed like the model check it replaces: no answer is a no. A no
        is remembered against the frame, so the same send asked for again on
        the same screen is refused without asking the person twice.

        `analysis` is the model check's evidence and has no use here: there is
        no vision pass without a model, and the driver was shown the frame
        itself. Nor does the prompt name the conversation -- that name came
        from reading the thread off the tree, which nothing does any more (see
        `conversation.py`); the driver asking the person knows whose it is.
        """
        key = f"{screen.exact_id}/{action.signature()}"
        if key in state.refused_sends:
            return state.refused_sends[key]
        draft = conversation.draft_in(action, screen)
        prompt = f"The agent wants to send with {control!r}"
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

        Nor is it using the phone. Once the driver has said nothing for
        `let_go_after_s` the phone is let go of (`_let_go_of_phone`), and the
        next message that needs it takes it back first. When somebody else has
        it by then, that call is answered with who, nothing is done, and the
        wait goes on: the driver is still there, and may try again.
        """
        began = time.monotonic()
        try:
            quiet_since = began
            while True:
                message = self._wait(state, quiet_since)
                if not self._let_go or not _needs_phone(message):
                    return message
                busy = self._take_back(state)
                if not busy:
                    return message
                self._post(replace(self._last_turn, refused=busy, happened=[],
                                   skill="", screenshot=None))
                quiet_since = time.monotonic()
        finally:
            state.paused_s += time.monotonic() - began

    def _wait(self, state: RunState, quiet_since: float) -> tuple:
        """The next message, letting go of the phone if it is slow to come."""
        ends = quiet_since + self.idle_timeout_s
        if (self.lock is not None and self.let_go_after_s > 0
                and not self._let_go and not self._released
                and quiet_since + self.let_go_after_s < ends):
            # Not while the person has the phone (`release`): the lock is what
            # keeps anything else from driving it under them.
            try:
                return self._inbox.get(timeout=max(
                    0.0, quiet_since + self.let_go_after_s - time.monotonic()))
            except queue.Empty:
                self._let_go_of_phone(state, time.monotonic() - quiet_since)
        try:
            return self._inbox.get(timeout=max(0.0, ends - time.monotonic()))
        except queue.Empty:
            raise self._abort(
                f"no instruction for {self.idle_timeout_s / 60:.0f} minutes; "
                f"the session ended and the phone was handed back") from None

    def _let_go_of_phone(self, state: RunState, idle: float) -> None:
        """Close the device session, then let go of the lock.

        In that order: closing puts back the animations, the rotation, the
        keyboard and the screen timeout, and that has to happen while nobody
        else can have started driving.
        """
        assert self.lock is not None
        log.info("no call for %.0fs; letting go of the phone", idle)
        try:
            self.dev.close()
        finally:
            self.lock.release()
            self._let_go = True
            self._let_go_idle = idle
        if self._rec is not None:
            self._rec.event("let_go", step=state.step, idle_s=round(idle, 1))

    def _take_back(self, state: RunState) -> str:
        """Take the phone back for the call in hand: "" when it is held again,
        or what to tell the driver when somebody else has it."""
        assert self.lock is not None
        quiet = _minutes(self._let_go_idle)
        try:
            self.lock.acquire()
        except PhoneBusy as exc:
            return (f"the phone is in use -- {exc}. This session let go of it "
                    f"after {quiet} without a call, so that others could use "
                    f"it, and takes it back on your next call once it is free. "
                    f"Nothing was done. Try again in a while, or end_session.")
        try:
            self.dev.open()
        except (DeviceLost, DeviceTimeout):
            self.lock.release()
            raise
        except Exception as exc:  # noqa: BLE001 -- the phone went away
            self.lock.release()
            raise DeviceLost(f"could not take the phone back: {exc}") from exc
        except BaseException:
            self.lock.release()
            raise
        self._let_go = False
        self._retaken = True
        log.info("took the phone back after %.0fs without a call",
                 self._let_go_idle)
        # Not a history line: those are numbered by step, and a batch keeps
        # only the last line of each step -- which would be the action's.
        self._news.append(f"after {quiet} without a call the session had let "
                          f"go of the phone; it took it back for this call")
        if self._rec is not None:
            self._rec.event("taken_back", step=state.step)
        return ""

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
            retaken, self._retaken = self._retaken, False
            moved = (retaken and self._on_screen is not None
                     and self._moved(self._on_screen))
            if kind_ == "answer" and message[1] == approval.id:
                if moved and message[2]:
                    # What was approved was asked about on a screen that has
                    # gone; doing it now would do it to whatever is there.
                    self._news.append("the screen changed while the session "
                                      "did not have the phone, so what was "
                                      "approved was not done; this is the "
                                      "screen now")
                    return False
                return bool(message[2])
            if kind_ == "look":
                self._post(replace(self._last_turn, happened=(
                    ["the screen changed while the session did not have the "
                     "phone; this picture is of the phone as it is now"]
                    if moved else []), screenshot=self._shot()))
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
        if self._news:
            # The session let go of the phone and took it back: whatever is
            # on it now may be somebody else's doing, so it comes with a frame.
            happened += self._news
            self._news = []
            if screenshot is None:
                screenshot = self._shot()
        goal = policy = ""
        if not self._briefed:
            self._briefed = True
            goal = kwargs.get("goal") or ""
            policy = prompts.policy_block(kwargs.get("policy") or "")
            happened = self._intro + happened
        return Turn(step=state.step, package=screen.package,
                    skeleton=screen.skeleton_id,
                    screen=kwargs.get("rendered") or "",
                    notes=self._fresh_notes(kwargs.get("note") or ""),
                    budget=kwargs.get("budget") or "",
                    goal=goal, policy=policy,
                    skill=fresh_skill,
                    happened=happened,
                    screenshot=screenshot,
                    failure=state.last_failure,
                    leg=self.leg, legs=self.legs, leg_name=self.leg_name)


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
                 send_limits: Optional[Dict[str, int]] = None,
                 legs: Sequence[Leg] = (),
                 let_go_after_s: Optional[float] = None):
        self.cfg = remote_config(cfg)
        self.driver = driver
        #: How long the driver may say nothing before the session lets go of
        #: the phone for others to use (`RemoteAgent._next`). 0 keeps it.
        self.let_go_after_s = (self.cfg.device.let_go_after_s
                               if let_go_after_s is None else let_go_after_s)
        #: Several policies' passes to work through in turn, each a run of its
        #: own. Empty for a session of one, which `policy` and `send_limits`
        #: describe -- as do the first leg's, for a session of several.
        self.legs: List[Leg] = list(legs)
        if self.legs:
            policy, send_limits = self.legs[0].policy, self.legs[0].send_limits
        #: The operator's instructions, verbatim, and the caps on what one
        #: session may send -- carried the way a watch pass carries them.
        self.policy = policy
        self.send_limits = dict(send_limits or {})
        #: Set by `end(skip_remaining=True)`: no further policy's pass starts.
        self._skip_rest = False
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
        #: What its driver names it by on every call after `start`. Random, and
        #: told to nobody else: one server can serve several conversations at
        #: once, and this is what keeps each to its own (`mcp_server.Phone`).
        self.id = secrets.token_hex(4)
        #: When it took the phone, to tell anyone else who has it.
        self.started = 0.0

    # -- state ---------------------------------------------------------------

    @property
    def live(self) -> bool:
        """Started, and its run has not ended -- whether or not anyone has read
        the ending yet."""
        if self._thread is None or (self._last and self._last.outcome):
            return False
        return self._thread.is_alive()

    @property
    def holds_phone(self) -> bool:
        """Whether the phone is this session's right now -- false once it has
        let go of it, until its driver's next call takes it back."""
        return self.lock.held

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
            self.goal = goal or " -> ".join(leg.name for leg in self.legs)
            self.lock.who = (f"an MCP session ({self.driver or 'client'}): "
                             f"{self.goal}")
            self.lock.acquire()
            self.started = time.time()
            self._thread = threading.Thread(target=self._run, args=(goal,),
                                             name="adbagent-session",
                                             daemon=True)
            self._thread.start()
            return replace(self._receive(), session=self.id)

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
            leg = turn.leg
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
                if turn.leg != leg:
                    # That policy's pass ended under the batch -- its budget,
                    # its own `done` -- and the next one's run has started.
                    # What was left was planned for the other app.
                    if i < len(actions) - 1:
                        happened.append(
                            f"stopped before the remaining "
                            f"{len(actions) - i - 1} action(s): that policy's "
                            f"pass is over, and the next one's first screen is "
                            f"below")
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

    def end(self, summary: str, failed: bool = False,
            skip_remaining: bool = False) -> Turn:
        """Finish the run with `done` (or `fail`) and hand the phone back.

        In a session of several policies this finishes the pass in hand, and
        what comes back is the next policy's first turn -- failed or not, as a
        watch of several goes on to its next policy. The phone goes back after
        the last, or now with `skip_remaining`.
        """
        action = AgentAction(observation="", reasoning="",
                             action="fail" if failed else "done",
                             text=summary or None)
        if skip_remaining:
            self._skip_rest = True
        before = self._last.leg if self._last is not None else 0
        turn = self.act([action])
        if not turn.outcome and turn.leg == before:
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
                turn = self._outbox.get_nowait()
            except queue.Empty:
                break
            if not turn.refused:
                self._last = turn

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
        if turn.refused:
            # Somebody else has the phone this session let go of. Nothing was
            # done, so the last turn the driver was given still stands.
            raise SessionError(turn.refused)
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

    def _hold_phone(self, dev: Any) -> str:
        """Have the phone for the next policy's pass: "" once it is held, or why
        that pass cannot run.

        A pass can end without the phone: a driver that went quiet long enough
        for the session to let go of it, and then ended the pass, never needed
        it back. The next one does.
        """
        if self.lock.held:
            return ""
        try:
            self.lock.acquire()
        except PhoneBusy as exc:
            return (f"the session had let go of the phone, and when their turn "
                    f"came it was in use -- {exc}")
        try:
            dev.open()
        except BaseException:
            self.lock.release()
            raise
        return ""

    def _run(self, goal: str) -> None:
        """The run, or the runs: one pass per policy, in turn, on one device.

        The next policy's run starts the moment the last one ends, and its first
        turn -- what it is shown, opening with how the last one went -- is the
        answer to whichever call ended the last. A pass that fails does not stop
        the ones after it, as in a watch; an abort does (the session closed, the
        driver went quiet, the person kept the phone), and so does
        `end(skip_remaining=True)`.
        """
        final = Turn(outcome="aborted")
        dev = None
        legs = self.legs or [Leg(goal=goal, policy=self.policy,
                                 send_limits=self.send_limits)]
        total = len(legs) if len(legs) > 1 else 0
        try:
            dev = self._open_device(self.cfg)
            intro: List[str] = []
            recap: List[str] = []
            # Standing advice is said once per session, not once per policy.
            advised: set = set()
            for n, leg in enumerate(legs, 1):
                agent = RemoteAgent(dev, self.cfg, inbox=self._inbox,
                                    outbox=self._outbox, driver=self.driver,
                                    idle_timeout_s=self.idle_timeout_s,
                                    policy=leg.policy,
                                    send_limits=leg.send_limits,
                                    leg=n if total else 0, legs=total,
                                    leg_name=leg.name, intro=intro,
                                    lock=self.lock,
                                    let_go_after_s=self.let_go_after_s)
                agent._advised = advised
                outcome, state = agent.run(leg.goal)
                final = Turn(outcome=outcome, step=state.step,
                             result=state.result,
                             evidence=self._why_it_ended(outcome, state, agent),
                             happened=agent._delta(state),
                             run_dir=str(runlog.run_dir(self.cfg, state.run_id)),
                             leg=n if total else 0, legs=total,
                             leg_name=leg.name)
                if not total:
                    break
                said = state.result or final.evidence
                recap.append(f"policy {n} of {total} ({leg.name}) ended: "
                             f"{outcome}" + (f" -- {said}" if said else "")
                             + f" [trace: {final.run_dir}]")
                rest = legs[n:]
                why_not = ("" if not rest
                           else "the session ended" if outcome == "aborted"
                           else "end_session skipped them" if self._skip_rest
                           else self._hold_phone(dev))
                if not rest or why_not:
                    if rest:
                        recap.append(f"not run: "
                                     f"{', '.join(r.name for r in rest)} -- "
                                     f"{why_not}")
                    final.happened = final.happened + recap
                    break
                intro = final.happened + [
                    recap[-1],
                    f"policy {n + 1} of {total} ({rest[0].name}) starts now, as "
                    f"a run of its own: its goal and its policy are below"]
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            log.exception("the session's run failed")
            final = Turn(outcome="aborted",
                         evidence=f"{type(exc).__name__}: {exc}")
        finally:
            # Only while the session has the phone. Once it has let go, its
            # device session is closed already, and nothing here may touch a
            # phone somebody else may be driving now.
            if dev is not None and self.lock.held:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001 - teardown logs its own
                    log.exception("closing the device failed")
            self.lock.release()
            self._outbox.put(final)
