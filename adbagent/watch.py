"""The unbounded monitor loop: poll, act when something changed, never die.

A run is bounded, may fail, and is over. A watch is expected to outlive
transient failures and still be going tomorrow. That difference drives every
decision in this file.

**One bounded iteration at a time.** The loop does not make the agent's inner
loop unbounded -- that would put it at war with the machinery that keeps a run
honest. `safety.LoopDetector` treats a repeated (screen, action) pair as being
stuck, and a poll loop is nothing but repeated (screen, action) pairs; the stall
ladder gives up after `stall_give_up_at` steps without learning anything, and a
quiet inbox teaches you nothing by design. So instead each pass is an ordinary
bounded run of ~25 steps with all of that intact, and the *supervisor* is what
never ends. Nothing inside the agent had to be weakened to make this work.

**Nothing changed means nothing spent.** Between passes the loop dumps the UI --
an adb round trip, no model call -- and compares a masked digest of the app's own
text against the screen the last pass left behind. Equal means no new message, so
it sleeps. That is what turns "an LLM call every 45 seconds forever" into "an LLM
call per actual new message", and it is also the honest answer to "how would you
know something arrived": you looked.

**Unless the work does not announce itself.** That probe asks "did anything
arrive?", which is the whole question for an inbox and the wrong question for a
goal whose work is generated somewhere the screen cannot show: a feed with more
items below the fold, a queue to take a few from each time, anything meant to
happen on a period. Those leave the screen exactly as the last pass left it and
still have work, so a purely reactive loop does one pass and sleeps forever after
it. `watch.sweep_s` is the second trigger -- run a pass at least this often
whatever the digest says -- and it is off by default, because which kind of goal
this is cannot be read off the app, only off what the operator asked for.

The comparison is against a remembered *anchor* -- package plus digest -- rather
than against the previous probe. Comparing consecutive probes would read a phone
sitting on the launcher as "nothing changed" and watch the wrong screen forever;
comparing against the anchor reads it as "not where I should be" and spends a
pass getting back. Self-healing falls out of that: a screen that went off, an
app that got killed, a notification shade left open all look like "not the
anchor", and the fix is the same pass that handles a new message.

**A rest is still a watch.** A policy's `snooze` rule names a screen that says
the work has run dry -- a feed with nobody left in it -- and how long to leave it
alone. The loop anchors on that screen and goes on probing it like any other; the
rule's time only says when to run a pass whatever the probe finds. Any change in
the meantime ends the rest early, because the work ran dry on that screen, not in
the app: a message that arrives while the feed is empty shows up as an unread
count on a tab, and a rest that never looked would sit on it for the full time.

**Failure is a pause, never an exit.** A failed pass doubles a backoff and the
loop continues. Only a keyboard interrupt stops it. A watch that exits because
the phone dropped off Wi-Fi for a minute is not a watch.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import conversation
from .agent import Agent, Outcome
from .config import Config
from .device import Device, DeviceLost, DeviceTimeout
from .fingerprint import mask_goal
from .llm import BudgetExceeded, LLMClient, LLMError
from .policies import instructions
from .screen import Screen

log = logging.getLogger("adbagent.watch")

#: What one pass of the loop is, appended to the operator's goal. Generic across
#: apps on purpose -- it says what a pass *is*, not what any app looks like.
#:
#: It used to say it in inbox nouns: get to the conversation list, find the new
#: incoming messages, reply, come back. That reads as a description of the task
#: rather than a description of a pass, and against any goal that was not an
#: inbox sweep -- work a feed, drain a queue, do a periodic check -- it competed
#: with the goal instead of framing it. The numbered shape is worth keeping, so
#: it stayed and the nouns went: the goal and the policy name the screen and say
#: what counts as work, and this says how much of it one pass does.
#:
#: The last line is what makes the cheap probe possible: a pass that ends back on
#: the list it worked from leaves an anchor the next probe can compare against,
#: so a quiet inbox costs one UI dump instead of a model call.
#:
#: "Do not start conversations" defers to the policy rather than overriding it.
#: Flatly forbidding openers here contradicted any policy that wanted one -- and
#: since `prompts.policy_block` tells the model the policy is what decides what
#: it sends, the two together produced a coin flip rather than a refusal. The
#: default is still no: an operator who wants openers has to write that down,
#: and the deferral is narrow enough that no other rule in this block moves.
ITERATION_CONTRACT = """\
This is ONE PASS of a monitoring loop that runs continuously. Do this pass only,
then stop:
  1. Get to the screen the goal works from -- the conversation list, the feed,
     whatever the goal and the REPLY POLICY describe.
  2. Find what needs handling there: a conversation with a new incoming message,
     an item not dealt with yet -- whatever the goal counts as work.
  3. Handle each one the way the REPLY POLICY says, then come back to that
     screen.
  4. When nothing is left, or when the goal's quota for one pass is met, report
     done.

Rules for this pass:
  - At most one reply per conversation. Never two.
  - If a send is refused, do not retry it and do not rephrase it. Leave that
    item and move on -- the refusal is the harness holding the send back (draft
    mode, a send limit, or a check against the REPLY POLICY), and it is always
    right.
  - Do not start conversations with anyone who has not messaged first, unless
    the REPLY POLICY explicitly says to open one. Silence in the policy means no.
  - Do not carry on past what this pass asks for. There is always another pass.
  - Finish on the screen you worked from, not buried inside one of the items. The
    next pass starts from wherever you leave the screen."""


#: A bare count, the way an unread badge draws one: ``3``, ``(12)``.
_COUNT = re.compile(r"\(?\d{1,4}\)?")


def _written(screen: Screen) -> List[Tuple[str, str]]:
    """``(as drawn, as compared)`` for each text the app has on screen, in order.

    Every node's text *and* its content description, containers included. A
    tab's unread count is rarely a node of its own: navigation bars put it in the
    tab's description -- ``Chats, 3 unread`` -- while the label drawn under the
    icon is a child, so reading only the leaves, or only a node's text when it
    has both, misses exactly the change a watch is waiting for.

    Compared through `mask_goal`, so a clock, a relative timestamp or a battery
    percentage moving on its own is not mistaken for news -- except a bare count,
    which `mask_goal` folds into ``<n>`` and which is kept as drawn: an unread
    badge going from 3 to 4 is a message arriving.
    """
    out: List[Tuple[str, str]] = []
    for node in conversation.app_nodes(screen):
        for raw in (node.text, node.content_desc):
            shown = " ".join(raw.split())
            if shown:
                out.append((shown, shown if _COUNT.fullmatch(shown)
                            else mask_goal(shown)))
    return out


def screen_digest(screen: Screen) -> str:
    """A digest of everything the app has written on screen, masked.

    The novelty signal. Deliberately the opposite of `skeleton_id`, which is
    content-free so that two visits to the same layout hash alike -- exactly the
    wrong instrument for "did a new message arrive", which is a question about
    content and nothing else. What counts as written, and what is masked, is
    `_written`'s to say.

    Read from the raw nodes, before pruning, for the reasons
    `conversation.app_nodes` gives.
    """
    h = hashlib.sha256()
    for _shown, compared in _written(screen):
        h.update(compared.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _quote(text: str, most: int = 60) -> str:
    return f"\"{text if len(text) <= most else text[:most - 3] + '...'}\""


def what_changed(before: Screen, now: Screen) -> str:
    """A few words on how `now` differs from `before`, for the log.

    Asked once a probe has found the two different, so that an operator reading
    a watch that keeps cutting its rests short can see what keeps doing it: the
    first text that is new, or failing that the first one that went.
    """
    if now.package != before.package:
        return f"{now.package or 'nothing'} is in front"
    was, nxt = _written(before), _written(now)
    added = Counter(c for _s, c in nxt) - Counter(c for _s, c in was)
    for shown, compared in nxt:
        if added[compared]:
            return f"{_quote(shown)} appeared"
    gone = Counter(c for _s, c in was) - Counter(c for _s, c in nxt)
    for shown, compared in was:
        if gone[compared]:
            return f"{_quote(shown)} went away"
    return "the same text, in a different order"


@dataclass
class Anchor:
    """The screen a successful pass left behind."""

    package: str = ""
    digest: str = ""

    def matches(self, screen: Screen) -> bool:
        return bool(self.package) and screen.package == self.package \
            and screen_digest(screen) == self.digest

    @classmethod
    def of(cls, screen: Screen) -> "Anchor":
        return cls(package=screen.package, digest=screen_digest(screen))


@dataclass(frozen=True)
class Snooze:
    """Words on screen that mean "nothing to do here for a while", and how long.

    From the policy's `snooze` front matter: ``You've seen everyone for now =
    15m``. Some screens say outright that the work has run dry -- a feed with
    nobody left in it, a daily limit reached -- and the novelty probe cannot
    tell that apart from "nothing new yet": the screen will not change on its
    own, so a reactive watch would never look again, and a sweep would spend a
    pass every `sweep_s` being told the same thing.

    So `seconds` is a deadline, not a nap: the probe goes on while the watch
    rests, and a pass runs when the screen changes or the time is up, whichever
    comes first. See `_Rest`.
    """

    phrase: str
    seconds: float


@dataclass
class _Rest:
    """A rest in progress: when it began, when it is over, and on what screen.

    The screen is kept whole, not as the anchor's digest, so that the line
    saying a rest was cut short can say what changed -- see `what_changed`.
    """

    started: float
    until: float
    screen: Screen


_SNOOZE_DURATION = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?)\s*$",
    re.I)
_SNOOZE_UNIT = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_snooze(text: str) -> List[Snooze]:
    """``phrase = 15m`` rules, one per line or `;`. Raises ValueError.

    Strict for the reason `conversation.parse_send_limits` is: a rule that is
    silently misread is a watch that never rests, found out about hours later.
    """
    rules: List[Snooze] = []
    for part in re.split(r"[;\n]", text or ""):
        part = part.strip()
        if not part:
            continue
        phrase, sep, duration = part.rpartition("=")
        phrase = phrase.strip()
        if len(phrase) >= 2 and phrase[0] == phrase[-1] and phrase[0] in "\"'":
            phrase = phrase[1:-1].strip()
        found = _SNOOZE_DURATION.match(duration)
        if not sep or not phrase or not found:
            raise ValueError(f"{part!r} is not a snooze rule -- write it as "
                             f"words on screen = 15m")
        unit = _SNOOZE_UNIT[found.group(2)[0].lower()]
        rules.append(Snooze(phrase=phrase, seconds=float(found.group(1)) * unit))
    return rules


def _fold(text: str) -> str:
    """Case, curly quotes and runs of whitespace folded away, for matching.

    The phrase is typed with a straight apostrophe and an app may draw a curly
    one; neither should decide whether the watch rests.
    """
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"'))
    return " ".join(text.casefold().split())


def snooze_for(screen: Screen, rules: Sequence[Snooze]) -> Optional[Snooze]:
    """The first rule whose phrase the app has on this screen, or None."""
    if not rules:
        return None
    texts = [_fold(e.best_text) for e in screen.content_elements if e.best_text]
    for rule in rules:
        wanted = _fold(rule.phrase)
        if any(wanted in t for t in texts):
            return rule
    return None


def _duration(seconds: float) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    return f"{seconds:g}s"


@dataclass
class Stats:
    """What the watch has done, for the periodic status line."""

    passes: int = 0
    skipped: int = 0
    #: Probes that found work but were held back by the rolling spend ceiling.
    #: Counted separately from `skipped` because they are the opposite situation:
    #: there *was* something to do.
    paused: int = 0
    #: Passes that ended on a screen a `snooze` rule names, and so were followed
    #: by that rule's rest rather than the usual interval.
    snoozed: int = 0
    #: Rests cut short because the screen changed under them. Read next to
    #: `snoozed`: how often the app had work again before the rule's time was up.
    woken: int = 0
    failures: int = 0
    usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at


class Watch:
    """Supervises bounded agent passes forever."""

    def __init__(self, dev: Device, llm: LLMClient, cfg: Config,
                 *, policy: str,
                 send_limits: Optional[Dict[str, int]] = None,
                 snooze: Sequence[Snooze] = (),
                 say: Optional[Callable[[str], None]] = None,
                 on_event: Optional[Callable[..., None]] = None,
                 make_agent: Optional[Callable[..., Agent]] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.dev = dev
        self.llm = llm
        self.cfg = cfg
        self.policy = policy
        #: The policy's `send_limits`, applied to each pass on its own: a pass
        #: is a run, and "at most 5 likes" is a promise about one of them.
        self.send_limits: Dict[str, int] = dict(send_limits or {})
        #: The policy's `snooze` rules: see `Snooze`.
        self.snooze: List[Snooze] = list(snooze)
        self.say = say or (lambda msg: None)
        self.on_event = on_event
        # Injected so the loop can be tested without a device or a model. The
        # clock comes with the sleep: a test whose `sleep` only records the delay
        # leaves real time barely moving, so anything measured in wall clock --
        # the sweep, the spend window -- would never come due.
        self._make_agent = make_agent or self._default_agent
        self._sleep = sleep or time.sleep
        self._now = clock or time.monotonic
        # A pass is bounded by `watch.max_steps`, not by `run.max_steps`: the
        # run-level budget is sized for a whole task and a pass is one sweep of an
        # inbox. Set here rather than left to the caller so that a `Watch`
        # constructed any way at all cannot run 60-step passes by accident.
        if cfg.watch.max_steps and cfg.run.max_steps != cfg.watch.max_steps:
            log.debug("bounding each pass to %d steps (run.max_steps was %d)",
                      cfg.watch.max_steps, cfg.run.max_steps)
            cfg.run.max_steps = cfg.watch.max_steps
        self.stats = Stats()
        self.anchor = Anchor()
        #: The `RunState` of the most recent pass, for whoever closes the trace
        #: off when the watch stops. `TraceCollector.app_traces` takes the step
        #: counts from its own per-package tally rather than from this, so a
        #: week's watch is not judged on the three steps its last pass took.
        self.last_state: Optional[Any] = None
        #: Whether a person took the phone during *any* pass. Sticky, and here
        #: rather than on a `RunState`, because that is per pass and this is not:
        #: a takeover in pass 3 of forty is still a takeover when the watch stops
        #: and the trace is closed off. Read by `cli` when it decides whether the
        #: app's skill may be written from what this watch saw -- and reading it
        #: off `last_state` would have asked the fortieth pass about the third.
        self.took_over = False
        #: (finished_at, usd) per pass, for the rolling spend ceiling.
        self._spend: List[Tuple[float, float]] = []
        #: When the last pass finished, for the sweep. None until one has.
        self._last_pass_at: Optional[float] = None
        #: The rest the loop is in, if a `snooze` rule put it in one.
        self._rest: Optional[_Rest] = None
        self._stop = False

    # -- construction ------------------------------------------------------

    def _default_agent(self) -> Agent:
        """A fresh agent per pass.

        Fresh on purpose: a pass is a run, and the per-run state -- history, loop
        detector, stall counters, scratchpad -- should not carry across.

        Nothing survives a pass any more. The reply ledger used to, and it was
        the one thing that did: it was read back from disk each pass so that a
        thread answered in pass 3 was still known to be answered in pass 40.
        What a pass now knows about what earlier passes said is whatever is
        visible in the thread on screen -- see `conversation.py`.

        No skill learning happens *per pass*, unlike `cmd_run`: rewriting the
        app's skill file every 45 seconds, mostly from passes that did nothing,
        would churn the file the next pass depends on. Learning instead happens
        once, when the watch stops, from a trace accumulated across every pass --
        see `cli.cmd_watch`. That is strictly the better trace anyway: fifty
        passes over an inbox and its threads tour the app far more thoroughly
        than any one of them does.
        """
        return Agent(self.dev, self.llm, self.cfg, policy=self.policy,
                     send_limits=self.send_limits, on_event=self._pass_event)

    def _pass_event(self, kind: str, **kw: Any) -> None:
        """Latch watch-wide facts before forwarding one pass's event.

        `released` is emitted at the moment the device session closes. Waiting
        for `Agent.run()` to return is too late when Stop interrupts that pass,
        and can let a trace influenced by a person's intervention be learned.
        """
        if kind == "released":
            self.took_over = True
        if self.on_event is not None:
            self.on_event(kind, **kw)

    # -- the loop ----------------------------------------------------------

    def stop(self) -> None:
        """Ask the loop to finish the pass it is on and return."""
        self._stop = True

    def run(self, goal: str, max_passes: int = 0) -> Stats:
        """Watch until interrupted. `max_passes` bounds it, for tests.

        Returns the stats rather than an exit code: the caller decides what a
        stopped watch means.
        """
        full_goal = f"{goal.strip()}\n\n{ITERATION_CONTRACT}"
        consecutive_failures = 0
        w = self.cfg.watch

        while not self._stop:
            s = self.stats
            if max_passes and s.passes + s.skipped + s.paused >= max_passes:
                return self.stats

            # -- is there anything to do? (no model call) ------------------
            try:
                screen = self.dev.observe()
            except (DeviceTimeout, DeviceLost) as exc:
                consecutive_failures += 1
                delay = self._backoff(consecutive_failures)
                log.warning("probe failed (%s); retrying in %.0fs", exc, delay)
                self.say(f"  device unreachable ({exc}); retrying in {delay:.0f}s")
                self._sleep(delay)
                continue

            rest = self._rest
            if self.anchor.matches(screen):
                if rest is not None:
                    left = rest.until - self._now()
                    if left > 0:
                        self.stats.skipped += 1
                        log.debug("resting on %s and nothing new; %.0fs left",
                                  screen.package, left)
                        self._sleep(min(w.interval_s, left))
                        continue
                    log.debug("the rest is over; a pass runs whatever the "
                              "screen says")
                elif not self._sweep_due():
                    self.stats.skipped += 1
                    log.debug("nothing new on %s; sleeping %.0fs",
                              screen.package, w.interval_s)
                    self._sleep(w.interval_s)
                    continue
                else:
                    log.debug("nothing new on %s, but the %.0fs sweep is due",
                              screen.package, w.sweep_s)
            elif rest is not None:
                # The screen changed while the watch rested. The app said there
                # was nothing to do *on that screen*; whatever it has written
                # since -- an unread count, a banner, another app in front -- is
                # a reason to look now rather than when the rule's time is up.
                self.stats.woken += 1
                change = what_changed(rest.screen, screen)
                into = self._now() - rest.started
                self.say(f"  rest cut short {into / 60:.1f}m into "
                         f"{_duration(rest.until - rest.started)}: {change}")
                log.info("the screen changed %.0fs into a %.0fs rest (%s)",
                         into, rest.until - rest.started, change)
            if rest is not None:
                # Over either way, and the anchor goes with it. It was the screen
                # saying there was nothing to do, and the pass this rest owes
                # must not be skipped because the screen still says so: a spend
                # pause between here and the pass would otherwise leave the
                # watch anchored on that screen for good.
                self._rest = None
                self.anchor = Anchor()

            # -- is the loop allowed to spend? -----------------------------
            paused = self._spend_pause()
            if paused > 0:
                self.stats.paused += 1
                self.say(f"  hourly spend ceiling reached; pausing {paused:.0f}s")
                log.warning("rolling spend ceiling ($%.2f/h) reached; pausing "
                            "%.0fs", w.max_usd_per_hour, paused)
                self._sleep(paused)
                continue

            # -- one bounded pass ------------------------------------------
            outcome, usd = self._one_pass(full_goal)
            self.stats.passes += 1
            self.stats.usd += usd
            self._spend.append((self._now(), usd))
            # Set for a failed pass too: the sweep says how often to *try*, and a
            # pass that failed already has the backoff deciding when to try next.
            self._last_pass_at = self._now()

            # The screen the pass left, read once. After a success it is the
            # next anchor, as it always was; after a failure it is read only
            # when a snooze rule has a question for it, so a watch without one
            # touches the phone exactly as often as it did.
            after = None
            if outcome in ("success", "needs_user") or self.snooze:
                after = self._screen_after_pass()
            rule = snooze_for(after, self.snooze) if after is not None else None
            if rule is not None:
                # The app has said the work has run dry. That is not a failure
                # to back off from, and not "nothing new" either -- the screen
                # will say the same thing until somebody looks again, which is
                # what the rest is for. So the watch anchors on it and probes it
                # like any other screen, and the top of the loop runs a pass when
                # it changes or when the rule's time is up, whichever is first.
                consecutive_failures = 0
                self.stats.snoozed += 1
                if outcome not in ("success", "needs_user"):
                    # Rested rather than backed off, but still reported: the
                    # status line is where an operator notices a pass that
                    # keeps failing on its way to this screen.
                    self.stats.failures += 1
                now = self._now()
                self.anchor = Anchor.of(after)
                self._rest = _Rest(started=now, until=now + rule.seconds,
                                   screen=after)
                self.say(f"  pass {self.stats.passes}: {outcome}; "
                         f"\"{rule.phrase}\" is on screen -- resting "
                         f"{_duration(rule.seconds)}, or until it changes")
                log.info("%r is on screen after pass %d; resting %.0fs, or "
                         "until the screen changes", rule.phrase,
                         self.stats.passes, rule.seconds)
                self._sleep(min(w.interval_s, rule.seconds))
                continue

            if outcome in ("success", "needs_user"):
                consecutive_failures = 0
                # Re-read rather than trusting the pass's last frame: the anchor
                # has to describe the screen as it is *now*, or the next probe
                # compares against something already stale.
                self.anchor = Anchor.of(after) if after is not None else Anchor()
                self.say(f"  pass {self.stats.passes}: {outcome} "
                         f"(${self.stats.usd:.4f})")
                # What the pass concluded. A watch prints one line per pass by
                # default, so without this the only thing a night of watching
                # leaves on the terminal is a column of "success". Safe to read
                # here: this branch is the one where `agent.run` returned, so
                # `last_state` is this pass's, not a previous one's.
                answer = getattr(self.last_state, "result", "")
                if answer:
                    self.say(f"    {answer}")
                if outcome == "needs_user":
                    # Not a failure -- the pass did what it could and stopped for
                    # a human. Said loudly because nobody is watching the log.
                    self.say("  a pass stopped and asked for a human; "
                             "continuing to watch")
                self._sleep(w.interval_s)
            else:
                consecutive_failures += 1
                self.stats.failures += 1
                delay = self._backoff(consecutive_failures)
                self.say(f"  pass {outcome} ({consecutive_failures} in a row); "
                         f"retrying in {delay:.0f}s")
                log.warning("pass %s; %d consecutive failure(s), backing off "
                            "%.0fs", outcome, consecutive_failures, delay)
                # The anchor is dropped on failure on purpose: a failed pass left
                # the screen somewhere unknown, and an anchor pointing at it would
                # make the next probe skip work it needs to do.
                self.anchor = Anchor()
                self._sleep(delay)

        return self.stats

    # -- one pass ----------------------------------------------------------

    def _one_pass(self, full_goal: str) -> Tuple[Outcome, float]:
        """Run one bounded agent pass. Never raises for an ordinary failure."""
        before = self.llm.ledger.total_usd
        agent = self._make_agent()
        try:
            outcome, state = agent.run(full_goal)
            self.last_state = state
            # Latches on and never clears. Every later pass would report False,
            # and the one that closes the trace is the one that gets asked.
            self.took_over = self.took_over or state.took_over
        except (BudgetExceeded, LLMError) as exc:
            # The session budget is a run-level guard; for a watch it is one more
            # thing to back off from rather than a reason to stop watching.
            log.warning("pass aborted: %s", exc)
            outcome = "aborted"
        except (DeviceTimeout, DeviceLost) as exc:
            log.warning("pass lost the device: %s", exc)
            outcome = "aborted"
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 -- a watch outlives surprises
            # Logged with a traceback and survived. An unhandled exception in one
            # pass is a bug to fix, not a reason for the watch to be gone when
            # somebody comes back to it tomorrow.
            log.exception("pass raised %s", type(exc).__name__)
            outcome = "failed"
        return outcome, self.llm.ledger.total_usd - before

    def _screen_after_pass(self) -> Optional[Screen]:
        try:
            return self.dev.observe()
        except (DeviceTimeout, DeviceLost) as exc:
            log.warning("could not read the screen the pass left (%s); the next "
                        "pass will run unconditionally", exc)
            return None

    # -- ceilings ----------------------------------------------------------

    def _backoff(self, consecutive: int) -> float:
        w = self.cfg.watch
        delay = w.backoff_initial_s * (2 ** max(0, consecutive - 1))
        return float(min(delay, w.backoff_max_s))

    def _sweep_due(self) -> bool:
        """Is a pass owed on time alone, whatever the screen says?

        Off unless the operator asked for it. The first probe of a watch is never
        held back by this -- an empty anchor never matches, so the loop is already
        past the check by the time it is asked.
        """
        every = self.cfg.watch.sweep_s
        if every <= 0:
            return False
        if self._last_pass_at is None:
            return True
        return self._now() - self._last_pass_at >= every

    def _spend_pause(self) -> float:
        """Seconds to wait before spending again, or 0."""
        ceiling = self.cfg.watch.max_usd_per_hour
        if ceiling <= 0:
            return 0.0
        now = self._now()
        window = [(t, u) for t, u in self._spend if now - t < 3600.0]
        self._spend = window
        if sum(u for _t, u in window) < ceiling:
            return 0.0
        oldest = min(t for t, _u in window)
        return max(1.0, 3600.0 - (now - oldest))

    # -- reporting ---------------------------------------------------------

    def status(self) -> str:
        """What the watch has done. No reply count: it was read off the ledger.

        `len(ledger)` was the one number here that said what the loop had
        actually *done to somebody* rather than what it had spent doing it, and
        nothing left in the process knows it -- a send leaves its trace in the
        pass's events and in the thread on the phone, neither of which this
        supervisor reads back.
        """
        s = self.stats
        bits = [f"{s.passes} pass(es)", f"{s.skipped} skipped"]
        if s.paused:
            bits.append(f"{s.paused} paused")
        if s.snoozed:
            bits.append(f"{s.snoozed} rested")
        if s.woken:
            bits.append(f"{s.woken} woken")
        bits += [f"{s.failures} failed", f"${s.usd:.4f}",
                 f"up {s.uptime_s / 3600:.1f}h"]
        return ", ".join(bits)


def load_policy(path: str) -> str:
    """The operator's reply instructions. Raises if unreadable.

    A watch without a policy is a loop that decides for itself what to say to
    people, so an unreadable policy file is fatal rather than a warning.

    Front matter is stripped (`policies.instructions`). It is a note about the
    policy -- which goal it was written for, what to call it -- and a model
    handed ``goal: ...`` in the middle of its reply instructions has been told
    something nobody meant to tell it. `policies.read` is the way to the
    metadata; this function is the way to the prompt.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"no policy file at {p}")
    text = instructions(p.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"the policy file {p} is empty")
    return text
