"""What earlier runs in an app know that the current one cannot.

A skill is written from one run's trace, and one run cannot tell a one-off from
a pattern. The synthesis prompt asks for exactly that judgement -- record the
dead end, not the slow load -- and until now it had no basis for making it: a
control tapped once that did nothing looks identical to a control that has done
nothing in every run for a week.

This module supplies the missing axis. It reads the recorded runs for one
package and reports only what *repeats*: the action that keeps failing
verification, the screen the loop detector keeps breaking out of, the control
that keeps having to be refused. Frequency is the whole product, so a signal
seen once is dropped rather than passed on -- the current run's trace already
carries it, and forwarding it again under the heading "history" would be
laundering a single observation into a trend.

Two sources, because they know different things:

* ``runs/<id>/events.jsonl`` -- what happened, step by step, including the
  verification grade the harness gave each action.
* the ``dead_end`` table -- what the agent already decided was a dud, keyed by
  screen and expiring after a day. Its `action_sig` carries element indices,
  which mean nothing across runs, so those are counted per screen and verb
  rather than quoted.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("adbagent.history")

#: How many recent runs to read. A tour of an app's quirks stabilises long
#: before this; reading every run ever recorded would grow the cost of finishing
#: a run without changing what the skill says.
DEFAULT_RUN_LIMIT = 12

#: How often something must have happened to count as a pattern. At 1 this
#: module would just be restating the current run back to itself.
MIN_OCCURRENCES = 2

#: Longest a goal line may be in the digest. Goals run to paragraphs.
MAX_GOAL_CHARS = 110


def _describe_action(action: Dict[str, Any]) -> str:
    """An action, named in a way that survives into another run.

    Element indices are deliberately dropped. ``tap #11`` identifies nothing a
    week later -- the same control is #4 on the next run, and a nuance written
    around the number is worse than none, because it reads as specific.
    """
    verb = str(action.get("action") or "?")
    target = action.get("target") or {}
    what = (action.get("text")
            or (target.get("text") if isinstance(target, dict) else "")
            or (target.get("resource_id") if isinstance(target, dict) else "")
            or action.get("direction") or "")
    what = " ".join(str(what).split())[:60]
    return f"{verb} {what!r}" if what else verb


#: Share of a run's steps a package needs before the run counts as being *about*
#: that app. Every run touches the launcher on its way somewhere, and a history
#: of "runs that glanced at this app" is not a history of the app.
_DOMINANCE = 0.2


def packages_in(events: Sequence[Dict[str, Any]]) -> Set[str]:
    """The app(s) a recorded run was actually working in.

    Three tiers, weakest last, because getting this wrong is not a near miss --
    it files one app's failures under another and the synthesis writes them into
    the wrong skill as fact:

    * ``run_end.package_steps`` -- where the steps went. Authoritative.
    * ``run_end.packages`` -- the set, for runs recorded before the counts were.
      Over-matches a run that merely passed through, which is tolerable.
    * the packages the run explicitly asked to open, for runs older still.

    ``active_skill`` looks like a fourth source and is not usable: it records
    the package on screen at the moment a skill loaded, which for a run starting
    from the home screen is the launcher, and for one starting on top of another
    app is that other app. It attributed a WhatsApp run to Bumble because Bumble
    happened to be open, so it is deliberately ignored.
    """
    steps: Dict[str, int] = {}
    packages: Set[str] = set()
    opened: Set[str] = set()

    for event in events:
        kind = event.get("kind")
        if kind == "run_end":
            counts = event.get("package_steps") or {}
            if isinstance(counts, dict):
                for pkg, n in counts.items():
                    if pkg:
                        steps[str(pkg)] = steps.get(str(pkg), 0) + int(n or 0)
            packages.update(str(p) for p in (event.get("packages") or []) if p)
        elif kind == "decide":
            action = event.get("action") or {}
            if isinstance(action, dict) and action.get("action") == "open_app":
                text = str(action.get("text") or "")
                if "." in text:
                    opened.add(text)

    if steps:
        total = sum(steps.values()) or 1
        return {pkg for pkg, n in steps.items() if n >= _DOMINANCE * total}
    return packages or opened


@dataclass
class History:
    """The repeated signals from earlier runs in one app."""

    package: str = ""
    runs: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    #: (action, times it failed verification, times it passed). The successes
    #: matter as much as the failures: "failed 3 times" reads as broken, and
    #: "failed 3 times, passed 40" reads as flaky, and only one of those is a
    #: nuance worth writing into a skill.
    failures: List[Tuple[str, int, int]] = field(default_factory=list)
    #: (screen skeleton, times the loop detector had to break out)
    stuck: List[Tuple[str, int]] = field(default_factory=list)
    #: (control label, times an irreversible action was refused there)
    refusals: List[Tuple[str, int]] = field(default_factory=list)
    #: (what led nowhere, times) from the dead-end table
    dead_ends: List[Tuple[str, int]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.failures or self.stuck or self.refusals
                    or self.dead_ends or self.runs > 1)

    def to_prompt_text(self) -> str:
        """The digest handed to skill synthesis. Empty when nothing repeated."""
        if not self:
            return ""
        lines = [f"WHAT EARLIER RUNS IN {self.package} SHOW "
                 f"(counted across the last {self.runs} run(s) in this app, "
                 f"including this one -- these are patterns, not single events):"]
        if self.outcomes:
            tally = ", ".join(f"{n} {name}" for name, n in
                              sorted(self.outcomes.items(), key=lambda kv: -kv[1]))
            lines.append(f"  outcomes: {tally}")
        for goal in self.goals:
            lines.append(f"  a goal it was asked for: {goal}")
        for what, failed, passed in self.failures:
            note = f" (and passed {passed} times)" if passed else " (never passed)"
            lines.append(f"  {what} failed verification {failed} times{note}")
        for screen, n in self.stuck:
            lines.append(f"  the loop detector broke out of screen {screen} {n} times")
        for label, n in self.refusals:
            lines.append(f"  an irreversible control {label!r} was reached {n} times")
        for what, n in self.dead_ends:
            lines.append(f"  {what} led nowhere {n} times")
        lines.append("  Turn a repeated one into a nuance with a way around it. "
                     "Something listed once or twice may still be circumstance.")
        return "\n".join(lines)


def _run_dirs(artifacts_dir: Path, limit: int) -> List[Path]:
    if not artifacts_dir.is_dir():
        return []
    dirs = [p for p in artifacts_dir.iterdir() if (p / "events.jsonl").is_file()]
    dirs.sort(key=lambda p: (p / "events.jsonl").stat().st_mtime, reverse=True)
    return dirs[:limit]


def _dead_ends_for(db_path: Path, package: str) -> List[Tuple[str, int]]:
    """Dud actions this app has already accumulated, per screen and verb.

    The stored signature is ``verb/#index/direction``; the index is dropped for
    the reason `_describe_action` gives, which collapses "every direction on
    this screen was tried" into one line that is actually usable.
    """
    if not db_path.is_file():
        return []
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = db.execute(
            "SELECT skeleton_id, action_sig FROM dead_end "
            "WHERE app_key=? AND expires_at > ?",
            (package, time.time())).fetchall()
        db.close()
    except sqlite3.Error as exc:
        log.warning("could not read dead ends for %s: %s", package, exc)
        return []

    counts: Counter = Counter()
    for skeleton, sig in rows:
        verb = str(sig).split("/")[0]
        counts[f"{verb} on screen {str(skeleton)[:8]}"] += 1
    return [(what, n) for what, n in counts.most_common(6) if n >= MIN_OCCURRENCES]


def for_package(cfg: Any, package: str, *,
                limit: int = DEFAULT_RUN_LIMIT) -> History:
    """Read what the recorded runs in `package` repeatedly showed."""
    from .replay import load_events

    history = History(package=package)
    if not package:
        return history

    outcomes: Counter = Counter()
    failures: Counter = Counter()
    successes: Counter = Counter()
    stuck: Counter = Counter()
    refusals: Counter = Counter()
    goals: List[str] = []

    for run_dir in _run_dirs(Path(cfg.run.artifacts_dir).expanduser(), limit):
        try:
            events = load_events(run_dir)
        except Exception as exc:  # noqa: BLE001 - a torn run must not stop the rest
            log.warning("skipping unreadable run %s: %s", run_dir.name, exc)
            continue
        if package not in packages_in(events):
            continue

        history.runs += 1
        # An action is described by the `decide` that chose it, but graded by the
        # `verify` that followed, so the two are joined on the step number.
        chosen: Dict[int, Dict[str, Any]] = {}
        for event in events:
            kind = event.get("kind")
            if kind == "decide" and isinstance(event.get("action"), dict):
                chosen[int(event.get("step") or 0)] = event["action"]
            elif kind == "verify" and event.get("grade") is not None:
                action = chosen.get(int(event.get("step") or 0))
                if action:
                    tally = successes if event["grade"] == "success" else failures
                    tally[_describe_action(action)] += 1
            elif kind == "loop_break" and event.get("exact_id"):
                stuck[str(event["exact_id"])[:8]] += 1
            elif kind == "refused" and event.get("label"):
                refusals[str(event["label"])] += 1
            elif kind == "run_end" and event.get("outcome"):
                outcomes[str(event["outcome"])] += 1
            elif kind == "run_start" and event.get("goal") and len(goals) < 5:
                # The same goal re-run five times is one thing people use the app
                # for, not five.
                goal = " ".join(str(event["goal"]).split())[:MAX_GOAL_CHARS]
                if goal not in goals:
                    goals.append(goal)

    history.outcomes = dict(outcomes)
    history.goals = goals
    history.failures = [(w, n, successes[w]) for w, n in failures.most_common(8)
                        if n >= MIN_OCCURRENCES]
    history.stuck = [(s, n) for s, n in stuck.most_common(5) if n >= MIN_OCCURRENCES]
    history.refusals = [(l, n) for l, n in refusals.most_common(5) if n >= MIN_OCCURRENCES]
    history.dead_ends = _dead_ends_for(Path(cfg.db_path).expanduser(), package)
    return history
