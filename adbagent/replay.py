"""Replay recorded decisions against a changed prompt, model or decoder.

Every run already writes down both halves of each decision: the exact messages
that were sent (`runs/<id>/step_NNN_decide_messages.json`) and the action that
came back (the `decide` events in `events.jsonl`). That is a regression set. This
module re-issues those messages and diffs the new answer against the recorded
one, so a change to `prompts.py` or to the reasoning budget can be measured
instead of guessed at.

Two modes, because they answer different questions:

* **verbatim** -- send the recorded messages unchanged. Holds the prompt fixed
  and varies the model, the temperature, the reasoning effort. This is the mode
  for "does thinking-off change any decision".
* **rebuilt system prompt** (`rebuild_system`) -- swap message[0] for whatever
  `prompts.system_prompt()` produces today and leave the run's own data alone.
  This is the mode for "did my prompt edit change any decision". The instructions
  all live in the system message; the rest of the conversation is observation.

What is deliberately *not* offered is a full re-render of the screen from the
recorded run. The dumps keep the rendered text, not the XML, so a rebuilt screen
block would be a re-quote rather than a re-render, and it would silently stop
testing `screen.py` the moment that module changed. Verbatim data plus a live
system prompt is honest about what it covers.

Divergence is not failure. The recorded action is only a baseline, and roughly
one step in twenty of a real run was graded `no_change` or worse -- diverging
from one of *those* is the outcome you were hoping for. So each case carries the
grade the recorded action earned, and the report splits the two apart.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("adbagent.replay")

#: `Recorder.dump_messages` writes this in place of the base64 payload, so a
#: dumped image cannot be replayed -- it is a description of an image, not one.
_STUB_IMAGE = re.compile(r"^\[base64 image payload:")

_DUMP_NAME = re.compile(r"^step_(\d+)_(?P<purpose>[a-z_]+)_messages\.json$")

#: (action, metrics) -- metrics is free-form and only used for reporting, so a
#: test can hand back an empty dict.
Decider = Callable[[List[Dict[str, Any]]], Tuple[Any, Dict[str, Any]]]


class ReplayError(RuntimeError):
    pass


def worked(grade: str) -> bool:
    """Whether a recorded step's verification counted as a success.

    An empty grade is a terminal action -- a `done` that stood, or a `fail` -- and
    counts as having worked, because the run ended on it deliberately. Stated once
    here because both the case and its result ask the same question, and two
    copies of this tuple would drift.
    """
    return grade in ("", "success", "soft_fail")


# ---------------------------------------------------------------------------
# Loading a run
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """One recorded decision, ready to re-issue."""

    step: int
    messages: List[Dict[str, Any]]
    #: The action the run actually took, as recorded (`AgentAction.model_dump()`).
    recorded: Dict[str, Any]
    #: What verification made of it: "success", "no_change", "hard_fail", ...
    #: Empty when the step never reached verification (a terminal action).
    grade: str = ""
    path: Optional[Path] = None

    @property
    def replayable(self) -> bool:
        return not has_stub_image(self.messages)

    @property
    def recorded_was_good(self) -> bool:
        """Whether the baseline is worth agreeing with."""
        return worked(self.grade)


def has_stub_image(messages: Sequence[Dict[str, Any]]) -> bool:
    """True when any image in `messages` is the dump placeholder, not real data."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if _STUB_IMAGE.match(url) or not url.startswith("data:"):
                return True
    return False


def events_path(run: Path) -> Path:
    return run / "events.jsonl" if run.is_dir() else run


def load_events(run: Path) -> List[Dict[str, Any]]:
    path = events_path(run)
    if not path.exists():
        raise ReplayError(f"no events at {path}")
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A run killed mid-write leaves a torn last line. Everything before
            # it is still a valid regression set.
            log.warning("skipping malformed event line in %s", path)
    return out


def load_cases(run: Path, purpose: str = "decide",
               steps: Optional[Sequence[int]] = None,
               limit: int = 0) -> List[Case]:
    """Pair each dumped prompt with the action it produced.

    `limit` samples evenly across the run rather than truncating: the interesting
    steps of a 136-step album walk are at the end, so taking the first 20 would
    replay nothing but the opening navigation.
    """
    run = Path(run).expanduser()
    directory = run if run.is_dir() else run.parent
    events = load_events(run)

    actions: Dict[int, Dict[str, Any]] = {}
    grades: Dict[int, str] = {}
    for event in events:
        step = event.get("step")
        if step is None:
            continue
        if event.get("kind") == "decide":
            actions[int(step)] = event.get("action") or {}
        elif event.get("kind") == "verify":
            grades[int(step)] = event.get("grade") or ""

    wanted = set(steps) if steps else None
    cases: List[Case] = []
    for path in sorted(directory.glob("step_*_messages.json")):
        match = _DUMP_NAME.match(path.name)
        if match is None or match.group("purpose") != purpose:
            continue
        step = int(match.group(1))
        if wanted is not None and step not in wanted:
            continue
        if step not in actions:
            # A dump with no decide event: the call was made but the reply never
            # validated, so there is no baseline to diff against.
            log.debug("step %d has a prompt dump but no recorded action", step)
            continue
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", path, exc)
            continue
        if not isinstance(messages, list) or not messages:
            log.warning("%s is not a message list", path)
            continue
        cases.append(Case(step=step, messages=messages, recorded=actions[step],
                          grade=grades.get(step, ""), path=path))

    cases.sort(key=lambda c: c.step)
    if limit and len(cases) > limit:
        stride = len(cases) / limit
        cases = [cases[int(i * stride)] for i in range(limit)]
    return cases


# ---------------------------------------------------------------------------
# Prompt rebuilding
# ---------------------------------------------------------------------------

def rebuild_system(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace the recorded system prompt with the one `prompts.py` builds now.

    Everything after message[0] is the run's own observations and is left alone.
    """
    from .actions import AgentAction
    from .llm import harden_schema
    from .prompts import system_prompt

    out = [dict(m) for m in messages]
    fresh = system_prompt(harden_schema(AgentAction))
    if out and out[0].get("role") == "system":
        out[0]["content"] = fresh
    else:
        out.insert(0, {"role": "system", "content": fresh})
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

#: Ordered by how much the divergence matters.
VERDICTS = ("match", "same_action", "differs", "error")


def _target_key(target: Optional[Dict[str, Any]]) -> str:
    """Compare targets the way the device would resolve them."""
    if not target:
        return ""
    if target.get("index") is not None:
        return f"#{target['index']}"
    if target.get("resource_id"):
        return f"id={target['resource_id']}"
    text = (target.get("text") or "").strip().lower()
    return f"text={text}"


def _as_dict(action: Any) -> Dict[str, Any]:
    if isinstance(action, dict):
        return action
    dump = getattr(action, "model_dump", None)
    return dump() if callable(dump) else dict(action)


def describe_action(action: Any) -> str:
    """Short, comparable rendering. Not `AgentAction.describe()`: that resolves
    against a live screen, and a recorded action has none."""
    data = _as_dict(action)
    bits = [str(data.get("action") or "?")]
    target = _target_key(data.get("target"))
    if target:
        bits.append(target)
    if data.get("x") is not None and data.get("y") is not None:
        bits.append(f"({data['x']:.2f},{data['y']:.2f})")
    for extra in ("key", "direction"):
        if data.get(extra):
            bits.append(str(data[extra]))
    return " ".join(bits)


@dataclass
class Result:
    """One case, replayed."""

    step: int
    verdict: str
    recorded: str
    replayed: str
    #: The grade the *recorded* action earned, so a divergence can be read as an
    #: improvement rather than a regression.
    grade: str = ""
    error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def regression_risk(self) -> bool:
        """Diverged from a step that had worked."""
        return self.verdict in ("differs", "error") and worked(self.grade)


def compare(recorded: Dict[str, Any], replayed: Any) -> str:
    """One of `VERDICTS`, ignoring the free-text fields.

    `observation`, `reasoning`, `notes` and `progress` are prose and will never
    match verbatim; grading them would drown the signal. What matters is whether
    the phone would have been driven the same way.
    """
    fresh = _as_dict(replayed)
    if (recorded.get("action") or "") != (fresh.get("action") or ""):
        return "differs"
    if _target_key(recorded.get("target")) != _target_key(fresh.get("target")):
        return "same_action"
    for extra in ("key", "direction"):
        if (recorded.get(extra) or "") != (fresh.get(extra) or ""):
            return "same_action"
    for extra in ("x", "y"):
        if (recorded.get(extra) or 0) != (fresh.get(extra) or 0):
            return "same_action"
    return "match"


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

@dataclass
class Report:
    results: List[Result] = field(default_factory=list)
    skipped: List[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    def count(self, verdict: str) -> int:
        return sum(1 for r in self.results if r.verdict == verdict)

    @property
    def agreement(self) -> float:
        """Fraction that would have driven the phone identically."""
        return self.count("match") / self.n if self.n else 0.0

    @property
    def regressions(self) -> List[Result]:
        return [r for r in self.results if r.regression_risk]

    def totals(self, key: str) -> int:
        return sum(int(r.metrics.get(key) or 0) for r in self.results)

    def median(self, key: str) -> float:
        values = sorted(float(r.metrics.get(key) or 0) for r in self.results
                        if r.metrics.get(key) is not None)
        if not values:
            return 0.0
        mid = len(values) // 2
        if len(values) % 2:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cases": self.n,
            "skipped": self.skipped,
            "agreement": round(self.agreement, 4),
            "verdicts": {v: self.count(v) for v in VERDICTS},
            "regressions": [r.step for r in self.regressions],
            "usd": round(sum(float(r.metrics.get("usd") or 0)
                             for r in self.results), 6),
            "median": {k: self.median(k) for k in
                       ("prompt_tokens", "cached_tokens", "completion_tokens",
                        "reasoning_tokens", "reasoning_chars", "latency_s")},
            "results": [
                {"step": r.step, "verdict": r.verdict, "grade": r.grade,
                 "recorded": r.recorded, "replayed": r.replayed,
                 "error": r.error, "metrics": r.metrics}
                for r in self.results
            ],
        }


def replay(cases: Sequence[Case], decide: Decider, *,
           rebuild_system_prompt: bool = False,
           on_result: Optional[Callable[[Result], None]] = None) -> Report:
    """Re-issue every case through `decide` and grade the answers.

    `decide` takes the messages and returns `(action, metrics)`. Errors are
    recorded as an `error` verdict rather than raised: one model refusing to
    produce valid JSON on step 40 should not throw away the other 126 results.
    """
    report = Report()
    for case in cases:
        if not case.replayable:
            # A dumped image is a placeholder string. Sending it would measure
            # the model's reaction to the words "[base64 image payload: 91234
            # chars]", which is worse than measuring nothing.
            report.skipped.append(case.step)
            continue
        messages = (rebuild_system(case.messages) if rebuild_system_prompt
                    else [dict(m) for m in case.messages])
        recorded = describe_action(case.recorded)
        try:
            action, metrics = decide(messages)
        except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
            result = Result(step=case.step, verdict="error", recorded=recorded,
                            replayed="", grade=case.grade, error=str(exc)[:300])
        else:
            result = Result(step=case.step,
                            verdict=compare(case.recorded, action),
                            recorded=recorded,
                            replayed=describe_action(action),
                            grade=case.grade,
                            metrics=metrics or {})
        report.results.append(result)
        if on_result is not None:
            on_result(result)
    return report
