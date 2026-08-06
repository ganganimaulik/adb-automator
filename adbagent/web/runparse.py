"""Parse a run's ``events.jsonl`` into summaries and stats for the web UI.

The event file is the single source of truth for both history and the live
view: the agent appends one structured line per decision as the run happens,
so reading it costs nothing and never goes stale mid-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..checkpoint import NAME as CHECKPOINT_NAME
from ..runlog import STREAM_NAME

EVENTS_NAME = "events.jsonl"


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * pct))]


def read_events(path: Path, name: str = EVENTS_NAME) -> List[Dict[str, Any]]:
    """Every parseable event in the file, oldest first. Tolerant of a torn
    last line, which is what a file being appended to looks like when read."""
    events_file = path / name if path.is_dir() else path
    events: List[Dict[str, Any]] = []
    try:
        lines = events_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _llm_totals(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Token/cost roll-up over every event that carries an ``llm`` block."""
    totals = {"n_calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
              "completion_tokens": 0, "reasoning_tokens": 0,
              "latency_s": 0.0, "usd": 0.0}
    for event in events:
        llm = event.get("llm")
        if not isinstance(llm, dict):
            continue
        for key in totals:
            value = llm.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
    return totals


def _scratchpad_text(events: List[Dict[str, Any]]) -> str:
    """The collected-data ledger, replayed from the ``notes`` deltas."""
    from .. import scratchpad
    try:
        return scratchpad.replay(events).plain()
    except Exception:  # noqa: BLE001 - a malformed note must not break the view
        return ""


def summarise(path: Path) -> Dict[str, Any]:
    """A run directory's headline fields, for the history list.

    Only the first and last meaningful events are consulted, so listing a
    directory of long runs stays cheap.
    """
    events = read_events(path)
    run_id = path.name if path.is_dir() else path.parent.name
    directory = path if path.is_dir() else path.parent
    summary: Dict[str, Any] = {
        "id": run_id,
        "goal": "",
        "model": "",
        "outcome": "unknown",
        # What the run answered, and why the harness believed it. Empty for a
        # run that was killed, and for runs recorded before `run_end` carried
        # them -- the detail view hides the block rather than showing a heading
        # over nothing.
        "result": "",
        "evidence": "",
        "steps": 0,
        "llm_calls": 0,
        "usd": 0.0,
        "started": 0.0,
        "duration_s": 0.0,
        "packages": [],
        # A checkpoint on disk means the run never finished and can be
        # continued -- the history view keys its resume button off this.
        "resumable": (directory / CHECKPOINT_NAME).is_file(),
        "n_events": len(events),
    }
    if not events:
        return summary

    first, last = events[0], events[-1]
    if first.get("kind") == "run_start":
        summary["goal"] = first.get("goal", "")
        summary["model"] = first.get("model", "")
        summary["started"] = first.get("t", 0.0)
    if last.get("kind") == "run_end":
        summary["outcome"] = last.get("outcome", "unknown")
        summary["steps"] = last.get("steps", 0)
        summary["llm_calls"] = last.get("llm_calls", 0)
        summary["usd"] = last.get("usd", 0.0)
        summary["packages"] = last.get("packages", [])
        summary["result"] = last.get("result", "")
        summary["evidence"] = last.get("evidence", "")
    else:
        # No run_end: either still running or killed mid-flight. The step
        # count and spend are still recoverable from what was recorded.
        summary["outcome"] = "interrupted"
        summary["steps"] = max((e.get("step", 0) for e in events), default=0)
        summary["usd"] = round(_llm_totals(events)["usd"], 6)
    t0, t1 = first.get("t", 0.0), last.get("t", 0.0)
    summary["duration_s"] = round(max(0.0, t1 - t0), 1)
    return summary


def fold_stream(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join each run of consecutive `llm_stream` chunks into one record.

    The file has one line per chunk because that is how the model talks: a
    measured 46-call run left 56,902 of them carrying 206 KB of text, in a 6 MB
    file. Replaying a finished run has no use for that granularity -- nobody
    watches a saved run arrive token by token -- and shipping it to the browser
    costs more than the text does by an order of magnitude.

    A run is broken by anything else: the start or end of a call, or the model
    switching between thinking and answering. The first chunk's timestamp is
    kept, so the joined record still sorts where the stream began.
    """
    folded: List[Dict[str, Any]] = []
    for record in records:
        if record.get("kind") != "llm_stream":
            folded.append(record)
            continue
        last = folded[-1] if folded else None
        if (last is not None and last.get("kind") == "llm_stream"
                and last.get("stream_type") == record.get("stream_type")):
            last["text"] = (last.get("text") or "") + (record.get("text") or "")
        else:
            folded.append(dict(record))
    return folded


def run_detail(path: Path) -> Dict[str, Any]:
    """Everything the history detail view renders: summary, cost-of-thinking
    stats (mirroring `adbagent report`), scratchpad, and the events.

    The feed itself is the decision events merged with the raw LLM stream --
    folded, see `fold_stream` -- so a finished run shows the same per-call
    thinking panels the live view did, all of them collapsed. Stats keep
    reading the decision events alone: stream lines carry no cost block and
    would only dilute them.
    """
    events = read_events(path)
    summary = summarise(path)

    decide_latencies = [e["wall_s"] for e in events
                        if e.get("kind") == "decide" and isinstance(e.get("wall_s"), (int, float))]
    sweep_reads = sum(
        1 for e in events for call in (e.get("llm") or {}).get("calls", [])
        if isinstance(call, dict) and call.get("purpose") == "read_item")
    totals = _llm_totals(events)

    stats = {
        "decisions": len(decide_latencies),
        "latency_median_s": round(_median(decide_latencies), 1),
        "latency_p90_s": round(_percentile(decide_latencies, 0.9), 1),
        "latency_total_s": round(sum(decide_latencies), 1),
        "sweep_reads": sweep_reads,
        "prompt_tokens": int(totals["prompt_tokens"]),
        "cached_tokens": int(totals["cached_tokens"]),
        "completion_tokens": int(totals["completion_tokens"]),
        "reasoning_tokens": int(totals["reasoning_tokens"]),
        "llm_calls": int(totals["n_calls"]),
        "usd": round(totals["usd"], 6),
    }
    feed = sorted(events + fold_stream(read_events(path, STREAM_NAME)),
                  key=lambda e: e.get("t", 0.0))
    return {"summary": summary, "stats": stats,
            "scratchpad": _scratchpad_text(events), "events": feed}


def list_runs(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Newest-first summaries of every run directory holding an events file."""
    if not artifacts_dir.is_dir():
        return []
    dirs = [d for d in artifacts_dir.iterdir()
            if d.is_dir() and (d / EVENTS_NAME).is_file()]
    dirs.sort(key=lambda d: (d / EVENTS_NAME).stat().st_mtime, reverse=True)
    return [summarise(d) for d in dirs]


def find_run(artifacts_dir: Path, run_id: str) -> Optional[Path]:
    """One run directory by id, refusing anything that escapes artifacts_dir."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    candidate = artifacts_dir / run_id
    if candidate.is_dir() and (candidate / EVENTS_NAME).is_file():
        return candidate
    return None
