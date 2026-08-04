"""Parse a run's ``events.jsonl`` into summaries and stats for the web UI.

The event file is the single source of truth for both history and the live
view: the agent appends one structured line per decision as the run happens,
so reading it costs nothing and never goes stale mid-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def read_events(path: Path) -> List[Dict[str, Any]]:
    """Every parseable event in the file, oldest first. Tolerant of a torn
    last line, which is what a file being appended to looks like when read."""
    events_file = path / EVENTS_NAME if path.is_dir() else path
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
    summary: Dict[str, Any] = {
        "id": run_id,
        "goal": "",
        "model": "",
        "outcome": "unknown",
        "steps": 0,
        "llm_calls": 0,
        "usd": 0.0,
        "started": 0.0,
        "duration_s": 0.0,
        "packages": [],
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
    else:
        # No run_end: either still running or killed mid-flight. The step
        # count and spend are still recoverable from what was recorded.
        summary["outcome"] = "interrupted"
        summary["steps"] = max((e.get("step", 0) for e in events), default=0)
        summary["usd"] = round(_llm_totals(events)["usd"], 6)
    t0, t1 = first.get("t", 0.0), last.get("t", 0.0)
    summary["duration_s"] = round(max(0.0, t1 - t0), 1)
    return summary


def run_detail(path: Path) -> Dict[str, Any]:
    """Everything the history detail view renders: summary, cost-of-thinking
    stats (mirroring `adbagent report`), scratchpad, and the events."""
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
    return {"summary": summary, "stats": stats,
            "scratchpad": _scratchpad_text(events), "events": events}


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
