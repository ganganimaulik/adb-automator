"""The log a run leaves behind, for the debugging that happens afterwards.

`events.jsonl` records what a run *decided*: one structured line per step, which
is what `report` and `replay` read back. This records what it *did* -- the adb
call that timed out, the screen that never settled, the swipe that was
retargeted to a pager, the LLM retry, the request field the provider rejected,
the recovery tier a lost device needed. Every module already writes those with
`logging`; until this they went to the console at WARNING and were gone with the
scrollback.

So there are two thresholds rather than one: the console shows what `-v` asked
for, and the file always takes DEBUG. An expensive, slow or intermittent run does
not come back on request just because this time you would have liked to watch it.

Only the `adbagent` tree is captured. `httpx` and `openai` at DEBUG dump the
request bodies, which on a vision turn means a base64 screenshot per line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: The file, inside `runs/<id>/`.
LOG_NAME = "run.log"

#: The raw LLM stream of a run, inside `runs/<id>/`: one line per `llm_start`,
#: `llm_stream` chunk and `llm_end`. Kept out of `events.jsonl` on purpose --
#: a reasoning turn is ~4,000 tokens arriving as hundreds of chunks, and that
#: file is what `report`, `replay` and the run history parse. This one is what
#: the web UI tails to show the model thinking live.
STREAM_NAME = "stream.jsonl"

#: Where each step's elements were, inside `runs/<id>/`: one line per decided
#: step, carrying the screen's size and the rectangle of every element the model
#: was shown. What it is for is drawing them -- the trace can say `tap #12` and
#: the picture beside it cannot say which thing #12 was, and matching the two was
#: done by eye.
#:
#: Kept out of `events.jsonl` for the same reason as the stream above: eighty
#: rectangles is a few kilobytes a step, and that file is the one `report`,
#: `replay`, the history list and the run detail all parse in full. A run
#: recorded before this file existed simply has no boxes to draw.
SCREENS_NAME = "screens.jsonl"

#: A frame a run actually showed a model, inside `runs/<id>/`. Only the
#: submitted ones are kept: the web UI shows the screenshot beside the call that
#: saw it, which is the difference between reading a vision read and checking it.
#:
#: Named to sit next to the prompt dump it belongs to
#: (`step_004_analyze_image_messages.json`), plus a digest of the bytes -- so one
#: frame shown twice is one file, and two different frames on one step (the
#: loop's vision read and the judge's) cannot overwrite each other.
def shot_name(step: int, purpose: str, data: bytes) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", (purpose or "shot").lower())
    return f"step_{step:03d}_{slug}_{hashlib.sha1(data).hexdigest()[:8]}.jpg"


#: What `shot_name` produces. A server asked for one of these can tell whether
#: the name is ours before it opens anything.
SHOT_RE = re.compile(r"step_\d{3,}_[a-z0-9_]+_[0-9a-f]{8}\.jpg")

#: The logger every module in the package hangs off.
ROOT = "adbagent"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LINE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"

#: How much of one event's fields goes into the text log. The whole of it is
#: already on the same file's neighbour, `events.jsonl`; this line exists to put
#: the decision in order next to the device traffic around it, and a 4,000-word
#: `reasoning` field in the middle of that stops the file being readable.
EVENT_CHARS = 400

log = logging.getLogger("adbagent.runlog")

#: Events are logged under their own name so a reader can grep for the decisions
#: alone (`grep adbagent.events run.log`) or filter them out.
events_log = logging.getLogger("adbagent.events")


class _Formatter(logging.Formatter):
    """The standard line, plus the thread name when it is not the main one.

    A vision read is prefetched into a worker thread so it overlaps the gesture
    it describes, which puts two interleaved stories in one file. Naming the
    thread only when there is something to name keeps that legible without
    widening the thousands of lines that came from the loop itself.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        line = super().formatMessage(record)
        if record.threadName and record.threadName != "MainThread":
            line += f"  [{record.threadName}]"
        return line


@dataclass(eq=False)  # identity: two logs on one directory are still two logs
class RunLog:
    """One open log file."""

    path: Path
    handler: logging.Handler

    def close(self) -> None:
        _release(self)


#: Live logs. Nesting is not the common case -- one run, one log -- but the
#: level must stay down while *any* of them is open.
_open: List[RunLog] = []
#: The level `adbagent` had before the first of them lowered it. Kept here
#: rather than per log so that closing them out of order still puts back what
#: the process started with, instead of the DEBUG an earlier log had set.
_baseline_level: Optional[int] = None
_lock = threading.Lock()


def run_dir(cfg: Any, run_id: str) -> Path:
    """Where a run's artifacts live. One definition, so nothing drifts."""
    return Path(cfg.run.artifacts_dir).expanduser() / run_id


def log_path(directory: Any) -> Path:
    return Path(directory).expanduser() / LOG_NAME


def attach(directory: Any, *, level: int = logging.DEBUG) -> Optional[RunLog]:
    """Start writing `<directory>/run.log` until the returned handle is closed.

    Returns None if there is nowhere to write or the file cannot be opened -- a
    read-only artifacts directory is a reason to run without a log, not a reason
    not to run.
    """
    if not directory:
        return None
    path = log_path(directory)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    except OSError as exc:
        log.warning("no run log at %s: %s", path, exc)
        return None
    handler.setLevel(level)
    handler.setFormatter(_Formatter(LINE_FORMAT, datefmt=DATE_FORMAT))

    global _baseline_level
    logger = logging.getLogger(ROOT)
    handle = RunLog(path=path, handler=handler)
    with _lock:
        if not _open:
            _baseline_level = logger.level
        # A logger set above DEBUG drops a debug record before any handler sees
        # it, so the file cannot be given the detail by asking the handler for
        # it -- the level has to come down here. What keeps the console at the
        # verbosity the flags asked for is the level on *its* handler; see
        # `cli.setup_logging`.
        if not logger.isEnabledFor(level):
            logger.setLevel(level)
        logger.addHandler(handler)
        _open.append(handle)
    return handle


def _release(handle: RunLog) -> None:
    global _baseline_level
    logger = logging.getLogger(ROOT)
    with _lock:
        logger.removeHandler(handle.handler)
        if handle in _open:
            _open.remove(handle)
        # Only the last one out puts the level back, and a leftover DEBUG here
        # would make every later command in this process a wall of text.
        if not _open and _baseline_level is not None:
            logger.setLevel(_baseline_level)
            _baseline_level = None
    try:
        handle.handler.close()
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def capture(directory: Any) -> Iterator[Optional[RunLog]]:
    """Append to a run's log for the duration of the block.

    Used for the work that happens either side of the loop -- the after-run skill
    write, mainly -- which spends real calls on the run's behalf and belongs in
    the run's log rather than nowhere. A falsy directory makes the block a no-op,
    so a caller that has no run to attribute the work to needs no branch.
    """
    handle = attach(directory)
    try:
        yield handle
    finally:
        if handle is not None:
            handle.close()


def preamble(**fields: Any) -> None:
    """The header a reader needs before the first line means anything.

    Which run, which goal, which models, which phone, and the settings that
    change how the loop behaves. All of it is knowable when the run starts and
    none of it is recoverable a week later from a shell history that has scrolled
    away -- and "was this the run with `never_screenshot` set?" decides whether
    the rest of the file is surprising or expected.
    """
    for key, value in fields.items():
        if value in ("", None, {}, [], ()):
            continue
        log.debug("%-16s %s", key, value)


#: Fields left out of the mirrored line. Token counts and per-call latencies are
#: what `report` is for; here they would push the action off the end of the line.
EVENT_SKIP = ("llm",)


def _set(value: Any) -> bool:
    """Was this field filled in? `False` and `0` were, an empty string was not."""
    return value is not None and value != ""


def _prune(value: Any) -> Any:
    """Drop the unset fields of an action. `AgentAction` has thirty-odd of them
    and any one decision uses three, so the blanks are most of the line."""
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if _set(v)}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def event(kind: str, fields: Dict[str, Any]) -> None:
    """Mirror one `events.jsonl` record into the text log, in place."""
    keep = {k: _prune(v) for k, v in fields.items()
            if k not in EVENT_SKIP and _set(v)}
    try:
        rendered = json.dumps(keep, default=str, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover -- default=str covers it
        rendered = repr(keep)
    if len(rendered) > EVENT_CHARS:
        rendered = f"{rendered[:EVENT_CHARS]}... (+{len(rendered) - EVENT_CHARS} chars)"
    events_log.debug("%s %s", kind, rendered)


#: A logged line at WARNING or worse, as `_Formatter` writes it.
_PROBLEM = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\.\d{3} "
                      r"(WARNING|ERROR|CRITICAL)\s+(.*)$")


def problems(directory: Any) -> List[str]:
    """The WARNING-and-worse lines of a run's log, in order.

    `report` ends with these, so "did anything go wrong in this run" does not
    require opening the file -- and the answer is not buried under the thousands
    of DEBUG lines that make the file worth having.
    """
    path = log_path(directory) if not str(directory).endswith(LOG_NAME) \
        else Path(directory)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [f"{m.group(1)} {m.group(2).lower()}: {m.group(3)}"
            for m in (_PROBLEM.match(line) for line in text.splitlines()) if m]
