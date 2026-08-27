"""Live reload for the web UI: watch what the UI is made of, tell the browser.

Everything the page is built from can change while it is open -- the Python
behind the API, the HTML/JS/CSS in front of it, the config, the skills, the
reply policy -- and each reaches the page a different way, because each costs a
different amount to apply:

* ``assets`` -- the files under ``static/``. The server reads those off disk on
  every request, so nothing has to restart: the page reloads and it has them.
* ``config``, ``skills``, ``policy`` -- read off disk per request too, but the
  page is holding the old copy in a form, so only that panel refetches. A
  reload of the whole page would be a reload out from under whatever is being
  typed.
* ``code`` -- any ``.py`` in the package. The process imported it once and will
  not import it again, so the only way to pick it up is to start over. `cmd_ui`
  re-execs; the browser notices the new process and reloads itself.

A code change is *deferred* while an agent is driving the phone. Re-exec
replaces this process, and the run and watch subprocesses are deliberately in
their own process groups, so they would survive it -- still tapping the screen,
still spending money, with nothing left that has a handle on them. So the
restart waits for the phone to be free and says so meanwhile.

Polled rather than hooked into an OS watch API: it is a few dozen files, a stat
apiece every half second is nothing beside what one LLM call costs, and it adds
no dependency that has to be installed before the UI will start.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

log = logging.getLogger("adbagent.web.reload")

#: How often the tree is re-stat'd. Below the threshold where a save feels like
#: it took a moment, above the point where the scan is worth noticing.
POLL_S = 0.4

#: Never walked. `runs` is the artifacts directory, which grows a file per step
#: of every run -- watching it would fire the reloader continuously while the
#: thing it is watching for is not happening.
SKIP_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".mypy_cache",
                       "node_modules", ".venv", "venv", "runs"})

CODE_SUFFIXES = frozenset({".py"})
ASSET_SUFFIXES = frozenset({".html", ".js", ".css", ".svg", ".png", ".ico"})
#: What counts as a policy in the policies directory -- `policies.SUFFIXES`,
#: duplicated rather than imported so this dev aid pulls in nothing at import
#: time. A drift here costs a panel that does not refresh, not a wrong reply.
POLICY_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

#: How long a restart frame is given to reach the browser before the process
#: goes. Not needed for correctness -- a page whose stream reconnects to a
#: different boot id reloads regardless -- but it turns a blank pause into a
#: sentence.
RESTART_GRACE_S = 0.3


@dataclass(frozen=True)
class Change:
    """One batch of files that changed together, as the browser will hear it."""

    version: int
    kind: str
    paths: Tuple[str, ...] = ()
    #: Only on ``restart``: why it is waiting, or "" when it is going now.
    note: str = ""

    def as_frame(self) -> Dict[str, object]:
        return {"version": self.version, "kind": self.kind,
                "paths": [Path(p).name for p in self.paths],
                "full": list(self.paths), "note": self.note}


@dataclass(frozen=True)
class _Watch:
    kind: str
    path: Path
    suffixes: Optional[FrozenSet[str]] = None

    def wanted(self, path: Path) -> bool:
        return self.suffixes is None or path.suffix.lower() in self.suffixes

    def files(self) -> Iterator[Path]:
        """Every file this watch covers, right now.

        A path that does not exist yet is not an error: a policy file is
        normally written after the UI is already open, and its arrival is
        exactly the change worth reporting.
        """
        try:
            if self.path.is_file():
                if self.wanted(self.path):
                    yield self.path
                return
            if not self.path.is_dir():
                return
            for root, dirs, names in os.walk(self.path):
                dirs[:] = [d for d in dirs
                           if d not in SKIP_DIRS and not d.startswith(".")]
                for name in names:
                    child = Path(root) / name
                    if self.wanted(child):
                        yield child
        except OSError:  # a directory removed mid-walk is a change, not a crash
            return


class LiveReload:
    """Watches the files the UI is made of and publishes what changed.

    Publishes rather than acts: the browser subscribes over SSE and decides
    what a change to each kind means for it. The one thing it cannot do for
    itself is restart the server, so that is handed back through `on_restart`
    -- and held until `busy` says the phone is free.
    """

    def __init__(self, *,
                 busy: Optional[Callable[[], str]] = None,
                 on_restart: Optional[Callable[[], None]] = None,
                 poll_s: float = POLL_S,
                 grace_s: float = RESTART_GRACE_S) -> None:
        #: Identifies this process to the page. A stream that reconnects and
        #: reports a different one is a server that restarted underneath it,
        #: which is the whole of how the browser detects a code reload.
        self.boot = uuid.uuid4().hex[:12]
        #: Why a restart cannot happen yet, or "". Consulted only for code.
        self.busy: Callable[[], str] = busy or (lambda: "")
        #: Called from the watcher thread when a code change is ready to apply.
        self.on_restart = on_restart
        self.poll_s = poll_s
        self.grace_s = grace_s

        self._watches: List[_Watch] = []
        self._seen: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None
        self._changes: List[Change] = []
        self._version = 0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._restart_pending = False
        self._blocked = ""

    # -- setup ----------------------------------------------------------

    def watch(self, kind: str, path, suffixes: Optional[Sequence[str]] = None) -> "LiveReload":
        """Cover `path` -- a file or a directory -- reporting it as `kind`."""
        self._watches.append(_Watch(
            kind, Path(path).expanduser(),
            frozenset(s.lower() for s in suffixes) if suffixes else None))
        return self

    def watching(self) -> List[Dict[str, str]]:
        return [{"kind": w.kind, "path": str(w.path)} for w in self._watches]

    @property
    def restarts(self) -> bool:
        """Whether a code change actually leads anywhere from here."""
        return self.on_restart is not None

    # -- what the stream reads ------------------------------------------

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def since(self, version: int) -> List[Change]:
        with self._lock:
            return [c for c in self._changes if c.version > version]

    # -- the loop -------------------------------------------------------

    def start(self) -> "LiveReload":
        if self._thread is not None:
            return self
        self.poll()  # prime, so the first tick reports changes and not the tree
        self._thread = threading.Thread(target=self._run, name="live-reload",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self.poll_s * 3)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.step()
            except Exception:  # noqa: BLE001 - a dev aid may not take the UI down
                log.exception("live reload poll failed")

    def step(self) -> List[Change]:
        """One pass: scan, publish, and restart if a code change is ready.

        Separate from the thread so a test can drive it a tick at a time.
        """
        changes = self.poll()
        if any(c.kind == "code" for c in changes):
            self._restart_pending = True
        if not self._restart_pending or self.on_restart is None:
            return changes

        why = self.busy()
        if why:
            # Said once per reason rather than once per tick: the page shows the
            # last one, and a banner rewritten twice a second is a flicker.
            if why != self._blocked:
                self._blocked = why
                changes.append(self._record("restart", note=why))
            return changes

        self._blocked = ""
        self._restart_pending = False
        changes.append(self._record("restart"))
        if self.grace_s:
            time.sleep(self.grace_s)
        self.on_restart()
        return changes

    def poll(self) -> List[Change]:
        """Stat everything watched and record what moved since last time."""
        found: Dict[Tuple[str, str], Tuple[int, int]] = {}
        for watch in self._watches:
            for path in watch.files():
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found[(watch.kind, str(path))] = (stat.st_mtime_ns, stat.st_size)

        if self._seen is None:  # first pass: this is the tree, not a change to it
            self._seen = found
            return []

        moved: Dict[str, List[str]] = {}
        for key, stamp in found.items():
            if self._seen.get(key) != stamp:
                moved.setdefault(key[0], []).append(key[1])
        for key in self._seen:
            if key not in found:  # a deleted skill is a change to the list
                moved.setdefault(key[0], []).append(key[1])
        self._seen = found
        return [self._record(kind, sorted(paths))
                for kind, paths in sorted(moved.items())]

    def _record(self, kind: str, paths: Sequence[str] = (), note: str = "") -> Change:
        with self._lock:
            self._version += 1
            change = Change(self._version, kind, tuple(paths), note)
            self._changes.append(change)
            # A page open for a day would otherwise hold every edit of that day.
            # Anything older than the tail is already applied or already missed.
            del self._changes[:-100]
            return change


def for_ui(package_dir: Path, *, config_path: Optional[Path] = None,
           skills_dir: Optional[Path] = None,
           policy_path: Optional[Path] = None,
           policies_dir: Optional[Path] = None, **kwargs) -> LiveReload:
    """The set the web UI is built from, in the kinds the page knows about."""
    reloader = LiveReload(**kwargs)
    reloader.watch("code", package_dir, CODE_SUFFIXES)
    reloader.watch("assets", package_dir / "web" / "static", ASSET_SUFFIXES)
    if config_path:
        reloader.watch("config", config_path)
    if skills_dir:
        reloader.watch("skills", skills_dir, {".json"})
    # Both, as one kind: the picker shows every policy there is and the editor
    # holds one of them, so a policy arriving in the directory and the open one
    # changing underneath are the same news to the same panel. The configured
    # policy is watched separately because it may live outside the directory.
    if policies_dir:
        reloader.watch("policy", policies_dir, POLICY_SUFFIXES)
    if policy_path:
        reloader.watch("policy", policy_path)
    return reloader


def in_source_checkout(package_dir: Path) -> bool:
    """Whether this is the repo rather than an installed copy.

    The question live reload is on by default for: watching files nobody is
    editing costs a stat every half second and can never fire, and re-execing
    site-packages under someone who did not ask for it is a surprise.
    """
    return (package_dir.parent / ".git").exists()
