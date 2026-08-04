"""Launch and supervise `adbagent` CLI subprocesses for the web UI.

Runs are spawned as real CLI invocations rather than driving the Agent
in-process: the CLI already handles KeyboardInterrupt by restoring the
phone's keyboard, animations, rotation and screen timeout, so stopping a
run from the browser is just a SIGINT away. It also means a destructive
action can never block the server on `input()` -- web runs are always
launched with `--unattended` (refuse) or `--allow-destructive`.

One run at a time: there is one phone, and two agents driving it at once
would each read the other's actions as their own.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunManager:
    def __init__(self, artifacts_dir: Path | str, *, config_path: str = ""):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_path = config_path
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._goal = ""
        self._started_at = 0.0
        self._run_dir: Optional[Path] = None
        self._returncode: Optional[int] = None
        self._output: List[str] = []  # ring buffer of the child's stdout
        self._dirs_before: set = set()

    # -- state -------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "running": running,
                "goal": self._goal,
                "run_id": self._run_dir.name if self._run_dir else "",
                "pid": self._proc.pid if self._proc and running else None,
                "started_at": self._started_at if running else 0.0,
                "returncode": None if running else self._returncode,
                "output_tail": list(self._output[-50:]),
            }

    def run_dir(self) -> Optional[Path]:
        with self._lock:
            return self._run_dir

    # -- lifecycle ----------------------------------------------------------

    def start(self, goal: str, *, max_steps: Optional[int] = None,
              budget_usd: Optional[float] = None, repeat: str = "1",
              dry_run: bool = False, allow_destructive: bool = False,
              no_learn: bool = False, serial: str = "",
              resume: str = "") -> Dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("a run is already in progress")

            if resume:
                # No goal argument: the checkpoint supplies it, along with the
                # run's history and where it stopped.
                argv = [sys.executable, "-m", "adbagent", "run",
                        "--resume", resume, "--repeat", "1"]
            else:
                argv = [sys.executable, "-m", "adbagent", "run", goal,
                        "--repeat", str(repeat or "1")]
            if max_steps:
                argv += ["--max-steps", str(max_steps)]
            if budget_usd is not None:
                argv += ["--budget-usd", str(budget_usd)]
            if dry_run:
                argv.append("--dry-run")
            # Never let the child reach a prompt: there is no tty behind it.
            argv.append("--allow-destructive" if allow_destructive else "--unattended")
            if no_learn:
                argv.append("--no-learn")
            if serial:
                argv += ["-d", serial]
            if self.config_path:
                argv += ["-c", self.config_path]

            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._dirs_before = {d.name for d in self.artifacts_dir.iterdir()
                                 if d.is_dir()}
            # A resumed run reuses its existing directory, which the new-dir
            # discovery below would never report as fresh -- so it is set
            # directly, and discovery has nothing left to do.
            self._run_dir = self.artifacts_dir / resume if resume else None
            self._returncode = None
            self._output = []
            self._goal = goal
            self._started_at = time.time()

            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            threading.Thread(target=self._watch, daemon=True).start()
            return {"pid": self._proc.pid, "argv": argv}

    def stop(self, timeout_s: float = 10.0) -> bool:
        """SIGINT the run so the agent restores the phone, then escalate."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def _watch(self) -> None:
        """Reap the child, drain its output, and locate its run directory."""
        proc = self._proc
        assert proc is not None
        assert proc.stdout is not None
        for line in proc.stdout:
            with self._lock:
                self._output.append(line.rstrip("\n"))
                self._output = self._output[-200:]
        self._returncode = proc.wait()
        # The run directory may appear a beat after the last stdout line on a
        # short run; give discovery one last sweep before declaring failure.
        self._discover_run_dir(deadline=time.monotonic() + 2.0)

    def _discover_run_dir(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            with self._lock:
                if self._run_dir is not None:
                    return
            try:
                current = {d.name: d for d in self.artifacts_dir.iterdir() if d.is_dir()}
            except OSError:
                current = {}
            fresh = [d for name, d in current.items() if name not in self._dirs_before]
            if fresh:
                newest = max(fresh, key=lambda d: d.stat().st_mtime)
                with self._lock:
                    self._run_dir = newest
                return
            time.sleep(0.25)

    def wait_for_run_dir(self, timeout_s: float = 60.0) -> Optional[Path]:
        """The new run's artifact directory, once the agent has created it."""
        self._discover_run_dir(deadline=time.monotonic() + timeout_s)
        with self._lock:
            return self._run_dir


class JobManager:
    """Fire-and-poll tracking for side tasks (`skills generate`) that are not
    phone runs: no exclusivity, no SIGINT choreography, just status and the
    tail of the output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def start(self, argv: List[str]) -> int:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
            self._jobs[job_id] = {"id": job_id, "argv": argv, "proc": proc,
                                  "output": [], "returncode": None}
        threading.Thread(target=self._drain, args=(job_id,), daemon=True).start()
        return job_id

    def _drain(self, job_id: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
        proc: subprocess.Popen = job["proc"]
        assert proc.stdout is not None
        for line in proc.stdout:
            with self._lock:
                job["output"].append(line.rstrip("\n"))
                job["output"] = job["output"][-200:]
        job["returncode"] = proc.wait()

    def get(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {"id": job_id, "argv": job["argv"],
                    "running": job["proc"].poll() is None,
                    "returncode": job["returncode"],
                    "output_tail": list(job["output"][-50:])}


def sse(payload: Dict[str, Any], event: str = "message") -> str:
    """One Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
