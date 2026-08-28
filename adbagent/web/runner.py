"""Launch and supervise `adbagent` CLI subprocesses for the web UI.

Runs are spawned as real CLI invocations rather than driving the Agent
in-process: the CLI already handles KeyboardInterrupt by restoring the
phone's keyboard, animations, rotation and screen timeout, so stopping a
run from the browser is just a SIGINT away. It also means a destructive
action can never block the server on `input()` -- web runs are always
launched with `--unattended` (refuse) or `--allow-destructive`.

One child at a time: there is one phone, and two agents driving it at once
would each read the other's actions as their own. That holds for `skills
generate` as much as for `run` -- a tour drives the phone the same way -- which
is why both are the same class here, discovering the `runs/<id>/` directories
they write so the browser can tail them.

`--repeat` makes one subprocess into several runs: the CLI pursues the goal
again from scratch each iteration, in a *new* `runs/<id>` directory, holding
one device connection and one spend ledger for the whole session. So the
manager tracks a list of directories rather than one, and keeps looking for
them for as long as the child lives.
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

#: How long to keep looking for run directories after the child has exited. The
#: last one can appear a beat after its final line of output on a short run.
POST_EXIT_SWEEP_S = 2.0


class ChildProcess:
    """One spawned CLI process, its output, and the runs it leaves on disk.

    Both kinds of child the UI starts are the same thing underneath: a
    subprocess that drives the phone and writes `runs/<id>/` as it goes. The
    live view tails those files, so finding them is every child's business
    rather than the run view's -- which is what lets a `skills generate` tour
    show the same feed, counters and screenshots as a goal run.
    """

    #: What this child is called when something has to be said about it. A watch
    #: refused a pause because it is not running should not be answered with a
    #: sentence about a run -- the browser shows that text verbatim, and a
    #: message about the wrong thing reads as a bug in the page.
    NOUN = "run"

    def __init__(self, artifacts_dir: Path | str):
        self.artifacts_dir = Path(artifacts_dir)
        # Re-entrant: a subclass's `start` holds it across the check-and-spawn,
        # and `_spawn` takes it again to publish what it set up.
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._started_at = 0.0
        self._returncode: Optional[int] = None
        self._output: List[str] = []  # ring buffer of the child's stdout
        #: Lines ever printed, which the ring buffer alone cannot say. A reader
        #: following the output needs a mark that survives lines falling off the
        #: front, so it can tell "nothing new" from "you missed some".
        self._output_seq = 0
        #: One entry per iteration, oldest first. `--repeat 1` leaves one.
        self._run_dirs: List[Path] = []
        self._dirs_before: set = set()
        #: A stop has been asked for and the child is on its way out -- still
        #: running, still holding the phone, but doing shutdown work rather than
        #: the job. Its own phase, because it is neither "running" nor "over" and
        #: showing it as either is what makes a stop look like it did nothing.
        self._stopping = False
        #: `_output_seq` when the signal went out, so a reader can replay the
        #: shutdown from its first line rather than from wherever it happened to
        #: look next.
        self._stop_seq = 0
        #: What this child has been told to do with itself: `run`, `pause` or
        #: `step`. Held here as well as on disk because it is a mode somebody
        #: switched on, and the file that carries it lives in one run directory
        #: -- so under `--repeat` it has to be re-applied to each new one, or a
        #: pause would quietly lapse at an iteration boundary.
        self._mode = "run"
        #: Counts commands rather than timestamping them, so the child can tell
        #: a new instruction from the same one read again a moment later.
        self._mode_seq = 0

    # -- state -------------------------------------------------------------

    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def state(self) -> Dict[str, Any]:
        """What the live view needs: whether it is going, which run it is
        writing now, and the tail of what it has printed."""
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            return {
                "running": running,
                "stopping": running and self._stopping,
                "run_id": self._run_dirs[-1].name if self._run_dirs else "",
                "iteration": len(self._run_dirs),
                "pid": self._proc.pid if self._proc and running else None,
                "started_at": self._started_at if running else 0.0,
                "returncode": None if running else self._returncode,
                "output_tail": list(self._output[-50:]),
                # What it has been asked to do with itself. Only meaningful
                # while it is going: a finished child is not paused, it is over.
                "mode": self._mode if running else "run",
            }

    def control(self, cmd: str) -> str:
        """Tell the child to hold, carry on, or take one more step.

        Writes into the directory it is writing now. A `step` leaves the mode at
        `step` here while the child spends it and returns to `pause` on its own
        -- the two answer different questions ("what was it last told" against
        "what is it doing"), and the child's own `control` events are what the
        page follows for the second.
        """
        from .. import control as controlmod

        with self._lock:
            if not (self._proc is not None and self._proc.poll() is None):
                raise RuntimeError(f"no {self.NOUN} in progress")
            if self._stopping:
                raise RuntimeError(f"the {self.NOUN} is stopping")
            if not self._run_dirs:
                raise RuntimeError(
                    f"the {self.NOUN} has not written a directory yet")
            self._mode = cmd
            self._mode_seq += 1
            controlmod.send(self._run_dirs[-1], cmd, self._mode_seq)
            return cmd

    def _apply_mode(self, run_dir: Path) -> None:
        """Carry a standing pause into a directory that has just appeared.

        Under `--repeat` each iteration writes its own, and the control file
        lives in one of them. Without this, pausing during iteration 3 would be
        forgotten the moment iteration 4 started -- which reads as a pause that
        was ignored rather than one that was scoped.
        """
        from .. import control as controlmod

        if self._mode == "run":
            return
        try:
            controlmod.send(run_dir, self._mode, self._mode_seq)
        except OSError:
            pass

    def stop_mark(self) -> int:
        """The output position the stop was asked for at."""
        with self._lock:
            return self._stop_seq

    def output_since(self, seq: int) -> tuple[int, List[str]]:
        """Lines printed since `seq`, and the mark to ask from next time.

        Clamped to what the ring buffer still holds: a reader that fell behind
        gets the newest of it rather than an exception, and one holding a mark
        from a previous child -- the counter restarts per spawn -- is simply
        moved to the current position instead of being handed the whole buffer
        as though it were new.
        """
        with self._lock:
            if seq >= self._output_seq:
                return self._output_seq, []
            missed = self._output_seq - seq
            lines = list(self._output[-missed:]) if missed <= len(self._output) \
                else list(self._output)
            return self._output_seq, lines

    def run_dir(self) -> Optional[Path]:
        """The iteration being written now -- the newest directory seen."""
        with self._lock:
            return self._run_dirs[-1] if self._run_dirs else None

    def run_dirs(self) -> List[Path]:
        with self._lock:
            return list(self._run_dirs)

    # -- lifecycle ----------------------------------------------------------

    def _spawn(self, argv: List[str], *, seed: str = "") -> subprocess.Popen:
        """Start `argv`, and start watching for the runs it writes.

        `seed` names a directory the child will *reuse* rather than create -- a
        resumed run -- which the new-directory sweep would never report as
        fresh, so it is seeded directly.
        """
        with self._lock:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._dirs_before = {d.name for d in self.artifacts_dir.iterdir()
                                 if d.is_dir()}
            self._run_dirs = [self.artifacts_dir / seed] if seed else []
            self._returncode = None
            self._output = []
            self._output_seq = 0
            self._stopping = False
            self._stop_seq = 0
            # A new child starts running. The sequence deliberately does not
            # reset: the agent clears its own control file on the way in, but a
            # counter that restarted could still have a stale file out-rank a
            # fresh command if anything ever raced the two.
            self._mode = "run"
            self._started_at = time.time()
            # Unbuffered because stdout is a pipe: Python block-buffers those, so
            # a child's lines would otherwise sit in an 8KB buffer until it
            # exited. Everything the child says outside its run files -- the
            # skill written up after the loop, a refusal before there is a run at
            # all -- is exactly the part nothing else reports, and holding it
            # back until the process is gone makes it arrive too late to be news.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                   "PYTHONUNBUFFERED": "1"}
            # On Windows, put the child in its own process group so we can
            # send it CTRL_BREAK_EVENT without killing the server too.
            flags = 0
            if sys.platform == "win32":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=flags,
            )
            proc = self._proc
        # The threads are handed the process rather than reading it back off
        # `self`: a later child would otherwise inherit its predecessor's
        # watchers, and both would write to the same output buffer.
        threading.Thread(target=self._watch, args=(proc,), daemon=True).start()
        threading.Thread(target=self._follow_dirs, args=(proc,), daemon=True).start()
        self._sweep()  # a run that started instantly is already on disk
        return proc

    def _watch(self, proc: subprocess.Popen) -> None:
        """Reap the child and drain its output."""
        assert proc.stdout is not None
        for line in proc.stdout:
            with self._lock:
                if self._proc is not proc:
                    return  # superseded: this is no longer the output on show
                self._output.append(line.rstrip("\n"))
                self._output = self._output[-200:]
                self._output_seq += 1
        code = proc.wait()
        with self._lock:
            if self._proc is proc:
                self._returncode = code

    def _follow_dirs(self, proc: subprocess.Popen) -> None:
        """Notice each iteration's directory for as long as the child lives.

        Discovery cannot stop at the first one it finds. Under `--repeat` the
        agent moves to a new directory per iteration, and a watcher that
        latched onto the first would leave the live view tailing files that
        had stopped growing while the phone was still being driven.
        """
        while proc.poll() is None:
            self._sweep()
            time.sleep(0.25)
        deadline = time.monotonic() + POST_EXIT_SWEEP_S
        while time.monotonic() < deadline:
            self._sweep()
            time.sleep(0.25)

    def _sweep(self) -> None:
        """Fold whatever directories have appeared since the last look into
        the iteration list, oldest first."""
        try:
            found = sorted(
                ((d.stat().st_mtime, d) for d in self.artifacts_dir.iterdir()
                 if d.is_dir()),
                key=lambda pair: pair[0])
        except OSError:
            return
        with self._lock:
            known = {d.name for d in self._run_dirs} | self._dirs_before
            fresh = [d for _, d in found if d.name not in known]
            self._run_dirs.extend(fresh)
            for d in fresh:
                self._apply_mode(d)

    def stop(self, timeout_s: float = 10.0) -> bool:
        """SIGINT the child so the agent restores the phone, then escalate.

        The signal rather than a kill is the whole reason runs are subprocesses:
        the CLI catches it and puts back the keyboard, the animations, the
        rotation and the screen timeout it changed. A tour changes the same
        things, so it is stopped the same way.

        Returns as soon as the signal is away; the waiting and the escalation
        happen on a thread behind it. Blocking here blocked the caller for as
        long as the child took to wind down -- up to `timeout_s`, and a watch
        gives that a full three minutes because it writes the app's skill on the
        way out. The browser's stop button sends this request and says "stopping"
        when it answers, so a blocking stop said nothing at all for the whole
        minute it was working: the button stayed live, the status line still read
        "watching", and the only honest reading of that screen was that the click
        had done nothing.

        Asked twice, it signals once. The second SIGINT would land in the
        shutdown -- during the skill write-up, which is not inside the loop's
        `KeyboardInterrupt` handler -- and take down the very work the long
        timeout exists to protect.
        """
        with self._lock:
            proc = self._proc
            # Order matters: a child that has already exited is a 409 for the
            # caller, whether or not a stop was ever asked for.
            if proc is None or proc.poll() is not None:
                return False
            if self._stopping:
                return True  # already on its way out
            self._stopping = True
            self._stop_seq = self._output_seq
        try:
            if sys.platform == "win32":
                # CTRL_BREAK_EVENT targets the child's own process group
                # (created with CREATE_NEW_PROCESS_GROUP in _spawn).
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            # Two different things wear this exception. The child exiting between
            # the poll above and the signal is over already, and there is nothing
            # left to escalate to. A signal that simply did not land -- which is
            # what Windows reports here -- is the opposite: the child is still
            # running and still holding the phone, so it needs the kill below
            # more than a healthy one does. Returning early on both left it
            # running forever behind a UI stuck on "stopping", and `_stopping`
            # latched, so asking again did nothing either.
            if proc.poll() is not None:
                return True
        threading.Thread(target=self._reap, args=(proc, timeout_s),
                         daemon=True).start()
        return True

    def _reap(self, proc: subprocess.Popen, timeout_s: float) -> None:
        """Give the child `timeout_s` to shut itself down, then insist."""
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                pass  # it went on its own between the timeout and the kill

    def wait_for_run_dir(self, timeout_s: float = 60.0) -> Optional[Path]:
        """The current iteration's artifact directory, once it exists.

        Gives up early on a child that has already exited without writing one.
        A tour refused before it touched the phone -- a locked screen, an app
        that is not installed -- is over in a second, and the live view has
        nothing to tail: it should say so at once rather than a minute later.
        """
        deadline = time.monotonic() + timeout_s
        gone_by: Optional[float] = None
        while time.monotonic() < deadline:
            self._sweep()
            with self._lock:
                if self._run_dirs:
                    return self._run_dirs[-1]
                dead = self._proc is None or self._proc.poll() is not None
            now = time.monotonic()
            if dead:
                # Wait out the same grace `_follow_dirs` gives a late directory
                # before calling it: the child may have exited between the
                # agent's last write and the file appearing.
                gone_by = now + POST_EXIT_SWEEP_S if gone_by is None else gone_by
                if now >= gone_by:
                    return None
            time.sleep(0.25)
        return self.run_dir()


class RunManager(ChildProcess):
    """The one goal run the UI is allowed to have in flight."""

    def __init__(self, artifacts_dir: Path | str, *, config_path: str = ""):
        super().__init__(artifacts_dir)
        self.config_path = config_path
        self._goal = ""
        self._repeat = "1"

    def state(self) -> Dict[str, Any]:
        state = super().state()
        with self._lock:
            state["goal"] = self._goal
            state["repeat"] = self._repeat
        return state

    # -- lifecycle ----------------------------------------------------------

    def start(self, goal: str, *, max_steps: Optional[int] = None,
              budget_usd: Optional[float] = None, repeat: str = "1",
              dry_run: bool = False, allow_destructive: bool = False,
              no_learn: bool = False, serial: str = "",
              assert_shell: str = "", assert_equals: str = "",
              assert_text: str = "", resume: str = "") -> Dict[str, Any]:
        with self._lock:
            if self.running():
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
            # An assertion is per-run, not config, so this is the only way the UI
            # can offer one at all.
            if assert_shell:
                argv += ["--assert-shell", assert_shell]
            if assert_equals:
                argv += ["--assert-equals", assert_equals]
            if assert_text:
                argv += ["--assert-text", assert_text]
            if serial:
                argv += ["-d", serial]
            if self.config_path:
                argv += ["-c", self.config_path]

            self._repeat = str(repeat or "1")
            self._goal = goal
            # A repeat *after* a resume still gets fresh directories, and the
            # sweep goes on finding those.
            proc = self._spawn(argv, seed=resume)
            return {"pid": proc.pid, "argv": argv}


class WatchManager(ChildProcess):
    """The one `adbagent watch` the UI is allowed to have in flight.

    A watch is the same shape as a run to everything here -- a subprocess that
    drives the phone and writes a `runs/<id>/` directory per pass -- so it
    inherits the directory sweep and the SIGINT stop unchanged. What differs is
    that it does not end on its own: `returncode` stays None for days, and the
    sweep goes on finding a new directory every pass. Both are already true of a
    `--repeat inf` run, which is why `ChildProcess` needed nothing new.

    It is a separate class from `RunManager` rather than a flag on it because the
    two must be able to *refuse* each other: one phone, and a watch that has been
    quietly displaced by a goal run is a watch that is no longer watching.
    """

    NOUN = "watch"

    def __init__(self, artifacts_dir: Path | str, *, config_path: str = ""):
        super().__init__(artifacts_dir)
        self.config_path = config_path
        self._goal = ""
        self._draft = False
        self._policy = ""

    def state(self) -> Dict[str, Any]:
        state = super().state()
        with self._lock:
            state["goal"] = self._goal
            state["draft"] = self._draft
            state["policy"] = self._policy
            # Passes, not iterations: the word the CLI and the docs both use.
            state["passes"] = len(self._run_dirs)
        return state

    def stop(self, timeout_s: float = 180.0) -> bool:
        """SIGINT, and allow a long shutdown before insisting.

        Much longer than a run's ten seconds, because a watch does real work on
        the way out: it folds everything it learned about the app across every
        pass into that app's skill, which is one call on `llm.model_skill` and can
        take a minute on a long trace. Killing at ten seconds would abandon it
        every time, and the learning would look like it silently did not happen.

        That minute is also why nobody may wait on this: the caller is an HTTP
        request from a browser that has a stop button to update.
        """
        return super().stop(timeout_s=timeout_s)

    def start(self, goal: str, *, policy: str, draft: bool = False,
              no_learn: bool = False,
              interval_s: Optional[float] = None,
              sweep_s: Optional[float] = None,
              max_steps: Optional[int] = None,
              replies_per_hour: Optional[int] = None,
              replies_per_conversation: Optional[int] = None,
              cooldown_s: Optional[float] = None,
              usd_per_hour: Optional[float] = None,
              ledger: str = "", serial: str = "") -> Dict[str, Any]:
        with self._lock:
            if self.running():
                raise RuntimeError("a watch is already running")

            argv = [sys.executable, "-m", "adbagent", "watch", goal,
                    "--policy", policy]
            if draft:
                argv.append("--draft")
            if no_learn:
                argv.append("--no-learn")
            if interval_s is not None:
                argv += ["--interval", str(interval_s)]
            if sweep_s is not None:
                argv += ["--sweep", str(sweep_s)]
            if max_steps is not None:
                argv += ["--steps-per-pass", str(max_steps)]
            if replies_per_hour is not None:
                argv += ["--replies-per-hour", str(replies_per_hour)]
            if replies_per_conversation is not None:
                argv += ["--replies-per-conversation",
                         str(replies_per_conversation)]
            if cooldown_s is not None:
                argv += ["--cooldown", str(cooldown_s)]
            if usd_per_hour is not None:
                argv += ["--usd-per-hour", str(usd_per_hour)]
            if ledger:
                argv += ["--ledger", ledger]
            if serial:
                argv += ["-d", serial]
            if self.config_path:
                argv += ["-c", self.config_path]

            self._goal = goal
            self._draft = draft
            self._policy = policy
            proc = self._spawn(argv)
            return {"pid": proc.pid, "argv": argv}


class Job(ChildProcess):
    """A side task the UI started -- `skills generate` -- watched like a run.

    Because it is one: the tour is an agent driving the phone against a
    look-around goal, writing the same events, thinking stream and screenshots
    into its own `runs/<id>/`. What is different is the ending -- the skill is
    written up after the loop stops, by the same process, with nothing left
    tailing the run directory -- which is why the raw output is kept too.
    """

    def __init__(self, job_id: int, argv: List[str], artifacts_dir: Path | str):
        super().__init__(artifacts_dir)
        self.id = job_id
        self.argv = list(argv)
        self._spawn(self.argv)

    def state(self) -> Dict[str, Any]:
        state = super().state()
        state["id"] = self.id
        state["argv"] = list(self.argv)
        return state


class JobManager:
    """Fire-and-poll tracking for side tasks that are not goal runs: no SIGINT
    choreography, just status, the tail of the output, and the run being
    written underneath."""

    def __init__(self, artifacts_dir: Path | str = "runs"):
        self.artifacts_dir = Path(artifacts_dir)
        self._lock = threading.Lock()
        self._jobs: Dict[int, Job] = {}
        self._next_id = 1

    def start(self, argv: List[str]) -> int:
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
        job = Job(job_id, argv, self.artifacts_dir)
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def job(self, job_id: int) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def get(self, job_id: int) -> Optional[Dict[str, Any]]:
        job = self.job(job_id)
        return job.state() if job is not None else None

    def active(self) -> Optional[Job]:
        """The job still holding the phone, if one is."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.running():
                return job
        return None


def sse(payload: Dict[str, Any], event: str = "message") -> str:
    """One Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
