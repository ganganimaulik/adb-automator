"""FastAPI app for the adbagent web UI.

The server is a thin shell over what already exists: runs are spawned as CLI
subprocesses (`runner.RunManager`), history is parsed off disk
(`runparse`), config and skills are read and written in the same files the
CLI uses. Nothing here knows how to drive a phone.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Generator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .. import policies as policymod
from ..config import Config, _set_path, find_config_file, load_config
from ..policies import same_file as _same_file
from ..runlog import SCREENS_NAME, SHOT_RE, STREAM_NAME
from ..skills import Skill, SkillRegistry
from . import runparse
from .reload import LiveReload
from .runner import ChildProcess, JobManager, RunManager, WatchManager, sse

STATIC_DIR = Path(__file__).parent / "static"

_MEDIA_TYPES = {".html": "text/html", ".js": "text/javascript",
                ".css": "text/css", ".png": "image/png"}

#: How long an `adb devices` answer is reused. The status line polls every few
#: seconds from every open tab, and what is plugged in does not change between
#: two of those; long enough to collapse the poll, short enough that unplugging
#: a phone shows up while your hand is still on the cable.
ATTACHED_TTL_S = 2.5

#: Ceiling on one `exec-out screencap`. A healthy phone answers in well under a
#: second over USB and a couple over wireless adb; past this the link is the
#: problem and the panel should say so rather than hang.
FRAME_TIMEOUT_S = 20.0

#: How long the server waits, on its way out, for the agents it started to put
#: the phone back. Deliberately far short of the three minutes a stop from the
#: browser allows a watch: there is somebody at a prompt waiting for it.
SHUTDOWN_GRACE_S = 15.0

#: What a policy created from the browser starts as. Not empty: an empty policy
#: is refused on save, so a blank editor is a dead end -- and the two things a
#: policy has to answer are easier to fill in than to think of.
NEW_POLICY_TEXT = """\
## What to reply to

- (which conversations, and which to leave alone)

## What to say

- (one short sentence; what never to promise)
"""


class RunRequest(BaseModel):
    goal: str = ""
    max_steps: Optional[int] = None
    budget_usd: Optional[float] = None
    repeat: str = "1"
    dry_run: bool = False
    allow_destructive: bool = False
    no_learn: bool = False
    serial: str = ""
    #: A machine-checkable definition of done. Free, instant, and it cannot be
    #: argued with -- and it removes the completion-judge call from the run.
    assert_shell: str = ""
    assert_equals: str = ""
    assert_text: str = ""
    #: A run id to continue from its checkpoint. When set, `goal` is ignored
    #: -- the checkpoint's own goal is the one being pursued.
    resume: str = ""


class AnswerRequest(BaseModel):
    #: What to tell a run that stopped on `ask_user`. Written into its
    #: checkpoint, where the resume reads it -- see `checkpoint.ANSWER`.
    text: str = ""


class WatchRequest(BaseModel):
    goal: str = ""
    #: Path to the reply policy. Required -- there is no default policy, for the
    #: same reason the CLI has none.
    policy: str = ""
    draft: bool = False
    no_learn: bool = False
    interval_s: Optional[float] = None
    sweep_s: Optional[float] = None
    max_steps: Optional[int] = None
    replies_per_hour: Optional[int] = None
    replies_per_conversation: Optional[int] = None
    cooldown_s: Optional[float] = None
    usd_per_hour: Optional[float] = None
    ledger: str = ""
    serial: str = ""


class PolicyUpdate(BaseModel):
    path: str = ""
    text: str = ""
    #: The goal this policy is written for, saved into its front matter. The
    #: pairing is the point: these instructions are only correct under that goal,
    #: so the editor saves both or neither.
    goal: str = ""
    title: str = ""


class PolicyCreate(BaseModel):
    name: str = ""
    goal: str = ""
    text: str = ""


class ConfigUpdate(BaseModel):
    sections: Dict[str, Dict[str, Any]]


class UseDeviceRequest(BaseModel):
    serial: str = ""


class GenerateRequest(BaseModel):
    name: str = ""
    tasks: str = ""
    max_steps: Optional[int] = None
    budget_usd: Optional[float] = None
    serial: str = ""


def _static(name: str, no_store: bool = False) -> Response:
    path = STATIC_DIR / name
    if not path.is_file() or path.parent != STATIC_DIR:
        raise HTTPException(status_code=404, detail="not found")
    media = _MEDIA_TYPES.get(path.suffix, "application/octet-stream")
    # Read off disk per request, so an edit is live the moment the page asks
    # again -- but only if the browser does ask. Nothing here carries a
    # validator, so a heuristically cached app.js would survive the reload that
    # was triggered by editing it. Under live reload, say not to keep it.
    headers = {"Cache-Control": "no-store"} if no_store else None
    return Response(content=path.read_bytes(), media_type=media, headers=headers)


def attached_serials() -> List[str]:
    """Serials adb can actually see, or [] when it cannot be asked.

    Separate from `/api/devices`, which also reads a model name and an Android
    version off every one of them: this is the cheap question -- is there a phone
    on the end of the cable -- and it is asked on every status poll, because a
    serial in config is not a device and only one of the two decides whether a
    run can start.
    """
    from .. import device as devmod
    try:
        return [d.serial for d in devmod.list_devices()]
    except Exception:  # noqa: BLE001 - adb absent or its server down
        return []


def screencap(serial: str = "", max_long_edge: int = 720,
              quality: int = 72) -> bytes:
    """A JPEG of the screen right now, without opening a device session.

    This is the one screenshot path that is safe to take *while an agent is
    driving the phone*, and that is the whole reason it exists. `Device.open()`
    zeroes the animation scales, locks rotation, selects its own IME and takes a
    stay-awake lock -- so `Device(...).screenshot()` cannot be used mid-run
    without changing the phone underneath the run. `exec-out screencap` is a
    plain read: no uiautomator server, no settings touched, nothing to restore.

    Downscaled here rather than shipped raw: a 1080x2400 `screencap -p` is 1-3 MB
    of PNG, and this is polled every couple of seconds.
    """
    import io

    from PIL import Image

    from ..device import adb_path

    argv = [adb_path()]
    if serial:
        argv += ["-s", serial]
    argv += ["exec-out", "screencap", "-p"]
    proc = subprocess.run(argv, capture_output=True, timeout=FRAME_TIMEOUT_S)
    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(detail or "screencap returned nothing")
    image = Image.open(io.BytesIO(proc.stdout))
    w, h = image.size
    factor = min(1.0, max_long_edge / max(w, h)) if max(w, h) else 1.0
    if factor < 1.0:
        image = image.resize((max(1, round(w * factor)), max(1, round(h * factor))),
                             Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def create_app(*, artifacts_dir: str = "runs", skills_dir: str = "",
               config_path: str = "",
               live_reload: Optional[LiveReload] = None) -> FastAPI:
    runs_dir = Path(artifacts_dir)
    manager = RunManager(runs_dir, config_path=config_path)
    # The same artifacts directory: a `skills generate` tour writes a run there
    # like any other, and the browser tails it through the same live view. A
    # watch writes one such directory per pass, and is tailed the same way.
    watcher = WatchManager(runs_dir, config_path=config_path)
    jobs = JobManager(runs_dir)

    def phone_busy() -> str:
        """Why the phone cannot be driven right now, or "".

        One phone, one agent. A tour and a goal run reading each other's taps
        as their own is the failure this prevents; that both are started from
        different corners of the page is what makes it easy to ask for.

        A child that has been asked to stop still holds the phone -- it is
        restoring the keyboard, the animations and the rotation, and writing up
        what it learned -- and the answer says that rather than "already in
        progress", which reads as though the stop had not been heard.
        """
        run = manager.state()
        if run["stopping"]:
            return "the run is stopping; the phone is free once it has"
        if run["running"]:
            return "a run is already in progress"
        watch = watcher.state()
        if watch["stopping"]:
            return "the watch is stopping; it is writing up what it learned"
        if watch["running"]:
            # Said with the remedy in it: a watch does not end on its own, so
            # "busy, try later" would be advice to wait forever.
            return ("a watch is running; stop it from the Watch tab before "
                    "driving the phone by hand")
        job = jobs.active()
        if job is not None:
            return f"a skill is being generated (job {job.id}); the phone is busy"
        return ""

    def load_cfg() -> Config:
        return load_config(config_path or None).config

    def registry() -> SkillRegistry:
        return SkillRegistry(skills_dir or load_cfg().skills.skills_dir)

    #: Set when the server is going down, so the live streams end themselves
    #: rather than being cancelled mid-frame -- which is what fills the console
    #: with `CancelledError` out of the middle of a `StreamingResponse`.
    #:
    #: Published on `app.state` because the only place it can usefully be set is
    #: the moment the signal lands: uvicorn waits for in-flight requests *before*
    #: it sends the lifespan shutdown, and a watch's stream never ends on its own,
    #: so a flag set from the lifespan is one the stream can never see. `cmd_ui`
    #: sets it from the signal handler; the lifespan below sets it too, for
    #: shutdowns that never went through a signal at all.
    shutting_down = threading.Event()

    def live_children() -> List[ChildProcess]:
        """Every child still driving the phone."""
        kids: List[ChildProcess] = [c for c in (manager, watcher) if c.running()]
        job = jobs.active()
        if job is not None:
            kids.append(job)
        return kids

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Take the children down with the server.

        Ctrl+C in the console the server was started from does not reach them on
        Windows: each child is spawned into its own process group -- it has to
        be, or signalling one would take the server down too -- and a new group
        has the console's Ctrl+C disabled. So the server would exit and leave a
        watch running: still driving the phone, still replying to people, with
        the only thing that could stop it now gone.

        They are asked, not killed, and then waited for -- briefly. A watch does
        real work on the way out and deserves the chance to finish it, but an
        operator who pressed Ctrl+C is waiting at a prompt, so the wait is
        seconds rather than the minute a stop from the browser allows.
        """
        yield
        shutting_down.set()
        kids = live_children()
        if not kids:
            return
        print(f"  stopping {len(kids)} agent(s) still driving the phone…")
        for kid in kids:
            kid.stop()
        deadline = time.monotonic() + SHUTDOWN_GRACE_S
        while time.monotonic() < deadline and any(k.running() for k in kids):
            await asyncio.sleep(0.25)
        if any(k.running() for k in kids):
            print("  one is still finishing -- the phone is being put back, and "
                  "a watch writes up what it learned. It will exit on its own.")

    app = FastAPI(title="adbagent ui", lifespan=lifespan)
    app.state.shutting_down = shutting_down
    #: Published for whoever started the server: live reload has to know when a
    #: restart would orphan an agent, and this is the same answer the API gives
    #: anyone else who wants the phone.
    app.state.phone_busy = phone_busy
    app.state.live_reload = live_reload

    # -- pages ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        return _static("index.html", no_store=live_reload is not None)

    @app.get("/static/{name}")
    def static_file(name: str) -> Response:
        return _static(name, no_store=live_reload is not None)

    # -- live reload -----------------------------------------------------

    @app.get("/api/dev/reload")
    def dev_reload() -> StreamingResponse:
        """What changed on disk, as it changes. Absent unless reload is on."""
        if live_reload is None:
            raise HTTPException(status_code=404, detail="live reload is off")
        return StreamingResponse(_reload_stream(live_reload, shutting_down),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # -- status & devices ------------------------------------------------

    #: `attached_serials()` behind a short TTL. Every open tab polls the status
    #: line, and none of them needs its own `adb devices`.
    attached: Dict[str, Any] = {"at": 0.0, "serials": []}

    def attached_now() -> List[str]:
        if time.monotonic() - attached["at"] > ATTACHED_TTL_S:
            attached.update(at=time.monotonic(), serials=attached_serials())
        return list(attached["serials"])

    @app.get("/api/status")
    def status() -> Dict[str, Any]:
        loaded = load_config(config_path or None)
        cfg = loaded.config
        # A generation in flight is reported alongside the run for the same
        # reason: a page reloaded mid-tour has to know there is something to
        # reattach to, and which job to ask for it.
        job = jobs.active()
        # What is configured and what is plugged in are two different facts, and
        # the header used to report the first as though it were the second: a
        # serial left in config.json read as "device 192.168.1.23:41207" with
        # nothing on the other end. Both travel, so the page can say which.
        serials = attached_now()
        return {
            "config_path": str(loaded.path) if loaded.path else "",
            "model": cfg.llm.model,
            "api_key_present": bool(cfg.api_key()),
            "device_serial": cfg.device.serial,
            "devices_attached": serials,
            "device_attached": (cfg.device.serial in serials if cfg.device.serial
                                else len(serials) == 1),
            "artifacts_dir": str(runs_dir),
            # Whether there is a reload stream worth opening. Asked here rather
            # than by opening it and seeing: a page that guessed wrong would
            # reconnect to a 404 every few seconds for as long as it stayed up.
            "live_reload": live_reload is not None,
            "run": manager.state(),
            # Reported alongside the run for the same reason a tour is: a page
            # reloaded hours later has to know there is a watch to reattach to.
            "watch": watcher.state(),
            "job": None if job is None else {
                "id": job.id,
                "running": True,
                "started_at": job.state()["started_at"],
                "run_id": job.state()["run_id"],
            },
        }

    @app.get("/api/devices")
    def devices() -> Dict[str, Any]:
        from .. import device as devmod
        try:
            found = [{"serial": d.serial,
                      "model": getattr(d.prop, "model", ""),
                      "android": d.getprop("ro.build.version.release") or ""}
                     for d in devmod.list_devices()]
        except Exception as exc:  # noqa: BLE001 - adb absent or server down
            return {"devices": [], "candidates": [], "error": str(exc)}
        try:
            candidates = devmod.mdns_candidates()
        except Exception:  # noqa: BLE001
            candidates = []
        return {"devices": found, "candidates": candidates, "error": ""}

    @app.post("/api/device/use")
    def use_device(req: UseDeviceRequest) -> Dict[str, Any]:
        """Point `device.serial` at a phone that is actually attached.

        The status line's job is to say whether a run can start, and it did —
        `configured, not attached` — and then left you to fix it four
        navigations away in the config form. A serial that adb is reporting
        right now is the one piece of config the page can offer to write for
        you, because there is nothing to get wrong about it: it is on the
        list.
        """
        serial = (req.serial or "").strip()
        if not serial:
            raise HTTPException(status_code=400, detail="no serial given")
        # Only ever a serial adb is reporting. Anything else is the config
        # form's job, where a typo is visible and reversible.
        if serial not in attached_now():
            raise HTTPException(
                status_code=409,
                detail=f"{serial} is not attached")
        return put_config(ConfigUpdate(sections={"device": {"serial": serial}}))

    @app.get("/api/device/frame")
    def device_frame(serial: str = "", max_long_edge: int = 720) -> Response:
        """The screen as it is now — polled while a run is happening.

        Deliberately not behind `phone_busy()`, unlike every other device call
        here. The reason those are refused is that opening a `Device` session
        resets the animation scales and the rotation on the way in and puts them
        back on the way out, which changes the phone under whatever is driving
        it. This path opens no session: it is one `exec-out screencap`, a read,
        and it is the only honest way to watch a phone that something else is
        holding.

        A serial that is not attached is a 404 rather than a 502 — it is a normal
        state, and the panel says so instead of showing the last frame it had.
        """
        wanted = serial or load_cfg().device.serial
        serials = attached_now()
        if not serials:
            raise HTTPException(
                status_code=404,
                detail="nothing attached: adb sees no device")
        if wanted and wanted not in serials:
            raise HTTPException(
                status_code=404,
                detail=f"{wanted} is configured but adb does not see it; "
                       f"attached: {', '.join(serials)}")
        if not wanted and len(serials) > 1:
            raise HTTPException(
                status_code=409,
                detail=f"{len(serials)} devices attached and no serial chosen: "
                       f"{', '.join(serials)}")
        try:
            jpeg = screencap(wanted or serials[0], max_long_edge=max_long_edge)
        except Exception as exc:  # noqa: BLE001 - a dead link, a slow phone
            raise HTTPException(status_code=502, detail=str(exc))
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/devices/screenshot")
    def screenshot(serial: str = "") -> Response:
        # `phone_busy` rather than the run alone: opening a `Device` session
        # zeroes the animation scales and forces portrait on entry and restores
        # them on exit, so doing it under a running agent -- a run, a watch or a
        # tour -- changes the phone beneath it.
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        from ..device import Device
        try:
            with Device(load_cfg(), serial) as dev:
                jpeg = dev.screenshot()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc))
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/apps")
    def apps(search: str = "", third_party: bool = False,
             serial: str = "") -> Dict[str, Any]:
        """Installed packages -- `adbagent apps`, as data.

        Worth having in the browser because it answers the question that comes up
        while writing a goal: what is this app actually called? A goal that names
        an app the phone does not have fails on step one.
        """
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        from ..device import Device
        try:
            with Device(load_cfg(), serial) as dev:
                pkgs = dev.list_apps(query=search, third_party_only=third_party)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc))
        return {"apps": pkgs, "count": len(pkgs), "search": search,
                "third_party": third_party}

    @app.get("/api/dump")
    def dump(serial: str = "", raw: bool = False) -> Dict[str, Any]:
        """Exactly what the model would be shown for the current screen.

        The single most useful thing for working out why a run did something
        strange: the tree the model saw, with the same pruning and the same
        indices it was told to aim at.
        """
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        from ..device import Device
        from ..screen import render
        try:
            with Device(load_cfg(), serial) as dev:
                screen = dev.observe(settle=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc))
        return {
            "package": screen.package,
            "activity": screen.activity,
            "width": screen.width,
            "height": screen.height,
            "skeleton_id": screen.skeleton_id,
            "elements": len(screen.elements),
            "nodes": len(screen.nodes),
            "keyboard_open": screen.keyboard_open,
            "rendered": render(screen),
            "xml": screen.xml if raw else "",
        }

    @app.get("/api/doctor")
    def doctor() -> Dict[str, Any]:
        """`adbagent doctor`, run as itself.

        Shelled out rather than reimplemented: the checks drift the moment there
        are two copies, and this is the one endpoint whose whole value is being
        the same answer the CLI gives. `Out` prints plain when it is not a
        terminal, so the text arrives without escape codes.
        """
        argv = [sys.executable, "-m", "adbagent", "doctor"]
        if config_path:
            argv += ["-c", config_path]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=120, errors="replace")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {"text": (proc.stdout or "") + (proc.stderr or ""),
                "ok": proc.returncode == 0, "returncode": proc.returncode}

    # -- config ----------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        loaded = load_config(config_path or None)
        return {"config": loaded.config.to_dict(),
                # What the settings would be with nothing configured at all, so
                # the form can mark the handful that were actually changed.
                # Sixty-two fields showing their defaults look identical to
                # sixty-two fields somebody set on purpose.
                "defaults": Config().to_dict(),
                "path": str(loaded.path) if loaded.path else "",
                "warnings": loaded.warnings,
                "api_key_present": bool(loaded.config.api_key())}

    @app.put("/api/config")
    def put_config(update: ConfigUpdate) -> Dict[str, Any]:
        # Validate every key against the dataclass schema first, so a typo
        # fails the whole save rather than landing as dead config.
        cfg = Config()
        errors = []
        for section, values in update.sections.items():
            for key, value in values.items():
                try:
                    _set_path(cfg, f"{section}.{key}", value)
                except (KeyError, ValueError, TypeError) as exc:
                    errors.append(str(exc))
        if errors:
            raise HTTPException(status_code=400, detail=errors)

        path = find_config_file(config_path or None)
        if path is None:
            path = Path.cwd() / "config.json"
        raw: Dict[str, Any] = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    raw = data
            except (OSError, json.JSONDecodeError):
                pass
        for section, values in update.sections.items():
            # Never persist the redaction marker as a real key.
            if section == "llm" and values.get("api_key") == "***":
                values = {k: v for k, v in values.items() if k != "api_key"}
            raw.setdefault(section, {}).update(values)
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        return {"saved": True, "path": str(path)}

    # -- models ----------------------------------------------------------

    #: The catalogue costs several paged HTTP calls, so hold the answer rather
    #: than page through it again for every visit to the config tab. Keyed on the
    #: provider alone -- two keys against one provider see the same serverless
    #: list -- and aged out, so a model released mid-session is at most this
    #: stale. `?refresh=1` fetches now.
    catalogue: Dict[str, Any] = {"provider": "", "at": 0.0, "models": []}
    catalogue_ttl_s = 600.0

    @app.get("/api/models")
    def models(refresh: bool = False) -> Dict[str, Any]:
        """What `llm.model` and its siblings can be set to, for the dropdowns.

        Never fails: no key yet, or a catalogue that cannot be reached, is a
        normal state for this endpoint -- the UI answers it by offering a text
        box instead of a list -- so the trouble travels in `error` rather than
        as a status code.
        """
        from ..llm import PROVIDERS, list_models, qualify

        cfg = load_cfg()
        answer: Dict[str, Any] = {"provider": cfg.llm.provider, "models": [],
                                  "cached": False, "error": ""}
        provider = PROVIDERS.get(cfg.llm.provider)
        if provider is None:
            answer["error"] = (f"unknown provider {cfg.llm.provider!r}; "
                               f"known: {', '.join(sorted(PROVIDERS))}")
            return answer
        if not cfg.api_key():
            answer["error"] = ("no API key: set llm.api_key below, or "
                               f"${cfg.llm.api_key_env} in the environment")
            return answer

        cached = (not refresh and catalogue["provider"] == cfg.llm.provider
                  and time.monotonic() - catalogue["at"] < catalogue_ttl_s)
        if not cached:
            try:
                found = list_models(provider, cfg.api_key())
            except Exception as exc:  # noqa: BLE001 - offline, 401, throttled
                answer["error"] = str(exc)
                return answer
            catalogue.update(
                provider=cfg.llm.provider, at=time.monotonic(),
                models=[{
                    "id": m.id,
                    # What the config file should hold: the id the wire wants,
                    # which on fireworks is the fully-qualified one.
                    "value": qualify(provider, m.id),
                    "display_name": m.display_name,
                    "context_length": m.context_length,
                    "vision": m.vision, "tools": m.tools,
                    "deprecated": m.deprecated,
                } for m in found])
        answer["models"] = catalogue["models"]
        answer["cached"] = cached
        return answer

    # -- runs ------------------------------------------------------------

    # Declared before /api/runs/{run_id} so the literal path wins.
    @app.get("/api/runs/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(_event_stream(manager, shutting_down),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs")
    def list_runs() -> Dict[str, Any]:
        return {"runs": runparse.list_runs(runs_dir), "active": manager.state()}

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> Dict[str, Any]:
        from .. import checkpoint as ckpt

        goal = req.goal.strip()
        if req.resume:
            path = runparse.find_run(runs_dir, req.resume)
            if path is None:
                raise HTTPException(status_code=404, detail="run not found")
            data = ckpt.load(path)
            if data is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"run {req.resume} has no checkpoint to resume from")
            goal = data.get("goal", "")
        if not goal:
            raise HTTPException(status_code=400, detail="goal is required")
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        try:
            return manager.start(goal, max_steps=req.max_steps,
                                 budget_usd=req.budget_usd, repeat=req.repeat,
                                 dry_run=req.dry_run,
                                 allow_destructive=req.allow_destructive,
                                 no_learn=req.no_learn, serial=req.serial,
                                 assert_shell=req.assert_shell,
                                 assert_equals=req.assert_equals,
                                 assert_text=req.assert_text,
                                 resume=req.resume)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/runs/stop")
    def stop_run() -> Dict[str, Any]:
        if not manager.stop():
            raise HTTPException(status_code=409, detail="no run in progress")
        return {"stopping": True}

    @app.post("/api/runs/{run_id}/answer")
    def answer_run(run_id: str, req: AnswerRequest) -> Dict[str, Any]:
        """Tell a run that stopped on `ask_user` what it wanted to know.

        Not a resume: this only writes the answer where the resume will find it,
        and the browser follows with the ordinary `POST /api/runs {resume}`. Two
        calls rather than one because they fail for different reasons and want
        different words -- there is nothing to answer, versus the phone is busy
        -- and because an answer typed against a phone that is in use is still
        worth keeping until it is free.

        The text is never echoed back. `ask_user` is what the agent does instead
        of typing a password or a one-time code, so what comes back through here
        is usually the credential; it goes to the one file that has to have it
        and no further.
        """
        from .. import checkpoint as ckpt

        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="an answer is required")
        path = runparse.find_run(runs_dir, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="run not found")
        try:
            written = ckpt.set_answer(path, text)
        except OSError as exc:
            raise HTTPException(status_code=500,
                                detail=f"could not save the answer: {exc}")
        if not written:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no checkpoint to answer into")
        return {"answered": True, "run_id": run_id}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> Dict[str, Any]:
        path = runparse.find_run(runs_dir, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="run not found")
        return runparse.run_detail(path)

    @app.get("/api/runs/{run_id}/shot/{name}")
    def run_shot(run_id: str, name: str) -> Response:
        """One frame the run showed a model, by the name its `llm_start` carries.

        The name is checked against the pattern the recorder writes rather than
        sanitised, so nothing in this directory that is not one of our JPEGs can
        be served -- `run.log` and the prompt dumps included.
        """
        path = runparse.find_run(runs_dir, run_id)
        if path is None or not SHOT_RE.fullmatch(name):
            raise HTTPException(status_code=404, detail="not found")
        shot = path / name
        if not shot.is_file():
            raise HTTPException(status_code=404, detail="not found")
        # Content-addressed: the bytes behind one of these names never change.
        return Response(content=shot.read_bytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=31536000, "
                                                  "immutable"})

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, tail: int = 40000) -> Dict[str, Any]:
        path = runparse.find_run(runs_dir, run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="run not found")
        log_file = path / "run.log"
        if not log_file.is_file():
            return {"text": ""}
        data = log_file.read_bytes()[-max(0, tail):]
        return {"text": data.decode("utf-8", errors="replace")}

    # -- watch -----------------------------------------------------------

    def policy_store() -> policymod.PolicyStore:
        """The policies on disk: the directory of them plus the configured one."""
        return policymod.store_for(load_cfg())

    def _policy_path(explicit: str = "") -> str:
        """The policy file `explicit` names, or the configured one, or "".

        Returns a string rather than a Path on purpose: `Path("")` is
        `PosixPath('.')`, whose string is "." -- truthy, so a "no policy set"
        guard written against the Path silently passes and the write lands on a
        directory.
        """
        return policy_store().resolve(explicit)

    def _writable(path: str) -> None:
        """Refuse a write outside the policies directory.

        The browser sends a path now -- it has to, since it picks between
        several -- and the endpoint used to write whatever arrived to wherever it
        pointed. A page reachable from another tab is not something to leave
        that open to, so a save has to land either in `watch.policies_dir` or on
        the policy `watch.policy` already names.
        """
        if not policy_store().owns(path):
            raise HTTPException(
                status_code=400,
                detail=f"{path} is outside the policies directory; save policies "
                       f"there, or point watch.policy at this file in Config")

    # Declared before any /api/watch/{...} route so the literal paths win, the
    # same ordering /api/runs/stream needs.
    @app.get("/api/watch/stream")
    def watch_stream() -> StreamingResponse:
        return StreamingResponse(_event_stream(watcher, shutting_down),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/api/watch")
    def watch_state() -> Dict[str, Any]:
        cfg = load_cfg()
        return {"active": watcher.state(),
                "defaults": dataclasses.asdict(cfg.watch),
                "policy_path": _policy_path(),
                "policies_dir": str(Path(cfg.watch.policies_dir).expanduser())
                                if cfg.watch.policies_dir else "",
                "ledger_path": str(Path(cfg.watch.ledger).expanduser())}

    @app.get("/api/watch/policies")
    def list_policies() -> Dict[str, Any]:
        """Every policy there is, with the goal each was written for.

        The list the picker is built from. The goal travels with the row rather
        than being fetched per selection, so choosing one can fill in the goal
        box without a round trip -- and so the picker can show what each policy
        is *for*, which is the part a filename does not say.
        """
        cfg = load_cfg()
        store = policy_store()
        current = _policy_path()
        return {
            "dir": str(Path(cfg.watch.policies_dir).expanduser())
                   if cfg.watch.policies_dir else "",
            "current": current,
            "policies": [{**p.to_dict(),
                          "current": _same_file(str(p.path), current)}
                         for p in store.list()],
        }

    @app.post("/api/watch/policies")
    def create_policy(req: PolicyCreate) -> Dict[str, Any]:
        """A new, empty-ish policy under `watch.policies_dir`.

        Refuses to overwrite: "new" that quietly replaced an existing policy
        would be a way to lose one, and the picker already offers every policy
        there is to edit instead.
        """
        store = policy_store()
        if store.dir is None:
            raise HTTPException(
                status_code=400,
                detail="no policies directory configured: set "
                       "watch.policies_dir in Config")
        try:
            path = store.path_for(req.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if path.exists():
            raise HTTPException(status_code=409,
                                detail=f"{path} already exists; open it from the "
                                       f"list instead")
        policy = store.write(str(path), req.text or NEW_POLICY_TEXT,
                             goal=req.goal)
        return {"created": True, **policy.to_dict()}

    @app.get("/api/watch/policy")
    def get_policy(path: str = "") -> Dict[str, Any]:
        """One policy, split into its goal and its instructions, for the editor.

        A missing file is not an error here -- it is the normal state before one
        has been written -- so it comes back as empty text with `exists: false`
        rather than a 404 the editor would have to special-case.

        `text` is the instructions alone, front matter stripped: the box holds
        what goes into the prompt, and the goal is a field of its own. A file
        with no front matter is unchanged by the round trip.
        """
        found = _policy_path(path)
        if not found:
            return {"path": "", "text": "", "exists": False, "goal": "",
                    "title": "", "name": "", "label": ""}
        policy = policymod.read(found)
        return {"path": found, "text": policy.body, "exists": policy.exists,
                "goal": policy.goal, "title": policy.title,
                "name": policy.name, "label": policy.label}

    @app.put("/api/watch/policy")
    def put_policy(update: PolicyUpdate) -> Dict[str, Any]:
        found = _policy_path(update.path)
        if not found:
            raise HTTPException(status_code=400,
                                detail="no policy path given, and none in config")
        if watcher.state()["running"]:
            # The child read the file once at startup, so a save now would take
            # effect at no predictable moment. Refusing says which it is.
            raise HTTPException(
                status_code=409,
                detail="stop the watch before editing its policy -- a running "
                       "watch has already read the file")
        if not update.text.strip():
            raise HTTPException(status_code=400,
                                detail="an empty policy would let the model "
                                       "decide for itself what to say")
        _writable(found)
        # `title or None` -- the editor has no title field, so an absent one is
        # not a request to drop the one somebody wrote into the file by hand.
        policy = policy_store().write(found, update.text, goal=update.goal,
                                      title=update.title or None)
        return {"saved": True, "path": found, "goal": policy.goal}

    @app.get("/api/watch/ledger")
    def get_ledger(limit: int = 100) -> Dict[str, Any]:
        """What has actually been sent, newest first.

        The watch's most important artifact by a distance: it is the only place
        that answers "what did it say to whom", and it is the record the
        never-double-reply guarantee is built on.
        """
        from ..ledger import ReplyLedger
        cfg = load_cfg()
        path = Path(cfg.watch.ledger).expanduser()
        led = ReplyLedger(path)
        return {
            "path": str(path),
            "exists": path.is_file(),
            "total": len(led),
            "threads": [{
                "thread_key": st.thread_key,
                "preview": st.preview,
                "last_attempt_at": st.last_attempt_at,
                "reply_count": st.reply_count,
                "confirmed": st.confirmed,
            } for st in led.recent(limit)],
        }

    @app.post("/api/watch")
    def start_watch(req: WatchRequest) -> Dict[str, Any]:
        cfg = load_cfg()
        policy = _policy_path(req.policy)
        if not policy:
            raise HTTPException(
                status_code=400,
                detail="a watch needs a policy file: the instructions that "
                       "decide what gets replied to and what it says")
        if not Path(policy).is_file():
            raise HTTPException(status_code=400,
                                detail=f"no policy file at {policy}")
        # Resolved before the goal is checked, because a policy carries the goal
        # it was written for and that is the one to run it under. The page fills
        # the box in on selection, so this is the fallback for a box that was
        # cleared -- and for anything driving the API directly.
        goal = req.goal.strip() or policymod.read(policy).goal
        if not goal:
            raise HTTPException(status_code=400, detail="goal is required")
        if not cfg.llm.model:
            raise HTTPException(status_code=400, detail="no model configured")
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        try:
            return watcher.start(
                goal, policy=policy, draft=req.draft, no_learn=req.no_learn,
                interval_s=req.interval_s, sweep_s=req.sweep_s,
                max_steps=req.max_steps,
                replies_per_hour=req.replies_per_hour,
                replies_per_conversation=req.replies_per_conversation,
                cooldown_s=req.cooldown_s, usd_per_hour=req.usd_per_hour,
                ledger=req.ledger, serial=req.serial)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/watch/stop")
    def stop_watch() -> Dict[str, Any]:
        if not watcher.stop():
            raise HTTPException(status_code=409, detail="no watch is running")
        return {"stopping": True}

    # -- skills ----------------------------------------------------------

    @app.get("/api/skills")
    def list_skills() -> Dict[str, Any]:
        reg = registry()
        return {"skills": [{
            "name": s.name, "packages": s.packages, "aliases": s.aliases,
            "description": s.description,
            "workflows": len(s.workflows), "nuances": len(s.nuances),
            "recommendations": len(s.recommendations),
        } for s in reg.list_skills()]}

    @app.get("/api/skills/{name}")
    def get_skill(name: str) -> Dict[str, Any]:
        skill = registry().find_by_name_or_alias(name)
        if skill is None:
            raise HTTPException(status_code=404, detail="skill not found")
        return skill.to_dict()

    @app.put("/api/skills/{name}")
    def put_skill(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            skill = Skill.from_dict(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid skill: {exc}")
        if not skill.name.strip():
            raise HTTPException(status_code=400, detail="skill needs a name")
        path = registry().save_skill(skill)
        return {"saved": True, "path": str(path)}

    @app.post("/api/skills/generate")
    def generate_skill(req: GenerateRequest) -> Dict[str, Any]:
        busy = phone_busy()
        if busy:
            raise HTTPException(status_code=409, detail=busy)
        argv = [sys.executable, "-m", "adbagent", "skills", "generate"]
        if req.name.strip():
            argv.append(req.name.strip())
        if req.tasks.strip():
            argv += ["--tasks", req.tasks.strip()]
        if req.max_steps:
            argv += ["--max-steps", str(req.max_steps)]
        if req.budget_usd is not None:
            argv += ["--budget-usd", str(req.budget_usd)]
        if req.serial:
            argv += ["-d", req.serial]
        if config_path:
            argv += ["-c", config_path]
        return {"job": jobs.start(argv)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: int) -> Dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: int) -> Dict[str, Any]:
        """Stop a generation. It holds the phone, so there has to be a way out
        of the browser -- otherwise a tour that will not finish blocks every run
        after it."""
        job = jobs.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not job.stop():
            raise HTTPException(status_code=409, detail="that job is already done")
        return {"stopping": True}

    @app.get("/api/jobs/{job_id}/stream")
    def job_stream(job_id: int) -> StreamingResponse:
        """The tour a generation is doing, as the run it is.

        The same frames off the same files as `/api/runs/stream`: a generation
        drives the phone through the same agent and leaves the same
        `events.jsonl` behind it, so the browser has no reason to settle for the
        child's stdout. What stdout still carries is the part that happens after
        the loop -- the skill written up from what the tour saw -- which is why
        `/api/jobs/{id}` stays.
        """
        job = jobs.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return StreamingResponse(_event_stream(job, shutting_down),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


#: How often the reload stream looks for new changes. A tenth of a second is
#: below the point where a save feels like it took a moment, and the look is an
#: integer comparison behind a lock.
RELOAD_POLL_S = 0.1


def _reload_stream(reloader: LiveReload,
                   shutting_down: Optional[threading.Event] = None,
                   ) -> Generator[str, None, None]:
    """Publish what the reloader saw, as SSE frames.

    The first frame identifies the process. That is how a code reload is seen
    from the browser at all: the restart takes this stream down with it, the
    page's `EventSource` reconnects on its own, and a `hello` bearing a boot id
    the page has not seen before means the server it was loaded from is gone.

    A plain generator, like `_event_stream`, and for the same reason: Starlette
    runs it on a thread, so `shutting_down` is what ends it -- without that it
    polls a reloader nobody is listening to while the shutdown tears the
    response down underneath it.
    """
    # Read before the frame is built, not after it is sent: a generator is only
    # resumed when the client reads, so a version taken on the far side of the
    # yield is the version at some later moment -- and everything that changed
    # in between is a change this stream would never mention.
    seen = reloader.version
    yield sse({"boot": reloader.boot, "version": seen,
               "restarts": reloader.restarts,
               "watching": reloader.watching()}, "hello")
    while shutting_down is None or not shutting_down.is_set():
        for change in reloader.since(seen):
            seen = change.version
            yield sse(change.as_frame(), "reload")
        time.sleep(RELOAD_POLL_S)
    yield sse({"reason": "server going away"}, "end")


def _event_stream(child: ChildProcess,
                  shutting_down: Optional[threading.Event] = None,
                  ) -> Generator[str, None, None]:
    """Replay then follow a child's run files as SSE frames.

    Any child that drives the phone: the goal run behind the Work tab, or the
    tour behind a `skills generate`. Both write the same files, so both are
    watched with the same frames -- and the browser renders them with one view.

    Three files, three frame types: `events.jsonl` (what was decided) arrives as
    `event`, `stream.jsonl` (the model's raw thinking and response as they
    happen) arrives as `llm`, and `screens.jsonl` (where the elements it was
    choosing between were) arrives as `screen`. A run recorded before either of
    the later two existed just yields the ones it has.

    A fourth arrives only while the child is stopping: `output`, one frame per
    line of its stdout. Both files go quiet the moment the loop ends, and for a
    watch that is where the work starts -- every pass it made is folded into the
    app's skill by one model call that can take a minute. None of that is written
    to a run directory, so without this the feed simply freezes between the stop
    and the exit, which reads as a stop that hung or did nothing.

    Under `--repeat` one subprocess writes several runs, each in its own
    directory, so the tail follows the move and announces it with a fresh
    `run` frame. The client rules off there; the session's own totals -- the
    spend the budget bounds -- carry across.

    `shutting_down` ends the stream when the server is going away. This is a
    plain generator run on a thread, so it cannot be cancelled: without the flag
    it goes on polling a run directory nobody is reading, and the shutdown tears
    the response down underneath it -- which is the `CancelledError` raised out
    of the middle of a `StreamingResponse` that a Ctrl+C prints to the console.
    """
    yield sse(child.state(), "state")
    state = child.state()
    if not state["running"] and not state["run_id"]:
        yield sse({"reason": "no active run"}, "end")
        return

    run_dir = child.run_dir() or child.wait_for_run_dir()
    if run_dir is None:
        yield sse({"reason": "run directory never appeared",
                   "output_tail": child.state()["output_tail"]}, "end")
        return

    def tails_for(path: Path):
        return [(path / runparse.EVENTS_NAME, "event"),
                (path / STREAM_NAME, "llm"),
                (path / SCREENS_NAME, "screen")]

    yield sse({"run_id": run_dir.name,
               "iteration": child.state()["iteration"] or 1}, "run")

    tails = tails_for(run_dir)
    offsets = [0] * len(tails)
    carries = [b""] * len(tails)  # an appended-to file's last line may be torn
    drain_passes = 0  # once the child exits, read twice more to catch the tail
    last_heartbeat = time.monotonic()
    output_seq: Optional[int] = None  # set when the stop is first seen
    while True:
        flowed = False
        for i, (path, frame) in enumerate(tails):
            try:
                with path.open("rb") as fh:
                    fh.seek(0, 2)
                    if fh.tell() < offsets[i]:
                        offsets[i] = 0  # truncated: start over, not lose events
                    fh.seek(offsets[i])
                    data = fh.read()
            except OSError:
                data = b""
            offsets[i] += len(data)

            data = carries[i] + data
            parts = data.split(b"\n")
            carries[i] = parts.pop() if parts else b""
            for line in parts:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                flowed = True
                yield sse(event, frame)

        # A repeat iteration starts in a directory of its own. Follow it, but
        # only once this one has gone quiet, so the move never abandons events
        # still being written -- and never before the last one has been read,
        # which is why a switch resets the drain count.
        if not flowed:
            newest = child.run_dir()
            if newest is not None and newest != run_dir:
                for i, (_, frame) in enumerate(tails):
                    if carries[i].strip():
                        try:
                            yield sse(json.loads(carries[i]), frame)
                        except json.JSONDecodeError:
                            pass
                run_dir = newest
                tails = tails_for(run_dir)
                offsets = [0] * len(tails)
                carries = [b""] * len(tails)
                drain_passes = 0
                yield sse({"run_id": run_dir.name,
                           "iteration": child.state()["iteration"]}, "run")
                continue

        state = child.state()

        # The shutdown, as the child tells it. Read from the mark the signal was
        # sent at rather than from wherever this loop happens to look first, so
        # the account starts at its own first line -- the child answers a SIGINT
        # in milliseconds and this poll is half a second wide.
        # Kept up once started, because `stopping` is only true while the child
        # is alive: the line that says whether the skill was written is the last
        # thing it prints, and the drain passes below are where it arrives.
        if state["stopping"] or output_seq is not None:
            if output_seq is None:
                output_seq = child.stop_mark()
                yield sse(state, "state")  # the button has a phase to show
            output_seq, lines = child.output_since(output_seq)
            for line in lines:
                yield sse({"line": line}, "output")

        if state["running"]:
            drain_passes = 0
        else:
            drain_passes += 1
            if drain_passes >= 2:
                for i, (path, frame) in enumerate(tails):
                    if carries[i].strip():  # a final line without its newline
                        try:
                            yield sse(json.loads(carries[i]), frame)
                        except json.JSONDecodeError:
                            pass
                state = child.state()
                yield sse(state, "state")
                yield sse({"reason": "finished",
                           "returncode": state["returncode"]}, "end")
                return
        if shutting_down is not None and shutting_down.is_set():
            yield sse({"reason": "the server is shutting down"}, "end")
            return
        if time.monotonic() - last_heartbeat > 15:
            yield sse({"t": time.time()}, "ping")
            last_heartbeat = time.monotonic()
        # Poll briskly while the model is talking so the stream reads as live;
        # fall back to a lazy cadence when the run is quiet.
        time.sleep(0.15 if flowed else 0.5)
