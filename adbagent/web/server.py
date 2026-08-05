"""FastAPI app for the adbagent web UI.

The server is a thin shell over what already exists: runs are spawned as CLI
subprocesses (`runner.RunManager`), history is parsed off disk
(`runparse`), config and skills are read and written in the same files the
CLI uses. Nothing here knows how to drive a phone.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import Config, _set_path, find_config_file, load_config
from ..runlog import SHOT_RE, STREAM_NAME
from ..skills import Skill, SkillRegistry
from . import runparse
from .runner import ChildProcess, JobManager, RunManager, sse

STATIC_DIR = Path(__file__).parent / "static"

_MEDIA_TYPES = {".html": "text/html", ".js": "text/javascript",
                ".css": "text/css", ".png": "image/png"}


class RunRequest(BaseModel):
    goal: str = ""
    max_steps: Optional[int] = None
    budget_usd: Optional[float] = None
    repeat: str = "1"
    dry_run: bool = False
    allow_destructive: bool = False
    no_learn: bool = False
    serial: str = ""
    #: A run id to continue from its checkpoint. When set, `goal` is ignored
    #: -- the checkpoint's own goal is the one being pursued.
    resume: str = ""


class ConfigUpdate(BaseModel):
    sections: Dict[str, Dict[str, Any]]


class GenerateRequest(BaseModel):
    name: str = ""
    tasks: str = ""
    max_steps: Optional[int] = None
    budget_usd: Optional[float] = None
    serial: str = ""


def _static(name: str) -> Response:
    path = STATIC_DIR / name
    if not path.is_file() or path.parent != STATIC_DIR:
        raise HTTPException(status_code=404, detail="not found")
    media = _MEDIA_TYPES.get(path.suffix, "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=media)


def create_app(*, artifacts_dir: str = "runs", skills_dir: str = "",
               config_path: str = "") -> FastAPI:
    runs_dir = Path(artifacts_dir)
    manager = RunManager(runs_dir, config_path=config_path)
    # The same artifacts directory: a `skills generate` tour writes a run there
    # like any other, and the browser tails it through the same live view.
    jobs = JobManager(runs_dir)

    def phone_busy() -> str:
        """Why the phone cannot be driven right now, or "".

        One phone, one agent. A tour and a goal run reading each other's taps
        as their own is the failure this prevents; that both are started from
        different corners of the page is what makes it easy to ask for.
        """
        if manager.state()["running"]:
            return "a run is already in progress"
        job = jobs.active()
        if job is not None:
            return f"a skill is being generated (job {job.id}); the phone is busy"
        return ""

    def load_cfg() -> Config:
        return load_config(config_path or None).config

    def registry() -> SkillRegistry:
        return SkillRegistry(skills_dir or load_cfg().skills.skills_dir)

    app = FastAPI(title="adbagent ui")

    # -- pages ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        return _static("index.html")

    @app.get("/static/{name}")
    def static_file(name: str) -> Response:
        return _static(name)

    # -- status & devices ------------------------------------------------

    @app.get("/api/status")
    def status() -> Dict[str, Any]:
        loaded = load_config(config_path or None)
        cfg = loaded.config
        # A generation in flight is reported alongside the run for the same
        # reason: a page reloaded mid-tour has to know there is something to
        # reattach to, and which job to ask for it.
        job = jobs.active()
        return {
            "config_path": str(loaded.path) if loaded.path else "",
            "model": cfg.llm.model,
            "api_key_present": bool(cfg.api_key()),
            "device_serial": cfg.device.serial,
            "artifacts_dir": str(runs_dir),
            "run": manager.state(),
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

    @app.get("/api/devices/screenshot")
    def screenshot(serial: str = "") -> Response:
        if manager.state()["running"]:
            raise HTTPException(status_code=409,
                                detail="screenshots are paused while a run is active")
        from ..device import Device
        try:
            with Device(load_cfg(), serial) as dev:
                jpeg = dev.screenshot()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc))
        return Response(content=jpeg, media_type="image/jpeg")

    # -- config ----------------------------------------------------------

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        loaded = load_config(config_path or None)
        return {"config": loaded.config.to_dict(),
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
        return StreamingResponse(_event_stream(manager),
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
                                 resume=req.resume)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/runs/stop")
    def stop_run() -> Dict[str, Any]:
        if not manager.stop():
            raise HTTPException(status_code=409, detail="no run in progress")
        return {"stopping": True}

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
        return StreamingResponse(_event_stream(job),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def _event_stream(child: ChildProcess) -> Generator[str, None, None]:
    """Replay then follow a child's run files as SSE frames.

    Any child that drives the phone: the goal run behind the Run tab, or the
    tour behind a `skills generate`. Both write the same files, so both are
    watched with the same frames -- and the browser renders them with one view.

    Two files, two frame types: `events.jsonl` (what was decided) arrives as
    `event`, and `stream.jsonl` (the model's raw thinking and response as they
    happen) arrives as `llm`. A run recorded before the stream file existed
    just yields the first.

    Under `--repeat` one subprocess writes several runs, each in its own
    directory, so the tail follows the move and announces it with a fresh
    `run` frame. The client rules off there; the session's own totals -- the
    spend the budget bounds -- carry across.
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
                (path / STREAM_NAME, "llm")]

    yield sse({"run_id": run_dir.name,
               "iteration": child.state()["iteration"] or 1}, "run")

    tails = tails_for(run_dir)
    offsets = [0] * len(tails)
    carries = [b""] * len(tails)  # an appended-to file's last line may be torn
    drain_passes = 0  # once the child exits, read twice more to catch the tail
    last_heartbeat = time.monotonic()
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

        if child.state()["running"]:
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
        if time.monotonic() - last_heartbeat > 15:
            yield sse({"t": time.time()}, "ping")
            last_heartbeat = time.monotonic()
        # Poll briskly while the model is talking so the stream reads as live;
        # fall back to a lazy cadence when the run is quiet.
        time.sleep(0.15 if flowed else 0.5)
