"""Web UI: the events parser, the run manager, and the FastAPI surface.

No phone and no API key: device calls are monkeypatched, runs are fake
subprocesses, and run artifacts are fixture files in a tmp directory.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adbagent.web import runparse
from adbagent.web.runner import RunManager
from adbagent.web.server import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_ID = "abc123def456"

RUN_EVENTS = [
    {"t": 1000.0, "kind": "run_start", "goal": "turn on wifi", "model": "m"},
    {"t": 1001.0, "kind": "decide", "step": 1,
     "action": {"action": "tap", "target": {"index": 3},
                "observation": "settings screen", "reasoning": "wifi is there",
                "notes": {"wifi": "on"}},
     "wall_s": 2.0,
     "llm": {"n_calls": 1, "prompt_tokens": 100, "cached_tokens": 0,
             "completion_tokens": 10, "reasoning_tokens": 0, "latency_s": 2.0,
             "usd": 0.001, "calls": [{"purpose": "decide", "usd": 0.001}]}},
    {"t": 1002.0, "kind": "verify", "step": 1, "grade": "worked",
     "reason": "screen changed"},
    {"t": 1003.0, "kind": "run_end", "outcome": "success", "steps": 1,
     "llm_calls": 1, "usd": 0.001, "packages": ["com.android.settings"]},
]


def make_run_dir(base: Path, run_id: str = RUN_ID,
                 events=RUN_EVENTS, with_log: bool = True,
                 with_checkpoint: bool = False) -> Path:
    d = base / run_id
    d.mkdir(parents=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    if with_log:
        (d / "run.log").write_text("line one\nline two\n", encoding="utf-8")
    if with_checkpoint:
        from adbagent import checkpoint
        (d / checkpoint.NAME).write_text(json.dumps(
            {"goal": "turn on wifi", "run_id": run_id, "step": 1}),
            encoding="utf-8")
    return d


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """A TestClient over a fully isolated app: tmp runs dir, tmp skills dir,
    tmp config file. The project cwd's own artifacts are never touched."""
    runs = tmp_path / "runs"
    skills = tmp_path / "skills"
    skills.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ADBAGENT_MODEL", "")  # keep env out of the way
    app = create_app(artifacts_dir=str(runs), skills_dir=str(skills),
                     config_path=str(config))
    return TestClient(app)


class FakeProc:
    """A Popen stand-in. Optionally materialises a run directory on spawn,
    the way the real CLI creates runs/<id> when the agent starts."""

    def __init__(self, argv, on_spawn=None, stay_running=False, **kwargs):
        self.argv = argv
        self.pid = 4321
        self._done = threading.Event()
        self._returncode = None
        self.stdout = iter([])
        self._stay = stay_running
        if on_spawn:
            on_spawn(argv)
        if not stay_running:
            self._done.set()

    def poll(self):
        return None if not self._done.is_set() else (self._returncode or 0)

    def wait(self, timeout=None):
        self._done.wait(timeout)
        self._returncode = 0
        return 0

    def send_signal(self, sig):
        self._returncode = 130
        self._done.set()


# ---------------------------------------------------------------------------
# runparse
# ---------------------------------------------------------------------------

def test_summarise_finished_run(tmp_path):
    d = make_run_dir(tmp_path)
    s = runparse.summarise(d)
    assert s["id"] == RUN_ID
    assert s["goal"] == "turn on wifi"
    assert s["outcome"] == "success"
    assert s["steps"] == 1
    assert s["usd"] == pytest.approx(0.001)
    assert s["duration_s"] == pytest.approx(3.0)
    assert s["packages"] == ["com.android.settings"]


def test_summarise_interrupted_run(tmp_path):
    # No run_end: killed mid-flight. Stats come from what was recorded.
    d = make_run_dir(tmp_path, events=RUN_EVENTS[:-1])
    s = runparse.summarise(d)
    assert s["outcome"] == "interrupted"
    assert s["steps"] == 1
    assert s["usd"] == pytest.approx(0.001)


def test_run_detail_stats_and_scratchpad(tmp_path):
    d = make_run_dir(tmp_path)
    detail = runparse.run_detail(d)
    assert detail["stats"]["decisions"] == 1
    assert detail["stats"]["latency_median_s"] == pytest.approx(2.0)
    assert detail["stats"]["prompt_tokens"] == 100
    assert "wifi" in detail["scratchpad"]
    assert len(detail["events"]) == 4


def test_read_events_tolerates_torn_tail(tmp_path):
    d = make_run_dir(tmp_path)
    with (d / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"t": 1004.0, "kind": "deci')  # mid-append
    assert len(runparse.read_events(d)) == 4


def write_stream(d: Path, records) -> None:
    with (d / "stream.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_run_detail_merges_the_llm_stream(tmp_path):
    d = make_run_dir(tmp_path)
    write_stream(d, [
        {"t": 1000.5, "kind": "llm_start", "step": 1, "purpose": "decide",
         "model": "m"},
        {"t": 1000.6, "kind": "llm_stream", "stream_type": "thinking",
         "text": "wifi is under network"},
        {"t": 1000.7, "kind": "llm_stream", "stream_type": "content",
         "text": '{"action": "tap"}'},
        {"t": 1000.9, "kind": "llm_end", "step": 1, "purpose": "decide",
         "elapsed": 1.8, "completion_tokens": 40},
    ])
    detail = runparse.run_detail(d)
    kinds = [e["kind"] for e in detail["events"]]
    # Interleaved by timestamp: the stream brackets the decision it produced.
    assert kinds == ["run_start", "llm_start", "llm_stream", "llm_stream",
                     "llm_end", "decide", "verify", "run_end"]
    # Stats still read the decision events alone.
    assert detail["stats"]["decisions"] == 1
    assert detail["stats"]["completion_tokens"] == 10
    assert detail["summary"]["n_events"] == 4


def test_list_runs_newest_first(tmp_path):
    older = make_run_dir(tmp_path, run_id="aaa")
    newer = make_run_dir(tmp_path, run_id="bbb")
    # mtime decides, not the name.
    now = time.time()
    os.utime(older / "events.jsonl", (now - 100, now - 100))
    os.utime(newer / "events.jsonl", (now, now))
    ids = [r["id"] for r in runparse.list_runs(tmp_path)]
    assert ids == ["bbb", "aaa"]


def test_find_run_rejects_traversal(tmp_path):
    make_run_dir(tmp_path)
    assert runparse.find_run(tmp_path, RUN_ID) is not None
    assert runparse.find_run(tmp_path, "../etc") is None
    assert runparse.find_run(tmp_path, "missing") is None


# ---------------------------------------------------------------------------
# pages & status
# ---------------------------------------------------------------------------

def test_index_and_static(web):
    res = web.get("/")
    assert res.status_code == 200
    assert "adbagent" in res.text
    res = web.get("/static/app.js")
    assert res.status_code == 200
    assert "text/javascript" in res.headers["content-type"]
    assert web.get("/static/nope.txt").status_code == 404


def test_status(web):
    st = web.get("/api/status").json()
    assert st["run"]["running"] is False
    assert "config_path" in st


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_round_trip(web):
    got = web.get("/api/config").json()
    assert got["config"]["safety"]["budget_usd"] == 2.0

    res = web.put("/api/config", json={
        "sections": {"safety": {"budget_usd": 5.0}, "llm": {"model": "m2"}}})
    assert res.status_code == 200

    got = web.get("/api/config").json()
    assert got["config"]["safety"]["budget_usd"] == 5.0
    assert got["config"]["llm"]["model"] == "m2"


def test_config_rejects_unknown_keys(web):
    res = web.put("/api/config", json={"sections": {"safety": {"nope": 1}}})
    assert res.status_code == 400
    # And a bad save writes nothing.
    assert web.get("/api/config").json()["config"]["safety"]["budget_usd"] == 2.0


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

def test_skills_crud(web):
    assert web.get("/api/skills").json()["skills"] == []

    skill = {"name": "WhatsApp", "packages": ["com.whatsapp"],
             "aliases": ["wa"], "description": "chat app",
             "workflows": [{"name": "send", "steps": "open chat, type, send"}],
             "nuances": ["long-press to react"], "recommendations": []}
    res = web.put("/api/skills/whatsapp", json=skill)
    assert res.status_code == 200

    listing = web.get("/api/skills").json()["skills"]
    assert len(listing) == 1
    assert listing[0]["name"] == "WhatsApp"
    assert listing[0]["workflows"] == 1

    got = web.get("/api/skills/wa").json()  # by alias
    assert got["packages"] == ["com.whatsapp"]

    assert web.put("/api/skills/x", json={"name": ""}).status_code == 400
    assert web.get("/api/skills/unknown").status_code == 404


# ---------------------------------------------------------------------------
# runs over the API
# ---------------------------------------------------------------------------

def test_start_run_requires_a_goal(web):
    assert web.post("/api/runs", json={"goal": ""}).status_code == 400


def test_stop_with_no_run_is_a_conflict(web):
    assert web.post("/api/runs/stop").status_code == 409


def test_run_lifecycle_against_fake_cli(web, tmp_path, monkeypatch):
    spawned = []

    def fake_popen(argv, **kwargs):
        proc = FakeProc(argv, stay_running=True, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)

    res = web.post("/api/runs", json={"goal": "turn on wifi", "max_steps": 5})
    assert res.status_code == 200
    argv = spawned[0].argv
    assert "turn on wifi" in argv
    assert "--unattended" in argv          # never blocks on a prompt
    assert "--allow-destructive" not in argv
    assert "--max-steps" in argv

    # One phone, one run.
    assert web.post("/api/runs", json={"goal": "again"}).status_code == 409

    assert web.post("/api/runs/stop").status_code == 200
    assert spawned[0].poll() is not None


# ---------------------------------------------------------------------------
# resuming from the UI
# ---------------------------------------------------------------------------

def test_summarise_flags_a_resumable_run(tmp_path):
    assert runparse.summarise(make_run_dir(tmp_path))["resumable"] is False
    d = make_run_dir(tmp_path, run_id="failed1", with_checkpoint=True)
    assert runparse.summarise(d)["resumable"] is True


def test_resume_requires_a_checkpoint(web, tmp_path):
    make_run_dir(tmp_path / "runs", run_id=RUN_ID)        # no checkpoint
    res = web.post("/api/runs", json={"resume": RUN_ID})
    assert res.status_code == 409
    assert "no checkpoint" in res.json()["detail"]
    assert web.post("/api/runs", json={"resume": "nope"}).status_code == 404


def test_resume_spawns_the_cli_with_resume_and_its_own_dir(web, tmp_path,
                                                           monkeypatch):
    make_run_dir(tmp_path / "runs", run_id=RUN_ID, with_checkpoint=True)
    spawned = []

    def fake_popen(argv, **kwargs):
        proc = FakeProc(argv, stay_running=True, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    res = web.post("/api/runs", json={"resume": RUN_ID, "max_steps": 10})
    assert res.status_code == 200

    argv = spawned[0].argv
    assert "--resume" in argv
    assert argv[argv.index("--resume") + 1] == RUN_ID
    assert "turn on wifi" not in argv     # the checkpoint supplies the goal
    assert "--unattended" in argv
    assert "--max-steps" in argv

    # Status shows the checkpoint's goal, not an empty string.
    st = web.get("/api/status").json()
    assert st["run"]["goal"] == "turn on wifi"
    assert st["run"]["run_id"] == RUN_ID

    web.post("/api/runs/stop")

    # The directory already exists, so the manager knew it without discovery:
    # the stream attaches to the old events and replays them.
    body = web.get("/api/runs/stream").text
    assert f'"run_id": "{RUN_ID}"' in body
    assert "turn on wifi" in body         # the first sitting's events replay


def test_stream_with_no_run_ends_immediately(web):
    body = web.get("/api/runs/stream").text
    assert "no active run" in body
    assert "event: end" in body


def test_stream_replays_and_finishes(web, tmp_path, monkeypatch):
    runs = tmp_path / "runs"

    def fake_popen(argv, **kwargs):
        def on_spawn(_argv):
            make_run_dir(runs, run_id="live1")
        return FakeProc(argv, on_spawn=on_spawn, **kwargs)

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    assert web.post("/api/runs", json={"goal": "turn on wifi"}).status_code == 200

    body = web.get("/api/runs/stream").text
    assert '"run_id": "live1"' in body
    assert "turn on wifi" in body           # run_start replayed
    assert "run_end" in body
    assert "event: end" in body


def test_stream_includes_llm_frames(web, tmp_path, monkeypatch):
    runs = tmp_path / "runs"

    def fake_popen(argv, **kwargs):
        def on_spawn(_argv):
            d = make_run_dir(runs, run_id="live2")
            write_stream(d, [
                {"t": 1000.5, "kind": "llm_start", "step": 1,
                 "purpose": "decide", "model": "m"},
                {"t": 1000.6, "kind": "llm_stream", "stream_type": "thinking",
                 "text": "wifi is under network"},
            ])
        return FakeProc(argv, on_spawn=on_spawn, **kwargs)

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    assert web.post("/api/runs", json={"goal": "turn on wifi"}).status_code == 200

    body = web.get("/api/runs/stream").text
    assert "event: event" in body            # the decisions still flow
    assert "event: llm" in body              # and the raw stream beside them
    assert '"kind": "llm_start"' in body
    assert "wifi is under network" in body
    assert "event: end" in body


def test_runs_list_detail_and_log(web, tmp_path):
    make_run_dir(tmp_path / "runs", run_id=RUN_ID)
    listing = web.get("/api/runs").json()
    assert [r["id"] for r in listing["runs"]] == [RUN_ID]

    detail = web.get(f"/api/runs/{RUN_ID}").json()
    assert detail["summary"]["goal"] == "turn on wifi"
    assert detail["stats"]["decisions"] == 1

    log = web.get(f"/api/runs/{RUN_ID}/log").json()
    assert "line one" in log["text"]

    assert web.get("/api/runs/nope").status_code == 404


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

def test_devices_lists_and_handles_adb_failure(web, monkeypatch):
    from adbagent import device as devmod

    class FakeProp:
        model = "Pixel 8"

    class FakeDevice:
        serial = "emulator-5554"
        prop = FakeProp()

        def getprop(self, key):
            return "15"

    monkeypatch.setattr(devmod, "list_devices", lambda: [FakeDevice()])
    monkeypatch.setattr(devmod, "mdns_candidates", lambda: ["1.2.3.4:5555"])
    d = web.get("/api/devices").json()
    assert d["devices"] == [{"serial": "emulator-5554", "model": "Pixel 8",
                             "android": "15"}]
    assert d["candidates"] == ["1.2.3.4:5555"]

    monkeypatch.setattr(devmod, "list_devices",
                        lambda: (_ for _ in ()).throw(RuntimeError("adb missing")))
    d = web.get("/api/devices").json()
    assert d["devices"] == []
    assert "adb missing" in d["error"]


def test_screenshot_refused_while_running(web, monkeypatch):
    monkeypatch.setattr(
        "adbagent.web.runner.subprocess.Popen",
        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))
    assert web.post("/api/runs", json={"goal": "hold"}).status_code == 200
    assert web.get("/api/devices/screenshot").status_code == 409


def test_screenshot_returns_jpeg(web, monkeypatch):
    from adbagent import device as devmod

    class FakeDev:
        def __init__(self, cfg, serial):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def screenshot(self):
            return b"\xff\xd8jpeg"

    monkeypatch.setattr(devmod, "Device", FakeDev)
    res = web.get("/api/devices/screenshot")
    assert res.status_code == 200
    assert res.content == b"\xff\xd8jpeg"


# ---------------------------------------------------------------------------
# skills generate jobs
# ---------------------------------------------------------------------------

def test_generate_skill_job(web, monkeypatch):
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(argv, **kw))
    res = web.post("/api/skills/generate", json={"name": "whatsapp"}).json()
    job = web.get(f"/api/jobs/{res['job']}").json()
    assert "skills" in job["argv"] and "generate" in job["argv"]
    assert "whatsapp" in job["argv"]
    assert web.get("/api/jobs/9999").status_code == 404


# ---------------------------------------------------------------------------
# RunManager unit behaviour
# ---------------------------------------------------------------------------

def test_run_manager_discovers_run_dir(tmp_path):
    def on_spawn(_argv):
        make_run_dir(tmp_path, run_id="newrun")

    import adbagent.web.runner as runner_mod
    real_popen = runner_mod.subprocess.Popen
    runner_mod.subprocess.Popen = lambda argv, **kw: FakeProc(
        argv, on_spawn=on_spawn, **kw)
    try:
        mgr = RunManager(tmp_path)
        mgr.start("goal")
        assert mgr.wait_for_run_dir(timeout_s=5) == tmp_path / "newrun"
    finally:
        runner_mod.subprocess.Popen = real_popen
