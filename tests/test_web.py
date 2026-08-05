"""Web UI: the events parser, the run manager, and the FastAPI surface.

No phone and no API key: device calls are monkeypatched, runs are fake
subprocesses, and run artifacts are fixture files in a tmp directory.
"""

from __future__ import annotations

import json
import os
import signal
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
        self.signals = []  # which one arrived matters: SIGINT restores the phone
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
        self.signals.append(sig)
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


def test_run_detail_folds_the_stream_into_one_record_per_run(tmp_path):
    """A saved run is not watched token by token, and the file has one line per
    token. Consecutive chunks of the same kind arrive joined."""
    d = make_run_dir(tmp_path)
    write_stream(d, [
        {"t": 1000.5, "kind": "llm_start", "step": 1, "purpose": "decide"},
        *[{"t": 1000.5 + i / 100, "kind": "llm_stream",
           "stream_type": "thinking", "text": f"tok{i} "} for i in range(500)],
        {"t": 1000.8, "kind": "llm_stream", "stream_type": "content",
         "text": '{"action":'},
        {"t": 1000.9, "kind": "llm_stream", "stream_type": "content",
         "text": ' "tap"}'},
        {"t": 1000.95, "kind": "llm_end", "step": 1, "purpose": "decide"},
    ])
    feed = runparse.run_detail(d)["events"]
    chunks = [e for e in feed if e["kind"] == "llm_stream"]
    assert len(chunks) == 2                       # one thinking, one content
    assert chunks[0]["text"].startswith("tok0 ") and "tok499" in chunks[0]["text"]
    assert chunks[1]["text"] == '{"action": "tap"}'
    # The join keeps the first chunk's timestamp, so it still sorts where the
    # model started talking -- after the call it belongs to, before the decision.
    assert chunks[0]["t"] == pytest.approx(1000.5)
    assert [e["kind"] for e in feed] == [
        "run_start", "llm_start", "llm_stream", "llm_stream", "llm_end",
        "decide", "verify", "run_end"]


def test_fold_stream_breaks_runs_at_call_boundaries():
    """Two calls' worth of thinking is two records, not one: a boundary event
    between them is what keeps the panels separate."""
    folded = runparse.fold_stream([
        {"kind": "llm_start"},
        {"kind": "llm_stream", "stream_type": "thinking", "text": "a"},
        {"kind": "llm_stream", "stream_type": "thinking", "text": "b"},
        {"kind": "llm_end"},
        {"kind": "llm_start"},
        {"kind": "llm_stream", "stream_type": "thinking", "text": "c"},
    ])
    assert [e.get("text") for e in folded if e["kind"] == "llm_stream"] == ["ab", "c"]
    assert len(folded) == 5


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
# the model catalogue behind the config dropdowns
# ---------------------------------------------------------------------------

def test_models_serves_the_catalogue_qualified_and_cached(web, monkeypatch):
    from adbagent import llm

    fetches = []

    def fake_list(provider, api_key, timeout=30.0):
        fetches.append(api_key)
        return [llm.ModelInfo(id="kimi-k2p6", context_length=262144, tools=True),
                llm.ModelInfo(id="qwen3-vl-235b", vision=True, tools=True)]

    monkeypatch.setattr(llm, "list_models", fake_list)
    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-test")

    d = web.get("/api/models").json()
    assert d["error"] == ""
    assert d["provider"] == "fireworks"
    # The value a dropdown saves is the one the wire wants, not the short id.
    assert [m["value"] for m in d["models"]] == [
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/qwen3-vl-235b"]
    assert [m["id"] for m in d["models"]] == ["kimi-k2p6", "qwen3-vl-235b"]
    assert [m["vision"] for m in d["models"]] == [False, True]

    # Several paged HTTP calls: the config tab must not pay for them twice.
    assert web.get("/api/models").json()["cached"] is True
    assert len(fetches) == 1
    assert web.get("/api/models?refresh=1").json()["cached"] is False
    assert len(fetches) == 2


def test_models_reports_trouble_instead_of_failing(web, monkeypatch):
    from adbagent import llm

    # No key: the UI answers this with a text box, so it is not an error status.
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    d = web.get("/api/models").json()
    assert d["models"] == []
    assert "API key" in d["error"]

    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "list_models", lambda *a, **kw: (_ for _ in ()).throw(
        llm.LLMError("catalogue rejected the API key (401)")))
    d = web.get("/api/models").json()
    assert d["models"] == []
    assert "401" in d["error"]

    web.put("/api/config", json={"sections": {"llm": {"provider": "nope"}}})
    d = web.get("/api/models").json()
    assert "unknown provider" in d["error"]
    assert "fireworks" in d["error"]  # and what it could have been


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
                 "purpose": "analyze_image", "model": "m", "screenshot": True,
                 "shot": "step_001_analyze_image_00c0ffee.jpg"},
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
    # The frame that call was shown, named so the page can fetch it while the
    # run is still going.
    assert "step_001_analyze_image_00c0ffee.jpg" in body
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


def test_a_submitted_frame_is_served_back(web, tmp_path):
    """The screenshot a call was shown, by the name its `llm_start` carries."""
    d = make_run_dir(tmp_path / "runs", run_id=RUN_ID)
    name = "step_004_analyze_image_00c0ffee.jpg"
    (d / name).write_bytes(b"\xff\xd8jpeg")

    res = web.get(f"/api/runs/{RUN_ID}/shot/{name}")
    assert res.status_code == 200
    assert res.content == b"\xff\xd8jpeg"
    assert res.headers["content-type"] == "image/jpeg"
    assert "immutable" in res.headers["cache-control"]

    # A name in the pattern but no such frame.
    assert web.get(f"/api/runs/{RUN_ID}/shot/"
                   "step_009_decide_deadbeef.jpg").status_code == 404
    assert web.get(f"/api/runs/nope/shot/{name}").status_code == 404


@pytest.mark.parametrize("name", [
    "run.log",                       # a real file in the directory, not a frame
    "events.jsonl",
    "step_001_decide_messages.json",
    "..%2f..%2fetc%2fpasswd",
    "step_001_decide_deadbeef.jpg.log",
])
def test_only_the_run_frames_are_served(web, tmp_path, name):
    """The name is matched against the pattern the recorder writes, so nothing
    else in a run directory can be fetched through this route."""
    make_run_dir(tmp_path / "runs", run_id=RUN_ID)
    assert web.get(f"/api/runs/{RUN_ID}/shot/{name}").status_code == 404


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


def test_generation_streams_its_tour_like_a_run(web, tmp_path, monkeypatch):
    """A tour writes the same files a run does, so it is watched the same way:
    the events, the thinking stream and the frame each call was shown."""
    runs = tmp_path / "runs"

    def fake_popen(argv, **kwargs):
        def on_spawn(_argv):
            d = make_run_dir(runs, run_id="tour1")
            write_stream(d, [
                {"t": 1000.5, "kind": "llm_start", "step": 1, "purpose": "decide",
                 "model": "m", "shot": "step_001_decide_00c0ffee.jpg"},
                {"t": 1000.6, "kind": "llm_stream", "stream_type": "thinking",
                 "text": "the composer is at the bottom"},
            ])
        return FakeProc(argv, on_spawn=on_spawn, **kwargs)

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    job = web.post("/api/skills/generate", json={"name": "whatsapp"}).json()["job"]

    body = web.get(f"/api/jobs/{job}/stream").text
    assert '"run_id": "tour1"' in body       # which run the frames come from
    assert "event: event" in body            # the steps it took
    assert "event: llm" in body              # and what it was thinking
    assert "the composer is at the bottom" in body
    assert "step_001_decide_00c0ffee.jpg" in body
    assert "event: end" in body

    assert web.get("/api/jobs/9999/stream").status_code == 404


def test_a_tour_that_never_reached_the_phone_ends_the_stream(web, monkeypatch):
    """A generation refused before it opened anything -- no key, no such app --
    writes no run at all. The stream says so rather than waiting a minute for a
    directory that is never coming."""
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(argv, **kw))
    job = web.post("/api/skills/generate", json={"name": "nope"}).json()["job"]
    body = web.get(f"/api/jobs/{job}/stream").text
    assert "event: end" in body
    assert "never appeared" in body or "no active run" in body


def test_a_generation_can_be_stopped(web, monkeypatch):
    """It holds the phone, and the guard above means a tour nobody can stop
    would block every run after it."""
    spawned = []

    def fake_popen(argv, **kwargs):
        proc = FakeProc(argv, stay_running=True, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    job = web.post("/api/skills/generate", json={"name": "whatsapp"}).json()["job"]

    assert web.post(f"/api/jobs/{job}/stop").status_code == 200
    assert spawned[0].poll() is not None
    # SIGINT, not a kill: the CLI catches it and puts the phone back as it was.
    assert spawned[0].signals == [signal.SIGINT]
    # And with the phone released, a run can start.
    assert web.post(f"/api/jobs/{job}/stop").status_code == 409
    assert web.post("/api/jobs/9999/stop").status_code == 404
    assert web.post("/api/runs", json={"goal": "turn on wifi"}).status_code == 200
    web.post("/api/runs/stop")


def test_one_phone_one_agent(web, monkeypatch):
    """A tour and a goal run drive the phone the same way, so they cannot
    overlap: each would read the other's taps as its own."""
    monkeypatch.setattr(
        "adbagent.web.runner.subprocess.Popen",
        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))

    job = web.post("/api/skills/generate", json={"name": "whatsapp"})
    assert job.status_code == 200
    res = web.post("/api/runs", json={"goal": "turn on wifi"})
    assert res.status_code == 409
    assert "generated" in res.json()["detail"]
    # And the other way about, once the generation has the phone released.
    assert web.get("/api/status").json()["job"]["id"] == job.json()["job"]
    assert web.post("/api/skills/generate",
                    json={"name": "again"}).status_code == 409


# ---------------------------------------------------------------------------
# RunManager unit behaviour
# ---------------------------------------------------------------------------

def test_repeat_reaches_the_cli(web, tmp_path, monkeypatch):
    spawned = []

    def fake_popen(argv, **kwargs):
        proc = FakeProc(argv, stay_running=True, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    assert web.post("/api/runs", json={"goal": "compare prices",
                                       "repeat": "inf",
                                       "budget_usd": 5.0}).status_code == 200
    argv = spawned[0].argv
    assert argv[argv.index("--repeat") + 1] == "inf"
    # The ceiling is what makes an unbounded repeat safe to start from a
    # browser, so it must survive the trip.
    assert argv[argv.index("--budget-usd") + 1] == "5.0"
    assert web.get("/api/status").json()["run"]["repeat"] == "inf"
    web.post("/api/runs/stop")


class RepeatProc(FakeProc):
    """A child that writes two iteration directories and then exits, the way
    `--repeat 2` does: one subprocess, two runs, two directories."""

    def __init__(self, argv, runs: Path, **kwargs):
        super().__init__(argv, stay_running=True, **kwargs)
        make_run_dir(runs, run_id="iter1")
        threading.Thread(target=self._rest, args=(runs,), daemon=True).start()

    def _rest(self, runs: Path) -> None:
        time.sleep(0.5)
        make_run_dir(runs, run_id="iter2", events=[
            {"t": 2000.0, "kind": "run_start", "goal": "second sitting",
             "model": "m"},
            {"t": 2003.0, "kind": "run_end", "outcome": "success", "steps": 1,
             "llm_calls": 1, "usd": 0.002},
        ])
        time.sleep(0.5)
        self._done.set()


def test_run_manager_follows_every_iteration(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: RepeatProc(argv, runs, **kw))
    mgr = RunManager(runs)
    mgr.start("goal", repeat="2")
    assert mgr.wait_for_run_dir(timeout_s=5) == runs / "iter1"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(mgr.run_dirs()) < 2:
        time.sleep(0.05)
    # Discovery does not stop at the first: the phone was still being driven.
    assert [d.name for d in mgr.run_dirs()] == ["iter1", "iter2"]
    assert mgr.run_dir() == runs / "iter2"
    assert mgr.state()["run_id"] == "iter2"
    assert mgr.state()["iteration"] == 2


def test_stream_follows_a_repeat_into_its_next_iteration(web, tmp_path,
                                                         monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: RepeatProc(argv, runs, **kw))
    assert web.post("/api/runs", json={"goal": "compare prices",
                                       "repeat": "2"}).status_code == 200

    body = web.get("/api/runs/stream").text
    # Both sittings arrive on the one connection, each announced by its own
    # `run` frame -- without which the feed would go silent after iter1.
    assert '"run_id": "iter1"' in body
    assert '"run_id": "iter2"' in body
    assert body.index('"run_id": "iter1"') < body.index('"run_id": "iter2"')
    assert '"iteration": 2' in body
    assert "turn on wifi" in body        # iter1's events
    assert "second sitting" in body      # iter2's, after the move
    assert "event: end" in body


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
