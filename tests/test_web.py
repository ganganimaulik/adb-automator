"""Web UI: the events parser, the run manager, and the FastAPI surface.

No phone and no API key: device calls are monkeypatched, runs are fake
subprocesses, and run artifacts are fixture files in a tmp directory.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adbagent.web import runparse
from adbagent.web.runner import RunManager
from adbagent.web.server import _event_stream, create_app

EXPECTED_STOP_SIGNAL = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT) if sys.platform == "win32" else signal.SIGINT

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
    # `watch.policies_dir` defaults to `policies`, which is a real directory in
    # this checkout: left alone, the policy endpoints would list -- and be
    # allowed to write -- the project's own policies from a test.
    config.write_text(json.dumps(
        {"watch": {"policies_dir": str(tmp_path / "policies")}}),
        encoding="utf-8")
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


def test_summarise_carries_what_the_run_answered(tmp_path):
    """The detail view leads with this. Without it the page opens on token
    counts and the answer is somewhere down in the step feed."""
    events = RUN_EVENTS[:-1] + [dict(RUN_EVENTS[-1],
                                     result="Wi-Fi is on: Home-5G.",
                                     evidence="the SSID is on screen")]
    s = runparse.summarise(make_run_dir(tmp_path, events=events))
    assert s["result"] == "Wi-Fi is on: Home-5G."
    assert s["evidence"] == "the SSID is on screen"


def test_summarise_a_run_recorded_before_the_answer_was_kept(tmp_path):
    """Empty, not missing: the page hides the block rather than showing a
    heading over nothing."""
    s = runparse.summarise(make_run_dir(tmp_path))
    assert s["result"] == "" and s["evidence"] == ""


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
# picking a device
# ---------------------------------------------------------------------------

def test_use_device_writes_the_serial(web, monkeypatch):
    """`configured, not attached` is the right diagnosis and was a dead end:
    the fix lived four navigations away in the config form, and the only thing
    it wanted typed was a serial adb was already reporting."""
    from adbagent.web import server as srv

    # A serial no real adb on any developer's machine would report, so this
    # passes only because the stub was consulted.
    monkeypatch.setattr(srv, "attached_serials", lambda: ["test-phone-0"])
    res = web.post("/api/device/use", json={"serial": "test-phone-0"})
    assert res.status_code == 200
    assert web.get("/api/config").json()["config"]["device"]["serial"] \
        == "test-phone-0"


def test_use_device_refuses_a_serial_adb_cannot_see(web, monkeypatch):
    """Only ever a phone on the list. Anything else is the config form's job,
    where a typo is visible and reversible."""
    from adbagent.web import server as srv

    monkeypatch.setattr(srv, "attached_serials", lambda: ["test-phone-0"])
    assert web.post("/api/device/use", json={"serial": "10.0.0.9:5555"}) \
        .status_code == 409
    assert web.post("/api/device/use", json={"serial": ""}).status_code == 400
    # And a refusal writes nothing.
    assert web.get("/api/config").json()["config"]["device"]["serial"] == ""


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


def test_config_carries_the_shipped_defaults_too(web):
    """Sixty-two fields showing their defaults look identical to sixty-two
    fields somebody set on purpose; the form marks the difference off this."""
    web.put("/api/config", json={"sections": {"safety": {"budget_usd": 9.0}}})
    got = web.get("/api/config").json()
    assert got["config"]["safety"]["budget_usd"] == 9.0
    assert got["defaults"]["safety"]["budget_usd"] == 2.0
    # Every section of the live config is in the defaults, or the form would
    # mark a whole section as changed the moment it appeared.
    assert set(got["defaults"]) == set(got["config"])


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
# the live frame: the one screenshot path that may be taken under a run
# ---------------------------------------------------------------------------
#
# It exists because `Device.open()` zeroes the animation scales, locks rotation
# and selects its own IME, so `/api/devices/screenshot` cannot be used while an
# agent holds the phone -- and a tool that drives a phone and never shows you the
# phone was the largest hole in it. `exec-out screencap` opens no session.


def _attach(monkeypatch, *serials):
    """Pretend adb sees these, with no TTL in the way."""
    monkeypatch.setattr("adbagent.web.server.ATTACHED_TTL_S", 0.0)
    monkeypatch.setattr("adbagent.web.server.attached_serials", lambda: list(serials))


def test_a_frame_says_when_there_is_no_phone(web, monkeypatch):
    _attach(monkeypatch)
    res = web.get("/api/device/frame")
    assert res.status_code == 404
    assert "no device" in res.json()["detail"]


def test_a_frame_will_not_pretend_a_configured_serial_is_attached(
        web, tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"device": {"serial": "192.168.1.23:41207"}}),
                   encoding="utf-8")
    _attach(monkeypatch, "emulator-5554")
    res = web.get("/api/device/frame")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert "192.168.1.23:41207" in detail and "emulator-5554" in detail


def test_a_frame_refuses_to_guess_between_two_phones(web, monkeypatch):
    _attach(monkeypatch, "one", "two")
    res = web.get("/api/device/frame")
    assert res.status_code == 409
    assert "no serial chosen" in res.json()["detail"]


def test_a_frame_comes_back_as_a_jpeg(web, monkeypatch):
    _attach(monkeypatch, "emulator-5554")
    seen = {}

    def fake(serial, max_long_edge=720):
        seen["serial"] = serial
        seen["edge"] = max_long_edge
        return b"\xff\xd8frame"

    monkeypatch.setattr("adbagent.web.server.screencap", fake)
    res = web.get("/api/device/frame?max_long_edge=360")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content == b"\xff\xd8frame"
    assert seen == {"serial": "emulator-5554", "edge": 360}


def test_a_frame_is_still_served_while_a_run_holds_the_phone(web, monkeypatch):
    """The whole point of the endpoint: every other device call is a 409 here."""
    _attach(monkeypatch, "emulator-5554")
    monkeypatch.setattr("adbagent.web.server.screencap",
                        lambda serial, max_long_edge=720: b"\xff\xd8live")
    monkeypatch.setattr(
        "adbagent.web.runner.subprocess.Popen",
        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))
    assert web.post("/api/runs", json={"goal": "hold"}).status_code == 200
    assert web.get("/api/devices/screenshot").status_code == 409
    live = web.get("/api/device/frame")
    assert live.status_code == 200
    assert live.content == b"\xff\xd8live"


def test_a_frame_reports_a_dead_link_rather_than_hanging_on_it(web, monkeypatch):
    _attach(monkeypatch, "emulator-5554")

    def boom(serial, max_long_edge=720):
        raise RuntimeError("device offline")

    monkeypatch.setattr("adbagent.web.server.screencap", boom)
    res = web.get("/api/device/frame")
    assert res.status_code == 502
    assert "device offline" in res.json()["detail"]


def test_screencap_downscales_a_png_into_a_jpeg(monkeypatch):
    """The raw `screencap -p` off a 1080x2400 phone is 1-3 MB, and this is
    polled every couple of seconds."""
    import io

    from PIL import Image

    from adbagent.web import server as srv

    raw = io.BytesIO()
    Image.new("RGB", (1080, 2400), (20, 30, 40)).save(raw, "PNG")

    seen = {}

    class Done:
        returncode, stdout, stderr = 0, raw.getvalue(), b""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return Done()

    monkeypatch.setattr(srv.subprocess, "run", fake_run)
    jpeg = srv.screencap("emulator-5554", max_long_edge=600)
    assert seen["argv"][1:] == ["-s", "emulator-5554", "exec-out", "screencap", "-p"]
    out = Image.open(io.BytesIO(jpeg))
    assert out.format == "JPEG"
    assert max(out.size) == 600
    assert out.size == (270, 600)   # aspect ratio kept, as the model needs


def test_screencap_reports_what_adb_said_when_it_fails(monkeypatch):
    from adbagent.web import server as srv

    class Failed:
        returncode, stdout, stderr = 1, b"", b"error: device offline\n"

    monkeypatch.setattr(srv.subprocess, "run", lambda argv, **kw: Failed())
    with pytest.raises(RuntimeError, match="device offline"):
        srv.screencap("emulator-5554")


def test_status_separates_a_configured_serial_from_an_attached_one(
        web, tmp_path, monkeypatch):
    """The header used to print the configured serial as though it were a phone."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"device": {"serial": "192.168.1.23:41207"}}),
                   encoding="utf-8")
    _attach(monkeypatch)
    st = web.get("/api/status").json()
    assert st["device_serial"] == "192.168.1.23:41207"
    assert st["devices_attached"] == []
    assert st["device_attached"] is False

    _attach(monkeypatch, "192.168.1.23:41207")
    st = web.get("/api/status").json()
    assert st["device_attached"] is True


def test_status_calls_one_unnamed_phone_attached(web, monkeypatch):
    """No serial configured and exactly one device is the everyday case, and it
    is the one adb resolves on its own."""
    _attach(monkeypatch, "emulator-5554")
    st = web.get("/api/status").json()
    assert st["device_serial"] == ""
    assert st["device_attached"] is True

    _attach(monkeypatch, "emulator-5554", "emulator-5556")
    assert web.get("/api/status").json()["device_attached"] is False


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
    # SIGINT/CTRL_BREAK_EVENT: the CLI catches it and puts the phone back as it was.
    assert EXPECTED_STOP_SIGNAL in spawned[0].signals
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


# ---------------------------------------------------------------------------
# the settings form against the schema it edits
# ---------------------------------------------------------------------------
#
# `CFG_SPEC` in app.js is a hand-written list, and nothing tied it to the
# dataclasses in `config.py`. So it drifted silently in both directions: a
# setting added to `RunConfig` never appeared on the form, and a setting renamed
# there would have left a field that saves a key the server rejects with a 400.
# The stall ladder is what made that visible -- a run started from the UI obeyed
# four settings the UI could not show.

APP_JS = Path(__file__).resolve().parent.parent / "adbagent" / "web" / "static" / "app.js"

#: Fields the form deliberately leaves out. Each needs a reason, so that
#: "not on the form" stays a decision rather than an oversight.
CFG_SPEC_OMISSIONS = {
    # Dump and connection mechanics. Wrong here breaks every run, and none of
    # it is tuned by eye.
    "device.max_depth", "device.compressed", "device.launch_timeout_s",
    "device.watchdog_s", "device.settle_interval_s", "device.launch_poll_s",
    # Transport tuning, not run behaviour.
    "llm.max_retries", "llm.temperature", "llm.read_timeout",
    # A ceiling on prompt rendering; nothing a user tunes from a form.
    "run.scratchpad_max_chars",
}


def cfg_spec_fields() -> dict:
    """`{section: [key, ...]}` as the settings form declares them.

    Depth-aware rather than a flat regex. The entries are nested three deep --
    ``[section, [[key, type, opts], ...]]`` -- and a regex for `["word"` also
    matches the option arrays, so `["reasoning_style", ["auto", ...]]` read as a
    field called `auto` and the real key went unchecked.
    """
    text = APP_JS.read_text(encoding="utf-8")
    start = text.index("const CFG_SPEC = [")
    body = re.sub(r"//[^\n]*", "", text[start + len("const CFG_SPEC = "):])

    out: dict = {}
    section = None
    depth = 0
    for i, ch in enumerate(body):
        if ch == "[":
            depth += 1
            opened = re.match(r'\[\s*"([^"]*)"', body[i:])
            if not opened:
                continue
            if depth == 2:                      # ["llm", [ ...
                section = opened.group(1)
                out.setdefault(section, [])
            elif depth == 4 and section:        # ...   ["provider", "text"],
                out[section].append(opened.group(1))
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
    assert out, "CFG_SPEC is no longer shaped the way this test parses it"
    return out


def test_the_settings_form_only_offers_real_config_keys():
    """A field naming a key the dataclass does not have saves as a 400."""
    from adbagent.config import Config
    cfg = Config()
    unknown = []
    for section, keys in cfg_spec_fields().items():
        holder = getattr(cfg, section, None)
        assert holder is not None, f"CFG_SPEC has no such section: {section}"
        unknown += [f"{section}.{k}" for k in keys if not hasattr(holder, k)]
    assert not unknown, f"the settings form edits keys that do not exist: {unknown}"


def test_every_config_key_is_on_the_settings_form_or_deliberately_not():
    from dataclasses import fields as dc_fields
    from adbagent.config import Config

    spec = cfg_spec_fields()
    missing = []
    for section in dc_fields(Config):
        holder = getattr(Config(), section.name)
        for f in dc_fields(holder):
            dotted = f"{section.name}.{f.name}"
            if f.name in spec.get(section.name, []):
                continue
            if dotted in CFG_SPEC_OMISSIONS:
                continue
            missing.append(dotted)
    assert not missing, (
        f"these settings exist but the UI cannot show them: {missing}. Add them "
        f"to CFG_SPEC in app.js, or to CFG_SPEC_OMISSIONS with a reason.")


# ---------------------------------------------------------------------------
# watch over the API
# ---------------------------------------------------------------------------

def _policy(tmp_path, text="reply only to people I follow") -> Path:
    p = tmp_path / "policy.md"
    p.write_text(text, encoding="utf-8")
    return p


def _configure_watch(tmp_path, **watch):
    """Point the app's config file at a policy and a ledger under tmp."""
    cfg = tmp_path / "config.json"
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data.setdefault("llm", {})["model"] = "fake/model"
    data.setdefault("watch", {}).update(watch)
    cfg.write_text(json.dumps(data), encoding="utf-8")


def test_watch_state_reports_defaults(web, tmp_path):
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    body = web.get("/api/watch").json()
    assert body["active"]["running"] is False
    assert body["defaults"]["interval_s"] == 45.0
    assert body["defaults"]["fail_closed"] is True
    assert body["policy_path"].endswith("policy.md")


def test_watch_needs_a_goal(web, tmp_path):
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    assert web.post("/api/watch", json={"goal": ""}).status_code == 400


def test_watch_needs_a_policy_file(web, tmp_path):
    _configure_watch(tmp_path, policy="")
    res = web.post("/api/watch", json={"goal": "watch dms"})
    assert res.status_code == 400
    assert "policy" in res.json()["detail"].lower()


def test_watch_rejects_a_policy_path_that_is_not_there(web, tmp_path):
    _configure_watch(tmp_path, policy=str(tmp_path / "nope.md"))
    res = web.post("/api/watch", json={"goal": "watch dms"})
    assert res.status_code == 400
    assert "no policy file" in res.json()["detail"]


def test_stopping_a_watch_that_is_not_running_is_a_conflict(web):
    assert web.post("/api/watch/stop").status_code == 409


def test_stopping_a_watch_that_exited_mid_signal_is_not_a_500(web, tmp_path,
                                                              monkeypatch):
    """The child can exit between poll() and send_signal(); the reaper in the
    _watch thread may have already collected it by then. On Python 3.10
    Popen.send_signal does not check returncode, so os.kill on the dead PID
    raises ProcessLookupError. That used to surface as a 500."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))

    class VanishedProc(FakeProc):
        # poll() says "still running" but send_signal says "gone" -- the race.
        def send_signal(self, sig):
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: VanishedProc(
                            argv, stay_running=True, **kw))
    web.post("/api/watch", json={"goal": "watch dms"})
    assert web.post("/api/watch/stop").status_code == 200


def test_stopping_a_watch_that_vanishes_after_timeout_is_not_a_500(
        web, tmp_path, monkeypatch):
    """The same race can hit proc.kill() after wait() times out: the child
    exits between the TimeoutExpired and the kill, and kill on the dead PID
    raises ProcessLookupError. The escalation happens on a thread now, so the
    request cannot see it either way -- what this holds is that a child which
    ignores the signal is still killed, and that the race is still swallowed."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    killed = threading.Event()

    class HungThenVanishedProc(FakeProc):
        # accept the SIGINT, but never exit -- so wait() times out -- and then
        # kill() raises because the PID was reaped between the two calls.
        def send_signal(self, sig):
            self.signals.append(sig)

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(self.argv, timeout)
            return 0

        def kill(self):
            killed.set()
            raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: HungThenVanishedProc(
                            argv, stay_running=True, **kw))
    web.post("/api/watch", json={"goal": "watch dms"})
    assert web.post("/api/watch/stop").status_code == 200
    assert killed.wait(timeout=5)


# -- stopping, which is its own phase ---------------------------------------
#
# A watch does not stop the moment it is asked to. It leaves the loop, restores
# the phone, and folds everything every pass learned about the app into that
# app's skill -- one model call, up to a minute, and none of it written to a run
# directory. The tests below hold the two halves of showing that: the request
# answers straight away, and what happens after it is streamed rather than
# silent.

def _sse_frames(body: str):
    """[(event name, payload)] from a finished SSE response."""
    frames = []
    for block in body.split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            frames.append((name, data))
    return frames


class WindingDownProc(FakeProc):
    """A child that answers a SIGINT by leaving its loop and working on."""

    LEARN_S = 0.4

    def __init__(self, argv, **kw):
        self._exit_at = None
        super().__init__(argv, stay_running=True, **kw)
        self.stdout = self._lines()

    def _lines(self):
        yield "  pass 1: success (0 repl(ies) sent so far, $0.0100)\n"
        while self._exit_at is None:
            time.sleep(0.02)
        yield "  stopped\n"                      # the loop is over here
        while time.time() < self._exit_at:       # ... and the skill is written
            time.sleep(0.02)
        yield "  skill 'instagram' updated from this run (3 workflows, 2 nuances)\n"

    def poll(self):
        return None if self._exit_at is None or time.time() < self._exit_at else 130

    def wait(self, timeout=None):
        deadline = time.time() + (timeout if timeout is not None else 60.0)
        while time.time() < deadline:
            if self.poll() is not None:
                return 130
            time.sleep(0.02)
        raise subprocess.TimeoutExpired(self.argv, timeout)

    def send_signal(self, sig):
        self.signals.append(sig)
        self._exit_at = time.time() + self.LEARN_S


def _winding_down_watch(web, tmp_path, monkeypatch, runs=None):
    """A started watch whose child takes `LEARN_S` to shut down."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    procs = []

    def fake_popen(argv, **kw):
        on_spawn = (lambda _argv: make_run_dir(runs, run_id="pass1")) \
            if runs is not None else None
        procs.append(WindingDownProc(argv, on_spawn=on_spawn, **kw))
        return procs[-1]

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen", fake_popen)
    assert web.post("/api/watch", json={"goal": "watch dms"}).status_code == 200
    # Let the pass line land, so the mark the shutdown is replayed from is a
    # real position in the output rather than the top of it.
    for _ in range(200):
        if web.get("/api/watch").json()["active"]["output_tail"]:
            break
        time.sleep(0.02)
    return procs


def test_stopping_a_watch_answers_before_the_shutdown_is_over(
        web, tmp_path, monkeypatch):
    """The browser says "stopping" when this request answers. Waiting here for
    the child meant waiting for its skill write-up -- three minutes are allowed
    for it -- and for all of that the page showed nothing at all: the button
    stayed live, the status line still read "watching", and a click that has no
    effect for a minute is a click that did nothing."""
    procs = _winding_down_watch(web, tmp_path, monkeypatch)

    t0 = time.monotonic()
    assert web.post("/api/watch/stop").status_code == 200
    assert time.monotonic() - t0 < WindingDownProc.LEARN_S / 2
    assert EXPECTED_STOP_SIGNAL in procs[0].signals    # sent, not merely scheduled

    # Still the phone's, and now saying which of the two states it is in.
    active = web.get("/api/watch").json()["active"]
    assert active["running"] is True
    assert active["stopping"] is True
    # Still the phone's, and the refusal says which of the two it is: "already
    # running" would read as though the stop had not been heard.
    res = web.post("/api/runs", json={"goal": "turn on wifi"})
    assert res.status_code == 409
    assert "stopping" in res.json()["detail"]
    procs[0].wait()


def test_a_signal_that_never_landed_still_gets_the_child_killed(tmp_path,
                                                                monkeypatch):
    """Windows reports an undelivered signal as an OSError, which is also what a
    child exiting a moment before the signal looks like. They need telling
    apart: one is over, the other is still running and still holding the phone.
    Treating both as over skipped the escalation, so the child ran on behind a
    UI stuck on "stopping" -- and the stop was latched, so clicking again did
    nothing either."""
    killed = threading.Event()

    class DeafProc(FakeProc):
        def send_signal(self, sig):
            raise OSError(22, "the signal could not be delivered")

        def wait(self, timeout=None):
            # Deaf to the signal, so the escalation is the only way out. Waits
            # the timeout out first, as Popen does -- raising instantly would
            # let the kill land before there was a phase to observe.
            if timeout is None:
                self._done.wait()
            elif not self._done.wait(timeout):
                raise subprocess.TimeoutExpired(self.argv, timeout)
            return self._returncode

        def kill(self):
            killed.set()
            self._returncode = 137
            self._done.set()

    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: DeafProc(argv, stay_running=True, **kw))
    manager = RunManager(tmp_path / "runs")
    manager.start("turn on wifi")
    assert manager.stop(timeout_s=0.5) is True
    # Still running, and saying so: the phase is not over just because the
    # signal bounced.
    assert manager.state()["stopping"] is True
    assert killed.wait(timeout=5)


def test_a_second_stop_does_not_signal_into_the_write_up(web, tmp_path,
                                                         monkeypatch):
    """Clicking stop twice used to send a second SIGINT. The first is caught by
    the loop's handler; the second lands after it, in the write-up -- so an
    impatient double click threw away everything the watch had learned."""
    procs = _winding_down_watch(web, tmp_path, monkeypatch)
    assert web.post("/api/watch/stop").status_code == 200
    assert web.post("/api/watch/stop").status_code == 200
    assert EXPECTED_STOP_SIGNAL in procs[0].signals
    procs[0].wait()


def test_the_stream_carries_the_shutdown_that_no_run_file_records(
        web, tmp_path, monkeypatch):
    """Between the stop and the exit, `events.jsonl` has nothing to say and the
    child is doing the learning. The feed sat still through all of it."""
    procs = _winding_down_watch(web, tmp_path, monkeypatch,
                                runs=tmp_path / "runs")
    assert web.post("/api/watch/stop").status_code == 200

    frames = _sse_frames(web.get("/api/watch/stream").text)
    assert any(name == "state" and data["stopping"] for name, data in frames)
    shutdown = [data["line"] for name, data in frames if name == "output"]
    assert "  stopped" in shutdown
    # The one line that says whether the learning happened is the last thing the
    # child prints, after the stream has seen it stop running.
    assert any("skill 'instagram' updated" in line for line in shutdown)
    # Replayed from the signal, not from the top: the pass above it already has
    # its own cards, and repeating the session's output under "stopping" would
    # bury the shutdown in it.
    assert not any("pass 1: success" in line for line in shutdown)
    assert frames[-1][0] == "end"
    procs[0].wait()


def test_the_server_takes_a_watch_down_with_it(tmp_path, monkeypatch):
    """Ctrl+C in the console does not reach the children on Windows: each one is
    spawned into its own process group -- it has to be, or signalling one would
    take the server down too -- and a new group has the console's Ctrl+C
    disabled. Without this the server exits and the watch keeps going: still
    driving the phone, still replying to people, with the only thing that could
    stop it now gone."""
    # Its own app rather than the `web` fixture's, because the lifespan only runs
    # when the client is used as a context manager.
    (tmp_path / "skills").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    procs = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: procs.append(
                            FakeProc(argv, stay_running=True, **kw)) or procs[-1])
    app = create_app(artifacts_dir=str(tmp_path / "runs"),
                     skills_dir=str(tmp_path / "skills"),
                     config_path=str(tmp_path / "config.json"))
    # As a context manager, so startup and shutdown actually run.
    with TestClient(app) as client:
        assert client.post("/api/watch",
                           json={"goal": "watch dms"}).status_code == 200
        assert client.get("/api/watch").json()["active"]["running"] is True
    assert EXPECTED_STOP_SIGNAL in procs[0].signals   # asked, so the phone is restored


def test_shutting_down_ends_the_live_streams(tmp_path, monkeypatch):
    """The stream is a plain generator on a thread, so nothing can cancel it. It
    has to notice the server leaving and end, or the shutdown tears the response
    down underneath it -- a CancelledError out of the middle of a
    StreamingResponse, which is what a Ctrl+C used to print to the console."""
    runs = tmp_path / "runs"
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(
                            argv, stay_running=True,
                            on_spawn=lambda _a: make_run_dir(runs, "live9"),
                            **kw))
    manager = RunManager(runs)
    manager.start("turn on wifi")
    leaving = threading.Event()
    frames = []

    def drain():
        for frame in _event_stream(manager, leaving):
            frames.append(frame)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    for _ in range(100):                       # let it get going
        if any("run_start" in f for f in frames):
            break
        time.sleep(0.02)
    leaving.set()
    reader.join(timeout=5)
    assert not reader.is_alive()               # ended itself, was not cancelled
    assert "the server is shutting down" in frames[-1]
    manager.stop()


def test_a_child_is_spawned_unbuffered(web, monkeypatch):
    """Python block-buffers stdout when it is a pipe. Everything a child says
    outside its run files -- the skill written after the loop, a refusal before
    there is a run at all -- would otherwise sit in that buffer until it exited,
    which is exactly too late for any of it to be news."""
    seen = {}
    monkeypatch.setattr(
        "adbagent.web.runner.subprocess.Popen",
        lambda argv, **kw: seen.update(kw) or FakeProc(argv, stay_running=True,
                                                       **kw))
    assert web.post("/api/runs", json={"goal": "turn on wifi"}).status_code == 200
    assert seen["env"]["PYTHONUNBUFFERED"] == "1"
    web.post("/api/runs/stop")


def test_watch_lifecycle_and_argv(web, tmp_path, monkeypatch):
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])

    res = web.post("/api/watch", json={
        "goal": "watch my instagram dms", "draft": True,
        "interval_s": 30, "max_steps": 12, "replies_per_hour": 5,
        "replies_per_conversation": 1, "cooldown_s": 300, "usd_per_hour": 0.5,
    })
    assert res.status_code == 200
    argv = spawned[0].argv
    assert argv[3] == "watch" or "watch" in argv
    assert "watch my instagram dms" in argv
    assert "--draft" in argv
    for flag, value in (("--interval", "30.0"), ("--steps-per-pass", "12"),
                        ("--replies-per-hour", "5"),
                        ("--replies-per-conversation", "1"),
                        ("--cooldown", "300.0"), ("--usd-per-hour", "0.5")):
        assert flag in argv, flag
        assert argv[argv.index(flag) + 1] == value, flag

    assert web.get("/api/watch").json()["active"]["running"] is True
    # One watch at a time.
    assert web.post("/api/watch", json={"goal": "again"}).status_code == 409
    assert web.post("/api/watch/stop").status_code == 200
    assert EXPECTED_STOP_SIGNAL in spawned[0].signals   # so the phone is restored


def test_live_watch_omits_the_draft_flag(web, tmp_path, monkeypatch):
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    web.post("/api/watch", json={"goal": "watch dms", "draft": False})
    assert "--draft" not in spawned[0].argv


def test_watch_no_learn_reaches_the_argv(web, tmp_path, monkeypatch):
    """The 'don't learn' checkbox: the watch's skill write-up on stop is
    opt-out, the same as a run's."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    web.post("/api/watch", json={"goal": "watch dms", "no_learn": True})
    assert "--no-learn" in spawned[0].argv


def test_a_watch_and_a_run_refuse_each_other(web, tmp_path, monkeypatch):
    """One phone. A watch quietly displaced by a run is no longer watching."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))

    assert web.post("/api/watch", json={"goal": "watch dms"}).status_code == 200
    res = web.post("/api/runs", json={"goal": "turn on wifi"})
    assert res.status_code == 409
    assert "watch" in res.json()["detail"].lower()
    # The screenshot, dump and apps endpoints hold the same line: they open a
    # Device session, which resets animations and rotation under the watch.
    for path in ("/api/devices/screenshot", "/api/dump", "/api/apps"):
        assert web.get(path).status_code == 409, path


def test_watch_is_reported_on_status(web, tmp_path, monkeypatch):
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))
    web.post("/api/watch", json={"goal": "watch dms", "draft": True})
    st = web.get("/api/status").json()
    assert st["watch"]["running"] is True
    assert st["watch"]["draft"] is True
    assert st["watch"]["goal"] == "watch dms"


# -- the policy file --------------------------------------------------------

def test_policy_round_trips(web, tmp_path):
    path = tmp_path / "policy.md"
    _configure_watch(tmp_path, policy=str(path))
    # Not written yet: an empty editor, not a 404.
    body = web.get("/api/watch/policy").json()
    assert body["exists"] is False and body["text"] == ""

    res = web.put("/api/watch/policy", json={"text": "be brief"})
    assert res.status_code == 200
    assert path.read_text(encoding="utf-8") == "be brief"
    assert web.get("/api/watch/policy").json()["text"] == "be brief"


def test_an_empty_policy_is_refused(web, tmp_path):
    _configure_watch(tmp_path, policy=str(tmp_path / "policy.md"))
    assert web.put("/api/watch/policy", json={"text": "  \n"}).status_code == 400


def test_policy_cannot_be_edited_under_a_running_watch(web, tmp_path, monkeypatch):
    """The child read the file at startup; a save now lands at no known moment."""
    _configure_watch(tmp_path, policy=str(_policy(tmp_path)))
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: FakeProc(argv, stay_running=True, **kw))
    web.post("/api/watch", json={"goal": "watch dms"})
    res = web.put("/api/watch/policy", json={"text": "something else"})
    assert res.status_code == 409
    assert "stop the watch" in res.json()["detail"]


def test_policy_with_no_path_anywhere_is_a_bad_request(web, tmp_path):
    _configure_watch(tmp_path, policy="", policies_dir="")
    assert web.put("/api/watch/policy", json={"text": "hi"}).status_code == 400


# -- several policies, each with the goal it was written for -----------------

def _write_policy(tmp_path, name, goal="", body="be brief") -> Path:
    """A policy in the tmp policies directory, in the file format on disk."""
    from adbagent import policies as policymod
    d = tmp_path / "policies"
    d.mkdir(exist_ok=True)
    path = d / f"{name}.md"
    path.write_text(policymod.with_front_matter({"goal": goal}, body),
                    encoding="utf-8")
    return path


def test_policies_are_listed_with_the_goal_each_was_written_for(web, tmp_path):
    """The goal travels with the row: it is what the picker shows, and what
    choosing a policy fills the goal box in with."""
    _write_policy(tmp_path, "hinge", goal="work through the feed")
    _write_policy(tmp_path, "insta", goal="watch my dms")
    _configure_watch(tmp_path, policy=str(tmp_path / "policies" / "hinge.md"))

    body = web.get("/api/watch/policies").json()
    rows = {p["name"]: p for p in body["policies"]}
    assert set(rows) == {"hinge", "insta"}
    assert rows["hinge"]["goal"] == "work through the feed"
    assert rows["insta"]["goal"] == "watch my dms"
    # Which one config names, said by the server: the two may be spelled
    # differently (relative here, absolute there) and name one file.
    assert rows["hinge"]["current"] is True and rows["insta"]["current"] is False


def test_the_editor_gets_the_instructions_without_the_front_matter(web, tmp_path):
    path = _write_policy(tmp_path, "hinge", goal="work the feed",
                         body="# Hinge\n\n- be brief")
    _configure_watch(tmp_path, policy=str(path))
    body = web.get("/api/watch/policy").json()
    assert body["goal"] == "work the feed"
    assert body["text"] == "# Hinge\n\n- be brief"
    assert "goal:" not in body["text"]


def test_a_named_policy_can_be_read_and_written(web, tmp_path):
    """Several policies means the browser sends which one, on every call."""
    one = _write_policy(tmp_path, "one", goal="goal one")
    two = _write_policy(tmp_path, "two", goal="goal two")
    _configure_watch(tmp_path, policy=str(one))

    assert web.get("/api/watch/policy",
                   params={"path": str(two)}).json()["goal"] == "goal two"
    res = web.put("/api/watch/policy",
                  json={"path": str(two), "text": "say less",
                        "goal": "goal two, revised"})
    assert res.status_code == 200
    from adbagent import policies as policymod
    reread = policymod.read(two)
    assert reread.goal == "goal two, revised" and reread.body == "say less"
    # And the other one is untouched.
    assert policymod.read(one).goal == "goal one"


def test_saving_a_policy_outside_the_directory_is_refused(web, tmp_path):
    """The browser sends a path now, so the endpoint cannot write to whatever
    arrives: a save lands in the policies directory or on the configured file."""
    _configure_watch(tmp_path, policy=str(_write_policy(tmp_path, "hinge")))
    res = web.put("/api/watch/policy",
                  json={"path": str(tmp_path / "elsewhere.md"), "text": "hi"})
    assert res.status_code == 400
    assert "outside the policies directory" in res.json()["detail"]
    assert not (tmp_path / "elsewhere.md").exists()


def test_a_new_policy_starts_from_the_goal_in_the_box(web, tmp_path):
    _configure_watch(tmp_path)
    res = web.post("/api/watch/policies",
                   json={"name": "insta", "goal": "watch my dms"})
    assert res.status_code == 200
    created = res.json()
    assert created["name"] == "insta" and created["goal"] == "watch my dms"
    assert Path(created["path"]) == tmp_path / "policies" / "insta.md"
    assert created["body"], "a new policy starts with something to fill in"
    assert web.get("/api/watch/policy",
                   params={"path": created["path"]}).json()["goal"] == "watch my dms"


def test_a_new_policy_never_overwrites_one(web, tmp_path):
    _write_policy(tmp_path, "hinge", goal="the original")
    _configure_watch(tmp_path)
    res = web.post("/api/watch/policies", json={"name": "hinge"})
    assert res.status_code == 409
    from adbagent import policies as policymod
    assert policymod.read(tmp_path / "policies" / "hinge.md").goal == "the original"


def test_a_new_policy_cannot_escape_the_directory(web, tmp_path):
    _configure_watch(tmp_path)
    res = web.post("/api/watch/policies", json={"name": "../escape"})
    # Either the name is scrubbed into the directory or it is refused outright;
    # what must not happen is a file above it.
    if res.status_code == 200:
        assert Path(res.json()["path"]).parent == tmp_path / "policies"
    else:
        assert res.status_code == 400
    assert not (tmp_path / "escape.md").exists()


def test_a_watch_started_with_no_goal_uses_the_policys_own(web, tmp_path,
                                                            monkeypatch):
    """The pairing, at the point it matters: these instructions are only correct
    under that goal, so the goal comes from the policy rather than the box."""
    path = _write_policy(tmp_path, "hinge", goal="work through the feed")
    _configure_watch(tmp_path, policy=str(path))
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    assert web.post("/api/watch", json={"goal": ""}).status_code == 200
    assert "work through the feed" in spawned[0].argv


def test_a_watch_can_name_the_policy_to_start(web, tmp_path, monkeypatch):
    _configure_watch(tmp_path, policy=str(_write_policy(tmp_path, "hinge")))
    other = _write_policy(tmp_path, "insta", goal="watch my dms")
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    res = web.post("/api/watch", json={"goal": "", "policy": str(other)})
    assert res.status_code == 200
    assert str(other) in spawned[0].argv
    assert "watch my dms" in spawned[0].argv


# -- the reply ledger -------------------------------------------------------

def test_ledger_is_empty_before_anything_is_sent(web, tmp_path):
    _configure_watch(tmp_path, ledger=str(tmp_path / "replies.jsonl"))
    body = web.get("/api/watch/ledger").json()
    assert body["exists"] is False
    assert body["total"] == 0 and body["threads"] == []


def test_ledger_lists_threads_newest_first(web, tmp_path):
    from adbagent.ledger import ReplyLedger, content_digest, thread_key
    path = tmp_path / "replies.jsonl"
    _configure_watch(tmp_path, ledger=str(path))
    led = ReplyLedger(path)
    led.record_attempt(thread_key("khushi"), content_digest(["hey"]),
                       preview="khushi: hey", at=1000)
    led.record_confirmed(thread_key("khushi"), content_digest(["hey", "hi"]),
                         preview="khushi: hi", at=1001)
    led.record_attempt(thread_key("shreya"), content_digest(["yo"]),
                       preview="shreya: yo", at=2000)

    body = web.get("/api/watch/ledger").json()
    assert body["exists"] is True
    assert body["total"] == 2                       # attempts, not confirmations
    assert [t["preview"] for t in body["threads"]] == ["shreya: yo", "khushi: hi"]
    assert body["threads"][0]["confirmed"] is False  # in doubt, and shown as such
    assert body["threads"][1]["confirmed"] is True


# ---------------------------------------------------------------------------
# apps / dump / doctor
# ---------------------------------------------------------------------------

class FakeDev:
    """Just the surface these three endpoints touch."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def list_apps(self, query="", third_party_only=False):
        pool = ["com.instagram.android", "com.whatsapp", "com.android.settings"]
        if third_party_only:
            pool = [p for p in pool if not p.startswith("com.android.")]
        return [p for p in pool if query in p]

    def observe(self, settle=False):
        from adbagent.fingerprint import attach
        from adbagent.screen import parse
        from . import xmlgen as X
        return attach(parse(X.settings_screen(), width=X.W, height=X.H))


def test_apps_lists_and_filters(web, monkeypatch):
    monkeypatch.setattr("adbagent.device.Device", FakeDev)
    # The endpoint lists everything by default; the form's checkbox is what asks
    # for third-party only, and it passes the flag explicitly.
    body = web.get("/api/apps").json()
    assert body["count"] == 3 and "com.android.settings" in body["apps"]
    body = web.get("/api/apps?third_party=true").json()
    assert body["count"] == 2 and "com.android.settings" not in body["apps"]
    body = web.get("/api/apps?search=insta").json()
    assert body["apps"] == ["com.instagram.android"]


def test_dump_returns_what_the_model_would_see(web, monkeypatch):
    monkeypatch.setattr("adbagent.device.Device", FakeDev)
    body = web.get("/api/dump").json()
    assert body["package"] == "com.android.settings"
    assert body["elements"] > 0 and body["nodes"] >= body["elements"]
    assert body["skeleton_id"]
    assert "#" in body["rendered"]        # the indices the model is told to aim at
    assert body["xml"] == ""              # not asked for


def test_dump_can_include_raw_xml(web, monkeypatch):
    monkeypatch.setattr("adbagent.device.Device", FakeDev)
    body = web.get("/api/dump?raw=true").json()
    assert body["xml"].startswith("<?xml")


def test_dump_reports_a_device_failure_as_a_bad_gateway(web, monkeypatch):
    class Boom(FakeDev):
        def __enter__(self):
            raise RuntimeError("adb is not there")
    monkeypatch.setattr("adbagent.device.Device", Boom)
    assert web.get("/api/dump").status_code == 502


def test_doctor_shells_out_to_the_real_command(web, monkeypatch):
    seen = {}

    class Done:
        stdout, stderr, returncode = "adbagent 0.1.0\nEnvironment\n", "", 0

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return Done()

    monkeypatch.setattr("adbagent.web.server.subprocess.run", fake_run)
    body = web.get("/api/doctor").json()
    assert body["ok"] is True
    assert "adbagent 0.1.0" in body["text"]
    assert seen["argv"][2:4] == ["adbagent", "doctor"]


# ---------------------------------------------------------------------------
# the run oracle, which had no UI at all
# ---------------------------------------------------------------------------

def test_assertions_reach_the_cli(web, monkeypatch):
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    web.post("/api/runs", json={
        "goal": "turn on airplane mode",
        "assert_shell": "settings get global airplane_mode_on",
        "assert_equals": "1",
    })
    argv = spawned[0].argv
    assert argv[argv.index("--assert-shell") + 1] == \
        "settings get global airplane_mode_on"
    assert argv[argv.index("--assert-equals") + 1] == "1"


def test_no_assertion_flags_when_none_given(web, monkeypatch):
    spawned = []
    monkeypatch.setattr("adbagent.web.runner.subprocess.Popen",
                        lambda argv, **kw: spawned.append(
                            FakeProc(argv, stay_running=True, **kw))
                        or spawned[-1])
    web.post("/api/runs", json={"goal": "turn on wifi"})
    assert "--assert-shell" not in spawned[0].argv
    assert "--assert-text" not in spawned[0].argv


# ---------------------------------------------------------------------------
# the static pair: every id the script reaches for must exist in the page
# ---------------------------------------------------------------------------

INDEX_HTML = Path(__file__).resolve().parents[1] / \
    "adbagent/web/static/index.html"

#: Counter ids `makeLive` derives that a surface deliberately does not have.
#: `paintCounters` guards on `iterWrap`, because a `skills generate` tour is one
#: pass by definition and has no iteration to count.
DERIVED_ID_OMISSIONS = {"gc-iter", "gc-iter-wrap"}

#: Ids the config form mints at load time -- `cfg-<section>-<key>` from
#: `CFG_SPEC`, plus what hangs off one -- so the page cannot ship them and the
#: form rebuilds them on every load. Only those a literal `$()` reaches for need
#: declaring here; the rest are built from the spec and never spelled out.
#:
#: `standing-resumable` is the same shape: the standing strip is painted from
#: the run list, and the resumable cell only exists when there is something to
#: resume -- a page that shipped the id would be claiming a button it has no
#: number for. Every `$()` that reaches for it is guarded on the null.
GENERATED_ID_OMISSIONS = {"cfg-llm-vision_in_decider-auto", "standing-resumable"}


def html_ids() -> set:
    return set(re.findall(r'id="([^"]+)"', INDEX_HTML.read_text(encoding="utf-8")))


def test_every_scripted_id_exists_in_the_page():
    """A typo'd `$("...")` is a null, and the first line to touch it throws.

    Cheap to check and impossible to notice by reading: the failure shows up as
    one dead button, in one tab, at the moment somebody needs it.
    """
    js = APP_JS.read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("([^"]+)"\)', js))
    missing = sorted(referenced - html_ids() - GENERATED_ID_OMISSIONS)
    assert not missing, f"app.js reaches for ids the page does not have: {missing}"


def test_every_derived_counter_id_exists_or_is_declared():
    """The `makeLive(prefix)` x `el(suffix)` grid, which no grep over literals
    would catch."""
    js = APP_JS.read_text(encoding="utf-8")
    prefixes = re.findall(r'makeLive\("([^"]*)"', js)
    suffixes = set(re.findall(r'el\("([^"]+)"\)', js))
    assert prefixes and suffixes, "the makeLive pattern moved; update this test"
    derived = {p + s for p in prefixes for s in suffixes}
    missing = sorted(derived - html_ids() - DERIVED_ID_OMISSIONS)
    assert not missing, (
        f"these counter ids are derived but absent: {missing}. Add them to the "
        f"page, or to DERIVED_ID_OMISSIONS with the reason.")


def test_every_tab_button_has_a_section_and_a_loader():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    tabs = set(re.findall(r'data-tab="([^"]+)"', html))
    sections = {i[4:] for i in html_ids() if i.startswith("tab-")}
    assert tabs == sections, f"tab buttons and sections disagree: {tabs ^ sections}"
    body = js[js.index("const tabLoaders = {"):]
    body = body[:body.index("};")]
    loaders = set(re.findall(r'^\s*(\w+):', body, re.M))
    assert tabs == loaders, (
        f"every tab needs a loader (a missing one throws on first click): "
        f"{tabs ^ loaders}")


def test_every_setup_pane_has_a_section_and_a_loader():
    """Setup's three panes are not tabs -- they are one tab's contents -- but
    they wire up the same way and break the same way."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    panes = set(re.findall(r'data-pane="([^"]+)"', html))
    sections = {i[5:] for i in html_ids() if i.startswith("pane-")}
    assert panes == sections, f"pane buttons and sections disagree: {panes ^ sections}"
    body = js[js.index("const setupLoaders = {"):]
    loaders = set(re.findall(r"(\w+):", body[:body.index("};")]))
    assert panes == loaders, f"every pane needs a loader: {panes ^ loaders}"


def test_every_folded_block_is_actually_hidden_when_folded():
    """A closed `<details>` hides its content through the UA stylesheet, and an
    author `display` on a direct child overrides it.

    Four folds had one — `.opts`, `.optgroups`, `.readout-rest` and
    `.panel-head` are all `display: flex` — and so rendered open whatever their
    marker said: the success assertion showed its three fields, the config's
    Advanced pane its search box, the run detail its cost table. Silent by
    construction, since the markup and the script are both right and only the
    pixels are wrong.

    Stated once for every fold rather than per class, so the next block given a
    `display` cannot reintroduce it.
    """
    css = (Path(__file__).resolve().parents[1]
           / "adbagent/web/static/style.css").read_text(encoding="utf-8")
    assert re.search(
        r"details:not\(\[open\]\)\s*>\s*\*:not\(summary\)\s*\{[^}]*"
        r"display:\s*none", css), (
        "the rule that makes a closed <details> actually fold is missing; "
        "any content with a display of its own will render while closed")


def test_the_density_toggle_only_offers_the_two_densities():
    """`body[data-density="story"] .trace-only` is what hides the trace; a third
    value on a button would simply show everything, silently."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = (Path(__file__).resolve().parents[1]
           / "adbagent/web/static/style.css").read_text(encoding="utf-8")
    offered = set(re.findall(r'data-density="([^"]+)"', html))
    assert offered == {"story", "trace"}, f"unexpected densities: {offered}"
    assert 'body[data-density="story"] .trace-only' in css
    # And the default the page ships in, so a reader who never touches it gets
    # the quiet view rather than the full trace.
    assert '<body data-density="story">' in html
