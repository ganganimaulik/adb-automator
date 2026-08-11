"""Live reload: what the watcher sees, and what the UI does about it.

No phone, no server: the watcher is driven a tick at a time, the stream is a
generator, and the restart is a callable that records that it was asked.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading

import pytest
from fastapi.testclient import TestClient

from adbagent import cli
from adbagent.web.reload import (ASSET_SUFFIXES, CODE_SUFFIXES, LiveReload,
                                 for_ui, in_source_checkout)
from adbagent.web.server import _reload_stream, create_app


def touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def kinds(changes):
    return sorted({c.kind for c in changes})


# ---------------------------------------------------------------------------
# What the watcher sees
# ---------------------------------------------------------------------------

def test_the_first_pass_is_the_tree_and_not_a_change_to_it(tmp_path):
    """Otherwise every page open reloads itself once: the watcher starts with
    nothing on record, so the whole source tree reads as new."""
    touch(tmp_path / "agent.py")
    reloader = LiveReload().watch("code", tmp_path, CODE_SUFFIXES)
    assert reloader.poll() == []
    assert reloader.version == 0


def test_an_edited_file_is_reported_under_the_kind_that_covers_it(tmp_path):
    touch(tmp_path / "agent.py")
    reloader = LiveReload().watch("code", tmp_path, CODE_SUFFIXES)
    reloader.poll()

    touch(tmp_path / "agent.py", "changed")
    changes = reloader.poll()
    assert kinds(changes) == ["code"]
    assert changes[0].as_frame()["paths"] == ["agent.py"]
    assert reloader.version == changes[0].version > 0


def test_static_files_are_assets_and_not_code(tmp_path):
    """`static/` lives inside the package, so both watches walk the same tree.
    Which one claims a file decides whether the page reloads or the server
    restarts -- and restarting to pick up a CSS edit throws away every run the
    server is following."""
    pkg = tmp_path / "adbagent"
    touch(pkg / "agent.py")
    touch(pkg / "web" / "static" / "app.js")
    reloader = (LiveReload()
                .watch("code", pkg, CODE_SUFFIXES)
                .watch("assets", pkg / "web" / "static", ASSET_SUFFIXES))
    reloader.poll()

    touch(pkg / "web" / "static" / "app.js", "edited")
    assert kinds(reloader.poll()) == ["assets"]

    touch(pkg / "agent.py", "edited")
    assert kinds(reloader.poll()) == ["code"]


def test_a_compiled_module_is_not_a_change(tmp_path):
    """`__pycache__` is rewritten by the import that follows every restart, so
    a watcher that counted it would restart the server forever."""
    pkg = tmp_path / "adbagent"
    touch(pkg / "agent.py")
    reloader = LiveReload().watch("code", pkg, CODE_SUFFIXES)
    reloader.poll()

    touch(pkg / "__pycache__" / "agent.cpython-310.pyc")
    assert reloader.poll() == []


def test_a_file_that_appears_or_disappears_is_a_change(tmp_path):
    """A policy is normally written after the UI is already open, and a skill
    can be deleted from under the list showing it."""
    skills = tmp_path / "skills"
    touch(skills / "tinder.json", "{}")
    reloader = LiveReload().watch("skills", skills, {".json"})
    reloader.poll()

    touch(skills / "bumble.json", "{}")
    assert kinds(reloader.poll()) == ["skills"]

    (skills / "bumble.json").unlink()
    changes = reloader.poll()
    assert kinds(changes) == ["skills"]
    assert changes[0].as_frame()["paths"] == ["bumble.json"]


def test_a_watch_on_a_path_that_is_not_there_yet_is_not_an_error(tmp_path):
    reloader = LiveReload().watch("policy", tmp_path / "nope.md")
    assert reloader.poll() == []
    touch(tmp_path / "nope.md", "reply politely")
    assert kinds(reloader.poll()) == ["policy"]


# ---------------------------------------------------------------------------
# The restart, and what holds it
# ---------------------------------------------------------------------------

def test_a_code_change_asks_for_a_restart(tmp_path):
    asked = []
    reloader = LiveReload(on_restart=lambda: asked.append(True), grace_s=0)
    reloader.watch("code", tmp_path, CODE_SUFFIXES)
    touch(tmp_path / "agent.py")
    reloader.poll()

    touch(tmp_path / "agent.py", "edited")
    changes = reloader.step()
    assert asked == [True]
    assert [c.kind for c in changes] == ["code", "restart"]
    assert changes[-1].note == ""          # "" means it is going now


def test_a_restart_waits_while_an_agent_is_driving_the_phone(tmp_path):
    """The run and watch children are in their own process groups so that
    signalling one does not take the server down. That is also why a re-exec
    would orphan them: they survive it, still tapping the screen, and the server
    that comes back has no handle on them."""
    asked, why = [], ["a run is already in progress"]
    reloader = LiveReload(busy=lambda: why[0],
                          on_restart=lambda: asked.append(True), grace_s=0)
    reloader.watch("code", tmp_path, CODE_SUFFIXES)
    touch(tmp_path / "agent.py")
    reloader.poll()

    touch(tmp_path / "agent.py", "edited")
    changes = reloader.step()
    assert asked == []
    assert changes[-1].kind == "restart"
    assert changes[-1].note == "a run is already in progress"   # and says why

    # Still waiting, and not saying so twice a second.
    assert reloader.step() == []
    assert asked == []

    why[0] = ""
    changes = reloader.step()
    assert asked == [True]
    assert changes[-1].note == ""


def test_a_change_to_anything_else_never_restarts_the_server(tmp_path):
    """A skill or a policy is read off disk per request. Restarting for one
    would stop whatever is running to apply a change that needed no restart."""
    asked = []
    reloader = LiveReload(on_restart=lambda: asked.append(True), grace_s=0)
    reloader.watch("skills", tmp_path, {".json"})
    touch(tmp_path / "tinder.json", "{}")
    reloader.poll()

    touch(tmp_path / "tinder.json", '{"name": "tinder"}')
    assert kinds(reloader.step()) == ["skills"]
    assert asked == []


def test_the_thread_starts_primed(tmp_path):
    """`start()` polls before the thread does, so files that were already there
    when the server came up are not a change the moment it is running."""
    touch(tmp_path / "agent.py")
    reloader = LiveReload(poll_s=0.01).watch("code", tmp_path, CODE_SUFFIXES)
    reloader.start()
    try:
        assert reloader.version == 0
    finally:
        reloader.stop()


# ---------------------------------------------------------------------------
# What reaches the browser
# ---------------------------------------------------------------------------

def frames(text):
    return [(chunk.splitlines()[0].removeprefix("event: "),
             json.loads(chunk.splitlines()[1].removeprefix("data: ")))
            for chunk in text.strip().split("\n\n") if chunk.strip()]


def test_the_stream_names_the_process_before_anything_else(tmp_path):
    """A page cannot tell a reconnect from a restart by itself: EventSource
    reconnects on its own either way. The boot id is the whole of the
    difference, so it is the first thing said."""
    reloader = LiveReload()
    stream = _reload_stream(reloader, threading.Event())
    event, data = frames(next(stream))[0]
    assert event == "hello"
    assert data["boot"] == reloader.boot
    assert data["restarts"] is False        # no supervisor: nothing will restart


def test_the_stream_carries_each_change_once(tmp_path):
    touch(tmp_path / "app.js")
    reloader = LiveReload().watch("assets", tmp_path, ASSET_SUFFIXES)
    reloader.poll()
    shutting_down = threading.Event()
    stream = _reload_stream(reloader, shutting_down)
    next(stream)                            # hello

    touch(tmp_path / "app.js", "edited")
    reloader.poll()
    event, data = frames(next(stream))[0]
    assert event == "reload"
    assert data["kind"] == "assets" and data["paths"] == ["app.js"]

    # And is not replayed to a page that already has it.
    shutting_down.set()
    assert frames(next(stream))[0][0] == "end"


def test_the_stream_ends_itself_when_the_server_is_going(tmp_path):
    """It is a plain generator on a thread, so it cannot be cancelled. Without
    the flag the shutdown waits for a stream that is waiting for the shutdown."""
    shutting_down = threading.Event()
    shutting_down.set()
    stream = _reload_stream(LiveReload(), shutting_down)
    assert frames(next(stream))[0][0] == "hello"
    assert frames(next(stream))[0][1] == {"reason": "server going away"}
    with pytest.raises(StopIteration):
        next(stream)


# ---------------------------------------------------------------------------
# The server surface
# ---------------------------------------------------------------------------

def test_the_page_is_told_whether_there_is_a_stream_to_open(tmp_path):
    """A page that guessed would reconnect to a 404 every few seconds for as
    long as it stayed open."""
    with TestClient(create_app(artifacts_dir=str(tmp_path))) as client:
        assert client.get("/api/status").json()["live_reload"] is False
        assert client.get("/api/dev/reload").status_code == 404

    app = create_app(artifacts_dir=str(tmp_path), live_reload=LiveReload())
    with TestClient(app) as client:
        assert client.get("/api/status").json()["live_reload"] is True


def test_static_files_are_not_kept_while_reload_is_on(tmp_path):
    """They carry no validator, so a heuristically cached app.js would survive
    the reload that editing app.js triggered."""
    with TestClient(create_app(artifacts_dir=str(tmp_path))) as client:
        assert "cache-control" not in client.get("/static/app.js").headers

    app = create_app(artifacts_dir=str(tmp_path), live_reload=LiveReload())
    with TestClient(app) as client:
        assert client.get("/static/app.js").headers["cache-control"] == "no-store"
        assert client.get("/").headers["cache-control"] == "no-store"


def test_the_phone_is_asked_through_the_same_answer_everyone_else_gets(tmp_path):
    app = create_app(artifacts_dir=str(tmp_path), live_reload=LiveReload())
    assert app.state.phone_busy() == ""      # nothing running in a fresh app


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def ui_args(tmp_path, **kw):
    return argparse.Namespace(host="127.0.0.1", port=1, config=None,
                              artifacts_dir=str(tmp_path / "runs"), **kw)


class FakeConfig:
    made = {}

    def __init__(self, app, **kw):
        FakeConfig.made = {"app": app, "kw": kw}


def fake_uvicorn(monkeypatch, run):
    import uvicorn

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def handle_exit(self, sig, frame):
            pass

        def run(self):
            run(self)

    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)


def test_the_ui_restarts_itself_when_code_changes(monkeypatch, tmp_path):
    """The process imported every module at startup and will not import them
    again, so an edited module is only picked up by starting over."""
    reloader = LiveReload(grace_s=0)
    monkeypatch.setattr(cli, "_ui_live_reload", lambda args, out: reloader)
    restarted = []
    monkeypatch.setattr(cli, "_reexec", lambda: restarted.append(True) or 0)
    # The watcher thread would call this; the fake server stands in for the tick.
    fake_uvicorn(monkeypatch, lambda server: reloader.on_restart())

    assert cli.cmd_ui(ui_args(tmp_path)) == 0
    assert restarted == [True]
    app = FakeConfig.made["app"]
    assert app.state.shutting_down.is_set()    # streams told, as on Ctrl+C
    assert reloader.busy == app.state.phone_busy


def test_a_ui_that_simply_exits_does_not_come_back(monkeypatch, tmp_path):
    """Ctrl+C is not a reload. Re-execing on every exit would make the server
    unkillable from the terminal it was started in."""
    reloader = LiveReload(grace_s=0)
    monkeypatch.setattr(cli, "_ui_live_reload", lambda args, out: reloader)
    restarted = []
    monkeypatch.setattr(cli, "_reexec", lambda: restarted.append(True) or 0)
    fake_uvicorn(monkeypatch, lambda server: None)

    assert cli.cmd_ui(ui_args(tmp_path)) == 0
    assert restarted == []


def test_no_reload_watches_nothing(monkeypatch, tmp_path):
    fake_uvicorn(monkeypatch, lambda server: None)
    assert cli.cmd_ui(ui_args(tmp_path, reload=False)) == 0
    assert FakeConfig.made["app"].state.live_reload is None


def test_reload_is_on_by_default_in_a_checkout_and_off_outside_one(tmp_path):
    """Files in the repo are being edited; files in site-packages are not, and
    re-execing a server nobody is working on is a surprise."""
    package_dir = tmp_path / "site-packages" / "adbagent"
    package_dir.mkdir(parents=True)
    assert in_source_checkout(package_dir) is False
    (tmp_path / "site-packages" / ".git").mkdir()
    assert in_source_checkout(package_dir) is True


def test_the_ui_watches_everything_the_page_is_made_of(tmp_path):
    reloader = for_ui(tmp_path / "adbagent",
                      config_path=tmp_path / "config.json",
                      skills_dir=tmp_path / "skills",
                      policy_path=tmp_path / "policy.md")
    assert sorted(w["kind"] for w in reloader.watching()) == [
        "assets", "code", "config", "policy", "skills"]


def test_a_restart_runs_the_same_command_again(monkeypatch):
    """Same port, same config, same artifacts directory: a server that came back
    on different terms would be a worse answer than not coming back."""
    called = []
    monkeypatch.setattr(os, "execv", lambda exe, argv: called.append((exe, argv)))
    monkeypatch.setattr(cli.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(cli.sys, "argv", ["/venv/bin/adbagent", "ui", "--port", "9"])
    cli._reexec()
    assert called == [("/usr/bin/python3",
                       ["/usr/bin/python3", "/venv/bin/adbagent", "ui", "--port", "9"])]

    # `python -m adbagent` has to go back through -m: running __main__.py as a
    # script puts the package's own directory on sys.path instead of its parent,
    # and the import of `adbagent.cli` inside it then fails.
    called.clear()
    monkeypatch.setattr(cli.sys, "argv", ["/repo/adbagent/__main__.py", "ui"])
    cli._reexec()
    assert called == [("/usr/bin/python3", ["/usr/bin/python3", "-m", "adbagent", "ui"])]


def test_a_restart_with_no_interpreter_to_run_is_not_a_crash(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "executable", "")
    monkeypatch.setattr(os, "execv", lambda exe, argv: pytest.fail("exec'd anyway"))
    assert cli._reexec() == 1
    assert "cannot restart" in capsys.readouterr().err
