"""What a run leaves behind for the person debugging it a week later.

The thing worth asserting is not that a file appears. It is that the file has the
detail the console was never shown, that it is complete for the crash which is
the one case you cannot reproduce on request, and that the logging configuration
is put back afterwards -- a run that leaves the `adbagent` logger at DEBUG turns
every later command in the same process into a wall of text.
"""

from __future__ import annotations

import json
import logging

import pytest

from adbagent import runlog
from adbagent.agent import Agent
from adbagent.cli import setup_logging
from adbagent.config import Config
from adbagent.llm import LLMError
from adbagent.memory import Memory

from . import fake

GOAL = "open the Wi-Fi settings screen"


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


@pytest.fixture(autouse=True)
def restore_logging():
    """Put the process's logging back however this test found it.

    These tests attach handlers and lower levels on a logger the whole package
    shares, and a leak would be invisible here and confusing everywhere else.
    """
    logger = logging.getLogger(runlog.ROOT)
    root = logging.getLogger()
    before = (logger.level, list(logger.handlers),
              [(h, h.level) for h in root.handlers])
    yield
    logger.level, logger.handlers = before[0], before[1]
    for handler, level in before[2]:
        handler.setLevel(level)
    runlog._open.clear()
    runlog._baseline_level = None


def run(dev, mem, cfg, policy=None, **kw):
    llm = fake.FakeLLM(dev, policy or fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    return Agent(dev, mem, llm, cfg, **kw).run(GOAL)


def log_of(cfg, state) -> str:
    return runlog.log_path(runlog.run_dir(cfg, state.run_id)).read_text()


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------

def test_a_run_writes_its_log_beside_its_events(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    _, state = run(dev, mem, cfg)
    path = tmp_path / "runs" / state.run_id / runlog.LOG_NAME
    assert path.is_file()
    assert (path.parent / "events.jsonl").is_file()


def test_the_header_says_which_run_under_which_settings(cfg, mem):
    """The settings are knowable when the run starts and nowhere afterwards."""
    cfg.run.never_screenshot = True
    dev = fake.FakeDevice(cfg)
    _, state = run(dev, mem, cfg)
    text = log_of(cfg, state)
    assert state.run_id in text
    assert GOAL in text
    assert "never=True" in text
    assert f"max_steps={cfg.run.max_steps}" in text


def test_the_log_takes_debug_though_the_console_was_asked_for_warnings(cfg, mem):
    """The whole point: no `-vv` rerun to diagnose a run that already happened."""
    setup_logging(0)
    assert all(h.level == logging.WARNING for h in logging.getLogger().handlers)

    dev = fake.FakeDevice(cfg)
    _, state = run(dev, mem, cfg)
    lines = log_of(cfg, state).splitlines()
    assert [l for l in lines if " DEBUG " in l]


def test_the_decisions_are_in_the_log_in_order_with_the_traffic(cfg, mem):
    """Two files correlated by timestamp is a worse tool than one file."""
    dev = fake.FakeDevice(cfg)
    _, state = run(dev, mem, cfg)
    kinds = [line.split("adbagent.events: ")[1].split(" ", 1)[0]
             for line in log_of(cfg, state).splitlines()
             if "adbagent.events: " in line]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    assert "decide" in kinds and "verify" in kinds


def test_an_events_line_drops_the_fields_the_decision_did_not_use(tmp_path):
    """`AgentAction` has thirty-odd fields and a tap uses three; the log is for
    reading, and `events.jsonl` still has all of them.

    `False` and `0` are kept -- a decision taken without a screenshot is a fact
    about the decision, and dropping it would read as a screenshot turn.
    """
    directory = tmp_path / "runs" / "pruned"
    with runlog.capture(directory):
        runlog.event("decide", {"step": 4, "screenshot": False, "reason": "",
                                "action": {"action": "tap", "text": None,
                                           "direction": "", "index": 0},
                                "llm": {"usd": 0.01, "n_calls": 2}})
    line = runlog.log_path(directory).read_text().strip()
    assert json.loads(line.split("decide ", 1)[1]) == {
        "step": 4, "screenshot": False,
        "action": {"action": "tap", "index": 0}}


def test_a_long_event_is_cut_rather_than_wrapping_the_file(cfg, mem, tmp_path):
    directory = tmp_path / "runs" / "cut"
    with runlog.capture(directory):
        runlog.event("decide", {"reasoning": "x" * 5_000})
    line = runlog.log_path(directory).read_text().strip()
    assert len(line) < runlog.EVENT_CHARS + 200
    assert "chars)" in line


def test_third_party_debug_stays_out(cfg, mem, tmp_path):
    """`httpx` at DEBUG prints the request body, which on a vision turn is a
    base64 screenshot per line."""
    directory = tmp_path / "runs" / "quiet"
    with runlog.capture(directory):
        logging.getLogger("httpx").debug("POST with a 2MB image in it")
        logging.getLogger("adbagent.device").debug("screen never settled")
    text = runlog.log_path(directory).read_text()
    assert "never settled" in text
    assert "2MB image" not in text


# ---------------------------------------------------------------------------
# Failure, which is the case the log exists for
# ---------------------------------------------------------------------------

def test_an_abort_leaves_the_stack_in_the_log(cfg, mem, monkeypatch):
    """"budget exceeded" on the console says what; the file has to say where."""
    def boom(self, state, rec):
        raise LLMError("the provider hung up")

    monkeypatch.setattr(Agent, "_loop", boom)
    dev = fake.FakeDevice(cfg)
    outcome, state = run(dev, mem, cfg)
    assert outcome == "aborted"
    text = log_of(cfg, state)
    assert "the provider hung up" in text
    assert "Traceback (most recent call last)" in text


def test_an_unhandled_crash_records_itself_before_the_log_closes(cfg, mem,
                                                                monkeypatch,
                                                                tmp_path):
    """The one run you cannot reproduce on request is the one that crashed."""
    def boom(self, state, rec):
        raise RuntimeError("something nobody expected")

    monkeypatch.setattr(Agent, "_loop", boom)
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    agent = Agent(dev, mem, llm, cfg)
    with pytest.raises(RuntimeError):
        agent.run(GOAL, run_id="crashed")

    run_dir = tmp_path / "runs" / "crashed"
    text = (run_dir / runlog.LOG_NAME).read_text()
    assert "something nobody expected" in text
    assert "Traceback (most recent call last)" in text
    # And in the machine-readable half, so `report` shows it too.
    errors = [json.loads(l) for l in
              (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    error = next(e for e in errors if e["kind"] == "error")
    assert "RuntimeError" in error["traceback"]


def test_a_log_that_cannot_be_opened_is_not_an_error(tmp_path):
    """A read-only artifacts directory is a reason to run without a log."""
    blocked = tmp_path / "runs"
    blocked.write_text("this is a file, not a directory")
    assert runlog.attach(blocked / "id") is None
    assert not logging.getLogger(runlog.ROOT).handlers


def test_nothing_is_attached_when_there_is_no_run_to_attribute_it_to():
    with runlog.capture(None) as handle:
        assert handle is None
    assert not logging.getLogger(runlog.ROOT).handlers


# ---------------------------------------------------------------------------
# Putting the logging configuration back
# ---------------------------------------------------------------------------

def test_the_run_does_not_leave_its_handler_behind(cfg, mem):
    logger = logging.getLogger(runlog.ROOT)
    before = logger.level
    dev = fake.FakeDevice(cfg)
    run(dev, mem, cfg)
    assert logger.handlers == []
    assert logger.level == before


def test_one_run_does_not_write_into_the_next_one(cfg, mem):
    dev = fake.FakeDevice(cfg)
    _, first = run(dev, mem, cfg)
    dev = fake.FakeDevice(cfg)
    _, second = run(dev, mem, cfg)
    assert first.run_id != second.run_id
    assert second.run_id not in log_of(cfg, first)
    assert first.run_id not in log_of(cfg, second)


@pytest.mark.parametrize("lifo", [True, False])
def test_the_level_comes_back_only_when_the_last_log_closes(tmp_path, lifo):
    """Two open logs are unusual -- one run, one log -- but the level has to stay
    down while either wants the detail, and go back to what the process started
    with whichever order they close in."""
    logger = logging.getLogger(runlog.ROOT)
    logger.setLevel(logging.WARNING)
    first = runlog.attach(tmp_path / "first")
    second = runlog.attach(tmp_path / "second")

    order = [second, first] if lifo else [first, second]
    order[0].close()
    assert logger.level == logging.DEBUG, "the other log still needs the detail"
    order[1].close()
    assert logger.level == logging.WARNING


def test_capture_appends_to_the_log_the_run_already_wrote(tmp_path):
    """The after-run learning is part of the same run, not a second file."""
    directory = tmp_path / "runs" / "shared"
    with runlog.capture(directory):
        logging.getLogger("adbagent.agent").info("the loop")
    with runlog.capture(directory):
        logging.getLogger("adbagent.skills").info("the skill it learned")
    text = runlog.log_path(directory).read_text()
    assert "the loop" in text and "the skill it learned" in text


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def test_problems_picks_the_warnings_out_in_order(tmp_path):
    """What `report` shows, so "did anything go wrong" needs no `grep`."""
    directory = tmp_path / "runs" / "noisy"
    with runlog.capture(directory):
        log = logging.getLogger("adbagent.device")
        log.debug("dumped the tree")
        log.warning("screen never settled within 2.0s")
        log.info("restored the keyboard")
        log.error("recovery tier 2 failed")
    found = runlog.problems(directory)
    assert found == [found[0], found[1]]
    assert "warning: adbagent.device: screen never settled within 2.0s" in found[0]
    assert "error: adbagent.device: recovery tier 2 failed" in found[1]
    assert len(found) == 2


def test_problems_of_a_run_that_has_no_log(tmp_path):
    """Every run recorded before this existed."""
    assert runlog.problems(tmp_path / "runs" / "old") == []
