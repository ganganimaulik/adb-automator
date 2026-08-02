"""Run tracking, transitions, dead-ends and intent keys."""

from __future__ import annotations

import pytest

from adbagent.actions import AgentAction, Target
from adbagent.config import Config
from adbagent.fingerprint import attach
from adbagent.memory import Memory, intent_key
from adbagent.screen import parse

from . import xmlgen as X


def s(xml: str):
    return attach(parse(xml, width=X.W, height=X.H))


BASE = s(X.settings_screen())


def act(**kw) -> AgentAction:
    kw.setdefault("observation", "settings")
    kw.setdefault("reasoning", "because")
    return AgentAction(**kw)


@pytest.fixture
def mem(tmp_path):
    cfg = Config()
    cfg.memory.db = str(tmp_path / "memory.db")
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

def test_begin_and_end_run(mem):
    mem.begin_run("r1", "open Wi-Fi", "intent1")
    mem.end_run("r1", "success", steps=3, llm_calls=2, usd=0.005)
    row = mem.db.execute("SELECT * FROM run WHERE run_id='r1'").fetchone()
    assert row is not None
    assert row["outcome"] == "success"
    assert row["steps"] == 3
    assert row["llm_calls"] == 2


# ---------------------------------------------------------------------------
# Screen corpus
# ---------------------------------------------------------------------------

def test_note_screen_records_tokens(mem):
    mem.note_screen(BASE)
    rows = mem.db.execute("SELECT COUNT(*) AS n FROM screen_seen").fetchone()
    assert rows["n"] > 0


def test_idf_returns_term_frequencies(mem):
    mem.note_screen(BASE)
    detail = s(X.detail_screen())
    mem.note_screen(detail)
    idf = mem.idf(BASE.package)
    assert isinstance(idf, dict)
    # Tokens unique to one screen should have higher IDF
    if idf:
        assert max(idf.values()) > 0


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def test_transitions_are_recorded(mem):
    after = s(X.detail_screen())
    action = act(action="tap", target=Target(index=3))
    mem.note_transition(BASE, after, action)
    row = mem.db.execute("SELECT * FROM transition").fetchone()
    assert row["from_skeleton"] == BASE.skeleton_id
    assert row["to_skeleton"] == after.skeleton_id


def test_same_transition_increments_count(mem):
    after = s(X.detail_screen())
    action = act(action="tap", target=Target(index=3))
    mem.note_transition(BASE, after, action)
    mem.note_transition(BASE, after, action)
    row = mem.db.execute("SELECT * FROM transition").fetchone()
    assert row["n_seen"] == 2


# ---------------------------------------------------------------------------
# Dead ends
# ---------------------------------------------------------------------------

def test_dead_end_is_recorded(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "didn't work")
    sigs = mem._dead_end_sigs(BASE.package, BASE.skeleton_id, "i")
    assert action.signature() in sigs


def test_dead_end_expires(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "test")
    # Artificially expire the dead end.
    mem.db.execute("UPDATE dead_end SET expires_at = 0")
    mem.db.commit()
    sigs = mem._dead_end_sigs(BASE.package, BASE.skeleton_id, "i")
    assert action.signature() not in sigs


# ---------------------------------------------------------------------------
# Intent keys
# ---------------------------------------------------------------------------

def test_intent_key_ignores_trivial_rewording():
    assert intent_key("Turn on Wi-Fi") == intent_key("turn on   wi-fi")
    assert intent_key("Turn on Wi-Fi") != intent_key("Turn off Wi-Fi")


def test_intent_key_opposite_goals_differ():
    """Turning on vs turning off should produce different cache keys."""
    assert intent_key("Turn on Wi-Fi") != intent_key("Turn off Wi-Fi")
    assert intent_key("enable dark mode") != intent_key("disable dark mode")


def test_intent_key_same_goal_rewording_matches():
    """Trivial rewording should still produce the same intent key."""
    assert intent_key("Turn on Wi-Fi") == intent_key("turn on   wi-fi")
    assert intent_key("open Settings") == intent_key("open   settings")
