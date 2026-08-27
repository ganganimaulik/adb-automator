"""Run tracking, dead ends and intent keys.

Memory holds only what something reads back. The dead-end table is the whole of
the cross-run knowledge: a control that did nothing on a screen is worth
remembering past the end of the process, so the next run does not rediscover it.
"""

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


def memory(tmp_path, name: str = "memory.db") -> Memory:
    cfg = Config()
    cfg.memory.db = str(tmp_path / name)
    return Memory(cfg, path=tmp_path / name)


@pytest.fixture
def mem(tmp_path):
    with memory(tmp_path) as m:
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
# Dead ends
# ---------------------------------------------------------------------------

def test_a_dead_end_is_recorded_with_its_reason(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "nothing on screen changed")
    found = mem.dead_ends(BASE, "i")
    assert found[action.signature()] == "nothing on screen changed"


def test_a_dead_end_expires(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "test")
    mem.db.execute("UPDATE dead_end SET expires_at = 0")
    mem.db.commit()
    assert mem.dead_ends(BASE, "i") == {}


def test_a_dead_end_belongs_to_one_goal(mem):
    """"This row does nothing" can be true of one goal and false of another."""
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "turn-on-wifi", action.signature(), "no change")
    assert mem.dead_ends(BASE, "turn-on-wifi")
    assert mem.dead_ends(BASE, "turn-off-bluetooth") == {}


def test_a_dead_end_belongs_to_one_screen(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "no change")
    assert mem.dead_ends(s(X.detail_screen()), "i") == {}


def test_recording_the_same_dead_end_twice_updates_it(mem):
    action = act(action="tap", target=Target(index=3))
    mem.record_dead_end(BASE, "i", action.signature(), "first reason")
    mem.record_dead_end(BASE, "i", action.signature(), "second reason")
    found = mem.dead_ends(BASE, "i")
    assert found == {action.signature(): "second reason"}


def test_dead_ends_outlive_the_process(tmp_path):
    """The point of putting them in SQLite. A dud control found in one run must
    not be rediscovered by the next one."""
    action = act(action="tap", target=Target(index=3))
    with memory(tmp_path) as first:
        first.record_dead_end(BASE, "i", action.signature(), "no change")
    with memory(tmp_path) as second:
        assert action.signature() in second.dead_ends(BASE, "i")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_tables_that_nothing_reads_are_dropped_on_open(tmp_path):
    """An unused table is indistinguishable from a broken feature to whoever
    opens the database next, so older versions' write-only tables go."""
    with memory(tmp_path, "old.db") as m:
        m.db.execute("CREATE TABLE screen_seen (app_key TEXT)")
        m.db.execute("CREATE TABLE transition (app_key TEXT)")
        m.db.execute("CREATE TABLE entry (id INTEGER)")
        m.db.execute("UPDATE meta SET v='3' WHERE k='schema_version'")
        m.db.commit()

    with memory(tmp_path, "old.db") as m:
        names = {r["name"] for r in m.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "screen_seen" not in names
        assert "transition" not in names
        assert "entry" not in names
        assert {"run", "dead_end", "meta"} <= names


def test_opening_an_existing_database_twice_is_harmless(tmp_path):
    action = act(action="tap", target=Target(index=3))
    with memory(tmp_path) as m:
        m.record_dead_end(BASE, "i", action.signature(), "no change")
    with memory(tmp_path) as m:
        assert m.dead_ends(BASE, "i")          # survived the migration pass
    with memory(tmp_path) as m:
        assert m.dead_ends(BASE, "i")


# ---------------------------------------------------------------------------
# The locate cache
# ---------------------------------------------------------------------------
#
# 577 `tap_at` actions across ``runs/`` named a control and resolve to 94
# distinct (skeleton, name) pairs: 37% of them repeat a pair located earlier in
# the same run, 84% one located in an earlier run. Each repeat was a screenshot
# and a vision call for an answer the harness already had.

def test_a_located_point_is_recalled(mem):
    mem.record_locate(BASE, "Send Priority Like", 0.7, 0.68)
    assert mem.recall_locate(BASE, "Send Priority Like") == (0.7, 0.68)


def test_recall_misses_when_nothing_was_stored(mem):
    assert mem.recall_locate(BASE, "Send Priority Like") is None


def test_the_name_is_matched_on_case_and_spacing_only(mem):
    mem.record_locate(BASE, "Send Priority Like", 0.7, 0.68)
    assert mem.recall_locate(BASE, "send   priority like") == (0.7, 0.68)
    # Two genuinely different wordings get their own rows rather than one
    # answering for the other: they may well be two different controls.
    assert mem.recall_locate(BASE, "the send button") is None


def test_relocating_the_same_control_moves_the_point(mem):
    mem.record_locate(BASE, "Send", 0.5, 0.5)
    mem.record_locate(BASE, "Send", 0.7, 0.9)
    assert mem.recall_locate(BASE, "Send") == (0.7, 0.9)
    assert mem.db.execute("SELECT COUNT(*) c FROM locate").fetchone()["c"] == 1


def test_a_located_point_expires(mem):
    mem.record_locate(BASE, "Send", 0.7, 0.68)
    mem.db.execute("UPDATE locate SET expires_at = 0")
    mem.db.commit()
    assert mem.recall_locate(BASE, "Send") is None


def test_forgetting_a_point_drops_it(mem):
    """The invalidation half. A cached point that taps nothing must not be
    served again, or one bad vision call becomes a bad answer all day."""
    mem.record_locate(BASE, "Send", 0.7, 0.68)
    mem.forget_locate(BASE, "Send")
    assert mem.recall_locate(BASE, "Send") is None


def test_forgetting_matches_the_same_way_recall_does(mem):
    mem.record_locate(BASE, "Send Priority Like", 0.7, 0.68)
    mem.forget_locate(BASE, "send priority like")
    assert mem.recall_locate(BASE, "Send Priority Like") is None


def test_a_point_is_not_shared_between_screens(mem):
    other = s(X.settings_screen(rows=3, labels=["A", "B", "C"]))
    assert other.skeleton_id != BASE.skeleton_id
    mem.record_locate(BASE, "Send", 0.7, 0.68)
    assert mem.recall_locate(other, "Send") is None


def test_a_point_is_shared_across_intents(mem):
    """Unlike a dead end. Where a control sits is a fact about the layout, and
    the goal being pursued has no bearing on it."""
    mem.record_locate(BASE, "Send", 0.7, 0.68)
    # `recall_locate` takes no intent at all -- that is the assertion.
    assert mem.recall_locate(BASE, "Send") == (0.7, 0.68)


def test_the_locate_table_survives_reopening(tmp_path):
    with memory(tmp_path) as m:
        m.record_locate(BASE, "Send", 0.7, 0.68)
    with memory(tmp_path) as m:
        assert m.recall_locate(BASE, "Send") == (0.7, 0.68)


# ---------------------------------------------------------------------------
# Intent keys
# ---------------------------------------------------------------------------

def test_intent_key_ignores_trivial_rewording():
    assert intent_key("Turn on Wi-Fi") == intent_key("turn on   wi-fi")
    assert intent_key("open Settings") == intent_key("open   settings")


def test_intent_key_opposite_goals_differ():
    assert intent_key("Turn on Wi-Fi") != intent_key("Turn off Wi-Fi")
    assert intent_key("enable dark mode") != intent_key("disable dark mode")
