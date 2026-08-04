"""Reading what earlier runs in an app repeatedly showed.

Attribution is the part that has to be right. A run filed under the wrong app
does not merely go missing -- its failures are handed to that app's synthesis as
fact, and end up written into the wrong skill for every later run to follow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.history import (MIN_OCCURRENCES, History, _describe_action,
                              for_package, packages_in)
from adbagent.memory import Memory

from . import fake


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    c.safety.unattended = True
    return c


def write_run(cfg, name, events):
    run_dir = Path(cfg.run.artifacts_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Which app was a run about?
# ---------------------------------------------------------------------------

def test_the_app_the_steps_went_to_is_the_app_the_run_was_about():
    """Every run touches the launcher on its way somewhere. A history of runs
    that glanced at an app is not a history of the app."""
    assert packages_in([{"kind": "run_end",
                         "package_steps": {"com.android.launcher3": 1,
                                           "com.whatsapp": 40}}]) == {"com.whatsapp"}


def test_a_run_that_genuinely_crossed_two_apps_counts_for_both():
    assert packages_in([{"kind": "run_end",
                         "package_steps": {"com.whatsapp": 20,
                                           "com.google.android.apps.docs": 15}}]) == {
        "com.whatsapp", "com.google.android.apps.docs"}


def test_the_skill_that_loaded_is_never_taken_as_the_app():
    """`active_skill` records the package on screen when a skill loaded, which
    for a run started on top of another app is that other app. Trusting it filed
    a WhatsApp run under Bumble."""
    events = [{"kind": "active_skill", "name": "WhatsApp", "package": "com.bumble.app"},
              {"kind": "run_end", "package_steps": {"com.whatsapp": 9}}]
    assert packages_in(events) == {"com.whatsapp"}
    # And with no counts at all it must still not believe the skill event.
    assert packages_in(events[:1]) == set()


def test_older_runs_fall_back_to_the_package_set_then_to_what_they_opened():
    assert packages_in([{"kind": "run_end",
                         "packages": ["com.whatsapp"]}]) == {"com.whatsapp"}
    assert packages_in([
        {"kind": "decide", "action": {"action": "open_app", "text": "com.whatsapp"}},
    ]) == {"com.whatsapp"}
    # A common name is not a package and cannot be trusted as one.
    assert packages_in([
        {"kind": "decide", "action": {"action": "open_app", "text": "whatsapp"}},
    ]) == set()


# ---------------------------------------------------------------------------
# What is worth reporting
# ---------------------------------------------------------------------------

def test_an_action_is_named_without_its_index():
    """`tap #11` identifies nothing a week later -- the same control is #4 next
    run, and a nuance written around the number reads as specific while being
    wrong."""
    described = _describe_action({"action": "tap", "target": {"index": 11,
                                                              "text": "Older chats"}})
    assert described == "tap 'Older chats'"
    assert "11" not in described
    assert _describe_action({"action": "scroll", "direction": "down"}) == "scroll 'down'"
    assert _describe_action({"action": "press_key"}) == "press_key"


def _run_with_failures(package, times, step_offset=0):
    events = [{"kind": "run_start", "goal": f"do the thing in {package}"}]
    for i in range(times):
        step = step_offset + i + 1
        events.append({"kind": "decide", "step": step,
                       "action": {"action": "tap", "target": {"index": step,
                                                              "text": "Older chats"}}})
        events.append({"kind": "verify", "step": step, "grade": "failed"})
    events.append({"kind": "run_end", "outcome": "failed",
                   "package_steps": {package: times}})
    return events


def test_only_what_repeated_is_reported(cfg):
    """A signal seen once is already in the current run's trace. Forwarding it
    again under the heading "history" would launder one observation into a
    trend."""
    write_run(cfg, "once", _run_with_failures("com.example.app", 1))
    assert not for_package(cfg, "com.example.app").failures

    write_run(cfg, "twice", _run_with_failures("com.example.app", MIN_OCCURRENCES))
    failures = for_package(cfg, "com.example.app").failures
    assert failures and failures[0][0] == "tap 'Older chats'"
    assert failures[0][1] >= MIN_OCCURRENCES


def test_only_runs_in_this_app_are_counted(cfg):
    write_run(cfg, "ours", _run_with_failures("com.example.app", 3))
    write_run(cfg, "theirs", _run_with_failures("com.other.app", 3))

    mine = for_package(cfg, "com.example.app")
    assert mine.runs == 1
    assert all("other" not in g for g in mine.goals)


def test_the_same_goal_re_run_is_one_thing_people_do_not_five(cfg):
    for i in range(5):
        write_run(cfg, f"run{i}", [
            {"kind": "run_start", "goal": "check the food weight group"},
            {"kind": "run_end", "outcome": "success",
             "package_steps": {"com.example.app": 5}}])
    assert for_package(cfg, "com.example.app").goals == ["check the food weight group"]


def test_outcomes_are_tallied_across_the_matching_runs(cfg):
    write_run(cfg, "a", [{"kind": "run_end", "outcome": "success",
                          "package_steps": {"com.example.app": 5}}])
    write_run(cfg, "b", [{"kind": "run_end", "outcome": "failed",
                          "package_steps": {"com.example.app": 5}}])
    assert for_package(cfg, "com.example.app").outcomes == {"success": 1, "failed": 1}


def test_nothing_to_say_renders_as_nothing(cfg):
    assert not History()
    assert History().to_prompt_text() == ""
    assert for_package(cfg, "com.never.seen").to_prompt_text() == ""
    assert for_package(cfg, "").to_prompt_text() == ""


def test_an_unreadable_run_does_not_stop_the_others(cfg):
    write_run(cfg, "good", _run_with_failures("com.example.app", 3))
    torn = write_run(cfg, "torn", [{"kind": "run_end", "outcome": "aborted",
                                    "package_steps": {"com.example.app": 2}}])
    (torn / "events.jsonl").write_text('{"kind": "run_end", "outc\n', encoding="utf-8")

    assert for_package(cfg, "com.example.app").runs >= 1


def test_dead_ends_come_from_the_database_keyed_by_app(cfg, tmp_path):
    from adbagent.screen import Screen

    with Memory(cfg, path=tmp_path / "memory.db") as mem:
        for i in range(3):
            screen = Screen(package="com.example.app")
            screen.skeleton_id = "abcd1234"
            mem.record_dead_end(screen, f"intent{i}", f"swipe/#{i}/left",
                                "swiping did not reveal new content")

    dead = for_package(cfg, "com.example.app").dead_ends
    assert dead and dead[0][1] == 3
    # Indices are collapsed, not quoted: #0, #1 and #2 are the same finding.
    assert "swipe on screen abcd1234" == dead[0][0]
    assert not for_package(cfg, "com.other.app").dead_ends


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_a_real_run_records_where_its_steps_went(cfg, tmp_path):
    """The whole feature rests on this field being written."""
    dev = fake.FakeDevice(cfg)
    with Memory(cfg, path=tmp_path / "memory.db") as mem:
        Agent(dev, mem, fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"])),
              cfg).run("open the Wi-Fi screen")

    history = for_package(cfg, "com.android.settings")
    assert history.runs == 1
    assert history.outcomes.get("success") == 1
