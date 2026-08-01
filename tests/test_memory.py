"""Anchors, admission gates, trust states and storage."""

from __future__ import annotations

import pytest

from adbagent import trust
from adbagent.actions import AgentAction, Postcondition, Target
from adbagent.config import Config
from adbagent.fingerprint import attach
from adbagent.memory import (Anchor, Memory, build_anchor, intent_key, resolve_anchor,
                             score_anchor)
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
# Trust statistics
# ---------------------------------------------------------------------------

def test_a_single_success_is_not_trust():
    """The whole point of Wilson: 1-of-1 is not evidence of reliability."""
    assert trust.wilson_lower_bound(1, 0) == pytest.approx(0.2065, abs=1e-3)
    assert trust.classify(trust.Stats(n_success=1)) == "probation"


def test_sustained_success_earns_trust():
    assert trust.classify(trust.Stats(n_success=4)) == "active"
    assert trust.classify(trust.Stats(n_success=12)) == "trusted"
    # A run with a blemish stays useful but unpromoted.
    assert trust.classify(trust.Stats(n_success=5, n_failure=1)) == "probation"
    assert trust.classify(trust.Stats(n_success=12, n_failure=1)) == "active"


def test_three_consecutive_failures_quarantine():
    stats = trust.Stats(n_success=20, n_failure=3, consecutive_failures=3)
    assert trust.classify(stats) == "quarantined"
    assert not trust.may_replay("quarantined")


def test_mostly_failing_entries_quarantine():
    assert trust.classify(trust.Stats(n_success=1, n_failure=6)) == "quarantined"


def test_decay_forgets_stale_evidence():
    fresh = trust.Stats(n_success=10)
    stale = trust.Stats(n_success=10, age_days=28)   # two half-lives
    assert stale.wilson() < fresh.wilson()
    assert trust.decay_factor(14) == pytest.approx(0.5)


def test_probation_always_verifies():
    assert trust.must_verify("probation")
    assert not trust.must_verify("trusted")


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def test_anchor_records_what_makes_an_element_findable():
    toggle = next(e for e in BASE.elements if e.kind() == "Toggle")
    anchor = build_anchor(toggle, BASE)
    assert anchor.resource_id == "switch_widget"
    assert anchor.class_eq == "Toggle"
    assert anchor.scroller_rid == "recycler_view"
    assert anchor.kind == "attributed"
    assert 0.0 <= anchor.bounds_frac[0] <= 1.0


def test_anchor_of_an_unidentifiable_element_is_coordinate_only():
    blank = X.N("android.view.View", (100, 700, 300, 800), clickable=True)
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.FrameLayout", (0, 700, X.W, 800), rid="canvas",
            children=[blank])]))
    el = next(e for e in scr.elements if e.cls == "android.view.View")
    assert build_anchor(el, scr).kind == "coordinate_only"


def test_anchor_scores_itself_perfectly():
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    assert score_anchor(build_anchor(el, BASE), el) == pytest.approx(1.0)


def test_missing_features_do_not_penalise():
    """An element with no resource-id must not be scored down for lacking one."""
    anchor = Anchor(text="Done", class_eq="Button")
    el = next(e for e in BASE.elements if e.best_text == "Done")
    assert score_anchor(anchor, el) == pytest.approx(1.0)


def test_text_matching_is_fuzzy_not_exact():
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    exact = Anchor(text="Network & internet", class_eq="TextView")
    drifted = Anchor(text="Network", class_eq="TextView")
    assert score_anchor(exact, el) == pytest.approx(1.0)
    assert 0.5 < score_anchor(drifted, el) < 1.0


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------

def resolve(anchor, screen, threshold=0.55, gap=0.08, **kw):
    return resolve_anchor(anchor, screen, threshold=threshold,
                          ambiguity_gap=gap, **kw)


def test_resolves_on_the_same_screen():
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    result = resolve(build_anchor(el, BASE), BASE)
    assert result.ok and result.element.resource_id == "action_bar_title"


def test_survives_a_scrolled_list():
    """The list moved; the row is still the row."""
    # The row's TextView is absorbed into the clickable row container, so the
    # addressable element is the row itself.
    row = next(e for e in BASE.elements
               if e.resource_id == "row_item" and e.best_text == "Option 5")
    anchor = build_anchor(row, BASE)
    scrolled = s(X.settings_screen(scroll=240))
    result = resolve(anchor, scrolled)
    assert result.ok and result.element.best_text == "Option 5"


def test_survives_chrome_drift():
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    anchor = build_anchor(el, BASE)
    later = s(X.settings_screen(clock="11:11", battery="12%", badge="9"))
    assert resolve(anchor, later).ok


def test_refuses_when_the_element_is_gone():
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    anchor = build_anchor(el, BASE)
    anchor.resource_id = "no_such_id"
    anchor.resource_id_raw = "com.android.settings:id/no_such_id"
    anchor.text = "Nothing like this"
    anchor.content_desc = ""
    result = resolve(anchor, BASE)
    assert not result.ok and "scored" in result.reason


def test_refuses_when_two_candidates_are_indistinguishable():
    """Tapping the wrong one of a pair is worse than asking the model again."""
    twins = [X.N("android.widget.Button", (100, 620 + i * 140, 400, 740 + i * 140),
                 text="Open", rid="open", clickable=True) for i in range(2)]
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.LinearLayout", (0, 600, X.W, 900), rid="pair",
            children=twins)]))
    el = next(e for e in scr.elements if e.resource_id == "open")
    anchor = build_anchor(el, scr)
    # Erase the one discriminating feature so both score identically.
    anchor.sibling_index = -1
    anchor.bounds_frac = (0.0, 0.0, 0.0, 0.0)
    result = resolve(anchor, scr)
    assert not result.ok and "ambiguous" in result.reason


def test_coordinate_only_anchors_are_gated_hard():
    anchor = Anchor(kind="coordinate_only", class_raw="android.view.View",
                    bounds_frac=(0.1, 0.3, 0.2, 0.05))
    assert not resolve(anchor, BASE).ok             # refused without an exact match
    assert resolve(anchor, BASE, dhash_ok=True).ok  # allowed only when gated


def test_clipped_element_is_refused():
    """u2 clips bounds to the display, so a half-scrolled row reports a small box."""
    el = next(e for e in BASE.elements if e.resource_id == "row_item")
    anchor = build_anchor(el, BASE)
    anchor.bounds_frac = (anchor.bounds_frac[0], anchor.bounds_frac[1], 0.9, 0.5)
    assert not resolve(anchor, BASE).ok


# ---------------------------------------------------------------------------
# The three admission gates, end to end
# ---------------------------------------------------------------------------

def learn(mem, screen, action, element, after, intent="i", visit=0):
    return mem.record(screen=screen, intent_id=intent, visit=visit, action=action,
                      element=element, postcondition=Postcondition(kind="screen_changed"),
                      after=after, run_id="r1")


def test_learn_then_replay_with_no_llm(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    action = act(action="tap", target=Target(index=el.index))
    learn(mem, BASE, action, el, s(X.detail_screen()))

    hit = mem.lookup(BASE, "i", 0)
    assert hit is not None
    bound = mem.rehydrate(hit, BASE)
    assert bound is not None and bound.action == "tap"
    # The replayed target points at the live element, not the recorded index.
    assert bound.target.resource_id == "action_bar_title"


def test_cache_misses_on_a_different_screen(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
          s(X.detail_screen()))
    assert mem.lookup(s(X.detail_screen()), "i", 0) is None


def test_cache_misses_for_a_different_goal(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
          s(X.detail_screen()))
    assert mem.lookup(BASE, "a-different-intent", 0) is None


def test_cache_still_hits_after_harmless_drift(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
          s(X.detail_screen()))
    later = s(X.settings_screen(clock="10:15", battery="61%", rows=12))
    assert mem.lookup(later, "i", 0) is not None


def test_gate_two_blocks_replay_when_something_irreversible_appeared(mem):
    """A confirm dialog must never be tapped through from cache."""
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    entry = learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
                  s(X.detail_screen()))
    mem.db.execute("UPDATE entry SET forbidden_tokens=? WHERE id=?",
                   ('["delete account?"]', entry.id))
    mem.db.commit()
    assert mem.lookup(BASE, "i", 0, forbidden_now=["delete account?"]) is None


def test_banned_signatures_are_skipped(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    action = act(action="tap", target=Target(index=el.index))
    learn(mem, BASE, action, el, s(X.detail_screen()))
    assert mem.lookup(BASE, "i", 0,
                      banned_signatures=[action.signature()]) is None


# ---------------------------------------------------------------------------
# Outcomes, versioning, maintenance
# ---------------------------------------------------------------------------

def test_repeated_failure_quarantines_and_stops_replaying(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    entry = learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
                  s(X.detail_screen()))
    for _ in range(3):
        mem.mark(entry, "hard_fail", "r1", reason="nothing changed")
    assert entry.state == "quarantined"
    assert mem.lookup(BASE, "i", 0) is None


def test_success_promotes_out_of_probation(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    entry = learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
                  s(X.detail_screen()))
    assert entry.state == "probation"
    for _ in range(10):
        mem.mark(entry, "success", "r1")
    assert entry.state == "trusted"
    assert mem.get(entry.id).state == "trusted"


def test_soft_fail_records_an_alternative_successor(mem):
    """Screens fan out; remember the new destination instead of punishing it."""
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    entry = learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
                  s(X.detail_screen()))
    mem.mark(entry, "soft_fail", "r1", observed_successor="other-screen")
    assert "other-screen" in mem.get(entry.id).alt_successors


def test_relearning_creates_a_new_version_rather_than_overwriting(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    a = learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
              s(X.detail_screen()))
    b = learn(mem, BASE, act(action="press_key", key="back"), None, s(X.detail_screen()))
    assert b.version == a.version + 1
    assert mem.get(a.id) is not None, "the old version must survive"


def test_transitions_are_recorded_for_navigation(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    after = s(X.detail_screen())
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el, after)
    row = mem.db.execute("SELECT * FROM transition").fetchone()
    assert row["from_skeleton"] == BASE.skeleton_id
    assert row["to_skeleton"] == after.skeleton_id


def test_gc_drops_quarantined_and_caps_versions(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    for _ in range(6):
        learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
              s(X.detail_screen()))
    mem.gc()
    versions = mem.db.execute(
        "SELECT COUNT(*) AS n FROM entry WHERE skeleton_id=?",
        (BASE.skeleton_id,)).fetchone()["n"]
    assert versions <= trust.MAX_VERSIONS


def test_disabled_memory_never_hits(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
          s(X.detail_screen()))
    mem.cfg.memory.enabled = False
    assert mem.lookup(BASE, "i", 0) is None


def test_summary_and_listing(mem):
    el = next(e for e in BASE.elements if e.resource_id == "action_bar_title")
    learn(mem, BASE, act(action="tap", target=Target(index=el.index)), el,
          s(X.detail_screen()))
    assert mem.stats_summary()["entries"] == 1
    assert len(mem.entries(app_key="com.android.settings")) == 1
    assert mem.forget(app_key="com.android.settings") == 1


# ---------------------------------------------------------------------------
# Intent keys
# ---------------------------------------------------------------------------

def test_intent_key_ignores_trivial_rewording():
    assert intent_key("Turn on Wi-Fi") == intent_key("turn on   wi-fi")
    assert intent_key("Turn on Wi-Fi") != intent_key("Turn off Wi-Fi")


def test_simhash_large_int_sqlite_compatibility(mem):
    """Test that a screen with simhash >= 2**63 does not overflow SQLite INTEGER."""
    screen_copy = s(X.settings_screen())
    screen_copy.simhash = 2**63 + 12345  # Unsigned 64-bit int exceeding SQLite signed max
    el = next(e for e in screen_copy.elements if e.resource_id == "action_bar_title")
    step = mem.record(
        screen=screen_copy,
        intent_id="i",
        visit=0,
        action=act(action="tap", target=Target(index=el.index)),
        element=el,
        postcondition=None,
        after=s(X.detail_screen()),
        run_id="run1",
    )
    assert step.id > 0
    fetched = mem.get(step.id)
    assert fetched is not None

