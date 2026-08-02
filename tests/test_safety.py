"""Guardrails: credentials, irreversible actions, interstitials, scope, loops,
and the read-only classification that keeps explore mode honest.
"""

from __future__ import annotations

import pytest

from adbagent import safety
from adbagent.actions import AgentAction, Target
from adbagent.config import Config
from adbagent.fingerprint import attach
from adbagent.screen import parse

from . import xmlgen as X


def s(xml: str):
    return attach(parse(xml, width=X.W, height=X.H))


BASE = s(X.settings_screen())


def act(**kw) -> AgentAction:
    kw.setdefault("observation", "o")
    kw.setdefault("reasoning", "r")
    return AgentAction(**kw)


def with_extra(*nodes):
    return s(X.settings_screen(extra_roots=list(nodes)))


# ---------------------------------------------------------------------------
# Credentials -- the agent must never handle these
# ---------------------------------------------------------------------------

def test_password_field_is_detected():
    screen = with_extra(X.N("android.widget.FrameLayout", (0, 600, X.W, 800),
                            rid="login", children=[
                                X.N("android.widget.EditText", (48, 620, 1030, 740),
                                    rid="password", hint="Password", password=True,
                                    clickable=True, focusable=True)]))
    finding = safety.sensitive_screen(screen)
    assert finding is not None and "password" in finding.reason


@pytest.mark.parametrize("hint", [
    "Enter your PIN", "Card number", "CVV", "One-time code",
    "Verification code", "Bank account number", "Social security number",
])
def test_sensitive_input_labels_are_detected(hint):
    screen = with_extra(X.N("android.widget.FrameLayout", (0, 600, X.W, 800),
                            rid="form", children=[
                                X.N("android.widget.EditText", (48, 620, 1030, 740),
                                    rid="field", hint=hint, clickable=True,
                                    focusable=True)]))
    assert safety.sensitive_screen(screen) is not None


def test_an_ordinary_screen_is_not_sensitive():
    assert safety.sensitive_screen(BASE) is None


def test_a_plain_search_box_is_not_sensitive():
    screen = with_extra(X.N("android.widget.FrameLayout", (0, 600, X.W, 800),
                            rid="search_bar", children=[
                                X.N("android.widget.EditText", (48, 620, 1030, 740),
                                    rid="search_src_text", hint="Search settings",
                                    clickable=True, focusable=True)]))
    assert safety.sensitive_screen(screen) is None


# ---------------------------------------------------------------------------
# Irreversible actions
# ---------------------------------------------------------------------------

def test_irreversible_button_is_flagged():
    screen = s(X.detail_screen())
    target = next(e for e in screen.elements if e.best_text == "Forget network")
    label = safety.irreversible(act(action="tap", target=Target(index=target.index)),
                                screen)
    assert label == "Forget network"


def test_ordinary_navigation_is_not_flagged():
    target = next(e for e in BASE.elements if e.best_text == "Wi-Fi") \
        if any(e.best_text == "Wi-Fi" for e in BASE.elements) else BASE.elements[0]
    assert safety.irreversible(
        act(action="tap", target=Target(index=target.index)), BASE) is None


def test_non_tap_actions_are_not_flagged():
    assert safety.irreversible(act(action="press_key", key="back"), BASE) is None
    assert safety.irreversible(act(action="scroll", direction="down"), BASE) is None


def test_unattended_runs_refuse_rather_than_guess():
    cfg = Config()
    cfg.safety.unattended = True
    assert safety.confirm("delete everything?", cfg) is False


def test_allow_destructive_skips_the_prompt():
    cfg = Config()
    cfg.safety.allow_destructive = True
    assert safety.confirm("delete everything?", cfg) is True


# ---------------------------------------------------------------------------
# Interstitials
# ---------------------------------------------------------------------------

def test_a_rating_nag_is_dismissable():
    screen = with_extra(X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
                            package="com.android.vending", rid="nag", children=[
                                X.N("android.widget.TextView", (100, 860, 980, 980),
                                    package="com.android.vending",
                                    text="Enjoying the app?"),
                                X.N("android.widget.Button", (620, 1240, 980, 1380),
                                    package="com.android.vending", text="Not now",
                                    rid="dismiss", clickable=True)]))
    found = safety.find_interstitial(screen, "com.android.settings")
    assert found is not None and found.best_text == "Not now"


def test_permission_dialogs_are_left_to_the_model():
    """Granting a permission is a decision, not noise -- never auto-tapped."""
    screen = with_extra(X.permission_dialog())
    assert safety.find_interstitial(screen, "com.android.settings") is None


def test_nothing_to_dismiss_on_a_normal_screen():
    assert safety.find_interstitial(BASE, "com.android.settings") is None


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def test_repeats_produce_a_hint_then_a_forced_back():
    loops = safety.LoopDetector()
    for _ in range(2):
        loops.record("screen-a", "tap/#1")
    assert loops.hint("screen-a") is None

    loops.record("screen-a", "tap/#1")
    assert "3 times" in (loops.hint("screen-a") or "")
    assert not loops.should_force_back("screen-a")

    for _ in range(2):
        loops.record("screen-a", "tap/#1")
    assert loops.should_force_back("screen-a")


def test_oscillation_between_two_screens_is_detected():
    loops = safety.LoopDetector()
    for _ in range(2):
        loops.record("a", "tap")
        loops.record("b", "back")
    assert loops.oscillating()


def test_a_normal_walk_is_not_oscillation():
    loops = safety.LoopDetector()
    for name in "abcdef":
        loops.record(name, "tap")
    assert not loops.oscillating()


def test_bans_are_per_screen():
    loops = safety.LoopDetector()
    loops.ban("screen-a", "tap/#4")
    assert "tap/#4" in loops.bans_for("screen-a")
    assert loops.bans_for("screen-b") == set()


def test_history_window_is_bounded():
    loops = safety.LoopDetector()
    for i in range(50):
        loops.record(f"s{i}", "tap")
    assert len(loops.history) <= safety.WINDOW


# ---------------------------------------------------------------------------
# Explore mode: read-only classification
# ---------------------------------------------------------------------------

def test_navigation_is_read_only():
    row = next(e for e in BASE.elements if e.resource_id == "row_item")
    ok, _ = safety.is_read_only(act(action="tap", target=Target(index=row.index)),
                                BASE)
    assert ok
    assert safety.is_read_only(act(action="scroll", direction="down"), BASE)[0]
    assert safety.is_read_only(act(action="press_key", key="back"), BASE)[0]


def test_typing_is_never_read_only():
    ok, why = safety.is_read_only(
        act(action="input_text", target=Target(index=1), text="hello"), BASE)
    assert not ok and "typing" in why


def test_toggling_a_setting_is_not_read_only():
    toggle = next(e for e in BASE.elements if e.kind() == "Toggle")
    ok, why = safety.is_read_only(
        act(action="tap", target=Target(index=toggle.index)), BASE)
    assert not ok and "state" in why


def test_pressing_an_irreversible_button_is_not_read_only():
    screen = s(X.detail_screen())
    target = next(e for e in screen.elements if e.best_text == "Forget network")
    ok, why = safety.is_read_only(
        act(action="tap", target=Target(index=target.index)), screen)
    assert not ok and "irreversible" in why


def test_long_press_is_not_read_only():
    ok, why = safety.is_read_only(
        act(action="long_press", target=Target(index=1)), BASE)
    assert not ok and "context menu" in why


def test_enter_key_is_not_read_only():
    """Enter submits forms."""
    ok, _ = safety.is_read_only(act(action="press_key", key="enter"), BASE)
    assert not ok


# ---------------------------------------------------------------------------
# Horizontal scroll oscillation
# ---------------------------------------------------------------------------

def test_left_right_oscillation_is_detected():
    """Alternating left/right scrolls should trigger scroll_oscillating."""
    loops = safety.LoopDetector()
    for _ in range(3):
        loops.record("screen-x", "scroll/left")
        loops.record("screen-x", "scroll/right")
    assert loops.scroll_oscillating()


def test_left_right_not_oscillating_when_too_few():
    loops = safety.LoopDetector()
    loops.record("screen-x", "scroll/left")
    loops.record("screen-x", "scroll/right")
    assert not loops.scroll_oscillating()


def test_scroll_context_mentions_horizontal_axis():
    loops = safety.LoopDetector()
    for _ in range(7):
        loops.record("screen-x", "scroll/right")
    ctx = loops.scroll_context()
    assert ctx is not None
    assert "horizontally" in ctx


def test_scroll_context_mentions_vertical_axis():
    loops = safety.LoopDetector()
    for _ in range(7):
        loops.record("screen-x", "scroll/down")
    ctx = loops.scroll_context()
    assert ctx is not None
    assert "vertically" in ctx
