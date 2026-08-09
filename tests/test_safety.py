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


def _lone_button(text: str, package: str = X.PKG):
    """A screen whose only dismiss-shaped candidate reads `text`."""
    return with_extra(X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
                          package=package, rid="overlay", children=[
                              X.N("android.widget.Button", (620, 1240, 980, 1380),
                                  package=package, text=text, rid="cta",
                                  clickable=True)]))


@pytest.mark.parametrize("label", ["Continue", "Next", "Close"])
def test_a_flow_control_is_left_to_the_model(label):
    """These advance or exit a flow rather than declining an offer.

    An app opening on an onboarding screen had its CTA pressed on every turn
    until the step budget ran out, with the model never consulted.
    """
    assert safety.find_interstitial(_lone_button(label), X.PKG) is None


def test_a_compose_screens_close_button_is_not_auto_tapped():
    """The sharpest case: one X described as "Close" next to a "Send" is a
    single dismiss-shaped candidate, so the old rule fired and discarded the
    draft -- and `irreversible` never saw it, since that grades only the actions
    the model chose."""
    screen = with_extra(X.N("android.widget.FrameLayout", (0, 600, X.W, 900),
                            rid="compose", children=[
                                X.N("android.widget.ImageButton", (24, 620, 144, 740),
                                    desc="Close", rid="close", clickable=True),
                                X.N("android.widget.Button", (860, 620, 1040, 740),
                                    text="Send", rid="send", clickable=True)]))
    assert safety.find_interstitial(screen, X.PKG) is None


@pytest.mark.parametrize("label", ["Not now", "No thanks", "Skip", "Got it",
                                   "Maybe later", "Don't show again"])
def test_declines_and_acknowledgements_are_still_dismissed(label):
    found = safety.find_interstitial(_lone_button(label), X.PKG)
    assert found is not None and found.best_text == label


def test_a_flow_control_from_another_package_is_left_alone_too():
    """The cross-package branch skips the foreground app's own controls, so a
    Play billing sheet's "Continue" was reachable there as well."""
    screen = _lone_button("Continue", package="com.android.vending")
    assert safety.find_interstitial(screen, X.PKG) is None


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
    for _ in range(5):
        loops.record("screen-a", "tap/#1")
    assert loops.hint("screen-a") is None

    loops.record("screen-a", "tap/#1")
    assert "6 times" in (loops.hint("screen-a") or "")
    assert not loops.should_force_back("screen-a")

    for _ in range(3):
        loops.record("screen-a", "tap/#1")
    assert loops.should_force_back("screen-a")


def test_oscillation_between_two_screens_is_detected():
    loops = safety.LoopDetector()
    # Needs 5 full repetitions of the 2-step cycle
    for _ in range(5):
        loops.record("a", "tap")
        loops.record("b", "back")
    assert loops.oscillating()


def test_walking_a_list_is_not_oscillation_though_the_screens_repeat():
    """The case that forces the action into the key.

    Opening ten grid items in turn bounces between exactly two screens, which
    is what a stuck two-cycle looks like if you only compare screen ids. The
    difference is entirely in the action: ``tap/#1``, ``tap/#2``, ``tap/#3``.
    """
    loops = safety.LoopDetector()
    for i in range(1, 6):
        loops.record("grid", f"tap/#{i}")
        loops.record("item", "press_key/back")
    assert not loops.oscillating()

    # Whereas the same walk stuck on one item is a loop, and this is the shape
    # `runs/2521862d7a23` ran for twenty steps without anything noticing.
    stuck = safety.LoopDetector()
    for _ in range(5):
        stuck.record("grid", "tap/#7")
        stuck.record("item", "press_key/back")
    assert stuck.oscillating()


def test_paging_one_gesture_is_not_oscillation():
    """Doing one thing over and over is not a cycle, and must not read as one.

    ``[A,A,A,A,...]`` satisfies the period-2 and period-3 patterns trivially, so
    without the distinct-entries rule an album walk -- twelve identical swipes
    on one screen -- was reported as oscillation. The remedy that fires on it is
    a back press, which ejects the agent from the album it is halfway through.

    Repeating one action is `should_force_back`'s question, and that one knows
    a browsing gesture from a thrashing one.
    """
    loops = safety.LoopDetector()
    for _ in range(12):
        loops.record("album", "swipe/#4/left")
    assert not loops.oscillating()
    assert not loops.should_force_back("album")

    # Vertical feeds page the same way and are excluded on the same grounds.
    feed = safety.LoopDetector()
    for _ in range(12):
        feed.record("reels", "scroll/up")
    assert not feed.oscillating()
    assert not feed.should_force_back("reels")


def test_force_back_counts_one_action_not_every_action():
    """Nine different taps on a hub screen are work; nine identical ones are not."""
    working = safety.LoopDetector()
    for i in range(1, 10):
        working.record("hub", f"tap/#{i}")
    assert not working.should_force_back("hub")

    stuck = safety.LoopDetector()
    for _ in range(9):
        stuck.record("hub", "tap/#3")
    assert stuck.should_force_back("hub")


def test_attempts_outlive_the_ring_buffer():
    """`history` keeps twenty entries; "have I tried this here" outlasts them."""
    loops = safety.LoopDetector()
    loops.record("grid", "tap/#7")
    for i in range(safety.WINDOW * 2):
        loops.record(f"elsewhere{i}", "tap/#1")
    assert not any(sid == "grid" for sid, _ in loops.history)
    assert loops.times_on("grid", "tap/#7") == 1
    assert loops.times_on("grid", "tap/#9") == 0


def test_tried_on_reports_the_worn_paths_first():
    loops = safety.LoopDetector()
    for _ in range(3):
        loops.record("grid", "tap/#7")
    loops.record("grid", "press_key/back")
    loops.record("other", "tap/#2")
    assert loops.tried_on("grid") == [("tap/#7", 3), ("press_key/back", 1)]


def test_a_normal_walk_is_not_oscillation():
    loops = safety.LoopDetector()
    for name in "abcdef":
        loops.record(name, "tap")
    assert not loops.oscillating()

    # Two repetitions of a cycle should NOT be enough anymore
    loops2 = safety.LoopDetector()
    for _ in range(2):
        loops2.record("a", "tap")
        loops2.record("b", "back")
    assert not loops2.oscillating()


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


def test_element_history_tracking_and_hints():
    loops = safety.LoopDetector()
    assert loops.element_history_hint("album_screen") is None

    loops.record_element_action("album_screen", 29, "tap/k=a3f1", "tap #6 [View photo 1 of 10]")
    hint = loops.element_history_hint("album_screen")
    assert hint is not None
    assert "PREVIOUS ACTIONS ON THIS SCREEN: step 29: tap #6 [View photo 1 of 10]" in hint
    assert "match on the k= value, not the #N" in hint
    assert loops.element_history_hint("other_screen") is None


def test_the_pager_exemption_only_excuses_the_pager_gesture():
    """Taken from ``runs/2521862d7a23``, which lost its goal to this hint.

    The agent was in a hard two-cycle -- tap #7, back, tap #7, back -- on a
    screen whose pager was a different element entirely. The hint recited five
    consecutive back presses and then told the agent that "repeating it is
    correct" and not to substitute anything else. Nothing in the loop touched the
    pager.

    `repeatable` is an `Element.key` rather than an ordinal now, matching the
    content-keyed form `AgentAction.signature` emits -- an ordinal would never
    match one of those, silently withdrawing the exemption from the one element
    it exists for.
    """
    loops = safety.LoopDetector()
    for step in (10, 12, 14, 16, 18):
        loops.record_element_action("reels", step, "press_key/back", "press_key back")

    hint = loops.element_history_hint("reels", repeatable="9c2b")
    assert "repeating it is correct" not in hint
    assert "match on the k= value, not the #N" in hint

    # But a real pager sweep still gets its exemption, which is why it exists.
    loops.record_element_action("reels", 20, "swipe/k=9c2b/left", "swipe #4 left")
    swept = loops.element_history_hint("reels", repeatable="9c2b")
    assert "repeating it is correct" in swept

    # And a swipe at a *different* element does not earn it.
    fresh = safety.LoopDetector()
    fresh.record_element_action("reels", 20, "swipe/k=0000/left", "swipe #7 left")
    assert "repeating it is correct" not in fresh.element_history_hint(
        "reels", repeatable="9c2b")


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
