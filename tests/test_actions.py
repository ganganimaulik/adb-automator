"""Action schema, target resolution and the verification DSL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adbagent.actions import (AgentAction, Postcondition, Target, check_postcondition,
                              describe_target, format_history_entry,
                              resolve_target, synthesise_postcondition, verify)
from adbagent.fingerprint import attach
from adbagent.screen import parse

from . import xmlgen as X


def s(xml: str):
    return attach(parse(xml, width=X.W, height=X.H))


BASE = s(X.settings_screen())


def act(**kw) -> AgentAction:
    kw.setdefault("observation", "a settings list")
    kw.setdefault("reasoning", "because")
    return AgentAction(**kw)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_is_flat_with_no_oneof():
    """oneOf is rejected by OpenAI strict mode and handled badly by OSS models."""
    schema = AgentAction.model_json_schema()
    assert "oneOf" not in repr(schema)
    props = list(schema["properties"])
    # Order is load-bearing: constrained decoders emit properties in order, so
    # the model reasons before it commits to an action.
    assert props[:4] == ["observation", "reasoning", "action", "target"]


def test_required_arguments_are_enforced():
    with pytest.raises(ValidationError):
        act(action="tap")                       # no target
    with pytest.raises(ValidationError):
        act(action="input_text", target=Target(index=1))   # no text
    with pytest.raises(ValidationError):
        act(action="press_key")                 # no key
    with pytest.raises(ValidationError):
        act(action="scroll")                    # no direction
    with pytest.raises(ValidationError):
        act(action="open_app")                  # no package
    with pytest.raises(ValidationError):
        Target()                                # nothing to locate by


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        act(action="press_key", key="back", coordinates=[10, 20])


def test_unknown_key_is_rejected():
    """Only names the on-device server accepts."""
    with pytest.raises(ValidationError):
        act(action="press_key", key="escape")


def test_signature_is_stable_and_distinguishing():
    a = act(action="tap", target=Target(index=3))
    b = act(action="tap", target=Target(index=3), observation="different words")
    c = act(action="tap", target=Target(index=4))
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_terminal_actions():
    assert act(action="done", text="ok").is_terminal
    assert act(action="fail", text="stuck").is_terminal
    assert act(action="ask_user", text="what pin?").is_terminal
    assert not act(action="press_key", key="back").is_terminal


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def test_resolve_by_index():
    el = resolve_target(Target(index=1), BASE)
    assert el is not None and el.index == 1


def test_resolve_by_resource_id():
    el = resolve_target(Target(resource_id="action_bar_title"), BASE)
    assert el is not None and el.text == "Network & internet"


def test_resolve_by_text_is_fuzzy_not_exact():
    """Labels drift between app versions and locales; `==` is too brittle."""
    assert resolve_target(Target(text="Network & internet"), BASE) is not None
    loose = resolve_target(Target(text="network"), BASE)
    assert loose is not None


def test_resolve_prefers_the_shortest_loose_match():
    el = resolve_target(Target(text="Option 1"), BASE)
    assert el is not None and "Option 1" in el.best_text


def test_resolve_returns_none_when_absent():
    assert resolve_target(Target(text="Nonexistent control"), BASE) is None
    assert resolve_target(Target(index=9999), BASE) is None


def test_ambiguous_resource_id_disambiguated_by_text():
    """Seven rows share resource-id 'title'; the text picks one."""
    el = resolve_target(Target(resource_id="title", text="Option 4"), BASE)
    assert el is not None and el.best_text == "Option 4"


# ---------------------------------------------------------------------------
# Postcondition synthesis -- the two silent-failure cases
# ---------------------------------------------------------------------------

def test_toggle_tap_checks_state_not_navigation():
    toggle = next(e for e in BASE.elements if e.kind() == "Toggle")
    post = synthesise_postcondition(act(action="tap", target=Target(index=toggle.index)),
                                    toggle)
    assert post.kind == "element_state" and post.field == "checked"
    assert post.value == "true"          # it was off, so it must end up on


def test_typing_must_not_require_navigation():
    field = next(e for e in BASE.elements if e.kind() == "Text")
    post = synthesise_postcondition(
        act(action="input_text", target=Target(index=1), text="hello"), field)
    assert post.kind == "element_state" and post.field == "text"


def test_open_app_checks_foreground_package():
    post = synthesise_postcondition(
        act(action="open_app", text="com.android.chrome"), None)
    assert post.kind == "app_is" and post.package == "com.android.chrome"


def test_plain_tap_falls_back_to_screen_changed():
    post = synthesise_postcondition(act(action="tap", target=Target(index=1)), None)
    assert post.kind == "screen_changed"


# ---------------------------------------------------------------------------
# Postcondition evaluation
# ---------------------------------------------------------------------------

def test_screen_changed():
    other = s(X.detail_screen())
    assert check_postcondition(Postcondition(kind="screen_changed"), BASE, other)[0]
    assert not check_postcondition(Postcondition(kind="screen_changed"), BASE, BASE)[0]


def test_element_state_checked():
    after = s(X.settings_screen(checked_row=0))
    post = Postcondition(kind="element_state", resource_id="switch_widget",
                         field="checked", value="true")
    assert check_postcondition(post, BASE, after)[0]
    # ...and the same check against the unflipped screen must fail.
    ok, why = check_postcondition(post, BASE, BASE)
    assert not ok and "expected true" in why


def test_app_is():
    post = Postcondition(kind="app_is", package="com.android.settings")
    assert check_postcondition(post, BASE, BASE)[0]
    bad = Postcondition(kind="app_is", package="com.android.chrome")
    ok, why = check_postcondition(bad, BASE, BASE)
    assert not ok and "com.android.chrome" in why


def test_text_present():
    assert check_postcondition(
        Postcondition(kind="text_present", text="Option 3"), BASE, BASE)[0]
    assert not check_postcondition(
        Postcondition(kind="text_present", text="Bluetooth pairing"), BASE, BASE)[0]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def test_navigational_tap_that_changes_nothing_is_no_change():
    outcome = verify(act(action="tap", target=Target(index=1)), BASE, BASE)
    assert outcome.grade == "no_change"
    assert not outcome.ok


def test_successful_navigation():
    outcome = verify(act(action="tap", target=Target(index=1)), BASE, s(X.detail_screen()))
    assert outcome.grade == "success" and outcome.ok


def test_toggle_flip_is_success_even_though_the_screen_barely_changed():
    toggle = next(e for e in BASE.elements if e.kind() == "Toggle")
    action = act(action="tap", target=Target(index=toggle.index))
    after = s(X.settings_screen(checked_row=0))
    post = synthesise_postcondition(action, toggle)
    assert verify(action, BASE, after, post).grade == "success"


def test_unexpected_successor_is_soft_not_hard_failure():
    """Screens legitimately fan out; note it, do not punish it."""
    after = s(X.detail_screen())
    outcome = verify(act(action="tap", target=Target(index=1)), BASE, after,
                     expected_skeleton="some-other-skeleton")
    assert outcome.grade == "soft_fail" and outcome.ok


def test_failed_postcondition_is_hard_failure():
    action = act(action="tap", target=Target(index=1))
    post = Postcondition(kind="text_present", text="Never appears")
    outcome = verify(action, BASE, s(X.detail_screen()), post)
    assert outcome.grade == "hard_fail" and not outcome.ok


def test_wait_always_succeeds():
    assert verify(act(action="wait"), BASE, BASE).grade == "success"


def test_describe_untruncated_text():
    long_msg = "Bumble matches checked: 5 matches visible — K, Z, A, D, and R (all Date matches), with recent conversation previews shown on the Chats screen."
    action = act(action="done", text=long_msg)
    assert action.describe() == f"done {long_msg}"


def test_describe_target_with_element():
    el = BASE.elements[0]
    target = Target(index=el.index)
    desc = describe_target(target, el)
    assert f"#{el.index}" in desc
    assert el.kind() in desc
    assert el.best_text in desc


def test_format_history_entry_rich_context():
    el = BASE.elements[0]
    action = act(action="tap", target=Target(index=el.index), observation="settings home")
    entry = format_history_entry(1, action, screen=BASE, element=el, grade="success")
    assert "1." in entry
    assert "tap #1" in entry
    assert f"in {BASE.package}" in entry
    assert "(Obs: settings home)" in entry
    assert "-> success" in entry


def test_format_history_entry_untruncated_observation():
    el = BASE.elements[0]
    long_obs = ("WhatsApp media viewer showing Krishna's 9:25 am photo: mixed nuts "
                "(almonds, cashews, walnuts) on scale reading 6g.")
    action = act(action="tap", target=Target(index=el.index), observation=long_obs)
    entry = format_history_entry(37, action, screen=BASE, element=el)
    assert f"(Obs: {long_obs})" in entry


# ---------------------------------------------------------------------------
# Multi-signal end-of-scroll detection
# ---------------------------------------------------------------------------

def test_scroll_no_change_with_identical_exact_id():
    """Signal 1: byte-identical hierarchy means nothing moved."""
    action = act(action="scroll", direction="down")
    outcome = verify(action, BASE, BASE)
    assert outcome.grade == "no_change"


def test_scroll_no_change_near_identical_screen():
    """Signal 2: same skeleton + simhash close + scroller content unchanged.

    A toggle flip changes exact_id (checked state is in exact_id) but not
    skeleton or simhash.  With the same scroller content, this should still
    be detected as a scroll no-op.
    """
    before = s(X.settings_screen(checked_row=-1))
    after = s(X.settings_screen(checked_row=0))
    # Verify the exact_ids are different (the checked state changed).
    assert before.exact_id != after.exact_id
    # But the skeleton is the same and simhash is close.
    assert before.skeleton_id == after.skeleton_id
    action = act(action="scroll", direction="down")
    outcome = verify(action, before, after)
    assert outcome.grade == "no_change"


def test_scroll_no_change_via_text_overlap():
    """Signal 3: ≥90% of scroller child texts are identical.

    A different tab is selected (changes exact_id and may move simhash a few
    bits), but the list content is the same.
    """
    labels = ["Wi-Fi", "Bluetooth", "Mobile data", "Airplane mode",
              "Hotspot", "VPN", "DNS"]
    before = s(X.settings_screen(labels=labels, selected_tab="network"))
    after = s(X.settings_screen(labels=labels, selected_tab="devices"))
    # The selected tab differs → exact_id differs.
    assert before.exact_id != after.exact_id
    action = act(action="scroll", direction="down")
    outcome = verify(action, before, after)
    assert outcome.grade == "no_change"


def test_genuine_scroll_is_detected_as_change():
    """When the screen genuinely changed, the scroll is NOT graded no_change.

    Uses a structurally different screen as the 'after' to simulate content
    that actually scrolled into view and changed the hierarchy.
    """
    before = s(X.settings_screen())
    after = s(X.detail_screen())
    action = act(action="scroll", direction="down")
    outcome = verify(action, before, after)
    # The screen genuinely changed — it should NOT be no_change.
    assert outcome.grade != "no_change"


# ---------------------------------------------------------------------------
# Horizontal scroll rendering
# ---------------------------------------------------------------------------

def test_horizontal_scroller_shows_scrollable_h_flag():
    from adbagent.screen import render_element
    screen = s(X.horizontal_scroll_screen())
    scroller = next(e for e in screen.elements if e.scrollable)
    rendered = render_element(scroller)
    assert "scrollable-h" in rendered


def test_vertical_scroller_shows_scrollable_flag():
    from adbagent.screen import render_element
    scroller = next(e for e in BASE.elements if e.scrollable)
    rendered = render_element(scroller)
    assert "scrollable" in rendered
    assert "scrollable-h" not in rendered


# ---------------------------------------------------------------------------
# Swipe action & Photo swiping
# ---------------------------------------------------------------------------

def test_swipe_action_validation():
    a = act(action="swipe", direction="left", scroll_amount=1.5, duration=0.15)
    assert a.action == "swipe"
    assert a.direction == "left"
    assert a.duration == 0.15

    with pytest.raises(ValidationError):
        act(action="swipe")  # no direction


def test_swipe_and_scroll_execution_target_bounds():
    from tests.fake import FakeDevice
    from adbagent.actions import execute
    
    dev = FakeDevice()
    screen = s(X.settings_screen())
    
    # Target element #1 (whether scrollable or not)
    action_swipe = act(action="swipe", target=Target(index=1), direction="left")
    execute(dev, action_swipe, screen)
    assert any("scroll(left)" in act for act in dev.actions)

    action_scroll = act(action="scroll", target=Target(index=1), direction="right")
    execute(dev, action_scroll, screen)
    assert any("scroll(right)" in act for act in dev.actions)


def test_textless_photo_screen_scroll_changed():
    """Screens without text inside scrollers (e.g. photo galleries) verify change on exact_id delta."""
    xml_photo_1 = """
    <hierarchy rotation="0">
      <node index="0" text="" content-desc="Photo 1" resource-id="com.whatsapp:id/photo" class="android.widget.ImageView" package="com.whatsapp" bounds="[0,0][1080,1920]" enabled="true" clickable="true" scrollable="false" />
    </hierarchy>
    """
    xml_photo_2 = """
    <hierarchy rotation="0">
      <node index="0" text="" content-desc="Photo 2" resource-id="com.whatsapp:id/photo2" class="android.widget.ImageView" package="com.whatsapp" bounds="[0,0][1080,1920]" enabled="true" clickable="true" scrollable="false" />
    </hierarchy>
    """
    before = s(xml_photo_1)
    after = s(xml_photo_2)
    assert before.exact_id != after.exact_id
    
    action = act(action="swipe", direction="left")
    outcome = verify(action, before, after)
    assert outcome.grade != "no_change"


def test_swipe_or_horizontal_scroll_on_identical_exact_id_is_not_no_change():
    """Swipe/horizontal scroll on identical XML trees (e.g. image gallery Bitmaps) should NOT grade as no_change."""
    xml_photo = """
    <hierarchy rotation="0">
      <node index="0" text="" content-desc="" resource-id="com.whatsapp:id/photo" class="android.widget.ImageView" package="com.whatsapp" bounds="[0,0][1080,1920]" enabled="true" clickable="true" scrollable="false" />
    </hierarchy>
    """
    before = s(xml_photo)
    after = s(xml_photo)
    assert before.exact_id == after.exact_id

    action_swipe = act(action="swipe", direction="left")
    outcome_swipe = verify(action_swipe, before, after)
    assert outcome_swipe.grade == "success"

    action_hscroll = act(action="scroll", direction="right")
    outcome_hscroll = verify(action_hscroll, before, after)
    assert outcome_hscroll.grade == "success"

    action_vscroll = act(action="scroll", direction="down")
    outcome_vscroll = verify(action_vscroll, before, after)
    assert outcome_vscroll.grade == "no_change"


def test_perceptual_dhash_signal_0_in_scroll_changed():
    """Signal 0: If dHash perceptual distance >= 4, verify visual change even if exact_id is identical."""
    from adbagent.actions import _scroll_changed
    xml_photo = """
    <hierarchy rotation="0">
      <node index="0" text="" content-desc="" resource-id="com.whatsapp:id/photo" class="android.widget.ImageView" package="com.whatsapp" bounds="[0,0][1080,1920]" enabled="true" clickable="true" scrollable="false" />
    </hierarchy>
    """
    before = s(xml_photo)
    after = s(xml_photo)
    assert before.exact_id == after.exact_id

    # Simulated distinct image dHashes (e.g. Photo 1 vs Photo 2)
    before.dhash = 0b1111000011110000111100001111000011110000111100001111000011110000
    after.dhash  = 0b0000111100001111000011110000111100001111000011110000111100001111

    # Visual bitmap change detected via dHash distance
    assert _scroll_changed(before, after) is True

    # Same visual image dHash
    after.dhash = before.dhash
    assert _scroll_changed(before, after) is False




def test_scroll_base_scale_validation():
    a = act(action="scroll", direction="down", base_scale=0.8)
    assert a.base_scale == 0.8

    with pytest.raises(ValidationError):
        act(action="scroll", direction="down", base_scale=0.05)  # < 0.1

    with pytest.raises(ValidationError):
        act(action="scroll", direction="down", base_scale=1.2)   # > 1.0


def test_scroll_execution_with_custom_base_scale():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())

    # Default base_scale (0.6)
    action_default = act(action="scroll", direction="down")
    execute(dev, action_default, screen)
    dev.scroll.assert_called_with("down", scale=0.6, box=None, duration=0.3)

    dev.scroll.reset_mock()

    # Custom base_scale (0.8)
    action_custom = act(action="scroll", direction="down", base_scale=0.8)
    execute(dev, action_custom, screen)
    dev.scroll.assert_called_with("down", scale=0.8, box=None, duration=0.3)


def test_scroll_describe_with_base_scale():
    a = act(action="scroll", direction="down", base_scale=0.8)
    desc = a.describe()
    assert "scroll down" in desc
    assert "base_scale=0.8" in desc


def test_list_apps_action():
    a = act(action="list_apps", text="whatsapp")
    assert a.action == "list_apps"
    assert a.text == "whatsapp"
    assert a.describe() == "list_apps whatsapp"

    b = act(action="list_apps")
    assert b.action == "list_apps"
    assert b.text is None
    assert b.describe() == "list_apps"


def test_list_apps_execution():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    dev.list_apps.return_value = ["com.whatsapp", "com.whatsapp.w4b"]
    screen = s(X.settings_screen())

    action = act(action="list_apps", text="whatsapp")
    execute(dev, action, screen)

    dev.list_apps.assert_called_once_with(query="whatsapp")
    assert getattr(action, "_result_summary") == "found 2 app(s): com.whatsapp, com.whatsapp.w4b"

    outcome = verify(action, screen, screen)
    assert outcome.grade == "success"
    assert "com.whatsapp" in outcome.reason


def test_clipboard_actions():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    dev.get_clipboard.return_value = "hello clipboard"
    screen = s(X.settings_screen())

    get_act = act(action="get_clipboard")
    execute(dev, get_act, screen)
    dev.get_clipboard.assert_called_once()
    assert verify(get_act, screen, screen).reason == "clipboard content: 'hello clipboard'"

    set_act = act(action="set_clipboard", text="new text")
    execute(dev, set_act, screen)
    dev.set_clipboard.assert_called_once_with("new text")
    assert verify(set_act, screen, screen).reason == "set clipboard to 'new text'"


def test_smart_open_app():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    dev.list_apps.return_value = ["com.spotify.music", "com.spotify.lite"]
    screen = s(X.settings_screen())

    # Common app name fuzzy resolution
    action = act(action="open_app", text="spotify")
    execute(dev, action, screen)

    dev.list_apps.assert_called_once_with(query="spotify")
    dev.open_app.assert_called_once_with("com.spotify.music")
    assert getattr(action, "_resolved_package") == "com.spotify.music"


def test_smarter_input_text():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())

    action = act(action="input_text", target=Target(index=1), text="search query", clear=False, press_enter=True)
    execute(dev, action, screen)

    dev.input_text.assert_called_once_with("search query", clear=False, press_enter=True)


def test_condition_based_wait():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())
    dev.observe.return_value = screen

    action = act(action="wait", wait_for_text="Option 1", timeout=2.0)
    execute(dev, action, screen)

    assert "found" in getattr(action, "_result_summary")
    assert verify(action, screen, screen).grade == "success"


def test_sleep_action():
    from unittest.mock import MagicMock, patch
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())

    action = act(action="sleep", duration=2.0)
    with patch("time.sleep") as mock_sleep:
        execute(dev, action, screen)
        mock_sleep.assert_called_once_with(2.0)

    assert getattr(action, "_result_summary") == "slept for 2.0s"
    assert verify(action, screen, screen).grade == "success"
    assert synthesise_postcondition(action, None).kind == "noop_ok"


def test_condition_based_sleep():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())
    dev.observe.return_value = screen

    action = act(action="sleep", wait_for_text="Option 1", timeout=2.0)
    execute(dev, action, screen)

    assert "found" in getattr(action, "_result_summary")
    assert verify(action, screen, screen).grade == "success"


def test_target_resolution_fallback():
    # Index 1 on BASE is status bar "9:41"
    # If target specifies index=1 but text="Option 4" (mismatch), fallback should search for text "Option 4"
    target = Target(index=1, text="Option 4", resource_id="btn_option4")
    el = resolve_target(target, BASE)
    assert el is not None
    assert "Option 4" in el.best_text




