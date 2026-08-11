"""Action schema, target resolution and the verification DSL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adbagent.actions import (AgentAction, Postcondition, Target, append_history,
                              check_postcondition, describe_target,
                              format_history_entry, resolve_target,
                              synthesise_postcondition, verify)
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


def test_a_signature_prefers_the_content_key_to_the_ordinal():
    """The signature is the primary key of everything the run remembers.

    `LoopDetector.attempts`, the per-screen ban list, the stall-tier refusal set,
    the pager exemption and the 24-hour cross-run `dead_end` rows all key on it,
    and it used to resolve to the bare `#N`. Of the 405 resource-ids seen more
    than once within a single run in ``runs/``, 192 (47%) appeared under more
    than one ordinal -- so a ban earned by `tap/#4` missed the same control when
    it was next listed as #1, and hit whatever else had landed on #4.
    """
    moved = act(action="tap", target=Target(index=1, key="a3f1"))
    same = act(action="tap", target=Target(index=9, key="a3f1"))
    other = act(action="tap", target=Target(index=1, key="b7c2"))

    assert moved.signature() == same.signature(), "the ordinal still decides"
    assert moved.signature() != other.signature(), "two controls share a key"
    # The model and the history still speak in #N -- only what is *remembered*
    # changed.
    assert moved.target.describe() == "#1"


def test_a_target_with_only_a_key_is_valid():
    t = Target(key="a3f1")
    assert t.describe() == "k=a3f1" and t.identity() == "k=a3f1"


def test_a_key_that_contradicts_the_index_re_resolves_to_the_key():
    """The list moved between the dump the model saw and the one being acted on."""
    wanted = BASE.elements[4]
    stale = Target(index=1, key=wanted.key)
    assert resolve_target(stale, BASE) is wanted


def test_a_key_that_is_gone_falls_through_rather_than_tapping_a_stranger():
    assert resolve_target(Target(index=None, key="ffff"), BASE) is None


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


def test_input_text_that_changes_nothing_is_no_change():
    """A byte-identical dump after typing means the field never took focus.

    The element_state postcondition cannot catch this on its own: when the
    field has no resource-id to find, `_find` answers None and a missing
    element is inconclusive-but-passing -- so "nothing was typed" graded a
    success, and the run proceeded as if the text were in.
    """
    action = act(action="input_text", target=Target(index=1), text="hello")
    outcome = verify(action, BASE, BASE)
    assert outcome.grade == "no_change" and not outcome.ok
    assert "never took focus" in outcome.reason


def test_input_text_that_lands_is_success():
    """The guard fires on a *byte-identical* tree only: a field whose text
    moved passes its postcondition exactly as before."""
    before = s(X.chat_thread(draft=""))
    after = s(X.chat_thread(draft="hello"))
    field = next(e for e in before.elements if e.editable)
    action = act(action="input_text", target=Target(index=field.index),
                 text="hello")
    post = synthesise_postcondition(action, field)
    assert post.kind == "element_state" and post.resource_id == "composer"
    assert verify(action, before, after, post).grade == "success"


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


def test_a_wait_is_graded_on_what_it_produced():
    """A wait used to be the one action that could never be wrong.

    `_loop` reads `outcome.ok` to zero `consecutive_failures` and clear
    `last_failure`, so an unconditional `success` let a wait launder the failure
    before it -- a fail/wait alternation could never reach
    `max_consecutive_failures`, nor trip the deeper-thinking and
    take-a-screenshot triggers, which key on the same counter. 13 of 103 turns
    across ``runs/`` were waits.
    """
    assert verify(act(action="wait"), BASE, BASE).grade == "no_change"
    assert verify(act(action="sleep"), BASE, BASE).grade == "no_change"
    # It really waited for something and the screen moved on: that is a success.
    moved = s(X.detail_screen())
    assert verify(act(action="wait"), BASE, moved).grade == "success"


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


def test_input_target_is_container():
    """What the loop's locate guard fires on: a scroller, or anything covering
    most of the screen -- but never an editable field, however big it is."""
    from adbagent.actions import input_target_is_container

    chat = s(X.chat_thread())
    scroller = next(e for e in chat.elements if e.kind() == "Scroller")
    assert input_target_is_container(scroller, chat)
    field = next(e for e in chat.elements if e.editable)
    assert not input_target_is_container(field, chat)
    send = next(e for e in chat.elements if e.best_text == "Send")
    assert not input_target_is_container(send, chat)

    # A big clickable surface that is not a field is a container too.
    canvas = s(X.dump(X.N("android.widget.FrameLayout", (0, 0, X.W, X.H),
                          clickable=True)))
    holder = next(e for e in canvas.elements if e.clickable)
    assert input_target_is_container(holder, canvas)

    # ...while a full-screen editor is the field itself: its centre focuses fine.
    editor = s(X.dump(X.N("android.widget.EditText", (0, 0, X.W, X.H),
                          clickable=True, focusable=True)))
    big_field = next(e for e in editor.elements if e.editable)
    assert not input_target_is_container(big_field, editor)


def test_input_text_taps_the_located_focus_point():
    """When the loop's vision locate placed the field, the focus tap goes to
    that point rather than the (container) target's centre."""
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.chat_thread())
    scroller = next(e for e in screen.elements if e.kind() == "Scroller")

    action = act(action="input_text", target=Target(index=scroller.index),
                 text="hello")
    setattr(action, "_focus_point", (0.42, 0.86))
    execute(dev, action, screen)

    dev.tap.assert_called_once_with(int(0.42 * X.W), int(0.86 * X.H))
    assert dev.tap.call_args.args != scroller.center
    dev.input_text.assert_called_once_with("hello", clear=True, press_enter=False)

    # Without the override the target's own centre is tapped, as before.
    dev2 = MagicMock()
    plain = act(action="input_text", target=Target(index=scroller.index),
                text="hello")
    execute(dev2, plain, screen)
    dev2.tap.assert_called_once_with(*scroller.center)


def test_condition_based_wait():
    from unittest.mock import MagicMock
    from adbagent.actions import execute

    dev = MagicMock()
    screen = s(X.settings_screen())
    dev.observe.return_value = screen

    action = act(action="wait", wait_for_text="Option 1", timeout=2.0)
    execute(dev, action, screen)

    assert "found" in getattr(action, "_result_summary")
    # The text was already there, so nothing changed while it waited.
    assert verify(action, screen, screen).grade == "no_change"


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
    assert verify(action, screen, screen).grade == "no_change"
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
    # The text was already there, so nothing changed while it waited.
    assert verify(action, screen, screen).grade == "no_change"


def test_target_resolution_fallback():
    # Index 1 on BASE is status bar "9:41"
    # If target specifies index=1 but text="Option 4" (mismatch), fallback should search for text "Option 4"
    target = Target(index=1, text="Option 4", resource_id="btn_option4")
    el = resolve_target(target, BASE)
    assert el is not None
    assert "Option 4" in el.best_text




# ---------------------------------------------------------------------------
# Folding repeated history
# ---------------------------------------------------------------------------
#
# Nine consecutive entries in one prompt from `runs/af76720d05c4`, identical but
# for the step number and the observation.

def swipe_entry(step: int, obs: str, grade: str = "success") -> str:
    return format_history_entry(
        step, act(observation=obs, action="swipe", direction="left",
                  target=Target(index=4)),
        screen=BASE, grade=grade)


def test_a_repeated_action_folds_into_one_line_with_a_count():
    history: list = []
    for step in range(30, 39):
        append_history(history, swipe_entry(step, f"photo {step}"))
    assert len(history) == 1
    assert history[0].startswith("30-38.")
    assert "[x9]" in history[0]


def test_folding_keeps_the_readings_because_they_are_the_data():
    history: list = []
    append_history(history, swipe_entry(30, "chicken 425g"))
    append_history(history, swipe_entry(31, "potatoes 403g"))
    assert "chicken 425g" in history[0]
    assert "potatoes 403g" in history[0]


def test_a_repeated_reading_is_not_repeated_in_the_fold():
    history: list = []
    for step in range(30, 35):
        append_history(history, swipe_entry(step, "the caption is hidden"))
    assert history[0].count("the caption is hidden") == 1


def test_a_different_outcome_is_a_different_line():
    history: list = []
    append_history(history, swipe_entry(30, "photo 1", grade="success"))
    append_history(history, swipe_entry(31, "photo 2", grade="no_change"))
    assert len(history) == 2


def test_a_different_action_is_a_different_line():
    history: list = []
    append_history(history, swipe_entry(30, "photo 1"))
    append_history(history, format_history_entry(
        31, act(observation="back to the list", action="press_key", key="back"),
        screen=BASE, grade="success"))
    assert len(history) == 2


def test_only_the_last_line_is_ever_rewritten():
    """`prompts.history_only_block` needs the block to stay append-only for the
    prompt prefix to be cacheable, so a fold must never touch an earlier line."""
    history: list = []
    append_history(history, format_history_entry(
        1, act(observation="home", action="tap", target=Target(index=1)),
        screen=BASE, grade="success"))
    before = list(history)
    for step in range(30, 40):
        append_history(history, swipe_entry(step, f"photo {step}"))
        assert history[:len(before)] == before


def test_the_readings_in_a_fold_are_bounded():
    history: list = []
    for step in range(30, 60):
        append_history(history, swipe_entry(step, f"reading number {step}"))
    assert "+" in history[0] and "more" in history[0]
    assert len(history[0]) < 900


def test_a_fold_survives_a_round_trip_through_its_own_format():
    """Folding is re-entrant: the folded line is parsed back out to fold again."""
    history: list = []
    append_history(history, swipe_entry(30, "photo A"))
    append_history(history, swipe_entry(31, "photo B"))
    append_history(history, swipe_entry(32, "photo C"))
    assert history[0].startswith("30-32.")
    assert "[x3]" in history[0]
    assert all(p in history[0] for p in ("photo A", "photo B", "photo C"))


def test_an_observation_containing_brackets_does_not_break_the_fold():
    history: list = []
    append_history(history, swipe_entry(30, "photo (blurry) shows 425g"))
    append_history(history, swipe_entry(31, "photo (clear) shows 426g"))
    assert len(history) == 1
    assert "425g" in history[0] and "426g" in history[0]


def test_a_line_with_no_step_number_is_never_folded():
    history = ["some free-form note the loop wrote"]
    append_history(history, "another free-form note")
    assert len(history) == 2


# ---------------------------------------------------------------------------
# tap_at -- the coordinate escape hatch
# ---------------------------------------------------------------------------

def test_tap_at_requires_coordinates_or_a_control_name():
    with pytest.raises(ValidationError):
        act(action="tap_at")                    # neither
    with pytest.raises(ValidationError):
        act(action="tap_at", x=0.5)             # only one coordinate
    a = act(action="tap_at", x=0.5, y=0.25)
    assert (a.x, a.y) == (0.5, 0.25)
    # Text mode: the control is named and the vision locate grounds it at act
    # time -- the decider never saw pixels, so it cannot have meant fractions.
    named = act(action="tap_at", text="the record button")
    assert named.x is None and named.text == "the record button"


def test_tap_at_coordinates_are_fractions_of_the_screen():
    with pytest.raises(ValidationError):
        act(action="tap_at", x=1.5, y=0.5)      # past the right edge
    with pytest.raises(ValidationError):
        act(action="tap_at", x=0.5, y=-0.1)     # above the top


def test_tap_at_signature_is_quantised_so_a_retried_guess_still_matches():
    """A blind tap retried a few pixels off is the same action to the loop
    detector -- otherwise the ban list would never catch a blind-tap loop."""
    a = act(action="tap_at", x=0.521, y=0.812)
    b = act(action="tap_at", x=0.524, y=0.814)
    c = act(action="tap_at", x=0.526, y=0.812)
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_tap_at_text_mode_signatures_key_on_the_control_name():
    """Until the locate grounds it, the name is the identity -- so asking for
    the same control twice reads as the repeat it is."""
    a = act(action="tap_at", text="the Record   button")
    b = act(action="tap_at", text="the record button")
    c = act(action="tap_at", text="the stop button")
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_tap_at_execution_converts_fractions_to_clamped_pixels():
    from tests.fake import FakeDevice
    from adbagent.actions import execute

    dev = FakeDevice()
    screen = s(X.settings_screen())             # X.W x X.H = 1080 x 2340

    execute(dev, act(action="tap_at", x=0.5, y=0.5), screen)
    assert dev.taps[-1] == (540, 1170)

    # The edges clamp into the screen -- and never to 0, which u2 would read
    # as a fraction rather than a pixel (device.tap nudges it, but the harness
    # should not rely on that).
    execute(dev, act(action="tap_at", x=0.0, y=1.0), screen)
    assert dev.taps[-1] == (1, X.H - 1)


def test_an_ungrounded_tap_at_never_taps_a_guess():
    """A text-mode tap_at reaches execute only when nothing grounded it; the
    agent loop's vision locate answers before execute, so this is the guard
    against a tap at (1,1)."""
    from tests.fake import FakeDevice
    from adbagent.actions import ActionError, execute

    dev = FakeDevice()
    with pytest.raises(ActionError):
        execute(dev, act(action="tap_at", text="the record button"),
                s(X.settings_screen()))
    assert dev.taps == []


def test_tap_at_needs_known_screen_dimensions():
    from tests.fake import FakeDevice
    from adbagent.actions import ActionError, execute

    dev = FakeDevice()
    # No nodes and no explicit size -> the parser's 0x0 fallback.
    empty = attach(parse("<hierarchy rotation=\"0\" />"))
    assert empty.width == 0 and empty.height == 0
    with pytest.raises(ActionError):
        execute(dev, act(action="tap_at", x=0.5, y=0.5), empty)


def test_tap_at_that_changes_nothing_grades_no_change():
    """The grade feeds the ban list, so the same dud blind tap is not retried."""
    action = act(action="tap_at", x=0.5, y=0.5)
    assert verify(action, BASE, BASE).grade == "no_change"

    other = s(X.settings_screen(rows=3, title="Bluetooth",
                                labels=["Pair new device", "Previously connected",
                                        "Bluetooth settings"]))
    assert verify(action, BASE, other).grade == "success"


def test_element_at_point_finds_the_button_sized_control_under_a_point():
    from adbagent.actions import element_at_point
    # The centre of the first row of the scripted settings screen.
    el = element_at_point(BASE, 0.5, 580 / X.H)
    assert el is not None and el.best_text == "Option 1"


def test_element_at_point_ignores_containers_too_big_to_be_the_target():
    """A map or video surface IS listed, but what is wanted inside it is not
    -- so landing on the surface alone is legitimate tap_at territory."""
    from adbagent.actions import element_at_point
    xml = X.dump(X.N("android.widget.FrameLayout", (0, 0, X.W, X.H),
                     rid="content", children=[
        X.N("android.view.View", (0, 0, X.W, 1900), rid="map", clickable=True),
        X.N("android.widget.Button", (400, 400, 680, 520), text="Go",
            rid="go", clickable=True)]))
    screen = s(xml)
    assert element_at_point(screen, 0.5, 460 / X.H).best_text == "Go"
    assert element_at_point(screen, 0.5, 0.8) is None
