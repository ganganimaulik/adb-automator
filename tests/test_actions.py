"""Action schema, target resolution and the verification DSL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adbagent.actions import (AgentAction, Postcondition, Target, check_postcondition,
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
