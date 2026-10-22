"""The send side of a chat screen: which actions send, and what has been sent.

Who said what in a thread is not read here any more -- the model reads it off the
screenshot (see `adbagent/conversation.py`). Thirty-odd tests of the layout
reader stood here, and went with it.
"""

from __future__ import annotations

import pytest

from adbagent.actions import AgentAction, Target
from adbagent.config import Config
from adbagent.conversation import (SendLog, draft_in, draft_refusal,
                                   has_text_field, parse_send_limits,
                                   send_label)
from adbagent.fingerprint import attach
from adbagent.screen import parse

from . import xmlgen as X


def screen(**kw):
    return attach(parse(X.chat_thread(**kw), width=X.W, height=X.H))


def tools(**kw):
    """A thread whose text field sits among the app's own controls."""
    return attach(parse(X.chat_with_composer_tools(**kw), width=X.W, height=X.H))


def cfg(**watch) -> Config:
    c = Config()
    for k, v in watch.items():
        setattr(c.watch, k, v)
    return c


def tap_send(s) -> AgentAction:
    """A tap on whatever the Send button was rendered as."""
    el = next(e for e in s.elements if e.resource_id == "send_button")
    return AgentAction(observation="in a thread", reasoning="send the reply",
                       action="tap", target=Target(index=el.index))


def tap_on(s, el) -> AgentAction:
    return AgentAction(observation="x", reasoning="y", action="tap",
                       target=Target(index=el.index))


# -- a screen worth a screenshot ------------------------------------------------

def test_a_screen_with_somewhere_to_type_has_a_text_field():
    """What `agent.needs_screenshot` keys on: every conversation has one."""
    assert has_text_field(screen())
    assert has_text_field(tools())
    settings = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    assert not has_text_field(settings)


# -- the controls around a text field ---------------------------------------------
#
# ``runs/425beba3ca6c``: the layout reader took a camera, a GIF and a voice control
# for messages, and the voice control -- hugging the right edge -- for the newest
# one, ours. Reading the thread is the model's now, off the screenshot; what is
# left to hold here is that none of those controls is mistaken for a send either.

def test_the_controls_around_a_text_field_are_not_sends():
    s = tools()
    controls = [e for e in s.elements
                if e.resource_id in ("attach_button", "gif_button",
                                     "action_button", "voice_record_icon")]
    assert len(controls) == 4, [e.best_text for e in s.elements]
    for el in controls:
        assert send_label(tap_on(s, el), s) == "", el.best_text


def test_a_bubble_is_not_a_send_even_when_it_is_a_button():
    """These bubbles are clickable, as they are in apps that open a menu on a
    long press. Being tappable does not make one a control that sends."""
    s = tools(messages=["can you send me the address?"])
    bubble = next(e for e in s.elements if e.resource_id == "bubble")
    assert bubble.interactive
    assert send_label(tap_on(s, bubble), s) == ""


def test_the_send_button_that_replaces_the_voice_control_is_a_send():
    s = tools(draft="sounds good, saturday?")
    assert send_label(tap_send(s), s) == "Send"
    assert draft_in(tap_send(s), s) == "sounds good, saturday?"


def test_a_placeholder_in_the_text_field_is_not_a_draft():
    s = tools()
    gif = next(e for e in s.elements if e.resource_id == "gif_button")
    assert draft_in(tap_on(s, gif), s) == ""


# -- which actions are sends ------------------------------------------------

def test_tap_on_send_is_a_send():
    s = screen()
    assert send_label(tap_send(s), s) == "Send"


def test_tap_elsewhere_is_not_a_send():
    s = screen()
    back = next(e for e in s.elements if e.resource_id == "back")
    assert send_label(tap_on(s, back), s) == ""


def test_enter_key_with_a_focused_composer_is_a_send():
    s = screen(composer_focused=True)
    act = AgentAction(observation="x", reasoning="y", action="press_key",
                      key="enter")
    assert send_label(act, s) == "enter"


def test_enter_key_without_a_focused_composer_is_not():
    s = screen(composer_focused=False)
    act = AgentAction(observation="x", reasoning="y", action="press_key",
                      key="enter")
    assert send_label(act, s) == ""


def _type_into_composer(s, **kw) -> AgentAction:
    el = next(e for e in s.elements if e.resource_id == "composer")
    return AgentAction(observation="x", reasoning="y", action="input_text",
                       target=Target(index=el.index), text="hi", **kw)


def test_input_text_with_press_enter_is_a_send():
    """Types and sends in one step -- the door gating the button would miss."""
    s = screen()
    assert send_label(_type_into_composer(s, press_enter=True), s) != ""


def test_input_text_without_press_enter_is_not():
    s = screen()
    assert send_label(_type_into_composer(s), s) == ""


def test_a_message_containing_the_word_send_is_not_a_control():
    """Refusing a tap on a bubble would strand the loop on a readable screen."""
    s = screen(messages=["can you send me the file when you get a chance"])
    for el in s.elements:
        if el.resource_id == "message_list":
            assert send_label(tap_on(s, el), s) == ""


# -- the only refusal left --------------------------------------------------
#
# Eight tests stood here against `reply_gate`: the same tail already replied to,
# a per-thread cooldown, a per-thread hourly ceiling, a global hourly ceiling,
# the rolling window those two counted in, and failing closed on a thread that
# could not be identified. Every one of them was a reading of the reply ledger,
# and all of it is gone -- whether a reply is owed is the model's call now, made
# from the screen. What is left refuses nothing except on request.

def test_draft_mode_never_sends():
    s = screen()
    why = draft_refusal(tap_send(s), s, cfg(draft=True))
    assert "draft mode is on" in why


def test_draft_mode_covers_the_keyboard_door_too():
    """Most chat apps send on the action key, so gating the button alone would
    leave the other door open."""
    s = screen(composer_focused=True)
    act = AgentAction(observation="x", reasoning="y", action="press_key",
                      key="enter")
    assert draft_refusal(act, s, cfg(draft=True))


def test_draft_mode_lets_everything_that_is_not_a_send_through():
    s = screen()
    act = AgentAction(observation="x", reasoning="y", action="scroll",
                      direction="down")
    assert draft_refusal(act, s, cfg(draft=True)) == ""


def test_a_live_send_is_never_refused():
    """Draft off is the whole of the condition. A second reply into a thread is
    not refused here -- that is the send check's call, made by a model with the
    screen in front of it (`Agent._check_send`)."""
    s = screen()
    assert draft_refusal(tap_send(s), s, cfg(draft=False)) == ""


# -- every door a send goes out through ----------------------------------------

def _like_sheet():
    return attach(parse(X.like_sheet(), width=X.W, height=X.H))


def test_a_send_command_longer_than_a_button_label_is_still_a_send():
    """31 characters. Under the old 24-character rule no like that carried its
    comment ever counted as a send, and draft mode let every one through."""
    s = _like_sheet()
    pill = next(e for e in s.elements
                if e.best_text == "Send priority like with message")
    act = tap_on(s, pill)
    assert send_label(act, s) == "Send priority like with message"
    assert draft_refusal(act, s, cfg(draft=True))


def test_a_tap_at_that_names_a_send_control_is_a_send():
    """The escape hatch was no door at all: a like sent by naming the pill went
    past draft mode untouched."""
    s = _like_sheet()
    act = AgentAction(observation="x", reasoning="y", action="tap_at",
                      text="the Send Priority Like pill")
    assert send_label(act, s) == "the Send Priority Like pill"
    assert draft_refusal(act, s, cfg(draft=True))


def test_a_tap_at_that_names_something_else_is_not_a_send():
    s = _like_sheet()
    act = AgentAction(observation="x", reasoning="y", action="tap_at",
                      text="the Edit comment field")
    assert send_label(act, s) == ""


def test_a_button_the_dump_does_not_flag_clickable_is_still_a_send():
    """Hinge dumps its like pill and the paid Rose beside it as Buttons with
    clickable="false", and a tap on either lands anyway. Asking for the flag
    kept all 617 likes in ``runs/`` out of `SendLog`: ``like=207`` never bound,
    and the driver was told "0 used" after thirteen likes
    (``runs/8f213eddda69``)."""
    s = _like_sheet()
    pill = next(e for e in s.elements
                if e.best_text == "Send priority like with message")
    rose = next(e for e in s.elements
                if e.best_text == "Send a Rose with message")
    assert not pill.interactive and not rose.interactive
    assert send_label(tap_on(s, pill), s) == "Send priority like with message"
    # ...which is what lets `rose=0` refuse the paid one.
    label = send_label(tap_on(s, rose), s)
    assert label == "Send a Rose with message"
    assert "forbids" in SendLog().over_limit(label, {"rose": 0})


def test_a_tap_at_landing_on_such_a_button_is_a_send_too():
    s = _like_sheet()
    pill = next(e for e in s.elements
                if e.best_text == "Send priority like with message")
    x, y = pill.center
    act = AgentAction(observation="x", reasoning="y", action="tap_at",
                      x=x / s.width, y=y / s.height)
    assert send_label(act, s) == "Send priority like with message"


def test_text_that_is_not_a_button_still_needs_the_flag():
    """The widening is to buttons alone: a bubble drawn as plain text that says
    "send it" is read, not pressed."""
    s = screen(messages=["send it", "ok"])
    bubble = next(e for e in s.elements if e.best_text == "send it")
    assert not bubble.pressable
    assert send_label(tap_on(s, bubble), s) == ""


def test_a_text_field_is_not_a_send_whatever_its_hint_says():
    """A field hinted "Send a message" is where a reply is typed, not what
    sends it, and a tap that only focused one went down as a reply sent
    (``runs/f94def19db5d``). Not by index, by point or by name."""
    s = tools(hint="Send a message")
    field = next(e for e in s.elements if e.resource_id == "message_input")
    assert field.best_text == "Send a message"
    assert send_label(tap_on(s, field), s) == ""
    x, y = field.center
    by_point = AgentAction(observation="x", reasoning="y", action="tap_at",
                           x=x / s.width, y=y / s.height)
    assert send_label(by_point, s) == ""
    by_name = AgentAction(observation="x", reasoning="y", action="tap_at",
                          text="Send a message")
    assert send_label(by_name, s) == ""
    assert draft_refusal(tap_on(s, field), s, cfg(draft=True)) == ""


# -- what a send says ------------------------------------------------------------

def test_the_draft_of_a_typed_send_is_what_will_be_typed():
    s = screen()
    act = _type_into_composer(s, press_enter=True)
    assert draft_in(act, s) == "hi"


def test_the_draft_of_a_tapped_send_is_what_sits_in_the_composer():
    s = screen(draft="on my way")
    assert draft_in(tap_send(s), s) == "on my way"


def test_an_empty_composer_has_no_draft_and_its_hint_is_not_one():
    s = screen()
    assert draft_in(tap_send(s), s) == ""


# -- what this run has sent, and what it may still send ----------------------------

def test_send_limits_parse_from_the_policy_front_matter():
    assert parse_send_limits("like=5, rose=0") == {"like": 5, "rose": 0}
    assert parse_send_limits("Like: 3") == {"like": 3}
    assert parse_send_limits("") == {}


def test_a_send_limit_that_does_not_parse_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="word=N"):
        parse_send_limits("at most five likes")


def test_the_send_log_counts_by_the_word_on_the_control():
    log = SendLog()
    log.record(3, "Send priority like with message")
    log.record(9, "the Send Priority Like pill")
    log.record(12, "Send message")
    assert len(log) == 3
    assert log.count("like") == 2
    assert log.count("rose") == 0
    assert log.count("lik") == 0          # a word, not a substring


def test_a_send_past_its_limit_is_refused_and_other_sends_are_not():
    log = SendLog()
    log.record(1, "Send priority like with message")
    log.record(2, "Send priority like with message")
    limits = {"like": 2}
    assert "allows 2" in log.over_limit("Send priority like", limits)
    assert log.over_limit("Send message", limits) == ""
    assert SendLog().over_limit("Send priority like", limits) == ""


def test_a_zero_limit_forbids_the_send_outright():
    """`rose=0` is how a policy keeps the paid button beside the like pill
    from ever being pressed."""
    refusal = SendLog().over_limit("Send a Rose with message", {"rose": 0})
    assert "forbids" in refusal


def test_the_send_log_renders_counts_and_limits():
    log = SendLog()
    log.record(1, "Send priority like with message")
    log.record(2, "Send priority like with message")
    log.record(5, "Send message")
    text = log.render({"like": 2, "rose": 0})
    assert text.startswith("SENT THIS RUN") and ": 3" in text
    assert '2 x "Send priority like with message"' in text
    assert '1 x "Send message"' in text
    assert 'LIMIT REACHED: 2 of 2 "like"' in text
    assert 'FORBIDDEN: "rose"' in text


def test_a_run_that_sends_nothing_and_limits_nothing_adds_nothing():
    assert SendLog().render() == ""
    assert "nothing yet" in SendLog().render({"like": 5})
    assert "5 left" in SendLog().render({"like": 5})
