"""Reading a chat screen: who said what, and which side said it."""

from __future__ import annotations

import pytest

from adbagent.actions import AgentAction, Target
from adbagent.config import Config
from adbagent.conversation import (OURS, THEIRS, SendLog, draft_in,
                                   draft_refusal, message_scroller,
                                   on_a_conversation, parse_send_limits,
                                   read_conversation, send_label, thread_on)
from adbagent.fingerprint import attach
from adbagent.prompts import conversation_block
from adbagent.screen import parse

from . import xmlgen as X


def screen(**kw):
    return attach(parse(X.chat_thread(**kw), width=X.W, height=X.H))


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


# -- reading ----------------------------------------------------------------

def test_reads_title_and_messages():
    c = read_conversation(screen())
    assert c.title == "khushi"
    assert c.texts[:2] == ["hey", "you around?"]
    assert c.readable


def test_messages_come_from_raw_nodes_not_the_pruned_view():
    """The extractor reads the tree, so it does not move when the render does.

    This used to assert the opposite of its first check: pruning folded the whole
    thread onto the scroller as one label, and reading raw nodes was the way
    around that. `_absorb_labels` no longer fires on a scroller, so the thread
    survives pruning -- but the extractor still must not depend on the pruned
    view, which collapses repeated messages and truncates at `RENDER_LIMIT`.
    """
    s = screen(messages=["one", "two", "three"])
    scroller = message_scroller(s)
    assert scroller is not None and scroller.scrollable
    assert scroller.label == "", \
        f"the scroller swallowed the thread again: {scroller.label!r}"
    assert read_conversation(s).texts[:3] == ["one", "two", "three"]


def nested(**kw):
    return attach(parse(X.chat_thread_nested(**kw), width=X.W, height=X.H))


def test_the_message_list_wins_over_a_full_screen_pager():
    """Observed live on Instagram: the biggest scrollable is the tab pager.

    Choosing it swallows the correspondent's name into the messages, so no title
    is found and every send is refused -- safe, and useless.
    """
    s = nested()
    assert message_scroller(s).resource_id == "message_list"
    c = read_conversation(s)
    assert c.title == "khushi"
    assert c.texts[:2] == ["hey", "you around?"]
    assert c.readable


def test_the_header_never_lands_in_the_conversation():
    s = nested()
    assert "Back" not in read_conversation(s).texts
    assert "khushi" not in read_conversation(s).texts


def test_nesting_does_not_change_what_was_read():
    """The same thread must read the same however the app lays it out.

    This used to compare `key` and `digest` -- the masked thread identity and
    the tail hash the reply ledger stored. Neither exists; what has to survive
    the layout is the dialogue itself.
    """
    flat, deep = read_conversation(screen()), read_conversation(nested())
    assert flat.title == deep.title
    assert flat.texts == deep.texts
    assert [m.side for m in flat.messages] == [m.side for m in deep.messages]


def test_a_short_nested_scroller_is_not_mistaken_for_the_messages():
    """An emoji tray or reaction strip inside the thread is deeper but tiny."""
    s = screen()
    scroller = message_scroller(s)
    assert scroller.height >= s.height * 0.25


def test_our_own_draft_is_not_something_anybody_said():
    """The composer holds our half-typed reply, which nobody has sent yet."""
    assert read_conversation(screen()).texts == \
           read_conversation(screen(draft="sure, one sec")).texts


# -- which side said it -----------------------------------------------------

def test_sides_come_from_which_edge_the_bubble_hugs():
    """The fixture alternates: incoming inset from the left, outgoing from the
    right. That is the only signal there is, and all of it."""
    c = read_conversation(screen(messages=["hey", "you around?", "hello?"]))
    assert [m.side for m in c.messages[:3]] == [THEIRS, OURS, THEIRS]


def test_a_full_width_line_belongs_to_nobody():
    """A date separator or a security notice spans the list, so its margins
    differ by less than the threshold. Empty is the honest answer -- labelling
    it as incoming would invent a message."""
    from adbagent.conversation import side_of
    from adbagent.screen import Element

    el = Element(bounds=(0, 400, X.W, 460))
    assert side_of(el, 0, X.W) == ""


def test_a_timestamp_under_the_last_bubble_does_not_flip_the_answer():
    """The bug this exists to stop. A relative timestamp is drawn hard against
    the left edge, so on geometry alone it reads as *their* message -- and a
    thread whose last real bubble is ours would then say they spoke last, which
    is the exact wrong answer on a reply pass. It is recognised by its
    resource id, keeps its place in the thread, and carries no side."""
    c = read_conversation(screen())          # ends: them, us, then "2m"
    assert c.texts[-1] == "2m"
    assert c.messages[-1].side == ""
    assert c.last_is_ours


def test_last_is_ours_is_false_when_they_spoke_last():
    c = read_conversation(screen(messages=["hey", "you around?", "hello?"]))
    assert not c.last_is_ours


# -- a list drawn from the bottom up ------------------------------------------
#
# Hinge's thread emits its bubbles newest-first (``runs/0fc8159ca26c`` step 3:
# y=1355, then 1194, then 934). Read in that order the oldest message came out
# last, and both readings of "who spoke last" were wrong.

def test_a_thread_drawn_bottom_up_is_still_read_oldest_first():
    c = read_conversation(screen(messages=["one", "two", "three"],
                                 newest_first=True))
    assert c.texts == ["one", "two", "three", "2m"]


def test_a_reply_already_sent_reads_as_ours_however_the_list_is_drawn():
    """The duplicate-reply direction: they opened, we answered last. Read in
    document order this thread came back as theirs with no reply under it."""
    for newest_first in (False, True):
        c = read_conversation(screen(messages=["hey", "on my way"],
                                     newest_first=newest_first))
        assert c.last_is_ours, f"newest_first={newest_first}"
        assert "Do NOT reply again" in conversation_block(c)


def test_a_message_they_sent_last_reads_as_theirs_however_the_list_is_drawn():
    for newest_first in (False, True):
        c = read_conversation(screen(messages=["hey", "on my way", "where are you"],
                                     newest_first=newest_first))
        assert not c.last_is_ours, f"newest_first={newest_first}"
        assert "is THEIRS" in conversation_block(c)


def test_no_messages_means_nobody_spoke_last():
    """`last_is_ours` must never be true by accident on an empty thread."""
    c = read_conversation(screen(messages=[], stamp=""))
    assert not c.last_is_ours


# -- the block the model decides on -----------------------------------------

def test_the_block_marks_every_speaker_and_states_the_reading():
    c = read_conversation(screen())
    block = conversation_block(c)
    assert "THIS CONVERSATION (khushi)" in block
    assert "them:" in block and "us:" in block
    assert "hey" in block and "you around?" in block
    # The derived line -- the one sentence that decides the pass.
    assert "last message in this thread is YOURS" in block
    assert "Do NOT reply again" in block


def test_the_block_says_when_a_reply_is_owed():
    c = read_conversation(screen(messages=["hey", "you around?", "hello?"]))
    block = conversation_block(c)
    assert "is THEIRS and has no reply under it yet" in block
    assert "Do NOT reply again" not in block


def test_nothing_is_rendered_for_a_screen_that_is_not_a_thread():
    """An unlabelled thread is one where "who is this" is unanswered, and a
    block opening with the wrong name is worse than no block."""
    assert conversation_block(read_conversation(screen(with_header=False))) == ""
    assert conversation_block(None) == ""
    settings = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    assert conversation_block(read_conversation(settings)) == ""


def test_a_long_thread_is_trimmed_from_the_front_and_says_so():
    from adbagent.prompts import CONVERSATION_MESSAGES

    many = [f"line {i}" for i in range(CONVERSATION_MESSAGES + 6)]
    block = conversation_block(read_conversation(screen(messages=many, stamp="")))
    assert "earlier message(s) above" in block
    assert "line 0" not in block          # trimmed
    assert many[-1] in block              # the newest always survives


def test_only_a_chat_screen_is_worth_reading():
    """The per-turn gate: a launcher or a settings page pays one pass over the
    nodes, not the scroller search and the side arithmetic."""
    assert on_a_conversation(screen())
    settings = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    assert not on_a_conversation(settings)


def test_nav_buttons_are_never_the_title():
    """Document order puts Back first; it must not win."""
    assert read_conversation(screen()).title == "khushi"


def test_title_without_a_telling_resource_id_falls_back_to_widest():
    c = read_conversation(screen(title_rid="tv_1"))
    assert c.title == "khushi"


def test_unreadable_header_is_reported_not_guessed():
    c = read_conversation(screen(with_header=False))
    assert not c.readable
    assert "no conversation name" in c.problem


def test_non_chat_screen_is_not_a_conversation():
    s = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    c = read_conversation(s)
    assert not c.readable
    assert "not a conversation" in c.problem


def test_a_thread_list_is_not_a_conversation():
    """Observed live: the Instagram inbox in multi-select mode has a scroller,
    rows and a plausible title ("0 selected"). Only the missing composer tells
    it apart from a thread.
    """
    s = screen(with_send=False)
    # Same screen minus the composer row entirely.
    rows = [e for e in s.elements if e.resource_id == "composer"]
    assert rows, "fixture sanity: the flat thread has a composer"
    listing = attach(parse(
        X.chat_thread().replace('resource-id="com.instagram.android:id/composer"',
                                'resource-id="x"')
        .replace('class="android.widget.EditText"',
                 'class="android.widget.TextView"'),
        width=X.W, height=X.H))
    c = read_conversation(listing)
    assert not c.readable
    assert "no message composer" in c.problem


# -- which actions are sends ------------------------------------------------

def test_tap_on_send_is_a_send():
    s = screen()
    assert send_label(tap_send(s), s) == "Send"


def test_tap_elsewhere_is_not_a_send():
    s = screen()
    back = next(e for e in s.elements if e.resource_id == "back")
    act = AgentAction(observation="x", reasoning="y", action="tap",
                      target=Target(index=back.index))
    assert send_label(act, s) == ""


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
            act = AgentAction(observation="x", reasoning="y", action="tap",
                              target=Target(index=el.index))
            assert send_label(act, s) == ""


# -- the only refusal left --------------------------------------------------
#
# Eight tests stood here against `reply_gate`: the same tail already replied to,
# a per-thread cooldown, a per-thread hourly ceiling, a global hourly ceiling,
# the rolling window those two counted in, and failing closed on a thread that
# could not be identified. Every one of them was a reading of the reply ledger,
# and all of it is gone -- whether a reply is owed is the model's call now, made
# from `conversation_block`. What is left refuses nothing except on request.

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
    thread in front of it (`Agent._check_send`)."""
    s = screen()
    assert draft_refusal(tap_send(s), s, cfg(draft=False)) == ""


def test_an_unreadable_thread_is_not_a_refusal():
    """`watch.fail_closed` used to refuse this send, on the grounds that a
    duplicate could not be ruled out. There is no gate to fail closed into:
    the thread simply does not reach the prompt, and the model decides without
    it."""
    s = screen(with_header=False)
    assert draft_refusal(tap_send(s), s, cfg(draft=False)) == ""
    assert conversation_block(read_conversation(s)) == ""


# -- every door a send goes out through ----------------------------------------

def _like_sheet():
    return attach(parse(X.like_sheet(), width=X.W, height=X.H))


def test_a_send_command_longer_than_a_button_label_is_still_a_send():
    """31 characters. Under the old 24-character rule no like that carried its
    comment ever counted as a send, and draft mode let every one through."""
    s = _like_sheet()
    pill = next(e for e in s.elements
                if e.best_text == "Send priority like with message")
    act = AgentAction(observation="x", reasoning="y", action="tap",
                      target=Target(index=pill.index))
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


def test_only_a_readable_conversation_has_a_thread():
    assert thread_on(screen()).title == "khushi"
    assert thread_on(_like_sheet()) is None
    assert thread_on(screen(with_header=False)) is None


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
    log.record(12, "Send message", thread="Alex")
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


def test_the_send_log_renders_counts_threads_and_limits():
    log = SendLog()
    log.record(1, "Send priority like with message")
    log.record(2, "Send priority like with message")
    log.record(5, "Send message", thread="Alex")
    text = log.render({"like": 2, "rose": 0})
    assert text.startswith("SENT THIS RUN") and ": 3" in text
    assert '2 x "Send priority like with message"' in text
    assert '1 x "Send message" in the conversation with Alex' in text
    assert 'LIMIT REACHED: 2 of 2 "like"' in text
    assert 'FORBIDDEN: "rose"' in text


def test_a_run_that_sends_nothing_and_limits_nothing_adds_nothing():
    assert SendLog().render() == ""
    assert "nothing yet" in SendLog().render({"like": 5})
    assert "5 left" in SendLog().render({"like": 5})
