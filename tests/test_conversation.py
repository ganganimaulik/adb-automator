"""Reading a chat screen, and the gate in front of every send."""

from __future__ import annotations

import pytest

from adbagent.actions import AgentAction, Target
from adbagent.config import Config
from adbagent.conversation import (read_conversation, reply_gate, send_label,
                                   message_scroller)
from adbagent.fingerprint import attach
from adbagent.ledger import ReplyLedger
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


@pytest.fixture()
def ledger(tmp_path):
    return ReplyLedger(tmp_path / "replies.jsonl")


# -- reading ----------------------------------------------------------------

def test_reads_title_and_messages():
    c = read_conversation(screen())
    assert c.title == "khushi"
    assert c.messages[:2] == ["hey", "you around?"]
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
    assert read_conversation(s).messages[:3] == ["one", "two", "three"]


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
    assert c.messages[:2] == ["hey", "you around?"]
    assert c.readable


def test_the_header_never_lands_in_the_conversation():
    s = nested()
    assert "Back" not in read_conversation(s).messages
    assert "khushi" not in read_conversation(s).messages


def test_nesting_does_not_change_a_conversation_identity():
    """The same thread must digest the same however the app lays it out."""
    flat, deep = read_conversation(screen()), read_conversation(nested())
    assert flat.key == deep.key
    assert flat.digest == deep.digest


def test_nested_layout_keeps_the_digest_properties():
    assert read_conversation(nested(stamp="2m")).digest == \
           read_conversation(nested(stamp="9m")).digest
    assert read_conversation(nested()).digest != \
           read_conversation(nested(messages=["hey", "you around?", "?"])).digest


def test_a_short_nested_scroller_is_not_mistaken_for_the_messages():
    """An emoji tray or reaction strip inside the thread is deeper but tiny."""
    s = screen()
    scroller = message_scroller(s)
    assert scroller.height >= s.height * 0.25


def test_a_ticking_timestamp_is_not_a_new_message():
    assert read_conversation(screen(stamp="2m")).digest == \
           read_conversation(screen(stamp="3m")).digest


def test_a_new_message_changes_the_digest():
    assert read_conversation(screen()).digest != \
           read_conversation(screen(messages=["hey", "you around?", "hello?"])).digest


def test_our_own_draft_is_excluded():
    """The composer holds our half-typed reply; it must not move the digest."""
    assert read_conversation(screen()).digest == \
           read_conversation(screen(draft="sure, one sec")).digest


def test_long_messages_differing_late_do_not_collide():
    """mask_text truncates at 32 chars; the digest must not."""
    a = read_conversation(screen(
        messages=["ok sounds good lets meet at 7 at the cafe near your place"]))
    b = read_conversation(screen(
        messages=["ok sounds good lets meet at 8 at the cafe near your place"]))
    assert a.digest != b.digest


def test_different_people_are_different_threads():
    assert read_conversation(screen()).key != \
           read_conversation(screen(title="shreya")).key


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


# -- the gate ---------------------------------------------------------------

def test_non_send_actions_pass(ledger):
    s = screen()
    act = AgentAction(observation="x", reasoning="y", action="scroll",
                      direction="down")
    assert reply_gate(act, s, ledger, cfg())


def test_first_reply_is_allowed(ledger):
    s = screen()
    assert reply_gate(tap_send(s), s, ledger, cfg())


def test_second_reply_to_the_same_tail_is_refused(ledger):
    s = screen()
    c = read_conversation(s)
    ledger.record_attempt(c.key, c.digest, at=1000)
    v = reply_gate(tap_send(s), s, ledger, cfg(), now=99999)
    assert not v and "already been replied to" in v.reason


def test_unreadable_conversation_is_refused_when_failing_closed(ledger):
    s = screen(with_header=False)
    v = reply_gate(tap_send(s), s, ledger, cfg(fail_closed=True))
    assert not v and "duplicate reply could not be ruled out" in v.reason


def test_unreadable_conversation_passes_when_failing_open(ledger):
    s = screen(with_header=False)
    assert reply_gate(tap_send(s), s, ledger, cfg(fail_closed=False))


def test_draft_mode_never_sends(ledger):
    s = screen()
    v = reply_gate(tap_send(s), s, ledger, cfg(draft=True))
    assert not v and "draft mode" in v.reason


def test_per_thread_hourly_ceiling(ledger):
    s = screen()
    c = read_conversation(s)
    ledger.record_attempt(c.key, "old-digest-1", at=1000)
    ledger.record_attempt(c.key, "old-digest-2", at=1100)
    v = reply_gate(tap_send(s), s, ledger,
                   cfg(max_replies_per_thread_per_hour=2,
                       thread_cooldown_s=0), now=1200)
    assert not v and "this hour" in v.reason


def test_global_hourly_ceiling(ledger):
    for i in range(3):
        ledger.record_attempt(f"thread-{i}", f"digest-{i}", at=1000 + i)
    s = screen()
    v = reply_gate(tap_send(s), s, ledger,
                   cfg(max_replies_per_hour=3, thread_cooldown_s=0), now=1100)
    assert not v and "this hour" in v.reason


def test_ceilings_only_count_a_rolling_hour(ledger):
    for i in range(5):
        ledger.record_attempt(f"thread-{i}", f"digest-{i}", at=1000 + i)
    s = screen()
    # Two hours later the window is clear again.
    assert reply_gate(tap_send(s), s, ledger,
                      cfg(max_replies_per_hour=3, thread_cooldown_s=0),
                      now=1000 + 7200)


def test_cooldown_refuses_a_genuinely_new_message(ledger):
    """Rate limiting wins over freshness: better late than twice."""
    s = screen()
    c = read_conversation(s)
    ledger.record_attempt(c.key, "some-older-digest", at=1000)
    ledger.record_confirmed(c.key, "another-digest", at=1001)
    v = reply_gate(tap_send(s), s, ledger,
                   cfg(thread_cooldown_s=600), now=1100)
    assert not v and "cooldown" in v.reason
