"""The never-double-reply guarantee: digests, cooldowns and crash recovery."""

from __future__ import annotations

import json

import pytest

from adbagent.ledger import (TAIL_MESSAGES, UNCONFIRMED_COOLDOWN_MULTIPLIER,
                             ReplyLedger, content_digest, thread_key)


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "replies.jsonl"


# -- identity ---------------------------------------------------------------

def test_thread_key_ignores_unread_counts_and_times():
    """The same person must not mint a second identity as their badge moves."""
    assert thread_key("khushi") == thread_key("khushi")
    assert thread_key("khushi") == thread_key("  Khushi  ")
    assert thread_key("khushi") != thread_key("shreya")


def test_thread_key_empty_when_unusable():
    assert thread_key("") == ""
    assert thread_key(None) == ""


def test_content_digest_survives_a_ticking_timestamp():
    """"2m ago" -> "3m ago" is not a new message."""
    a = content_digest(["hey there", "2m ago"])
    b = content_digest(["hey there", "3m ago"])
    assert a == b != ""


def test_content_digest_changes_on_a_new_message():
    a = content_digest(["hey there"])
    b = content_digest(["hey there", "you around?"])
    assert a != b


def test_content_digest_only_reads_the_tail():
    """Scrolling up must not rewrite the digest."""
    tail = [f"msg {i}" for i in range(TAIL_MESSAGES)]
    assert content_digest(tail) == content_digest(["older", *tail])


def test_content_digest_empty_means_could_not_read():
    assert content_digest([]) == ""
    assert content_digest(["", "   "]) == ""


# -- the core rule ----------------------------------------------------------

def test_fresh_thread_is_allowed(path):
    led = ReplyLedger(path)
    assert led.check(thread_key("khushi"), content_digest(["hi"]),
                     cooldown_s=60, now=1000)


def test_same_tail_is_refused_after_a_reply(path):
    led = ReplyLedger(path)
    key, digest = thread_key("khushi"), content_digest(["hi"])
    led.record_attempt(key, digest, at=1000)
    v = led.check(key, digest, cooldown_s=60, now=9999)
    assert not v and "already been replied to" in v.reason


def test_a_new_message_reopens_the_thread(path):
    led = ReplyLedger(path)
    key = thread_key("khushi")
    before = content_digest(["hi"])
    led.record_attempt(key, before, at=1000)
    led.record_confirmed(key, content_digest(["hi", "sure, one sec"]), at=1001)
    # They say something new: past the cooldown this must be answerable.
    fresh = content_digest(["hi", "sure, one sec", "you free tonight?"])
    assert led.check(key, fresh, cooldown_s=60, now=2000)


def test_our_own_reply_is_not_read_as_new_content(path):
    """The post-send tail is remembered, so the next poll skips it."""
    led = ReplyLedger(path)
    key = thread_key("khushi")
    led.record_attempt(key, content_digest(["hi"]), at=1000)
    after = content_digest(["hi", "sure, one sec"])
    led.record_confirmed(key, after, at=1001)
    v = led.check(key, after, cooldown_s=60, now=9999)
    assert not v and "already been replied to" in v.reason


def test_cooldown_blocks_a_different_tail(path):
    led = ReplyLedger(path)
    key = thread_key("khushi")
    led.record_attempt(key, content_digest(["hi"]), at=1000)
    led.record_confirmed(key, content_digest(["hi", "ok"]), at=1001)
    v = led.check(key, content_digest(["hi", "ok", "and again"]),
                  cooldown_s=600, now=1100)
    assert not v and "cooldown" in v.reason


def test_unidentifiable_thread_is_refused(path):
    led = ReplyLedger(path)
    assert not led.check("", content_digest(["hi"]), cooldown_s=60)
    v = led.check(thread_key("khushi"), "", cooldown_s=60)
    assert not v and "could not be read" in v.reason


def test_already_acted_never_licenses_a_send_on_empty_input(path):
    led = ReplyLedger(path)
    assert led.already_acted("", "") is False
    assert led.already_acted(thread_key("k"), "") is False


# -- crash recovery ---------------------------------------------------------

def test_state_survives_a_reopen(path):
    key, digest = thread_key("khushi"), content_digest(["hi"])
    ReplyLedger(path).record_attempt(key, digest, preview="khushi", at=1000)
    again = ReplyLedger(path)
    assert not again.check(key, digest, cooldown_s=60, now=9999)
    assert again.state(key).preview == "khushi"


def test_unconfirmed_send_gets_the_long_cooldown(path):
    """Crash between the tap and the confirmation: the thread is in doubt."""
    key = thread_key("khushi")
    ReplyLedger(path).record_attempt(key, content_digest(["hi"]), at=1000)
    led = ReplyLedger(path)
    assert led.state(key).confirmed is False
    # A tail we have never seen -- which is exactly what our own unrecorded
    # reply looks like -- is still refused inside the multiplied window.
    unseen = content_digest(["hi", "sure, one sec"])
    v = led.check(key, unseen, cooldown_s=60, now=1100)
    assert not v and "unconfirmed" in v.reason
    # ...and allowed once the long window has passed.
    assert led.check(key, unseen, cooldown_s=60,
                     now=1000 + 60 * UNCONFIRMED_COOLDOWN_MULTIPLIER + 1)


def test_confirmation_lifts_the_doubt(path):
    key = thread_key("khushi")
    led = ReplyLedger(path)
    led.record_attempt(key, content_digest(["hi"]), at=1000)
    led.record_confirmed(key, content_digest(["hi", "ok"]), at=1001)
    assert ReplyLedger(path).state(key).confirmed is True


def test_truncated_final_line_is_dropped_not_fatal(path):
    key, digest = thread_key("khushi"), content_digest(["hi"])
    ReplyLedger(path).record_attempt(key, digest, at=1000)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"thread_key": "half-writ')      # kill -9 mid-append
    led = ReplyLedger(path)
    assert not led.check(key, digest, cooldown_s=60, now=9999)
    assert len(led) == 1


def test_records_are_folded_in_timestamp_order(path):
    """Out-of-order lines must not make a confirmation lose to its attempt."""
    key = thread_key("khushi")
    rows = [
        {"thread_key": key, "digest": "bbb", "at": 1001, "confirmed": True},
        {"thread_key": key, "digest": "aaa", "at": 1000, "confirmed": False},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    led = ReplyLedger(path)
    st = led.state(key)
    assert st.confirmed is True                  # the later record wins
    assert st.digests == frozenset({"aaa", "bbb"})
    assert st.reply_count == 1                   # the confirmation is not an attempt


# -- rate limiting ----------------------------------------------------------

def test_replies_since_counts_attempts_globally_and_per_thread(path):
    led = ReplyLedger(path)
    led.record_attempt(thread_key("khushi"), content_digest(["a"]), at=1000)
    led.record_confirmed(thread_key("khushi"), content_digest(["a", "b"]), at=1001)
    led.record_attempt(thread_key("shreya"), content_digest(["c"]), at=1002)
    assert led.replies_since(0) == 2                              # not 3
    assert led.replies_since(0, thread_key("khushi")) == 1
    assert led.replies_since(1002) == 1


def test_recent_is_newest_first(path):
    led = ReplyLedger(path)
    led.record_attempt(thread_key("khushi"), content_digest(["a"]), at=1000)
    led.record_attempt(thread_key("shreya"), content_digest(["b"]), at=2000)
    assert [s.thread_key for s in led.recent()] == [thread_key("shreya"),
                                                    thread_key("khushi")]


def test_record_refuses_an_empty_key(path):
    led = ReplyLedger(path)
    with pytest.raises(ValueError):
        led.record_attempt("", "digest")
    with pytest.raises(ValueError):
        led.record_confirmed("", "digest")
