"""Notifications to the owner: the `notify` action, and the ledger that keeps
any one line from reaching their phone twice.

Nothing here talks to Telegram. The transport is a recorder (`Phone`), or
`httpx.post` replaced for the tests of the transport itself -- a test that
reached the real bot would put a message on somebody's phone.
"""

from __future__ import annotations

import json

import pytest

from adbagent import notify, runlog
from adbagent.actions import ActionError, AgentAction, execute
from adbagent.agent import Agent
from adbagent.config import Config, load_config
from adbagent.conversation import SendLog, parse_send_limits
from adbagent.notify import Notifier, NotifyError, Telegram, lines_of, same

from . import fake

TOKEN = "123456:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"


class Phone:
    """A transport that records what it was asked to deliver, and can fail."""

    def __init__(self) -> None:
        self.sent: list = []
        self.fail: Exception | None = None

    def send(self, text: str) -> None:
        if self.fail is not None:
            raise self.fail
        self.sent.append(text)


def ledger(directory) -> list:
    path = directory / notify.LEDGER_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# what counts as the same line
# ---------------------------------------------------------------------------

def test_the_same_message_read_two_ways_is_one_line():
    """The element list draws an emoji as two dots, the screenshot draws the
    emoji: a message read off one on this pass and the other on the next must
    not reach the phone twice."""
    assert same("Hinge: new message from Alex: Hii 😊",
                "hinge: new message from alex: hii ..")
    assert same("  Bumble:  replied to Sam: Up for movie at my place? ",
                "Bumble: replied to Sam: up for movie at my place")
    # Different people are different lines, whatever they said.
    assert not same("Hinge: new message from Alex: hi",
                    "Hinge: new message from Sam: hi")


def test_emoji_are_part_of_what_a_message_says():
    """Ishi's "😂😂" and a later "😍" are two messages. Compared on words alone
    both were "Tinder: new message from Ishi", and the second never went out."""
    assert not same("Tinder: new message from Ishi: 😂😂",
                    "Tinder: new message from Ishi: 😍")
    assert not same("Hinge: new message from Alex: hi 😊",
                    "Hinge: new message from Alex: hi 😊😊")
    assert not same("Hinge: new message from Alex: hi ❤", "Hinge: new message from Alex: hi ☺")
    assert same("🔥", " 🔥 ")
    # One emoji drawn with and without a skin tone is still one emoji.
    assert same("Bumble: new message from Sam: 👍🏽", "Bumble: new message from Sam: 👍")


def test_an_emoji_the_list_cannot_draw_is_known_by_count_not_face():
    """P's one "😊", left unanswered, sat on screen as `P ..` pass after pass;
    the model read its face off the screenshot each time and saw 😊, 🤗, 🥰,
    😄, 😋 and 😉 -- six notifications of one message. The tree says how many
    emoji there were, never which."""
    assert same("Hinge: new message from P: 😊", "Hinge: new message from P: 😉")
    assert same("Hinge: new message from Alex: hi 😊", "Hinge: new message from Alex: hi 😘")
    # Emoji the tree carries as they are can still be told apart.
    assert not same("Hinge: new message from Alex: ❤", "Hinge: new message from Alex: ✨")
    assert not same("Hinge: new message from Alex: ❤", "Hinge: new message from Alex: 😊")


def test_one_unanswered_emoji_reaches_the_owner_once_however_it_is_read(tmp_path):
    phone = Phone()
    notifier = Notifier(tmp_path, phone)
    notifier.notify("Hinge: new message from P: 😊")
    for guess in "🤗🥰😄😋😉":
        assert notifier.notify(f"Hinge: new message from P: {guess}").repeats
    assert phone.sent == ["Hinge: new message from P: 😊"]


def test_two_emoji_told_apart_in_one_notify_both_go_out(tmp_path):
    phone = Phone()
    outcome = Notifier(tmp_path, phone).notify(
        "Hinge: new message from P: 😂\nHinge: new message from P: 😍")
    assert len(outcome.sent) == 2


def test_two_emoji_only_messages_from_one_person_both_reach_the_owner(tmp_path):
    phone = Phone()
    notifier = Notifier(tmp_path, phone)
    notifier.notify("Tinder: new message from Ishi: 😂😂")
    later = notifier.notify("Tinder: new message from Ishi: 😍")
    assert later.sent == ["Tinder: new message from Ishi: 😍"]
    assert notifier.notify("Tinder: new message from Ishi: 😂😂").repeats
    assert phone.sent == ["Tinder: new message from Ishi: 😂😂",
                          "Tinder: new message from Ishi: 😍"]


def test_a_message_read_off_the_element_list_is_not_sent_again(tmp_path):
    """`..` is an emoji the list could not draw: it matches whichever one the
    screenshot showed, either way round."""
    phone = Phone()
    notifier = Notifier(tmp_path, phone)
    notifier.notify("Hinge: new message from Alex: hey 😊")
    assert notifier.notify("Hinge: new message from Alex: hey ..").repeats
    notifier.notify("Bumble: new message from Sam: yo ..")
    assert notifier.notify("Bumble: new message from Sam: yo 🔥").repeats
    assert len(phone.sent) == 2


def test_one_notify_carries_one_notification_per_line():
    text = ("Tinder: new match: Ana\n\n   Tinder: new match:   Bo  \n"
            "TINDER: new match: ana!")
    assert lines_of(text) == ["Tinder: new match: Ana", "Tinder: new match: Bo"]


def test_a_line_is_a_notification_not_a_transcript():
    (only,) = lines_of("x" * (notify.LINE_MAX_CHARS + 50))
    assert len(only) == notify.LINE_MAX_CHARS


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------

def test_with_no_bot_nothing_is_sent_or_remembered(tmp_path):
    out = Notifier(tmp_path, None).notify("Hinge: new message from Alex: hi")
    assert out.unconfigured and not out.worked
    assert out.grade == "not sent"
    assert "not set up" in out.summary()
    # Nothing written down: once a bot is set up, the line is still owed.
    assert ledger(tmp_path) == []


def test_a_line_reaches_the_owner_once_ever(tmp_path):
    """Every pass is a fresh run with a fresh notifier, and an unanswered
    message is still on screen at the next one: the file is what remembers."""
    phone = Phone()
    first = Notifier(tmp_path, phone).notify("Hinge: new message from Alex: hii 😊")
    assert first.sent == ["Hinge: new message from Alex: hii 😊"]
    assert first.grade == "sent" and first.worked

    again = Notifier(tmp_path, phone).notify("hinge: new message from alex: Hii ..")
    assert again.sent == [] and again.repeats == ["hinge: new message from alex: Hii .."]
    assert again.grade == "already sent" and again.worked
    assert "not repeated" in again.summary()
    assert phone.sent == ["Hinge: new message from Alex: hii 😊"]


def test_only_the_new_lines_of_a_notify_go_out_together(tmp_path):
    phone = Phone()
    notifier = Notifier(tmp_path, phone)
    notifier.notify("Tinder: new match: Ana")

    out = notifier.notify("Tinder: new match: Ana\nTinder: new match: Bo\n"
                          "Tinder: new message from Cy: hey")
    assert out.repeats == ["Tinder: new match: Ana"]
    assert out.sent == ["Tinder: new match: Bo", "Tinder: new message from Cy: hey"]
    # One message, one buzz.
    assert phone.sent[-1] == "Tinder: new match: Bo\nTinder: new message from Cy: hey"
    assert "2 sent" in out.summary() and "1 already sent before" in out.summary()


def test_a_line_that_was_not_delivered_goes_out_with_the_next_one(tmp_path):
    """A pass that sent a reply never comes back to that moment, so nothing
    else would bring its notification back."""
    phone = Phone()
    phone.fail = NotifyError("Telegram answered: Bad Gateway")
    clock = [1000.0]
    notifier = Notifier(tmp_path, phone, now=lambda: clock[0])

    lost = notifier.notify("Hinge: replied to Alex: up for movie at my place?")
    assert lost.kept == ["Hinge: replied to Alex: up for movie at my place?"]
    assert lost.grade == "not delivered" and not lost.worked
    assert "Bad Gateway" in lost.summary() and "do not send it again" in lost.summary()

    phone.fail = None
    clock[0] += 600
    later = notifier.notify("Bumble: new message from Sam: hey")
    assert later.caught_up == ["Hinge: replied to Alex: up for movie at my place?"]
    assert later.sent == ["Bumble: new message from Sam: hey"]
    assert phone.sent == ["Hinge: replied to Alex: up for movie at my place?\n"
                          "Bumble: new message from Sam: hey"]
    assert "1 kept from earlier went out" in later.summary()

    # Delivered now, so never again.
    assert notifier.notify("Hinge: replied to Alex: up for movie at my place?").repeats
    assert len(phone.sent) == 1


def test_a_kept_line_delivered_in_another_spelling_is_not_owed_any_more(tmp_path):
    """Kept as the element list spelled it, then notified again as the
    screenshot spells it and delivered: the kept copy must not go out too."""
    phone = Phone()
    phone.fail = NotifyError("Telegram did not answer")
    notifier = Notifier(tmp_path, phone)
    notifier.notify("Hinge: new message from Alex: hey ..")
    phone.fail = None
    notifier.notify("Hinge: new message from Alex: hey 😊")
    later = notifier.notify("Bumble: new message from Sam: yo")
    assert later.caught_up == []
    assert phone.sent == ["Hinge: new message from Alex: hey 😊",
                          "Bumble: new message from Sam: yo"]


def test_a_line_kept_for_more_than_a_day_is_let_go(tmp_path):
    phone = Phone()
    phone.fail = NotifyError("Telegram did not answer")
    clock = [0.0]
    notifier = Notifier(tmp_path, phone, now=lambda: clock[0])
    notifier.notify("Hinge: replied to Alex: up for movie at my place?")

    phone.fail = None
    clock[0] += notify.PENDING_TTL_S + 1
    out = notifier.notify("Bumble: new message from Sam: hey")
    assert out.caught_up == []
    assert phone.sent == ["Bumble: new message from Sam: hey"]


def test_a_bot_that_keeps_failing_does_not_grow_the_ledger(tmp_path):
    """Only the first failure is written down: it is what the line ages from."""
    phone = Phone()
    phone.fail = NotifyError("Telegram did not answer")
    notifier = Notifier(tmp_path, phone)
    for _ in range(3):
        notifier.notify("Hinge: new message from Alex: hi")
    entries = ledger(tmp_path)
    assert len(entries) == 1 and entries[0]["sent"] is False


def test_lines_too_many_for_one_message_go_out_as_several(tmp_path):
    phone = Phone()
    lines = [f"Tinder: new message from P{i}: " + "y" * 900 for i in range(9)]
    out = Notifier(tmp_path, phone).notify("\n".join(lines))
    assert out.sent == lines
    assert len(phone.sent) > 1
    assert all(len(message) <= notify.MESSAGE_MAX_CHARS for message in phone.sent)
    assert "\n".join(phone.sent).split("\n") == lines


def test_what_went_before_a_failure_is_recorded_as_sent(tmp_path):
    """Several messages, and the network goes away after the first: the lines
    in it reached the phone and must not be sent again."""
    class Flaky(Phone):
        def send(self, text: str) -> None:
            if self.sent:
                raise NotifyError("Telegram did not answer")
            self.sent.append(text)

    phone = Flaky()
    lines = [f"Tinder: new message from P{i}: " + "y" * 900 for i in range(9)]
    out = Notifier(tmp_path, phone).notify("\n".join(lines))
    went = phone.sent[0].split("\n")
    assert out.sent == went and out.kept == lines[len(went):]
    assert Notifier(tmp_path, Phone()).notify(lines[0]).repeats == [lines[0]]


def test_a_ledger_line_cut_short_by_a_crash_is_skipped(tmp_path):
    (tmp_path / notify.LEDGER_NAME).write_text(
        '{"at": 1, "text": "Tinder: new match: Ana", "sent": true}\n{"at": 2, "te',
        encoding="utf-8")
    phone = Phone()
    out = Notifier(tmp_path, phone).notify("Tinder: new match: Ana\nTinder: new match: Bo")
    assert out.repeats == ["Tinder: new match: Ana"]
    assert out.sent == ["Tinder: new match: Bo"]
    assert [line.text for line in Notifier(tmp_path, phone).read_back().found] == \
        ["Tinder: new match: Ana", "Tinder: new match: Bo"]


# ---------------------------------------------------------------------------
# reading back what was notified
# ---------------------------------------------------------------------------

def notified(tmp_path, *lines: str, at: float = 1_000_000.0) -> Notifier:
    """A ledger holding `lines`, notified a minute apart from `at`."""
    clock = [at]
    notifier = Notifier(tmp_path, Phone(), now=lambda: clock[0])
    for line in lines:
        notifier.notify(line)
        clock[0] += 60
    return notifier


def test_a_name_finds_what_that_person_said_and_nobody_else():
    """Whole words: a list can show a match by an initial, and S sits beside
    Sam and Samir on the same screen."""
    assert notify.matches("Hinge: new message from S: maybe tomorrow .",
                          "Hinge: new message from S")
    assert not notify.matches("Hinge: new message from Sam: hi",
                              "Hinge: new message from S")
    assert notify.matches("Hinge: new message from Samir: Oh…! How about coffee first?",
                          "new message from samir")
    # The words in order, ignoring case and punctuation, as `same` reads them.
    assert notify.matches("Bumble: new message from T: We can go for a walk",
                          "bumble new message from t we can go")
    assert not notify.matches("Bumble: new message from T: We can go",
                              "Bumble: new message from D")
    # Nothing to look for finds nothing.
    assert not notify.matches("Tinder: new match: 🔥", "🔥")


def test_a_pass_can_see_the_newest_message_was_read_before(tmp_path):
    """A Matches row like "S maybe tomorrow 1 day ago. Reply?", and the two
    lines an earlier pass notified from S's chat."""
    notifier = notified(tmp_path,
                        "Hinge: new message from S: haha ok.",
                        "Hinge: new message from S: maybe tomorrow .",
                        "Hinge: new message from Sam: see you.",
                        "Bumble: new message from S: hey")
    back = notifier.read_back("Hinge: new message from S")
    assert [line.text for line in back.found] == [
        "Hinge: new message from S: haha ok.", "Hinge: new message from S: maybe tomorrow ."]
    assert back.grade == "found"
    summary = back.summary()
    assert summary.startswith("2 notified before, oldest first: ")
    assert "Hinge: new message from S: haha ok. | " in summary
    assert "Sam" not in summary and "Bumble" not in summary
    # Each with when, for a row that says how long ago its message came.
    assert back.found[0].shown().endswith(" Hinge: new message from S: haha ok.")
    assert back.found[0].shown() != back.found[0].text


def test_one_lookup_covers_every_row_on_screen(tmp_path):
    notifier = notified(tmp_path,
                        "Hinge: new message from S: maybe tomorrow .",
                        "Hinge: new message from K: maybe next week",
                        "Hinge: new message from Kim: yo.",
                        "Hinge: replied to Kim: up for movie at my place?")
    back = notifier.read_back("Hinge: new message from S\n\n"
                              "Hinge: new message from K\n"
                              "Hinge: new message from G")
    assert [line.text for line in back.found] == [
        "Hinge: new message from S: maybe tomorrow .",
        "Hinge: new message from K: maybe next week"]
    assert back.filters == ["Hinge: new message from S", "Hinge: new message from K",
                            "Hinge: new message from G"]


def test_a_message_never_notified_is_not_found(tmp_path):
    notifier = notified(tmp_path, "Hinge: new message from S: maybe tomorrow .")
    back = notifier.read_back("Hinge: new message from Robin")
    assert back.found == [] and back.grade == "nothing found"
    assert back.summary() == "nothing notified before matches that"
    assert Notifier(tmp_path / "elsewhere", Phone()).read_back().summary() == \
        "nothing has been notified yet"


def test_with_no_bot_and_no_ledger_the_read_says_why_it_is_empty(tmp_path):
    """No bot means `notify` writes nothing down, so an empty read is no sign
    that nothing was read: the model must not take it for one."""
    back = Notifier(tmp_path, None).read_back("Hinge: new message from S")
    assert back.unconfigured and "not set up" in back.summary()
    # A ledger written while a bot was set up is still there to read.
    notified(tmp_path, "Hinge: new message from S: maybe tomorrow .")
    assert Notifier(tmp_path, None).read_back("from S").found


def test_nothing_to_look_for_reads_everything(tmp_path):
    lines = ["Tinder: new match: Ana", "Bumble: new message from Sam: yo",
             "Hinge: replied to Alex: up for movie at my place?"]
    notifier = notified(tmp_path, *lines)
    for asked in ("", "  \n ", "🔥"):
        back = notifier.read_back(asked)
        assert [line.text for line in back.found] == lines, repr(asked)
    assert notifier.read_back().summary().startswith("3 notified so far")


def test_a_kept_line_was_read_all_the_same(tmp_path):
    """Not delivered yet, but the pass that read it is over: it counts, and says
    so. Delivered later, it is dated by the pass that read it."""
    phone = Phone()
    phone.fail = NotifyError("Telegram did not answer")
    clock = [1_000_000.0]
    notifier = Notifier(tmp_path, phone, now=lambda: clock[0])
    notifier.notify("Hinge: new message from Alex: hii")
    (kept,) = notifier.read_back("from Alex").found
    assert not kept.sent and kept.shown().endswith("[not delivered yet]")

    phone.fail = None
    clock[0] += 3600
    notifier.notify("Bumble: new message from Sam: yo")
    (delivered,) = notifier.read_back("from Alex").found
    assert delivered.sent and delivered.at == 1_000_000.0
    assert "not delivered" not in delivered.shown()


def test_a_long_ledger_shows_its_newest_lines_and_says_what_it_left_out(tmp_path):
    lines = [f"Tinder: new match: P{i}" for i in range(notify.READ_MAX_LINES + 7)]
    back = notified(tmp_path, *lines).read_back("Tinder: new match")
    assert len(back.found) == len(lines)
    assert [line.text for line in back.shown] == lines[7:]
    assert "(+7 older not shown: name the people you are looking for)" in back.summary()


def test_a_time_no_clock_can_show_is_no_reason_to_fail(tmp_path):
    (tmp_path / notify.LEDGER_NAME).write_text(
        '{"at": 1e300, "text": "Tinder: new match: Ana", "sent": true}\n'
        '{"at": "soon", "text": "Tinder: new match: Bo", "sent": true}\n',
        encoding="utf-8")
    back = Notifier(tmp_path, Phone()).read_back("Tinder: new match")
    assert "? Tinder: new match: Ana | ? Tinder: new match: Bo" in back.summary()


def test_the_answer_fits_one_line_of_history(tmp_path):
    lines = [f"Hinge: new message from P{i}: " + "y" * 900 for i in range(9)]
    back = notified(tmp_path, *lines).read_back()
    assert len(back.summary()) <= notify.READ_MAX_CHARS + 200
    assert back.shown and back.shown[-1].text == lines[-1]
    assert "\n" not in back.summary()


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

class Reply:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> dict:
        return self._body


def test_a_notification_goes_to_the_chat_as_plain_text(monkeypatch):
    calls = []

    def post(url, data=None, timeout=None):
        calls.append((url, data))
        return Reply(200, {"ok": True, "result": {}})

    monkeypatch.setattr("httpx.post", post)
    Telegram(TOKEN, "42").send("Hinge: new message from Alex: *hi* _there_")
    ((url, data),) = calls
    assert url.endswith("/sendMessage")
    assert data["chat_id"] == "42"
    assert data["text"] == "Hinge: new message from Alex: *hi* _there_"
    # A match's message is not markup: no parse mode, so nothing in it is
    # read as formatting or can make Telegram refuse the message.
    assert "parse_mode" not in data


def test_a_refused_token_is_not_asked_again(monkeypatch):
    tries = []

    def post(url, data=None, timeout=None):
        tries.append(url)
        return Reply(401, {"ok": False, "description": "Unauthorized"})

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(NotifyError) as err:
        Telegram(TOKEN, "42", sleep=lambda s: None).send("x")
    assert err.value.refused and len(tries) == 1
    assert "Unauthorized" in str(err.value)


def test_a_network_error_is_tried_again_and_never_quotes_the_token(monkeypatch):
    """The token is in the URL, and an HTTP error can quote the URL -- into
    `events.jsonl` and the model's history."""
    import httpx

    tries = []

    def post(url, data=None, timeout=None):
        tries.append(url)
        raise httpx.ConnectError(f"could not reach {url}")

    monkeypatch.setattr("httpx.post", post)
    naps = []
    with pytest.raises(NotifyError) as err:
        Telegram(TOKEN, "42", attempts=2, sleep=naps.append).send("x")
    assert len(tries) == 2 and naps == [1.0]
    assert not err.value.refused
    assert TOKEN not in str(err.value) and "***" in str(err.value)


def test_flood_control_waits_as_long_as_telegram_asks(monkeypatch):
    replies = [Reply(429, {"ok": False, "description": "Too Many Requests",
                           "parameters": {"retry_after": 3}}),
               Reply(200, {"ok": True})]
    monkeypatch.setattr("httpx.post",
                        lambda url, data=None, timeout=None: replies.pop(0))
    naps = []
    Telegram(TOKEN, "42", sleep=naps.append).send("x")
    assert naps == [3.0]


def test_the_bot_is_checked_without_sending_anything(monkeypatch):
    calls = []

    def post(url, data=None, timeout=None):
        calls.append(url)
        return Reply(200, {"ok": True, "result": {"username": "my_bot"}})

    monkeypatch.setattr("httpx.post", post)
    assert Telegram(TOKEN, "42").whoami() == "my_bot"
    assert [url.rsplit("/", 1)[-1] for url in calls] == ["getMe"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_the_bot_token_never_leaves_through_the_config_dict():
    cfg = Config()
    cfg.notify.telegram_token = TOKEN
    cfg.notify.telegram_chat_id = "42"
    shown = cfg.to_dict()
    assert shown["notify"]["telegram_token"] == "***"
    assert shown["notify"]["telegram_chat_id"] == "42"
    assert cfg.telegram_token() == TOKEN
    # A traceback or a failed assertion prints the whole config.
    assert TOKEN not in repr(cfg)


def test_the_environment_supplies_what_config_json_does_not(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    cfg = Config()
    assert cfg.telegram_token() == TOKEN and cfg.telegram_chat_id() == "42"
    assert isinstance(Notifier.from_config(cfg).transport, Telegram)


def test_no_bot_or_no_chat_means_notifications_are_off(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    cfg = Config()
    cfg.notify.telegram_token = TOKEN
    assert Notifier.from_config(cfg).transport is None


def test_a_chat_id_written_as_a_number_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"notify": {"telegram_token": TOKEN,
                                           "telegram_chat_id": 1589}}))
    assert load_config(str(path)).config.telegram_chat_id() == "1589"


# ---------------------------------------------------------------------------
# the action
# ---------------------------------------------------------------------------

def act(**kw) -> AgentAction:
    return AgentAction(observation="", reasoning="", **kw)


def test_notify_needs_something_to_say():
    with pytest.raises(ValueError, match="notify requires"):
        act(action="notify", text="   ")


def test_two_notifications_are_two_actions_with_one_history_line_each():
    two = act(action="notify", text="Tinder: new match: Ana\nTinder: new match: Bo")
    other = act(action="notify", text="Tinder: new match: Cy")
    assert two.signature() != other.signature()
    shown = two.describe()
    assert "\n" not in shown
    assert "Tinder: new match: Ana | Tinder: new match: Bo" in shown


def test_notify_never_reaches_the_device():
    dev = fake.FakeDevice()
    with pytest.raises(ActionError, match="does not act on the phone"):
        execute(dev, act(action="notify", text="x"), dev.observe())
    assert dev.taps == [] and dev.actions == []


def test_read_notified_needs_nothing_and_never_reaches_the_device():
    everything = act(action="read_notified")
    assert everything.describe() == "read_notified"
    dev = fake.FakeDevice()
    with pytest.raises(ActionError, match="does not act on the phone"):
        execute(dev, everything, dev.observe())
    assert dev.taps == [] and dev.actions == []


def test_two_lookups_are_two_actions_with_one_history_line_each():
    rows = act(action="read_notified",
               text="Hinge: new message from S\nHinge: new message from K")
    other = act(action="read_notified", text="Hinge: new message from G")
    assert rows.signature() != other.signature()
    assert rows.signature() != act(action="notify", text=rows.text).signature()
    shown = rows.describe()
    assert "\n" not in shown
    assert "Hinge: new message from S | Hinge: new message from K" in shown


# ---------------------------------------------------------------------------
# in the loop
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 10
    c.safety.unattended = True
    return c


def notifying(text: str):
    """A model that notifies once on the first screen, then reports done."""
    told = []

    def policy(screen, llm):
        if not told:
            told.append(text)
            return AgentAction(observation="a new message", reasoning="the "
                               "policy says to tell the owner", action="notify",
                               text=text)
        return AgentAction(observation="told", reasoning="nothing left",
                           action="done", text="told the owner")
    return policy


def run_with(cfg, policy, phone):
    dev = fake.FakeDevice(cfg)
    agent = Agent(dev, fake.FakeLLM(dev, policy), cfg)
    agent.notifier = Notifier(cfg.run.artifacts_dir, phone)
    outcome, state = agent.run("tell the owner about new messages")
    return dev, outcome, state


def events_of(cfg, state) -> list:
    path = runlog.run_dir(cfg, state.run_id) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_notify_step_tells_the_owner_and_leaves_the_phone_alone(cfg):
    phone = Phone()
    dev, outcome, state = run_with(cfg, notifying(
        "Hinge: new message from Alex: hii"), phone)

    assert outcome == "success"
    assert phone.sent == ["Hinge: new message from Alex: hii"]
    assert dev.taps == [] and not any(a.startswith("press") for a in dev.actions)
    line = next(h for h in state.history if " notify " in h)
    assert "-> sent" in line and "1 sent to the owner's phone" in line
    # Something the run had not done before, which the harness watched happen.
    assert state.last_progress == "it sent 1 notification(s)"

    events = events_of(cfg, state)
    told = [e for e in events if e["kind"] == "notify"]
    assert len(told) == 1 and told[0]["sent"] == ["Hinge: new message from Alex: hii"]
    # Nothing was verified, so nothing is graded as if it had been: a verify
    # grade feeds the app's record of what fails there.
    assert not [e for e in events if e["kind"] == "verify"
                and e["step"] == told[0]["step"]]


def test_the_next_pass_does_not_tell_the_owner_again(cfg):
    phone = Phone()
    run_with(cfg, notifying("Hinge: new message from Alex: hii"), phone)
    _, outcome, state = run_with(cfg, notifying("Hinge: new message from Alex: Hii!"), phone)

    assert outcome == "success"
    assert phone.sent == ["Hinge: new message from Alex: hii"]
    line = next(h for h in state.history if " notify " in h)
    assert "-> already sent" in line
    # A repeat is not progress: it buys the stall ladder nothing.
    assert "notification" not in state.last_progress


def test_an_undelivered_notification_is_not_a_failed_step(cfg):
    phone = Phone()
    phone.fail = NotifyError("Telegram answered: Bad Gateway")
    _, outcome, state = run_with(cfg, notifying("Bumble: new message from Sam: hey"), phone)

    assert outcome == "success"
    assert state.consecutive_failures == 0
    line = next(h for h in state.history if " notify " in h)
    assert "-> not delivered" in line and "kept" in line


def test_without_a_bot_the_run_carries_on(cfg):
    _, outcome, state = run_with(cfg, notifying("Tinder: new match: Ana"), None)
    assert outcome == "success"
    line = next(h for h in state.history if " notify " in h)
    assert "-> not sent" in line and "not set up" in line


def test_a_dry_run_does_not_notify(cfg):
    cfg.run.dry_run = True
    phone = Phone()
    _, _, state = run_with(cfg, notifying("Tinder: new match: Ana"), phone)
    assert phone.sent == []
    assert any("(dry run) notify" in h for h in state.history)


def reading(text: str):
    """A model that reads back what was notified, then reports done."""
    read = []

    def policy(screen, llm):
        if not read:
            read.append(text)
            return AgentAction(observation="the chat list", reasoning="see which "
                               "rows an earlier pass already read",
                               action="read_notified", text=text)
        return AgentAction(observation="read", reasoning="nothing new",
                           action="done", text="nothing new to open")
    return policy


def test_a_pass_reads_back_what_an_earlier_one_notified(cfg):
    phone = Phone()
    run_with(cfg, notifying("Hinge: new message from S: maybe tomorrow ."), phone)
    dev, outcome, state = run_with(cfg, reading("Hinge: new message from S"), phone)

    assert outcome == "success"
    assert dev.taps == [] and not any(a.startswith("press") for a in dev.actions)
    line = next(h for h in state.history if " read_notified " in h)
    assert "-> found (1 notified before, oldest first: " in line
    assert line.rstrip(")").endswith("Hinge: new message from S: maybe tomorrow .")
    # Reading what was already known is not progress.
    assert "notification" not in state.last_progress
    assert phone.sent == ["Hinge: new message from S: maybe tomorrow ."]

    (event,) = [e for e in events_of(cfg, state) if e["kind"] == "read_notified"]
    assert event["matched"] == 1 and event["grade"] == "found"
    assert event["shown"] == ["Hinge: new message from S: maybe tomorrow ."]
    assert event["filters"] == ["Hinge: new message from S"]


def test_a_dry_run_reads_back_all_the_same(cfg):
    """It touches nothing, and a dry run deciding without the answer would
    decide differently from a real one."""
    run_with(cfg, notifying("Hinge: new message from S: maybe tomorrow ."), Phone())
    cfg.run.dry_run = True
    _, _, state = run_with(cfg, reading("from S"), Phone())
    line = next(h for h in state.history if " read_notified " in h)
    assert "(dry run)" not in line and "-> found" in line


# ---------------------------------------------------------------------------
# a policy that must not send at all
# ---------------------------------------------------------------------------

def test_send_and_enter_caps_close_every_door_a_message_goes_out_through():
    """What a notifications-only policy puts in its front matter: the button,
    the keyboard's action key, and typing with press_enter are all refused."""
    limits = parse_send_limits("like=0, send=0, enter=0")
    sends = SendLog()
    for label in ("Send message", "Send", "enter", "input_text with press_enter"):
        assert "forbids" in sends.over_limit(label, limits), label
