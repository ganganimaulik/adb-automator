"""The never-double-reply guarantee, exercised through the real agent loop.

`test_conversation.py` checks the gate in isolation. These run it where it
actually has to work: inside `Agent._loop`, against a scripted phone, with a real
ledger on disk.
"""

from __future__ import annotations

import json

import pytest

from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.conversation import read_conversation
from adbagent.ledger import ReplyLedger
from adbagent.memory import Memory

from . import fake
from . import xmlgen as X

SENT_REPLY = "sure, one sec"


def chat_app():
    """A thread where tapping Send adds our reply to the conversation."""
    return {
        "thread": fake.FakeScreen(
            xml=X.chat_thread(title="khushi", messages=["hey", "you around?"]),
            taps={"Send": "thread_sent"}, back="inbox"),
        "thread_sent": fake.FakeScreen(
            xml=X.chat_thread(title="khushi",
                              messages=["hey", "you around?", SENT_REPLY]),
            taps={"Send": "thread_sent"}, back="inbox"),
        "inbox": fake.FakeScreen(
            xml=X.chat_thread(title="khushi", messages=["hey", "you around?"],
                              with_send=False),
            taps={}),
    }


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 6
    c.safety.unattended = True
    return c


@pytest.fixture()
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


@pytest.fixture()
def ledger(tmp_path):
    return ReplyLedger(tmp_path / "replies.jsonl")


def run_taps_send(dev, mem, cfg, ledger, policy_text="be brief"):
    llm = fake.FakeLLM(dev, fake.tap_label("Send"))
    agent = Agent(dev, mem, llm, cfg, ledger=ledger, policy=policy_text)
    outcome, state = agent.run("reply to new messages")
    return outcome, state, llm


#: Where `xmlgen.chat_thread` draws the Send button. Counting taps by coordinate
#: rather than by label because `FakeDevice.actions` records only `tap(x,y)`.
SEND_BOUNDS = (880, 1960, 1040, 2080)


def sends(dev) -> int:
    """How many taps actually landed on the Send control."""
    left, top, right, bottom = SEND_BOUNDS
    return sum(1 for x, y in dev.taps
               if left <= x <= right and top <= y <= bottom)


# -- the guarantee ----------------------------------------------------------

def test_first_reply_is_sent_and_recorded(cfg, mem, ledger):
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    run_taps_send(dev, mem, cfg, ledger)
    assert dev.state == "thread_sent", "the reply never went out"
    # One attempt and one confirmation, both on disk.
    rows = [json.loads(l) for l in
            ledger.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["confirmed"] for r in rows] == [False, True]


def test_the_second_send_in_the_same_run_is_refused(cfg, mem, ledger):
    """The policy taps Send on every screen that has one; only the first lands."""
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    run_taps_send(dev, mem, cfg, ledger)
    assert sends(dev) == 1, f"more than one send landed: {dev.actions}"


def test_a_later_run_will_not_answer_the_same_tail(cfg, mem, ledger):
    """A fresh pass, a fresh Agent, the same conversation: nothing goes out."""
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    run_taps_send(dev, mem, cfg, ledger)
    before = sends(dev)

    # The phone is now where the first pass left it, and the ledger is re-read
    # from disk exactly as the next watch pass would.
    dev2 = fake.FakeDevice(cfg, start="thread_sent", app=chat_app())
    run_taps_send(dev2, mem, cfg, ReplyLedger(ledger.path))
    assert sends(dev2) == 0, "it replied twice across passes"
    assert before == 1


def test_a_new_message_is_answered_in_a_later_run(cfg, mem, ledger):
    """The gate must not be a one-way door: fresh content earns a reply."""
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    run_taps_send(dev, mem, cfg, ledger)

    app = chat_app()
    app["thread_sent"] = fake.FakeScreen(
        xml=X.chat_thread(title="khushi",
                          messages=["hey", "you around?", SENT_REPLY,
                                    "still there?"]),
        taps={"Send": "thread_replied_again"}, back="inbox")
    app["thread_replied_again"] = fake.FakeScreen(
        xml=X.chat_thread(title="khushi",
                          messages=["hey", "you around?", SENT_REPLY,
                                    "still there?", "yes!"]),
        taps={}, back="inbox")
    dev2 = fake.FakeDevice(cfg, start="thread_sent", app=app)
    # Past the cooldown, so only the digest rule is in play.
    cfg.watch.thread_cooldown_s = 0
    run_taps_send(dev2, mem, cfg, ReplyLedger(ledger.path))
    assert dev2.state == "thread_replied_again", "a new message went unanswered"


# -- the other refusal paths, through the loop ------------------------------

def test_draft_mode_sends_nothing(cfg, mem, ledger):
    cfg.watch.draft = True
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    run_taps_send(dev, mem, cfg, ledger)
    assert sends(dev) == 0
    assert dev.state == "thread"
    assert len(ledger) == 0, "draft mode must not record a reply either"


def test_an_unidentifiable_conversation_sends_nothing(cfg, mem, ledger):
    """No thread name on screen means a duplicate cannot be ruled out."""
    app = {
        "thread": fake.FakeScreen(
            xml=X.chat_thread(title="khushi", messages=["hey"],
                              with_header=False),
            taps={"Send": "sent"}),
        "sent": fake.FakeScreen(xml=X.chat_thread(title="khushi",
                                                 messages=["hey", "hi"])),
    }
    dev = fake.FakeDevice(cfg, start="thread", app=app)
    run_taps_send(dev, mem, cfg, ledger)
    assert sends(dev) == 0
    assert dev.state == "thread"


def test_failing_open_lets_it_through(cfg, mem, ledger):
    cfg.watch.fail_closed = False
    app = {
        "thread": fake.FakeScreen(
            xml=X.chat_thread(title="khushi", messages=["hey"],
                              with_header=False),
            taps={"Send": "sent"}),
        "sent": fake.FakeScreen(xml=X.chat_thread(title="khushi",
                                                 messages=["hey", "hi"],
                                                 with_header=False)),
    }
    dev = fake.FakeDevice(cfg, start="thread", app=app)
    run_taps_send(dev, mem, cfg, ledger)
    assert dev.state == "sent"


def test_the_refusal_is_explained_to_the_model(cfg, mem, ledger):
    """A refusal the model cannot read is a refusal it will retry forever."""
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    _outcome, state, _llm = run_taps_send(dev, mem, cfg, ledger)
    refusals = [h for h in state.history if "refused" in h.lower()]
    assert refusals, f"no refusal reached the history: {state.history}"


# -- an ordinary run is untouched -------------------------------------------

def test_without_a_ledger_nothing_is_gated(cfg, mem):
    """`adbagent run` must behave exactly as it did before any of this."""
    dev = fake.FakeDevice(cfg, start="thread", app=chat_app())
    llm = fake.FakeLLM(dev, fake.tap_label("Send"))
    agent = Agent(dev, mem, llm, cfg)          # no ledger, no policy
    agent.run("reply to new messages")
    assert dev.state == "thread_sent"
    assert sends(dev) >= 1


def test_no_policy_block_without_a_policy(cfg, mem):
    from adbagent import prompts
    assert prompts.policy_block("") == ""
    assert "REPLY POLICY" in prompts.policy_block("be brief")
