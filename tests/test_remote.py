"""A run decided from outside the process, against a scripted phone.

`remote.Session` runs the ordinary agent loop on a thread and takes each
decision from its caller -- an MCP client, in `mcp_server.py`. These tests drive
it directly, the way that client does.
"""

from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path

import pytest

from adbagent import phonelock, prompts
from adbagent.actions import AgentAction
from adbagent.agent import Agent, RunState
from adbagent.config import Config
from adbagent.remote import RemoteAgent, Session, SessionError, Turn

from . import fake
from . import xmlgen as X

GOAL = "open the Wi-Fi settings screen"


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    return c


def start(cfg, dev, goal=GOAL, **kw):
    session = Session(cfg, open_device=lambda _cfg: dev, driver="test",
                      idle_timeout_s=kw.pop("idle_timeout_s", 30), **kw)
    return session, session.start(goal)


def element(turn: Turn, label: str) -> dict:
    """{index, key} of the element whose label is `label`, read off the screen."""
    for line in turn.screen.splitlines():
        if f'"{label}"' in line:
            index = int(re.match(r"#(\d+)", line).group(1))
            key = re.search(r"\bk=(\w+)", line).group(1)
            return {"index": index, "key": key}
    raise AssertionError(f"{label!r} is not on screen:\n{turn.screen}")


def tap(target: dict) -> AgentAction:
    return AgentAction(observation="", reasoning="", action="tap", target=target)


def back() -> AgentAction:
    return AgentAction(observation="", reasoning="", action="press_key",
                       key="back")


def events(cfg) -> list:
    [run_dir] = [p for p in Path(cfg.run.artifacts_dir).iterdir() if p.is_dir()]
    return [json.loads(line) for line in
            (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# A session from start to end
# ---------------------------------------------------------------------------

def test_start_shows_the_first_screen_and_holds_the_phone(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        assert turn.step == 1 and not turn.outcome
        assert '"Wi-Fi"' in turn.screen
        assert "the agent wants" not in turn.render()
        assert "test" in phonelock.holder(cfg.run.artifacts_dir)
    finally:
        session.close()
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_an_action_goes_through_the_loop_and_comes_back_graded(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    turn = session.act([tap(element(turn, "Wi-Fi"))])
    assert dev.state == "wifi"
    [line] = turn.happened
    assert line.startswith("1. tap #") and "-> success" in line
    assert '"Wi-Fi details"' in turn.screen and turn.step == 2
    session.end("the Wi-Fi screen is open")


def test_ending_records_the_answer_and_gives_the_phone_back(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    session.act([tap(element(turn, "Wi-Fi"))])
    turn = session.end("the Wi-Fi screen is open")

    assert turn.outcome == "success"
    assert turn.result == "the Wi-Fi screen is open"
    assert "SESSION ENDED: success" in turn.render()
    assert dev.closes == 1
    assert phonelock.holder(cfg.run.artifacts_dir) == ""
    # Recorded like any run, with the driver where the model would be.
    recorded = events(cfg)
    assert recorded[0]["kind"] == "run_start"
    assert recorded[0]["model"] == "test (mcp)"
    assert recorded[-1]["kind"] == "run_end"
    assert recorded[-1]["outcome"] == "success"
    assert recorded[-1]["result"] == "the Wi-Fi screen is open"
    with pytest.raises(SessionError, match="has ended"):
        session.act([back()])


def test_failing_says_why(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = start(cfg, dev)
    turn = session.end("Wi-Fi is managed by the admin", failed=True)
    assert turn.outcome == "failed"
    assert turn.result == "Wi-Fi is managed by the admin"


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

def test_a_batch_stops_at_the_first_action_that_does_not_work(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = start(cfg, dev)
    try:
        turn = session.act([tap({"index": 99}), back()])
        assert "press(back)" not in dev.actions
        assert any("-> failed" in line for line in turn.happened)
        assert turn.failure
        assert "no element matches" in turn.notes
    finally:
        session.close()


def test_a_batch_stops_before_a_target_read_off_a_screen_that_is_gone(cfg):
    """`#N` and x/y were read off the first screen; after a navigation they
    would land on whatever the new screen has there."""
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        wifi = element(turn, "Wi-Fi")
        bluetooth = element(turn, "Bluetooth")
        turn = session.act([tap(wifi), tap({"index": bluetooth["index"]})])
        assert dev.state == "wifi"
        assert len(dev.taps) == 1
        assert turn.happened[-1].startswith("stopped before the remaining 1")
    finally:
        session.close()


def test_a_target_named_by_a_key_still_on_screen_survives_a_change(cfg):
    """Seen on a real feed: a down-fling changed the layout and held back the
    up-fling aimed at the same scroller, whose key was still on screen."""
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        wifi = element(turn, "Wi-Fi")
        # The status bar's clock is on both screens, with the same key.
        clock = re.search(r"#(\d+)\s+\[Text\]\s+\"9:41\".*?k=(\w+)",
                          turn.screen)
        target = {"index": 99, "key": clock.group(2)}
        turn = session.act([tap(wifi), tap(target)])
        assert dev.state == "wifi"
        assert len(dev.taps) == 2
        assert not any(line.startswith("stopped before") for line in turn.happened)
    finally:
        session.close()


def test_a_batch_on_one_screen_runs_through(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        toggle = {"index": int(re.search(r"#(\d+)\s+\[Toggle\]",
                                         turn.screen).group(1))}
        turn = session.act([tap(toggle), tap(toggle)])
        assert len(dev.taps) == 2
        assert [l for l in turn.happened if "-> success" in l] and \
            len(turn.happened) == 2
    finally:
        session.close()


# ---------------------------------------------------------------------------
# What only the person may decide
# ---------------------------------------------------------------------------

def _to_wifi(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    turn = session.act([tap(element(turn, "Wi-Fi"))])
    return dev, session, turn


def test_an_irreversible_press_waits_for_the_person(cfg):
    dev, session, turn = _to_wifi(cfg)
    try:
        forget = element(turn, "Forget network")
        turn = session.act([tap(forget)])
        assert turn.pending is not None and turn.pending.kind == "irreversible"
        assert "Forget network" in turn.pending.prompt
        assert "APPROVAL NEEDED (request_id 1)" in turn.render()
        assert len(dev.taps) == 1            # not pressed yet

        turn = session.answer(turn.pending.id, True)
        assert len(dev.taps) == 2
        assert turn.pending is None
        assert any('"Forget network"' in line for line in turn.happened)
    finally:
        session.close()


def test_a_declined_press_is_refused_and_said_so(cfg):
    dev, session, turn = _to_wifi(cfg)
    try:
        turn = session.act([tap(element(turn, "Forget network"))])
        turn = session.answer(turn.pending.id, False)
        assert len(dev.taps) == 1
        assert any("refused to press 'Forget network'" in line
                   for line in turn.happened)
        assert "was refused" in turn.notes
    finally:
        session.close()


def test_any_call_but_an_answer_declines_the_approval(cfg):
    dev, session, turn = _to_wifi(cfg)
    try:
        turn = session.act([tap(element(turn, "Forget network"))])
        with pytest.raises(SessionError, match="no approval 7"):
            session.answer(7, True)
        turn = session.act([back()])
        assert turn.happened[0].startswith("approval 1 was declined")
        assert dev.state == "home" and len(dev.taps) == 1
    finally:
        session.close()


def test_allow_destructive_does_not_ask(cfg):
    cfg.safety.allow_destructive = True
    dev, session, turn = _to_wifi(cfg)
    try:
        turn = session.act([tap(element(turn, "Forget network"))])
        assert turn.pending is None and len(dev.taps) == 2
    finally:
        session.close()


def _chat(cfg):
    return fake.FakeDevice(cfg, start="chat", app={
        "chat": fake.FakeScreen(xml=X.chat_thread(draft="see you at 7"))})


def test_a_send_waits_for_the_person_and_shows_what_it_says(cfg):
    dev = _chat(cfg)
    session, turn = start(cfg, dev, goal="reply to khushi")
    try:
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is not None and turn.pending.kind == "send"
        # The control, the app and the draft. Not the conversation's name: it
        # was guessed off the tree, and the driver asking knows whose it is.
        assert "'Send'" in turn.pending.prompt
        assert "com.instagram.android" in turn.pending.prompt
        assert "see you at 7" in turn.pending.prompt
        assert not dev.taps
        turn = session.answer(turn.pending.id, True)
        assert len(dev.taps) == 1
    finally:
        session.close()


def test_a_send_the_person_turned_down_is_not_asked_about_again(cfg):
    dev = _chat(cfg)
    session, turn = start(cfg, dev, goal="reply to khushi")
    try:
        send = element(turn, "Send")
        turn = session.act([tap(send)])
        turn = session.answer(turn.pending.id, False)
        assert not dev.taps
        turn = session.act([tap(send)])
        assert turn.pending is None and not dev.taps
        assert "the person did not allow this send" in turn.notes
    finally:
        session.close()


def test_draft_mode_holds_a_send_before_anyone_is_asked(cfg):
    cfg.watch.draft = True
    dev = _chat(cfg)
    session, turn = start(cfg, dev, goal="reply to khushi")
    try:
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is None and not dev.taps
        assert "draft mode is on" in turn.notes
    finally:
        session.close()


def test_without_the_send_check_a_send_just_goes(cfg):
    cfg.safety.check_sends = False
    dev = _chat(cfg)
    session, turn = start(cfg, dev, goal="reply to khushi")
    try:
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is None and len(dev.taps) == 1
    finally:
        session.close()


def test_a_password_screen_ends_the_session_unseen(cfg):
    login = X.dump(X.N("android.widget.FrameLayout", (0, 0, X.W, X.H),
                       rid="content", children=[
                           X.status_bar(),
                           X.N("android.widget.EditText", (40, 400, 1040, 520),
                               rid="password", hint="Password", password=True,
                               clickable=True, focusable=True)]))
    dev = fake.FakeDevice(cfg, start="login",
                          app={"login": fake.FakeScreen(xml=login)})
    session, turn = start(cfg, dev, goal="log in")
    assert turn.outcome == "needs_user"
    assert "password" in turn.evidence.lower()
    assert "Password" not in turn.render()
    assert dev.closes == 1


# ---------------------------------------------------------------------------
# Handing the phone over, and a driver that goes away
# ---------------------------------------------------------------------------

def test_the_person_can_have_the_phone_and_give_it_back(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        session.release()
        assert dev.closes == 1 and not dev.session_open
        with pytest.raises(SessionError, match="the person has the phone"):
            session.act([back()])
        dev.state = "bluetooth"      # what they did with it
        turn = session.reclaim()
        assert dev.opens == 1
        assert any("the person took the phone" in line for line in turn.happened)
        assert '"Pair new device"' in turn.screen
    finally:
        session.close()


def test_a_driver_that_goes_quiet_loses_the_phone(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = start(cfg, dev, idle_timeout_s=0.2)
    time.sleep(1.0)
    turn = session.look()
    assert turn.outcome == "aborted"
    assert "no instruction for" in turn.evidence
    assert dev.closes == 1
    assert phonelock.holder(cfg.run.artifacts_dir) == ""


def test_time_spent_waiting_for_the_driver_is_not_work(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    time.sleep(0.3)
    session.end("done")
    [end] = [e for e in events(cfg) if e["kind"] == "run_end"]
    assert end["paused_s"] >= 0.3


def test_look_adds_a_fresh_screenshot_and_repeats_nothing(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        shots = dev.screenshots
        seen = session.look()
        assert seen.screenshot and dev.screenshots == shots + 1
        assert seen.happened == [] and seen.screen == turn.screen
        assert seen.step == turn.step          # nothing changed, no step spent
    finally:
        session.close()


def test_look_reads_again_a_screen_that_changed_under_the_turn(cfg):
    """Seen against a real phone: a dialog came up after the turn was read, and
    a fresh screenshot beside the old list put `#N` on a screen that was gone."""
    dev = fake.FakeDevice(cfg)
    session, turn = start(cfg, dev)
    try:
        dev.state = "bluetooth"              # the phone moved on its own
        seen = session.look()
        assert '"Pair new device"' in seen.screen
        assert seen.step == turn.step + 1
        assert seen.happened == ["the screen changed since the last turn; "
                                 "this is it now"]
        assert seen.screenshot
        assert not seen.failure              # a look is not a failed step
    finally:
        session.close()


def test_the_phone_is_one_session_at_a_time(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = start(cfg, dev)
    try:
        with pytest.raises(phonelock.PhoneBusy, match="MCP session"):
            Session(cfg, open_device=lambda _cfg: dev).start(GOAL)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# What the driver is shown
# ---------------------------------------------------------------------------

def _remote(cfg) -> RemoteAgent:
    return RemoteAgent(fake.FakeDevice(cfg), cfg, inbox=queue.Queue(),
                       outbox=queue.Queue())


def test_standing_advice_is_sent_once_a_session(cfg):
    agent = _remote(cfg)
    note = f"{prompts.SCROLLING_ADVICE}\n\nscrolling down did not reveal new content"
    assert prompts.SCROLLING_ADVICE in agent._fresh_notes(note)
    again = agent._fresh_notes(note)
    assert prompts.SCROLLING_ADVICE not in again
    assert again == "scrolling down did not reveal new content"


def test_a_batch_reports_a_folded_repeat_once():
    """Seen against a real phone: two backs in one batch came back as
    "3. back" and then "3-4. back [x2]"."""
    from adbagent.remote import _merge

    happened = ["approval 1 was declined: x"]
    _merge(happened, ["3. press_key back -> success"])
    _merge(happened, ["3-4. press_key back -> success [x2]"])
    _merge(happened, ["5. tap #2 -> success"])
    assert happened == ["approval 1 was declined: x",
                        "3-4. press_key back -> success [x2]",
                        "5. tap #2 -> success"]


def test_only_new_history_is_sent_and_a_folded_line_is_sent_again(cfg):
    agent = _remote(cfg)
    state = RunState(goal=GOAL, run_id="r", intent_id="i")
    state.history.append("1. swipe #4 left -> success")
    assert agent._delta(state) == ["1. swipe #4 left -> success"]
    assert agent._delta(state) == []
    # `append_history` folds a repeat into the line before it.
    state.history[-1] = "1-2. swipe #4 left -> success [x2]"
    state.history.append("3. tap #2 -> success")
    assert agent._delta(state) == ["1-2. swipe #4 left -> success [x2]",
                                   "3. tap #2 -> success"]


def test_what_the_harness_did_on_its_own_is_reported_once(cfg):
    """A forced `back` leaves no history line; a refusal does, and is not doubled."""
    agent = _remote(cfg)
    state = RunState(goal=GOAL, run_id="r", intent_id="i", step=4)
    agent._state = state
    agent._heard("loop_warning", message="step 4: stuck in a loop; going back")
    state.step = 5
    agent._heard("loop_warning", message="step 5: refusing tap/#3 (tried 2x here)")
    state.history.append("5. tap #3 -> refused (already tried 2x here)")
    assert agent._delta(state) == [
        "4. the harness: stuck in a loop; going back",
        "5. tap #3 -> refused (already tried 2x here)"]


def test_the_loop_takes_its_decisions_from_the_seam(cfg):
    """Any `_decide` drives the loop -- and None means look again, not a failure."""
    answers = [None, AgentAction(observation="", reasoning="", action="done",
                                 text="nothing to do")]

    class Scripted(Agent):
        def _decide(self, state, rec, screen, **kwargs):
            return answers.pop(0)

    outcome, state = Scripted(fake.FakeDevice(cfg), None, cfg).run(GOAL)
    assert outcome == "success" and state.result == "nothing to do"
    assert state.consecutive_failures == 0 and state.step == 2


def test_a_session_that_ended_unread_frees_the_phone_for_the_next(cfg):
    """An idle timeout ends the run with nobody there to read it; a new session
    must not be refused as though the old one were still going."""
    from adbagent.mcp_server import Phone

    dev = fake.FakeDevice(cfg)
    phone = Phone(cfg, open_device=lambda _cfg: dev, idle_timeout_s=0.2)
    phone.start(GOAL, "test")
    time.sleep(1.0)
    first = phone.session
    assert not first.live
    turn = phone.start(GOAL, "test")
    assert turn.step == 1 and phone.session is not first
    phone.close()
    first.close()
    assert first.last.outcome == "aborted"


def test_a_policy_is_shown_once_and_its_send_caps_hold(cfg):
    dev = _chat(cfg)
    session = Session(cfg, open_device=lambda _cfg: dev, driver="test",
                      idle_timeout_s=30, policy="Only ever reply 'hi'.",
                      send_limits={"send": 0})
    turn = session.start("reply to khushi")
    try:
        assert "REPLY POLICY" in turn.render()
        assert "Only ever reply 'hi'." in turn.render()
        assert "GOAL:\nreply to khushi" in turn.render()
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is None and not dev.taps     # capped, not asked
        assert "REPLY POLICY" not in turn.render()       # said once
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Several policies, one pass of each in turn
# ---------------------------------------------------------------------------

def _legs(**caps):
    from adbagent.remote import Leg
    return [Leg(goal="open the Wi-Fi settings", policy="Policy one says this.",
                send_limits=caps.get("one", {}), name="One"),
            Leg(goal="open the Bluetooth settings",
                policy="Policy two says that.",
                send_limits=caps.get("two", {}), name="Two")]


def _start_several(cfg, dev, legs):
    session = Session(cfg, open_device=lambda _cfg: dev, driver="test",
                      idle_timeout_s=30, legs=legs)
    return session, session.start("")


def test_several_policies_take_turns_in_one_session(cfg):
    dev = fake.FakeDevice(cfg)
    session, turn = _start_several(cfg, dev, _legs())
    try:
        first = turn.render()
        assert "POLICY 1 of 2: One" in first
        assert "GOAL:\nopen the Wi-Fi settings" in first
        assert "Policy one says this." in first
        assert "test" in phonelock.holder(cfg.run.artifacts_dir)

        turn = session.act([tap(element(turn, "Wi-Fi"))])
        assert turn.leg == 1 and not turn.outcome
        turn = session.end("Wi-Fi is open")

        # Not the end of the session: the next policy's first turn.
        assert not turn.outcome and turn.leg == 2
        second = turn.render()
        assert "POLICY 2 of 2: Two" in second
        assert "GOAL:\nopen the Bluetooth settings" in second
        assert "Policy two says that." in second
        assert "Policy one says this." not in second
        assert any("policy 1 of 2 (One) ended: success -- Wi-Fi is open" in line
                   for line in turn.happened)
        assert any("policy 2 of 2 (Two) starts now" in line
                   for line in turn.happened)
        # The phone was never let go in between.
        assert "test" in phonelock.holder(cfg.run.artifacts_dir)

        turn = session.end("Bluetooth is open")
        assert turn.outcome == "success"
        ended = turn.render()
        assert "SESSION ENDED: success" in ended
        assert "policy 1 of 2 (One) ended: success" in ended
        assert "policy 2 of 2 (Two) ended: success" in ended
    finally:
        session.close()
    assert phonelock.holder(cfg.run.artifacts_dir) == ""
    assert dev.closes == 1                 # one device session for the two
    runs = [p for p in Path(cfg.run.artifacts_dir).iterdir() if p.is_dir()]
    assert len(runs) == 2                  # and a run of its own for each


def test_each_policy_brings_its_own_send_caps(cfg):
    """Policy one forbids the send; policy two leaves it to the person."""
    dev = _chat(cfg)
    session, turn = _start_several(cfg, dev, _legs(one={"send": 0}))
    try:
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is None and not dev.taps      # capped, not asked
        turn = session.end("nothing to send here")
        assert turn.leg == 2
        turn = session.act([tap(element(turn, "Send"))])
        assert turn.pending is not None                     # asked this time
    finally:
        session.close()


def test_a_failed_pass_still_hands_on_to_the_next_policy(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = _start_several(cfg, dev, _legs())
    try:
        turn = session.end("Wi-Fi is managed by the admin", failed=True)
        assert not turn.outcome and turn.leg == 2
        assert any("policy 1 of 2 (One) ended: failed" in line
                   for line in turn.happened)
    finally:
        session.close()


def test_skip_remaining_ends_the_session_with_the_pass_in_hand(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = _start_several(cfg, dev, _legs())
    turn = session.end("that will do", skip_remaining=True)
    assert turn.outcome == "success"
    assert "not run: Two -- end_session skipped them" in turn.render()
    assert phonelock.holder(cfg.run.artifacts_dir) == ""
    runs = [p for p in Path(cfg.run.artifacts_dir).iterdir() if p.is_dir()]
    assert len(runs) == 1


def test_a_batch_stops_where_a_policys_pass_ends(cfg):
    """What was left of the batch was planned for the last policy's app."""
    cfg.run.max_steps = 2
    dev = fake.FakeDevice(cfg)
    session, turn = _start_several(cfg, dev, _legs())
    try:
        turn = session.act([tap(element(turn, "Wi-Fi")), back(), back()])
        assert turn.leg == 2 and not turn.outcome
        assert dev.actions.count("press(back)") == 1
        assert any("stopped before the remaining 1 action(s)" in line
                   for line in turn.happened)
        assert any("policy 1 of 2 (One) ended: failed" in line
                   for line in turn.happened)
    finally:
        session.close()


def test_closing_mid_chain_runs_none_of_the_rest(cfg):
    dev = fake.FakeDevice(cfg)
    session, _ = _start_several(cfg, dev, _legs())
    session.close("the client went away")
    assert session.last.outcome == "aborted"
    assert "not run: Two -- the session ended" in session.last.render()
    runs = [p for p in Path(cfg.run.artifacts_dir).iterdir() if p.is_dir()]
    assert len(runs) == 1
