"""The whole loop, end to end, against a scripted phone.

The agent always uses the LLM for decisions. These tests verify the core loop:
perceive → ask the LLM → guard → act → verify → learn → repeat.
"""

from __future__ import annotations

import json

import pytest

from adbagent.actions import AgentAction
from adbagent.agent import Agent, Oracle, RunState, needs_screenshot
from adbagent.config import Config
from adbagent.device import DeviceTimeout
from adbagent.memory import Memory

from . import fake
from . import xmlgen as X


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 25
    c.safety.unattended = True      # never block a test on input()
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


GOAL = "open the Wi-Fi settings screen"


def run(dev, mem, cfg, policy, **kw):
    llm = fake.FakeLLM(dev, policy)
    agent = Agent(dev, mem, llm, cfg, **kw)
    outcome, state = agent.run(GOAL)
    return outcome, state, llm


# ---------------------------------------------------------------------------
# Basic operation
# ---------------------------------------------------------------------------

def test_run_uses_the_llm_and_succeeds(cfg, mem):
    dev = fake.FakeDevice(cfg)
    outcome, state, llm = run(dev, mem, cfg,
                              fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.calls >= 2          # at least one decide plus the completion judge


def test_every_turn_is_told_what_day_the_phone_thinks_it_is(cfg, mem):
    """A goal bounded in time cannot be read without it.

    ``runs/963a4f4ae96c`` -- "check today and yesterday's messages" -- had no
    date in the prompt at all, and walked a recency-ordered list of chats from
    today's down through Sunday, Saturday and 27 Jul.
    """
    dev = fake.FakeDevice(cfg)
    _, _, llm = run(dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    # Every decide turn, and the same date on each -- this message sits above
    # the goal, so a value that moved would evict everything after it.
    assert llm.dates_seen and set(llm.dates_seen) == {fake.TODAY}
    # Read once for the run, not once a turn: it is an adb round trip.
    assert dev.date_reads == 1


def test_a_phone_that_will_not_give_its_date_is_not_guessed_for(cfg, mem):
    """The host clock is not a fallback: it can be a day out, and the prompt
    states this as fact with nothing on screen to check it against."""
    dev = fake.FakeDevice(cfg)
    dev.date = ""
    _, _, llm = run(dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert llm.dates_seen and set(llm.dates_seen) == {""}


# ---------------------------------------------------------------------------
# tap_at with a named control, grounded by the vision locate
# ---------------------------------------------------------------------------

def test_a_named_tap_at_is_grounded_by_the_vision_locate(cfg, mem):
    """The decider names a control it has no pixels for; the locate places it.

    Works in every model configuration because the decider never handles
    coordinates itself: the point it taps is the one the vision model read off
    this turn's own frame.
    """
    dev = fake.FakeDevice(cfg)
    expected = []

    def policy(screen, llm):
        if dev.state == "wifi":
            return AgentAction(observation="arrived", reasoning="goal reached",
                               action="done", text="reached wifi")
        wanted = next(e for e in screen.elements if e.best_text == "Wi-Fi")
        # What a vision model would answer for the named control on this frame.
        llm.location = (wanted.center[0] / screen.width,
                        wanted.center[1] / screen.height)
        expected.append(wanted.center)
        # The name must NOT resolve to a listed element -- "Wi-Fi" itself is in
        # the tree, and naming it would be refused with its #N.
        return AgentAction(observation="the tree does not list it",
                           reasoning="name it and have it located",
                           action="tap_at", text="the Wi-Fi row icon")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.locates == 1 and llm.locates_seen == ["the Wi-Fi row icon"]
    # The tap that went out is the located point, within a pixel of the
    # control's centre (fractions truncate, they do not round).
    cx, cy = expected[0]
    assert abs(dev.taps[0][0] - cx) <= 1 and abs(dev.taps[0][1] - cy) <= 1


def _locator(cfg, mem, dev, location):
    """An agent, a state and a recorder, for calling `_locate_cached` directly."""
    from adbagent.agent import Recorder

    llm = fake.FakeLLM(dev, lambda *a: None)
    llm.location = location
    return (Agent(dev, mem, llm, cfg), llm,
            RunState(goal=GOAL, run_id="r", intent_id="i"),
            Recorder(cfg, "r"))


def test_the_second_locate_of_the_same_control_is_free(cfg, mem):
    """A locate is a screenshot plus a vision call, and it is asked the same
    question over and over: across ``runs/``, 577 named `tap_at`s resolve to 94
    distinct (skeleton, name) pairs, and "send priority like" on one Hinge
    skeleton was located 134 separate times."""
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    agent, llm, state, rec = _locator(cfg, mem, dev, (0.5, 0.9))
    try:
        first = agent._locate_cached(state, rec, screen, "the send pill")
        second = agent._locate_cached(state, rec, screen, "the send pill")
        # Case and inner spacing are not a different control.
        third = agent._locate_cached(state, rec, screen, "The  Send  Pill")
    finally:
        rec.close()

    assert first == second == third == (0.5, 0.9)
    assert llm.locates == 1


def test_a_different_control_or_screen_is_located_afresh(cfg, mem):
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    agent, llm, state, rec = _locator(cfg, mem, dev, (0.5, 0.9))
    try:
        agent._locate_cached(state, rec, screen, "the send pill")
        agent._locate_cached(state, rec, screen, "the attach button")
        assert llm.locates == 2                     # a different name
        dev.state = "wifi"
        other = dev.observe()
        assert other.skeleton_id != screen.skeleton_id
        agent._locate_cached(state, rec, other, "the send pill")
        assert llm.locates == 3                     # a different screen
    finally:
        rec.close()


def test_a_locate_that_misses_is_not_cached(cfg, mem):
    """There is nothing to remember, and caching "not found" would stop the
    next turn looking on a screen that may since have drawn the control."""
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    agent, llm, state, rec = _locator(cfg, mem, dev, None)
    try:
        assert agent._locate_cached(state, rec, screen, "the send pill") is None
        assert agent._locate_cached(state, rec, screen, "the send pill") is None
    finally:
        rec.close()
    assert llm.locates == 2
    assert mem.recall_locate(screen, "the send pill") is None


def test_a_cached_point_that_taps_nothing_is_forgotten(cfg, mem):
    """The invalidation half. Without it the cache is worse than paying for the
    locate: a vision call that misses costs one turn, a cached miss would cost
    every remaining turn that named the same control."""
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    # A point on bare canvas: tapping it changes nothing on the scripted phone.
    mem.record_locate(screen, "the ghost button", 0.5, 0.97)
    assert mem.recall_locate(screen, "the ghost button") is not None

    tries = []

    def policy(scr, llm):
        if len(tries) >= 1:
            return AgentAction(observation="that did nothing",
                               reasoning="give up on it", action="done",
                               text="the point was dead")
        tries.append(True)
        llm.location = (0.5, 0.97)
        return AgentAction(observation="aiming at the remembered point",
                           reasoning="tap where it was last seen",
                           action="tap_at", text="the ghost button")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 0          # the cached point was used, not re-derived
    # ...and having changed nothing, it is no longer remembered.
    assert mem.recall_locate(screen, "the ghost button") is None


def test_a_cached_point_on_the_ban_list_is_dropped_and_located_again(cfg, mem):
    """Something has tapped there since and nothing happened. Falling through
    to a real locate is the whole reason to check the ban list first."""
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    mem.record_locate(screen, "the ghost button", 0.5, 0.97)

    state = RunState(goal=GOAL, run_id="r", intent_id="i")
    state.loops.ban(screen.skeleton_id, "tap_at/0.50,0.97")

    llm = fake.FakeLLM(dev, lambda *a: None)
    llm.location = (0.4, 0.4)
    agent = Agent(dev, mem, llm, cfg)

    from adbagent.agent import Recorder
    rec = Recorder(cfg, "r")
    try:
        where = agent._locate_cached(state, rec, screen, "the ghost button")
    finally:
        rec.close()

    assert where == (0.4, 0.4)       # the fresh locate, not the banned point
    assert llm.locates == 1
    assert mem.recall_locate(screen, "the ghost button") == (0.4, 0.4)


def test_a_tap_at_naming_a_listed_element_is_refused_with_its_index(cfg, mem):
    """The escape hatch is not a shortcut: what the list can name, the list taps."""
    dev = fake.FakeDevice(cfg)
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="refused", reasoning="moving on",
                               action="done", text="taught the index")
        tried.append(True)
        return AgentAction(observation="Wi-Fi is right there",
                           reasoning="lazy coordinate tap", action="tap_at",
                           text="Wi-Fi")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 0             # refused before any locate was paid for
    assert dev.taps == []
    assert "tap_at refused" in (state.last_failure or "")
    assert "#" in (state.last_failure or "")


def test_a_tap_at_naming_something_inside_a_container_label_is_located(cfg, mem):
    """The text half of the refusal guard takes the point half's size rule:
    a name that resolves only as a substring of a big container's aggregated
    label is naming something inside it that has no element of its own, and
    refusing it with "tap it by index" points at the container -- a dead end,
    not guidance. runs/8213dc5e6bf3: Hinge's like-comment composer is one
    full-screen scroller whose label mentions the "Send Priority Like" pill;
    the named tap_at was refused twice with the scroller's #1 before the
    watch was stopped, and no locate ever ran."""
    # The composer as the tree reports it: one full-screen scroller whose
    # label aggregates the composer's contents, the pill among them.
    composer = X.dump(X.N("android.widget.ScrollView", (0, 0, X.W, X.H),
                          desc="Edit comment Send a Rose with message "
                               "Send priority like with message Vac's photo",
                          scrollable=True))
    dev = fake.FakeDevice(cfg, start="composer",
                          app={"composer": fake.FakeScreen(xml=composer)})
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="the composer closed",
                               reasoning="the like went out",
                               action="done", text="like sent")
        tried.append(True)
        # What the vision model answers for the pill on this frame.
        llm.location = (0.59, 0.67)
        return AgentAction(observation="the send pill has no element of its own",
                           reasoning="name it and have it located",
                           action="tap_at", text="Send Priority Like")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 1 and llm.locates_seen == ["Send Priority Like"]
    assert "tap_at refused" not in (state.last_failure or "")
    # The located point, not the container's centre, is what was tapped.
    assert dev.taps == [(int(0.59 * X.W), int(0.67 * X.H))]


def test_a_locate_is_told_ruled_out_points_and_a_repeat_is_not_tapped(cfg, mem):
    """A locate call is stateless: asked again for the same control on an
    unchanged screen, the model re-derives the same wrong point. The harness
    knows that point is dead -- the tap landed and nothing changed -- so the
    next locate is told the ruled-out points, and one answered anyway is the
    same miss as "not found", not another tap. runs/6fc2c7bbddeb paid for
    four locates of Hinge's "Send Priority Like" pill, got (0.60, 0.52) three
    times, and tapped the dead photo there twice."""
    composer = X.dump(X.N("android.widget.ScrollView", (0, 0, X.W, X.H),
                          desc="Edit comment Send a Rose with message "
                               "Send priority like with message Vac's photo",
                          scrollable=True))
    dev = fake.FakeDevice(cfg, start="composer",
                          app={"composer": fake.FakeScreen(xml=composer)})

    def policy(screen, llm):
        if llm.locates >= 2:
            return AgentAction(observation="the pill cannot be placed",
                               reasoning="stop poking the same dead spot",
                               action="done", text="could not send")
        # The vision model's wrong answer, repeated verbatim, as in the run.
        llm.location = (0.60, 0.52)
        return AgentAction(observation="the send pill has no element",
                           reasoning="name it and have it located",
                           action="tap_at", text="Send Priority Like")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 2
    # The second locate was told what the first one's tap ruled out.
    assert llm.locate_misses_seen == [(), ((0.60, 0.52),)]
    # The repeated dead point was not tapped again.
    assert dev.taps == [(int(0.60 * X.W), int(0.52 * X.H))]
    assert "keeps placing" in (state.last_failure or "")


def test_a_tap_at_landing_on_a_listed_control_is_refused(cfg, mem):
    dev = fake.FakeDevice(cfg)
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="refused", reasoning="moving on",
                               action="done", text="taught the index")
        tried.append(True)
        # The centre of the "Wi-Fi" row, in fractions: (540, 580) on 1080x2340.
        return AgentAction(observation="saw it on the screenshot",
                           reasoning="lazy coordinate tap", action="tap_at",
                           x=0.5, y=0.248)

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert dev.taps == []
    assert "tap_at refused" in (state.last_failure or "")


def test_a_tap_at_on_bare_canvas_is_not_refused(cfg, mem):
    """The guard refuses listed controls, not coordinates: a point that hits
    nothing button-sized in the list goes straight through."""
    dev = fake.FakeDevice(cfg)
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="done probing", reasoning="finished",
                               action="done", text="probed")
        tried.append(True)
        # (540, 468): below the tabs, above the list -- nothing interactive.
        return AgentAction(observation="a canvas with no elements",
                           reasoning="press where the control is drawn",
                           action="tap_at", x=0.5, y=0.2)

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert dev.taps == [(540, 468)]


def test_a_locate_miss_is_a_failed_step_never_a_tapped_guess(cfg, mem):
    dev = fake.FakeDevice(cfg)
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="moving on", reasoning="giving up",
                               action="done", text="could not press it")
        tried.append(True)
        return AgentAction(observation="no such control in the list",
                           reasoning="name it anyway", action="tap_at",
                           text="the record button")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 1            # the locate was asked...
    assert dev.taps == []              # ...and its miss was not tapped
    assert "could not locate" in (state.last_failure or "")


def test_a_blind_deciders_guessed_point_is_replaced_by_the_locate(cfg, mem):
    """A blind decider never saw a frame, so the x/y it writes are guesses by
    construction: when the control is also named, the locate answers and its
    point -- not the guess -- is what goes out.

    runs/467405879436: a split pair (deepseek deciding, inkling reading the
    screenshots) and Hinge's "Send Priority Like" pill, which the tree folds
    into the composer scroller's label. The decider read the image-analysis
    block as "a screenshot is attached" and guessed (0.5, 0.55), (0.5, 0.62),
    (0.5, 0.45), (0.5, 0.58) -- around the pill every time, on it never --
    and the run aborted with the like unsent.
    """
    dev = fake.FakeDevice(cfg)
    expected = []

    def policy(screen, llm):
        if dev.state == "wifi":
            return AgentAction(observation="arrived", reasoning="goal reached",
                               action="done", text="reached wifi")
        wanted = next(e for e in screen.elements if e.best_text == "Wi-Fi")
        # What a vision model would answer for the named control on this frame.
        llm.location = (wanted.center[0] / screen.width,
                        wanted.center[1] / screen.height)
        expected.append(wanted.center)
        # Coordinates AND a name, as the blind decider in the run sent them.
        # (540, 468) is bare canvas, so the listed-element guard passes; the
        # located row centre is the point that must actually be tapped.
        return AgentAction(observation="the tree does not list it",
                           reasoning="tap where I believe the control is",
                           action="tap_at", x=0.5, y=0.2,
                           text="the Wi-Fi row icon")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.locates == 1 and llm.locates_seen == ["the Wi-Fi row icon"]
    cx, cy = expected[0]
    assert abs(dev.taps[0][0] - cx) <= 1 and abs(dev.taps[0][1] - cy) <= 1


def test_a_seeing_deciders_named_point_is_kept(cfg, mem):
    """The override is for blind deciders only: a model shown the frame taps
    the point it read off it, and no locate is paid for."""
    cfg.llm.vision_in_decider = True
    dev = fake.FakeDevice(cfg)
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="done probing", reasoning="finished",
                               action="done", text="probed")
        tried.append(True)
        # (540, 468): below the tabs, above the list -- nothing interactive.
        return AgentAction(observation="saw the canvas on the screenshot",
                           reasoning="press where the control is drawn",
                           action="tap_at", x=0.5, y=0.2,
                           text="the record button")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 0
    assert dev.taps == [(540, 468)]


# ---------------------------------------------------------------------------
# input_text aimed at a container: the vision locate finds the field
# ---------------------------------------------------------------------------

def test_input_text_aimed_at_a_scroller_is_grounded_by_the_locate(cfg, mem):
    """A composer the model aims at through the message-list scroller must not
    be typed into via the scroller's centre: nothing there takes focus, the
    keys go nowhere, and the tree keeps rendering the field's old text. The
    same vision locate that grounds a named tap_at places the real field, and
    the focus tap goes to that point."""
    dev = fake.FakeDevice(cfg, start="thread",
                          app={"thread": fake.FakeScreen(xml=X.chat_thread())})
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="the draft is in",
                               reasoning="send it", action="done",
                               text="message typed")
        tried.append(True)
        scroller = next(e for e in screen.elements if e.kind() == "Scroller")
        field = next(e for e in screen.elements if e.editable)
        # What the vision model answers for the composer on this frame.
        llm.location = (field.center[0] / screen.width,
                        field.center[1] / screen.height)
        return AgentAction(observation="the composer folds into the list",
                           reasoning="type into it", action="input_text",
                           target={"index": scroller.index}, text="hello there")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 1 and llm.locates_seen == ["the text input field"]
    # The located composer centre is what was tapped -- the scroller's centre
    # (540, 1100) never was.
    cx, cy = (450, 2020)     # the composer's centre in the fixture
    assert len(dev.taps) == 1
    assert abs(dev.taps[0][0] - cx) <= 1 and abs(dev.taps[0][1] - cy) <= 1
    assert any(a.startswith("input_text('hello there'") for a in dev.actions)


def test_an_input_text_container_locate_miss_is_never_tapped(cfg, mem):
    """A locate that cannot find the field is a failed step, never a tap at
    the container's centre -- that tap would focus nothing and type into the
    void."""
    dev = fake.FakeDevice(cfg, start="thread",
                          app={"thread": fake.FakeScreen(xml=X.chat_thread())})
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="no field could be placed",
                               reasoning="stop poking", action="done",
                               text="gave up")
        tried.append(True)
        scroller = next(e for e in screen.elements if e.kind() == "Scroller")
        # llm.location stays None: the field is not on screen.
        return AgentAction(observation="the composer folds into the list",
                           reasoning="type into it", action="input_text",
                           target={"index": scroller.index}, text="hello")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"
    assert llm.locates == 1
    assert dev.taps == []      # the container's centre was never tapped
    assert not any(a.startswith("input_text") for a in dev.actions)
    assert "could not locate" in (state.last_failure or "")


def test_an_input_text_that_changes_nothing_is_a_failed_step(cfg, mem):
    """Failed focus, caught after the fact: the tap that was meant to focus
    the field hit nothing editable, the keys went nowhere, and the dump is
    byte-identical. It used to grade a success -- the field's stale text was
    still there to read -- and the run walked on believing the text was in."""
    dev = fake.FakeDevice(cfg, start="thread",
                          app={"thread": fake.FakeScreen(xml=X.chat_thread())})
    tried = []

    def policy(screen, llm):
        if tried:
            return AgentAction(observation="typing failed", reasoning="stop",
                               action="done", text="could not type")
        tried.append(True)
        # A legitimate, small target -- so no locate is paid for -- whose tap
        # still focuses nothing: the fake phone's screen never moves.
        send = next(e for e in screen.elements if e.best_text == "Send")
        return AgentAction(observation="type the reply", reasoning="...",
                           action="input_text", target={"index": send.index},
                           text="on my way")

    outcome, state, llm = run(dev, mem, cfg, policy)

    assert outcome == "success"   # the run goes on to decide what comes next
    assert llm.locates == 0
    assert state.consecutive_failures >= 1
    assert "never took focus" in (state.last_failure or "")
    assert any("-> no_change" in line for line in state.history)


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def test_a_programmatic_assertion_ends_the_run_with_no_judge(cfg, mem):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    outcome, state = Agent(dev, mem, llm, cfg,
                           oracle=Oracle(text="Forget network")).run(GOAL)
    assert outcome == "success"
    assert llm.judges == 0


def test_shell_assertion(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.shell_replies["settings get global wifi_on"] = "1"
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    oracle = Oracle(shell="settings get global wifi_on", equals="1")
    outcome, _ = Agent(dev, mem, llm, cfg, oracle=oracle).run(GOAL)
    assert outcome == "success"


def test_premature_done_is_rejected_by_the_judge(cfg, mem):
    """Claiming success too early is a documented failure of every mobile agent."""
    dev = fake.FakeDevice(cfg)

    calls = {"n": 0}

    def policy(screen, llm):
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentAction(observation="home", reasoning="claiming early",
                               action="done", text="I think it's done")
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="really do it",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="there", reasoning="arrived",
                           action="done", text="done for real")

    llm = fake.FakeLLM(dev, policy, judge_result=False)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert llm.judges >= 1
    assert outcome in ("failed", "success")
    assert any("rejected" in line for line in state.history)


def test_assertion_overrules_a_premature_done(cfg, mem):
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        if dev.state == "wifi":
            return AgentAction(observation="arrived", reasoning="ok",
                               action="done", text="done")
        if llm.calls <= 1:
            return AgentAction(observation="home", reasoning="too early",
                               action="done", text="premature")
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="go",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="?", reasoning="back", action="press_key",
                           key="back")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg,
                           oracle=Oracle(text="Forget network")).run(GOAL)
    assert outcome == "success"
    assert dev.state == "wifi"
    assert llm.judges == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_credential_screen_hands_over_and_learns_nothing(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.app["home"] = fake.FakeScreen(xml=X.dump(
        X.N("android.widget.FrameLayout", (0, 0, X.W, X.H), rid="content", children=[
            X.N("android.widget.EditText", (48, 700, 1030, 820), rid="password",
                hint="Password", password=True, clickable=True, focusable=True),
            X.N("android.widget.Button", (48, 900, 520, 1020), text="Sign in",
                rid="signin", clickable=True)])))
    dev._xml = lambda: dev.app[dev.state].xml  # type: ignore[assignment]

    llm = fake.FakeLLM(dev, fake.tap_label("Sign in"))
    outcome, state = Agent(dev, mem, llm, cfg).run("log in")
    assert outcome == "needs_user"
    assert llm.calls == 0, "the model must never even see a credential screen"


def test_irreversible_action_is_refused_when_unattended(cfg, mem):
    dev = fake.FakeDevice(cfg, start="wifi")
    llm = fake.FakeLLM(dev, fake.tap_label("Forget network"))
    outcome, state = Agent(dev, mem, llm, cfg).run("forget this network")
    assert "Forget network" not in "".join(dev.actions)
    assert any("refused" in line for line in state.history)


def test_interstitial_is_dismissed_without_an_llm_call(cfg, mem):
    dev = fake.FakeDevice(cfg)
    dev.app["home"] = fake.FakeScreen(
        xml=X.settings_screen(extra_roots=[
            X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
                package="com.android.vending", rid="nag", children=[
                    X.N("android.widget.TextView", (100, 860, 980, 980),
                        package="com.android.vending", text="Rate this app!"),
                    X.N("android.widget.Button", (620, 1240, 980, 1380),
                        package="com.android.vending", text="Not now",
                        rid="dismiss", clickable=True)])]),
        taps={"Wi-Fi": "wifi"})
    dev._xml = lambda: dev.app[dev.state].xml  # type: ignore[assignment]

    llm = fake.FakeLLM(dev, fake.tap_label("Wi-Fi"))
    Agent(dev, mem, llm, cfg).run(GOAL)
    # The dismiss happened before any model call.
    assert dev.taps, "the nag should have been tapped"


def test_loop_breaker_stops_a_stuck_agent(cfg, mem):
    """A model that keeps choosing a dud action must not burn the whole budget."""
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 12

    def useless(screen, llm):
        el = next(e for e in screen.elements if e.best_text == "Data usage")
        return AgentAction(observation="home", reasoning="press it again",
                           action="tap", target={"index": el.index})

    llm = fake.FakeLLM(dev, useless)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "failed"
    assert state.step <= cfg.run.max_steps
    # The dud action was banned rather than retried forever.
    assert any(state.loops.bans_for(k) for k in state.loops.banned)


def test_step_budget_is_enforced(cfg, mem):
    dev = fake.FakeDevice(cfg)
    cfg.run.max_steps = 3
    cfg.run.max_consecutive_failures = 99

    def wander(screen, llm):
        return AgentAction(observation="x", reasoning="y", action="scroll",
                           direction="down")

    llm = fake.FakeLLM(dev, wander)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "failed" and state.step == 3


def test_dry_run_touches_nothing(cfg, mem):
    cfg.run.dry_run = True
    cfg.run.max_steps = 3
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.tap_label("Wi-Fi"))
    Agent(dev, mem, llm, cfg).run(GOAL)
    assert dev.taps == []
    assert dev.state == "home"


# ---------------------------------------------------------------------------
# Screenshot policy
# ---------------------------------------------------------------------------

def test_xml_first_no_screenshot_on_a_normal_screen(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    state = RunState(goal="g", run_id="r", intent_id="i")
    want, _ = needs_screenshot(state, screen, cfg)
    assert not want


def test_screenshot_for_a_degenerate_tree(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.webview_screen(), width=X.W, height=X.H))
    want, note = needs_screenshot(RunState(goal="g", run_id="r", intent_id="i"),
                                  screen, cfg)
    assert want and "WebView" in note


def test_screenshot_after_a_failure(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    screen = attach(parse(X.settings_screen(), width=X.W, height=X.H))
    state = RunState(goal="g", run_id="r", intent_id="i", consecutive_failures=1)
    want, _ = needs_screenshot(state, screen, cfg)
    assert want


def test_never_screenshot_wins(cfg):
    from adbagent.fingerprint import attach
    from adbagent.screen import parse

    cfg.run.never_screenshot = True
    screen = attach(parse(X.webview_screen(), width=X.W, height=X.H))
    want, _ = needs_screenshot(RunState(goal="g", run_id="r", intent_id="i"),
                               screen, cfg)
    assert not want


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def test_run_writes_artifacts(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)
    events = (tmp_path / "runs" / state.run_id / "events.jsonl")
    assert events.exists()
    lines = [l for l in events.read_text().splitlines() if l.strip()]
    kinds = {__import__("json").loads(l)["kind"] for l in lines}
    assert {"run_start", "decide", "verify", "run_end"} <= kinds


def test_run_mirrors_the_llm_stream(cfg, mem, tmp_path):
    # The web UI tails stream.jsonl to show the model thinking live. A fake
    # LLM never chunks, but the loop's own llm_start/llm_end still land there.
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)
    stream = tmp_path / "runs" / state.run_id / "stream.jsonl"
    assert stream.exists()
    records = [json.loads(l) for l in stream.read_text().splitlines() if l.strip()]
    kinds = [r["kind"] for r in records]
    assert "llm_start" in kinds and "llm_end" in kinds
    starts = [r for r in records if r["kind"] == "llm_start"]
    assert any(r["purpose"] == "decide" and r["step"] >= 1 for r in starts)
    ends = [r for r in records if r["kind"] == "llm_end"]
    assert all("elapsed" in r and "completion_tokens" in r for r in ends)
    # And the decision file is not polluted with stream records.
    events = (tmp_path / "runs" / state.run_id / "events.jsonl").read_text()
    assert '"llm_stream"' not in events


def test_stream_tap_whitelists_fields(cfg, tmp_path):
    from adbagent.agent import Recorder, _stream_tap

    class Verdict:  # a pydantic look-alike the tap must not try to serialise
        satisfied = True

    class Call:
        prompt_tokens = 100
        completion_tokens = 40
        reasoning_tokens = 30

    seen = []
    rec = Recorder(cfg, "run_stream_tap")
    tap = _stream_tap(rec, lambda kind, **kw: seen.append((kind, kw)))
    tap("llm_start", step=2, purpose="decide", model="m", screenshot=True,
        shot="step_002_decide_deadbeef.jpg", effort="high",
        hard_because="new screen")
    tap("llm_stream", stream_type="thinking", text="hmm ")
    tap("llm_stream", stream_type="content", text='{"action":')
    tap("llm_end", step=2, purpose="decide", elapsed=1.5, call=Call(),
        verdict=Verdict())
    tap("perceive", step=2, elapsed=0.1)  # not an llm event: no record
    rec.close()

    # Everything still flows through to the wrapped reporter.
    assert [k for k, _ in seen] == ["llm_start", "llm_stream", "llm_stream",
                                    "llm_end", "perceive"]
    records = [json.loads(l)
               for l in (tmp_path / "runs" / "run_stream_tap" / "stream.jsonl")
               .read_text().splitlines() if l.strip()]
    assert [r["kind"] for r in records] == ["llm_start", "llm_stream",
                                            "llm_stream", "llm_end"]
    assert records[0]["screenshot"] is True and records[0]["effort"] == "high"
    # The kept frame travels with the call, so the UI can show what it was shown.
    assert records[0]["shot"] == "step_002_decide_deadbeef.jpg"
    assert "hard_because" not in records[0]
    assert records[1]["stream_type"] == "thinking" and records[1]["text"] == "hmm "
    end = records[3]
    assert end["completion_tokens"] == 40 and end["reasoning_tokens"] == 30
    assert "verdict" not in end and "call" not in end


def test_recorder_dump_messages(cfg, tmp_path):
    from adbagent.agent import Recorder
    rec = Recorder(cfg, "run_test_123")
    msgs = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ABCDEF"}}]}
    ]
    path_str = rec.dump_messages(1, msgs, purpose="decide")
    from pathlib import Path
    import json
    p = Path(path_str)
    assert p.exists()
    assert p.name == "step_001_decide_messages.json"
    data = json.loads(p.read_text())
    assert data[0]["content"] == "sys prompt"
    assert "base64 image payload" in data[1]["content"][1]["image_url"]["url"]
    rec.close()


def test_recorder_keeps_a_submitted_screenshot(cfg, tmp_path):
    """The frames a model was shown land beside that step's prompt dump, one file
    per distinct frame however many calls saw it."""
    from adbagent.agent import Recorder
    rec = Recorder(cfg, "run_shots")
    d = tmp_path / "runs" / "run_shots"

    name = rec.screenshot(4, b"\xff\xd8jpeg-one", "analyze_image")
    assert name.startswith("step_004_analyze_image_") and name.endswith(".jpg")
    assert (d / name).read_bytes() == b"\xff\xd8jpeg-one"

    # Same bytes, same name: one frame shown twice is not two files.
    assert rec.screenshot(4, b"\xff\xd8jpeg-one", "analyze_image") == name
    # Different bytes on the same step do not overwrite it (the judge's frame is
    # not the loop's), and nothing to show returns nothing.
    other = rec.screenshot(4, b"\xff\xd8jpeg-two", "analyze_image")
    assert other != name and (d / other).is_file()
    assert rec.screenshot(4, b"", "analyze_image") == ""
    assert len(list(d.glob("*.jpg"))) == 2
    rec.close()


def _stream_records(tmp_path, run_id, kind, purpose):
    path = tmp_path / "runs" / run_id / "stream.jsonl"
    return [r for r in (json.loads(l) for l in path.read_text().splitlines()
                        if l.strip())
            if r["kind"] == kind and r.get("purpose") == purpose]


def test_the_decider_keeps_the_frame_when_it_is_the_one_looking(cfg, mem, tmp_path):
    """`vision_in_decider` sends the image to the deciding model itself, so that
    is the call the kept frame belongs to and the UI hangs it off."""
    cfg.run.always_screenshot = True
    cfg.llm.vision_in_decider = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    kept = sorted(p.name for p in (tmp_path / "runs" / state.run_id).glob("*.jpg"))
    assert kept and all("_decide_" in name for name in kept)
    starts = _stream_records(tmp_path, state.run_id, "llm_start", "decide")
    assert starts and all(r["shot"] in kept for r in starts)


def test_a_blind_decider_is_given_no_frame_of_its_own(cfg, mem, tmp_path):
    """Without it the decider reads prose, and the vision pass is the call that
    was shown the screenshot -- so the decide panel must not claim one."""
    cfg.run.always_screenshot = True          # vision_in_decider stays off
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert not list((tmp_path / "runs" / state.run_id).glob("*_decide_*.jpg"))
    starts = _stream_records(tmp_path, state.run_id, "llm_start", "decide")
    assert starts and all(r["shot"] == "" for r in starts)


def test_the_plan_accumulates_instead_of_being_overwritten():
    """A step the model stops mentioning is still in the plan.

    This is the property the old field did not have. `progress` was one string
    the model rewrote from scratch every turn and `progress_log` was assigned a
    fresh one-element list from it, so a sub-step dropped from the rewrite was
    gone -- the same failure `scratchpad` was rebuilt to make unrepresentable,
    left in place for the one field nothing was checking.
    """
    from adbagent.agent import RunState
    from adbagent.actions import AgentAction

    state = RunState(goal="multi step test", run_id="r1", intent_id="i1")

    declare = AgentAction(action="wait", observation="s1", reasoning="r1",
                          progress=[{"id": "1", "text": "open the app"},
                                    {"id": "2", "text": "find the contact"},
                                    {"id": "3", "text": "send the message"}])
    # A turn that mentions one step and says nothing about the other two.
    finish_one = AgentAction(action="wait", observation="s2", reasoning="r2",
                             progress=[{"id": "1", "status": "done"}])

    state.plan.update(declare.progress, 1)
    state.plan.update(finish_one.progress, 2)

    assert state.plan.plain() == ("[x] open the app\n"
                                 "[ ] find the contact\n"
                                 "[ ] send the message")
    assert state.plan.outstanding() == ["find the contact", "send the message"]


def test_list_apps_in_agent_loop(cfg, mem):
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="home screen", reasoning="find whatsapp package",
                               action="list_apps", text="whatsapp")
        elif step_count["n"] == 2:
            return AgentAction(observation="home screen", reasoning="open whatsapp",
                               action="open_app", text="com.whatsapp")
        return AgentAction(observation="whatsapp opened", reasoning="done",
                           action="done", text="opened whatsapp")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("open whatsapp")
    assert outcome == "success"
    assert "list_apps('whatsapp')" in dev.actions
    assert "open_app(com.whatsapp)" in dev.actions


def test_scroll_swipe_no_change_does_not_ban_action(cfg, mem):
    """When a scroll or swipe action encounters no_change, it must not be added to the ban list."""
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="feed", reasoning="scroll down",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="done",
                           action="done", text="finished")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll feed")
    assert outcome == "success"
    # Even if scroll down was graded no_change, scroll/down must not be in banned
    assert "scroll/down" not in state.loops.bans_for(dev.observe().skeleton_id)


def test_a_scroll_that_revealed_nothing_is_refused_the_second_time(cfg, mem):
    """The stop is enforced in code, not left as advice the model can ignore.

    The first scroll down grades no_change on the unchanged screen. When the
    model proposes the identical gesture again on the identical frame, the
    harness refuses it before it reaches the device -- and the detection that
    feeds the refusal is hierarchy comparison, not an LLM reading an image.
    """
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] <= 2:
            return AgentAction(observation="feed", reasoning="scroll down",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="done",
                           action="done", text="finished")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll the feed")
    assert outcome == "success"
    # The device saw exactly one scroll; the second was refused in code.
    assert dev.actions.count("scroll(down)") == 1
    assert "refused" in state.last_failure


def test_a_dead_scroll_rearms_after_the_screen_changes(cfg, mem):
    """The refusal is keyed on the exact frame, not the screen's shape.

    A feed that loads more content is the same skeleton with a different
    exact_id, and the gesture must be legal there again -- otherwise one
    premature end-of-list verdict would lock the direction for the whole run.
    """
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        n = step_count["n"]
        if n <= 2:
            return AgentAction(observation="feed", reasoning="scroll down",
                               action="scroll", direction="down")
        if n == 3:
            # Stand in for the feed loading more content: a tap that toggles
            # a checkbox changes the tree, and therefore the exact_id.
            el = next(e for e in screen.elements if e.checkable)
            return AgentAction(observation="feed", reasoning="tap the toggle",
                               action="tap", target={"index": el.index})
        if n == 4:
            return AgentAction(observation="feed", reasoning="scroll down again",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="done",
                           action="done", text="finished")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll the feed")
    assert outcome == "success"
    # Both post-change scrolls reached the device: step 2's was refused, but
    # the toggle changed the frame, so step 4's ran and was graded on its own
    # merits -- the first scroll executed, then the re-armed one.
    assert dev.actions.count("scroll(down)") == 2




# ---------------------------------------------------------------------------
# Screenshots are captured once per screen
# ---------------------------------------------------------------------------

def test_a_screen_is_photographed_once_however_many_times_it_is_needed(cfg, mem):
    """Verification's screenshot IS the next turn's screenshot.

    `observe(settle=True)` has already waited for the tree to stop moving, and
    every path that touches the device between verifying one step and deciding
    the next (dismissing an interstitial, breaking a loop with `back`) drops the
    screen and re-observes. So a second capture buys nothing and costs a device
    round trip -- and leaves the pager comparing a `dhash` taken from different
    pixels than the ones the model was shown.
    """
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    step_count = {"n": 0}

    def policy(screen, llm):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return AgentAction(observation="feed", reasoning="scroll",
                               action="scroll", direction="down")
        return AgentAction(observation="feed", reasoning="finished",
                           action="done", text="done")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run("scroll the feed")
    assert outcome == "success"
    # Step 1 decides (1) and verifies (2). Step 2 reuses the verify screenshot,
    # and so does the completion judge. Four captures would mean the fix regressed.
    assert dev.screenshots == 2


def test_the_screenshot_the_model_saw_is_the_one_the_dhash_came_from(cfg, mem):
    from adbagent.agent import Agent as _Agent

    dev = fake.FakeDevice(cfg)
    agent = _Agent(dev, mem, None, cfg)
    screen = dev.observe()
    first = agent._ensure_screenshot(screen)
    assert dev.screenshots == 1
    assert agent._ensure_screenshot(screen) is first
    assert dev.screenshots == 1


# ---------------------------------------------------------------------------
# Token instrumentation
# ---------------------------------------------------------------------------

def _events(tmp_path, run_id):
    path = tmp_path / "runs" / run_id / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def test_every_decision_records_what_it_cost(cfg, mem, tmp_path):
    """Latency work needs per-step tokens in the artifact, not just a total."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides
    for event in decides:
        metrics = event["llm"]
        assert metrics["n_calls"] >= 1
        assert metrics["prompt_tokens"] == 1000     # from FakeLLM's ledger
        assert metrics["completion_tokens"] == 50
        assert "wall_s" in event


def test_a_vision_turn_is_attributed_both_of_its_calls(cfg, mem, tmp_path):
    """A screenshot turn is an image analysis *then* a decision. Charging the
    step for only the last one is how a 2-call turn reads as a 1-call turn."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides[0]["llm"]["n_calls"] == 2
    purposes = [c["purpose"] for c in decides[0]["llm"]["calls"]]
    assert purposes == ["analyze_image", "decide"]
    # 500 + 1000 prompt, 100 + 50 completion.
    assert decides[0]["llm"]["prompt_tokens"] == 1500
    assert decides[0]["llm"]["completion_tokens"] == 150


def test_a_decision_records_the_element_its_ordinal_resolved_to(cfg, mem, tmp_path):
    """`tap #3` names a position in a list that is not in the artifact.

    So the trace -- and the live feed the web UI builds off it -- could say what
    the run did but not what it did it *to*, and reading a step back meant
    matching an ordinal against the screenshot by eye.
    """
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    taps = [e for e in _events(tmp_path, state.run_id)
            if e["kind"] == "decide" and e["action"]["action"] == "tap"]
    assert taps
    target = taps[0]["target_element"]
    assert target["index"] == taps[0]["action"]["target"]["index"]
    assert target["text"] == "Wi-Fi"
    assert target["kind"] and target["center"]


def test_an_action_without_a_target_records_no_element(cfg, mem, tmp_path):
    """The field is absent rather than null, so a reader can tell "nothing to
    resolve" from "resolved to nothing" -- which for a tap is the reason the
    step is about to fail."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    untargeted = [e for e in _events(tmp_path, state.run_id)
                  if e["kind"] == "decide" and not e["action"].get("target")]
    assert untargeted, "the run never took an action without a target"
    assert all("target_element" not in e for e in untargeted)


def test_a_target_that_matches_nothing_is_recorded_as_such(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        if llm.calls == 1:
            return AgentAction(observation="settings", reasoning="tap a ghost",
                               action="tap", target={"index": 999})
        return AgentAction(observation="settings", reasoning="give up",
                           action="fail", text="no such element")

    _, state = Agent(dev, mem, fake.FakeLLM(dev, policy), cfg).run(GOAL)

    first = next(e for e in _events(tmp_path, state.run_id)
                 if e["kind"] == "decide")
    assert first["target_element"] is None


def test_the_judge_is_costed_separately_from_the_step_that_proposed_done(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    judges = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "judge"]
    assert len(judges) == 1
    assert judges[0]["step"] == state.step
    assert "llm" in judges[0] and "wall_s" in judges[0]


def test_the_run_total_omits_the_per_call_breakdown(cfg, mem, tmp_path):
    """Each call already has its own event; repeating them all in `run_end`
    would double the size of the artifact for nothing."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    end = next(e for e in _events(tmp_path, state.run_id) if e["kind"] == "run_end")
    assert "calls" not in end["llm"]
    assert end["llm"]["n_calls"] >= 2
    assert end["llm"]["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Repeated history, folded in the loop
# ---------------------------------------------------------------------------

def test_a_repeated_action_folds_in_the_history_the_loop_keeps(cfg, mem):
    """`prompts.history_only_block` renders whatever the loop appended, so the
    fold has to happen on the way in -- see `actions.append_history`."""
    cfg.run.max_steps = 8

    def policy(screen, llm):
        if llm.calls > 5:
            return AgentAction(observation="enough", reasoning="stop",
                               action="done", text="done waiting")
        return AgentAction(observation=f"waiting, turn {llm.calls}",
                           reasoning="let the screen settle",
                           action="wait", duration=0.05)

    _, state, _ = run(fake.FakeDevice(cfg), mem, cfg, policy)

    waits = [line for line in state.history if "wait" in line]
    assert len(waits) == 1, f"identical waits were not folded: {waits}"
    assert "[x5]" in waits[0]
    # The readings are what a fold must never discard.
    assert "waiting, turn 1" in waits[0] and "waiting, turn 5" in waits[0]


def test_the_situational_advice_only_shows_up_when_it_applies(cfg, mem):
    """The scrolling and app-switching blocks are a big chunk of what the system
    prompt used to be, and irrelevant on a turn like this one."""
    dev = fake.FakeDevice(cfg)
    _, _, llm = run(dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert llm.notes, "no NOTE block was built at all"
    assert not any("SWITCHING APPS" in note for note in llm.notes)


def test_a_screenshot_turn_is_timed_for_both_of_its_calls(cfg, mem, tmp_path):
    """`report`'s latency/step reads `wall_s`, so a step whose cost covers two
    calls must have a clock that covers them too."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    assert decides[0]["llm"]["n_calls"] == 2
    assert decides[0]["wall_s"] > 0


# ---------------------------------------------------------------------------
# Dead ends survive the run that found them
# ---------------------------------------------------------------------------

def test_a_dud_control_found_in_one_run_is_not_retried_in_the_next(cfg, mem):
    """The only knowledge in this system that outlives the process.

    Failures were being written to SQLite on every step and read back by nothing,
    so every run rediscovered the same dud control on the same screen.
    """
    from adbagent.memory import intent_key

    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    dud = AgentAction(observation="settings", reasoning="try it",
                      action="tap", target={"index": 4})
    mem.record_dead_end(screen, intent_key(GOAL), dud.signature(),
                        "nothing changed")

    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    Agent(dev, mem, llm, cfg).run(GOAL)

    note = "\n".join(llm.notes)
    assert "KNOWN DEAD ENDS here from earlier runs" in note
    assert dud.signature() in note
    assert "nothing changed" in note


def test_a_dead_end_from_a_different_goal_is_not_mentioned(cfg, mem):
    dev = fake.FakeDevice(cfg)
    screen = dev.observe()
    dud = AgentAction(observation="settings", reasoning="try it",
                      action="tap", target={"index": 4})
    mem.record_dead_end(screen, "some-other-intent", dud.signature(), "no change")

    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    Agent(dev, mem, llm, cfg).run(GOAL)
    assert "KNOWN DEAD ENDS" not in "\n".join(llm.notes)


def test_this_runs_bans_are_not_repeated_as_remembered_ones(cfg, mem):
    """Both lists come from the same failure once it has been recorded, and
    saying it twice in one prompt reads as two separate findings."""
    dev = fake.FakeDevice(cfg)
    calls = {"n": 0}

    def policy(screen, llm):
        calls["n"] += 1
        if calls["n"] <= 2:
            # #7 is "VPN", a row the fake never navigates from: no change.
            return AgentAction(observation="settings", reasoning="try VPN",
                               action="tap", target={"index": 7})
        return AgentAction(observation="settings", reasoning="done",
                           action="done", text="finished")

    llm = fake.FakeLLM(dev, policy)
    Agent(dev, mem, llm, cfg).run(GOAL)
    for note in llm.notes:
        signature = "tap/#7"
        if "BANNED ACTIONS" in note and signature in note:
            after = note.split("BANNED ACTIONS", 1)[1]
            assert after.count(signature) == 1, note


# ---------------------------------------------------------------------------
# How hard to think about this turn
# ---------------------------------------------------------------------------
#
# Reasoning tokens are the run's wall clock, so the default is shallow. The whole
# safety of that rests on the loop noticing the turns that are not routine, from
# evidence it already has.

def deep(cfg):
    cfg.llm.reasoning_effort = "none"
    cfg.llm.reasoning_effort_hard = "high"
    return cfg


def effort(cfg, **kw):
    from adbagent.agent import needs_reasoning

    state = RunState(goal="g", run_id="r", intent_id="i")
    # Mid-run unless the case says otherwise: the opening turn is hard by its own
    # clause, and a default of step 0 would make every case here look hard.
    state.step = 5
    for key, value in kw.items():
        if key not in ("visit", "blocked", "hint"):
            setattr(state, key, value)
    return needs_reasoning(state, cfg, visit=kw.get("visit", 1),
                           blocked=kw.get("blocked", False),
                           hint=kw.get("hint", ""))


def test_the_feature_is_off_until_it_is_configured(cfg):
    """An unset depth must leave every request exactly as it was."""
    assert effort(cfg) == ("", "")
    assert effort(cfg, consecutive_failures=3) == ("", "")


def test_a_routine_turn_thinks_shallowly(cfg):
    assert effort(deep(cfg)) == ("none", "")


def test_a_failure_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), consecutive_failures=1)
    assert chosen == "high"
    assert "did not work" in why


def test_a_new_screen_does_not_buy_deeper_thinking(cfg):
    """Novelty is the default state of exploring, not evidence of difficulty.

    This clause used to fire on 57 of 103 decide calls across ``runs/`` -- 78% of
    every escalation, ~430s, 17.5% of all wall clock -- and it was pointed the
    wrong way: on a run walking a set, the novel screens are the items ("read it,
    then go back") while the screen the run keeps *returning* to is the index,
    which is the only screen it can stop from.
    """
    assert effort(deep(cfg), visit=0) == ("none", "")


def test_a_screen_the_run_keeps_returning_to_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), visit=2)
    assert chosen == "high"
    assert "3 times" in why


def test_the_opening_turn_is_hard_on_its_own_account(cfg):
    """One turn per run, not fifty-seven."""
    chosen, why = effort(deep(cfg), step=1)
    assert chosen == "high"
    assert "first step" in why


def test_saying_it_was_unsure_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), want_screenshot=True)
    assert chosen == "high"
    assert "unsure" in why


def test_a_known_dead_end_here_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), blocked=True)
    assert chosen == "high"
    assert "lead nowhere" in why


def test_a_loop_warning_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), hint="you have been here before")
    assert chosen == "high"
    assert "loop detector" in why


def test_a_rejected_action_buys_deeper_thinking(cfg):
    chosen, why = effort(deep(cfg), last_failure="your 'done' was rejected")
    assert chosen == "high"
    assert "rejected" in why


def test_the_first_step_of_a_run_is_always_a_hard_one(cfg, mem, tmp_path):
    """visit == 0 on the opening screen, which is the turn that picks the whole
    approach -- exactly the wrong one to skimp on."""
    deep(cfg)
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    first = next(e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide")
    assert first["effort"] == "high"
    assert first["hard_because"]


def test_a_settled_walk_runs_shallow(cfg, mem, tmp_path):
    """Revisiting a screen that has caused no trouble costs the floor.

    Every step here succeeds, so the only escalation is the opening turn -- the
    one that picks the approach. Walking into a screen for the first time used to
    escalate too, which is how 73 of 103 decide calls across ``runs/`` ended up at
    `high` on a ladder documented as "shallow by default".
    """
    deep(cfg)
    dev = fake.FakeDevice(cfg)
    calls = {"n": 0}

    def policy(screen, llm):
        calls["n"] += 1
        if calls["n"] == 1:                       # home (new) -> wifi
            el = next(e for e in screen.elements if e.best_text == "Wi-Fi")
            return AgentAction(observation="settings", reasoning="open Wi-Fi",
                               action="tap", target={"index": el.index})
        if calls["n"] == 2:                       # wifi (new) -> back to home
            return AgentAction(observation="wifi", reasoning="go back",
                               action="press_key", key="back")
        return AgentAction(observation="settings", reasoning="finished",
                           action="done", text="done")

    llm = fake.FakeLLM(dev, policy)
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    decides = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "decide"]
    shown = [(e["step"], e["effort"], e["hard_because"]) for e in decides]
    assert decides[0]["effort"] == "high", shown    # the turn that picks the approach
    # Everything after it is routine, including the screen it had never seen.
    assert decides[1]["effort"] == "none", shown
    assert decides[1]["hard_because"] == "", shown
    assert decides[2]["effort"] == "none", shown
    assert decides[2]["hard_because"] == "", shown


def test_the_chosen_depth_is_recorded_for_every_decision(cfg, mem, tmp_path):
    """Without this in the artifact there is no way to tell whether a run that
    got slower got deeper, or a run that got cheaper got shallower."""
    deep(cfg)
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    _, state = Agent(dev, mem, llm, cfg).run(GOAL)

    for event in _events(tmp_path, state.run_id):
        if event["kind"] == "decide":
            assert event["effort"] in ("none", "high")
            assert "hard_because" in event


# ---------------------------------------------------------------------------
# Device recovery
# ---------------------------------------------------------------------------
#
# The perceive-path handler sets `screen = None`, so the next turn re-observes.
# The three handlers that fire *after* the action has gone out did not, and a
# recovered device was then driven from the frame that predated the action --
# tapping element centres that had already moved.

class _FlakyDevice(fake.FakeDevice):
    """Raises `DeviceTimeout` once, from whichever observe the test names."""

    def __init__(self, *a, fail_settled: bool = True, **kw):
        super().__init__(*a, **kw)
        self.fail_settled = fail_settled
        self.blew_up = False

    def observe(self, settle: bool = False):
        if settle == self.fail_settled and not self.blew_up:
            self.blew_up = True
            raise DeviceTimeout("dump_hierarchy exceeded 60s")
        return super().observe(settle=settle)


def _rendered_per_turn(dev, mem, cfg, policy):
    """(device state, rendered activity) for every decide call of a run."""
    seen = []

    class _Spy(fake.FakeLLM):
        def decide(self, *, rendered, **kw):
            seen.append((self.dev.state, rendered.splitlines()[0]))
            return super().decide(rendered=rendered, **kw)

    llm = _Spy(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    return outcome, seen


def test_a_recovered_device_is_re_observed_before_the_next_decision(cfg, mem):
    dev = _FlakyDevice(cfg)          # the post-action settle blows up
    outcome, seen = _rendered_per_turn(
        dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert dev.blew_up               # the recovery path really was taken
    assert outcome == "success"
    for state_name, header in seen:
        # The phone moved to `wifi`; the model must not still be shown `home`.
        assert f".{state_name.title()}Activity" in header, seen


def test_recovery_during_the_act_call_also_re_observes(cfg, mem):
    """`execute` can raise after the gesture reached the phone."""
    dev = fake.FakeDevice(cfg)
    real_tap = dev.tap
    fired = {"n": 0}

    def tap(x, y):
        real_tap(x, y)               # the tap lands...
        fired["n"] += 1
        if fired["n"] == 1:
            raise DeviceTimeout("click exceeded 60s")   # ...then the call dies

    dev.tap = tap
    outcome, seen = _rendered_per_turn(
        dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert fired["n"] >= 1
    for state_name, header in seen:
        assert f".{state_name.title()}Activity" in header, seen


# ---------------------------------------------------------------------------
# Rejected completions
# ---------------------------------------------------------------------------

def test_repeated_rejected_dones_give_up_instead_of_burning_the_budget(cfg, mem):
    """Each rejection costs a screenshot, a vision pass and a high-effort judge
    call. They were not counted as failures, so `max_consecutive_failures` never
    fired and the run went all the way to the step budget."""
    cfg.run.max_consecutive_failures = 3
    dev = fake.FakeDevice(cfg)

    def policy(screen, llm):
        return AgentAction(observation="home", reasoning="claiming early",
                           action="done", text="I think it's done")

    llm = fake.FakeLLM(dev, policy, judge_result=False)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "failed"
    assert state.step == cfg.run.max_consecutive_failures < cfg.run.max_steps
    assert llm.judges == cfg.run.max_consecutive_failures


def test_progress_between_rejections_resets_the_give_up_counter(cfg, mem):
    """A run that keeps working after a premature `done` is not a stuck one."""
    cfg.run.max_consecutive_failures = 2
    dev = fake.FakeDevice(cfg)
    calls = {"n": 0}

    def policy(screen, llm):
        calls["n"] += 1
        if calls["n"] in (1, 3):     # premature, real work, premature, ...
            return AgentAction(observation="home", reasoning="too early",
                               action="done", text="premature")
        for el in screen.elements:
            if el.best_text == "Wi-Fi" and el.interactive:
                return AgentAction(observation="home", reasoning="go",
                                   action="tap", target={"index": el.index})
        return AgentAction(observation="?", reasoning="back",
                           action="press_key", key="back")

    llm = fake.FakeLLM(dev, policy, judge_result=False)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert state.step > cfg.run.max_consecutive_failures


# ---------------------------------------------------------------------------
# Dismissing interstitials
# ---------------------------------------------------------------------------

def _nag_screen(label: str, package: str = X.PKG) -> str:
    """A settings screen with one dismiss-shaped control over it."""
    return X.settings_screen(extra_roots=[
        X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
            package=package, rid="nag", children=[
                X.N("android.widget.Button", (620, 1240, 980, 1380),
                    package=package, text=label, rid="dismiss", clickable=True)])])


def test_a_dismiss_that_never_works_hands_the_screen_to_the_model(cfg, mem):
    """A "Skip" that is disabled, or sits on a WebView where the tap lands
    nowhere, used to be pressed until the step budget ran out with the model
    never consulted once."""
    dev = fake.FakeDevice(cfg)
    dev.app["home"] = fake.FakeScreen(xml=_nag_screen("Skip"), taps={})
    dev._xml = lambda: dev.app[dev.state].xml  # type: ignore[assignment]

    def policy(screen, llm):
        return AgentAction(observation="the nag is part of this screen",
                           reasoning="nothing else to do", action="done",
                           text="fin")

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    from adbagent.agent import MAX_DISMISS_TRIES
    assert len(dev.taps) == MAX_DISMISS_TRIES
    assert llm.calls >= 1                       # the model did get a turn
    assert state.step < cfg.run.max_steps
    assert any("harness dismissed 'Skip'" in line for line in state.history)


def test_a_dismissal_that_works_is_not_held_against_the_next_one(cfg, mem):
    """Two nags in a row are two dismissals, not one dismissal and a refusal.

    The memo is keyed on `exact_id` for this: `skeleton_id` is content-free, so
    a second card with different text hashes the same as the first.
    """
    dev = fake.FakeDevice(cfg)
    nags = ["Not now", "Got it"]
    dev.app["home"] = fake.FakeScreen(xml=_nag_screen(nags[0]), taps={})

    def tap(x, y):
        dev.taps.append((x, y))
        if nags:
            nags.pop(0)
            dev.app["home"] = fake.FakeScreen(
                xml=_nag_screen(nags[0]) if nags else X.settings_screen(), taps={})

    dev._xml = lambda: dev.app["home"].xml    # type: ignore[assignment]
    dev.tap = tap                             # type: ignore[assignment]

    llm = fake.FakeLLM(dev, lambda s, l: AgentAction(
        observation="clear", reasoning="done", action="done", text="fin"))
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert len(dev.taps) == 2, "both nags should have been dismissed"


# ---------------------------------------------------------------------------
# The stall ladder
# ---------------------------------------------------------------------------
#
# `consecutive_failures` counts actions that failed. The failure that actually
# loses runs is every action succeeding while the run goes nowhere, and none of
# the guards that existed before this could see it -- in `runs/2521862d7a23` a
# two-cycle ran for twenty steps, every step graded `success`, until the person
# watching pressed Ctrl-C.


def _two_cycle_policy(dev):
    """tap into the detail screen, press back, forever -- and it all works.

    The point of the fixture is that nothing here fails. Every tap opens a
    screen, every back returns, `verify` grades all of it `success`, and the
    run learns nothing after step 2.
    """
    def policy(screen, llm):
        if dev.state == "home":
            el = next(e for e in screen.elements
                      if e.best_text == "Wi-Fi" and e.interactive)
            return AgentAction(observation="the settings list",
                               reasoning="open Wi-Fi", action="tap",
                               target={"index": el.index})
        return AgentAction(observation="the Wi-Fi screen",
                           reasoning="go back for the next one",
                           action="press_key", key="back")
    return policy


def test_a_two_cycle_where_every_step_succeeds_is_stopped(cfg, mem, tmp_path):
    """The regression test for `runs/2521862d7a23`."""
    cfg.run.max_steps = 60
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "failed"
    # Well inside the budget: the old loop only stopped at `max_steps`.
    assert state.step < 20, f"took {state.step} steps to notice"

    kinds = [e["kind"] for e in _events(tmp_path, state.run_id)]
    assert "stall_block" in kinds, "the repeat was never refused"
    assert "stalled_out" in kinds, "the run never gave up"
    assert llm.replans == 1, "one stall episode should buy exactly one replan"


def test_the_stall_is_put_to_the_model_before_anything_is_refused(cfg, mem):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    Agent(dev, mem, llm, cfg).run(GOAL)

    nudges = [n for n in llm.notes if "NO PROGRESS FOR" in n]
    assert nudges, "the model was never told it had stopped getting anywhere"
    # The first one arrives before the harness starts refusing things.
    assert "REFUSING" not in nudges[0]
    assert any("REFUSING" in n for n in nudges), "the refusal was never named"


def test_the_replan_is_shown_what_has_been_tried(cfg, mem):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.replans_seen, "no replan ran"
    tried = dict(llm.replans_seen[0])
    assert any(sig.startswith("tap/") for sig in tried), tried
    assert max(tried.values()) >= 2, "it was not shown the repetition"


def test_the_agreed_strategy_is_carried_into_later_turns(cfg, mem):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    llm.replan_strategy = "open Bluetooth from the list instead"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert any("open Bluetooth from the list instead" in n for n in llm.notes)


def test_a_replan_that_abandons_ends_the_run_there(cfg, mem, tmp_path):
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    llm.replan_abandon = True
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "failed"
    replans = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "replan"]
    assert replans and replans[-1]["abandon"] is True
    # It stopped on the replan rather than running on to the give-up tier.
    assert "stalled_out" not in [e["kind"] for e in _events(tmp_path, state.run_id)]


def test_rewording_the_progress_note_does_not_buy_the_run_more_time(cfg, mem, tmp_path):
    """The model must not hold the reset switch of the guard that bounds it.

    `_loop` used to call `note_progress` whenever `action.progress` differed from
    the previous turn's, compared as a raw string. Measured across ``runs/``: the
    field was present on 76 of 103 turns and its text changed on 72 of them, so
    70% of all steps reset the ladder on model prose alone -- and `stalled` never
    once exceeded 3 against block=5, replan=8, give_up=14. In
    ``runs/c1d57cc79d9c`` steps 18-22 the "Done:" clause is byte-identical five
    turns running while only the "Next:" clause is reworded.
    """
    cfg.run.max_steps = 60
    dev = fake.FakeDevice(cfg)
    inner = _two_cycle_policy(dev)
    reworded = iter(f"Done: nothing. Next: attempt number {n}" for n in range(1, 500))

    def policy(screen, llm):
        action = inner(screen, llm)
        # Same two-cycle as ever, plus a fresh progress sentence every turn.
        # Re-validated rather than `model_copy`d, so the prose goes through the
        # same validator a model's reply would -- which is the thing under test.
        return AgentAction.model_validate(
            {**action.model_dump(), "progress": next(reworded)})

    outcome, state = Agent(dev, mem, llm := fake.FakeLLM(dev, policy), cfg).run(GOAL)

    assert outcome == "failed"
    kinds = [e["kind"] for e in _events(tmp_path, state.run_id)]
    assert "stalled_out" in kinds, "rewording the progress note kept the run alive"
    assert state.step < 20, f"took {state.step} steps to notice"
    # The note itself is still working memory and still reaches the model. It
    # lands in the one reserved entry prose goes to, which is overwritten rather
    # than accumulated and can never be marked done -- so 15 rewrites are one
    # line, and none of them is a step the ladder could be paid for.
    assert "attempt number" in state.plan.plain()
    assert len(state.plan) == 0 and state.plan.credited == set()


def test_finishing_a_plan_step_resets_the_stall_ladder(cfg, mem, tmp_path):
    """The other half of the change: a real milestone must count as progress.

    Every signal `note_progress` listens to is about the device -- a screen not
    seen before, content that moved, a setting that flipped -- or about data
    collection. None is about the goal's own structure, so a run working
    steadily through a plan in an app it has already mapped reads as a stall and
    is stopped by the same ladder that exists to catch a two-cycle going nowhere.

    So this is the two-cycle from the test above, unchanged and still learning
    nothing about the device, except that it finishes one plan step every third
    turn. It must outlive `stall_give_up_at`, which the bare two-cycle does not.
    """
    cfg.run.max_steps = 40
    dev = fake.FakeDevice(cfg)
    inner = _two_cycle_policy(dev)
    turn = {"n": 0}

    def policy(screen, llm):
        turn["n"] += 1
        action = inner(screen, llm)
        if turn["n"] == 1:
            progress = [{"id": str(n), "text": f"step {n}"} for n in range(1, 9)]
        elif turn["n"] % 3 == 0:
            progress = [{"id": str(turn["n"] // 3), "status": "done"}]
        else:
            return action
        return AgentAction.model_validate(
            {**action.model_dump(), "progress": progress})

    outcome, state = Agent(dev, mem, llm := fake.FakeLLM(dev, policy), cfg).run(GOAL)

    kinds = [e["kind"] for e in _events(tmp_path, state.run_id)]
    assert "stalled_out" not in kinds, "a run finishing plan steps was stopped"
    assert state.step > cfg.run.stall_give_up_at, \
        f"gave up after {state.step} steps despite finishing plan steps"
    assert state.plan.done_count >= 4
    # Credited for what it finished, and only for that.
    assert state.plan.credited == {s.id for s in state.plan.steps
                                   if s.status == "done"}
    # The turn that declared all eight is the one a model would use to buy time,
    # and it bought none: the first plan event completes nothing.
    plans = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "plan"]
    assert plans[0]["total"] == 8 and plans[0]["completed"] == []
    assert sum(len(e["completed"]) for e in plans) == state.plan.done_count


def test_one_stall_episode_can_buy_more_than_one_replan(cfg, mem, tmp_path):
    """`replanned_at` counts steps, not stall depth.

    Holding the stall value meant that after a first fire at stalled=8 the next
    needed stalled>=16 -- and `stall_give_up_at`=14 ends the run first, so one
    replan per run was the structural maximum.
    """
    cfg.run.max_steps = 80
    cfg.run.stall_replan_at = 3
    cfg.run.stall_give_up_at = 40
    cfg.run.stall_block_at = 0          # let the cycle run rather than refusing it
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.replans >= 2, (
        f"a stall lasting {cfg.run.stall_give_up_at} steps bought only "
        f"{llm.replans} replan(s)")


def test_the_agreed_strategy_survives_the_run_getting_somewhere(cfg, mem):
    """A plan outlives the turn that asked for it.

    `note_progress` used to clear `strategy`, and `strategy` reached the prompt
    only through `stall_note`, which renders only while `stalled >=
    stall_nudge_at`. Between them the new approach was deleted by the first step
    it succeeded on -- so it was visible exactly while it was not working.
    """
    from adbagent.agent import RunState

    state = RunState(goal=GOAL, run_id="r", intent_id="i")
    state.strategy = "open Bluetooth from the list instead"
    state.steps_since_progress = 7

    state.note_progress("it reached a screen it had not seen before")

    assert state.steps_since_progress == 0
    assert state.strategy == "open Bluetooth from the list instead"


def test_the_strategy_is_its_own_block_and_not_part_of_the_stall_note(cfg, mem):
    """The two have different lifetimes, so they cannot share a render.

    The stall note describes a condition that is true right now; a strategy is a
    decision that outlives the condition that bought it. Rendered from inside
    `stall_note` it appeared only while `stalled >= stall_nudge_at`.
    """
    from adbagent import prompts

    note = prompts.stall_note(9, tried=[("tap/#7", 4)], refused=["tap/#7"])
    assert "open Bluetooth" not in note
    assert prompts.strategy_block("") == ""

    carried = prompts.strategy_block("open Bluetooth from the list instead")
    assert "open Bluetooth from the list instead" in carried
    assert "NO PROGRESS" not in carried

    # And the loop actually renders it.
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _two_cycle_policy(dev))
    llm.replan_strategy = "open Bluetooth from the list instead"
    Agent(dev, mem, llm, cfg).run(GOAL)
    assert any("AGREED NEW APPROACH" in n for n in llm.notes), \
        "the strategy never reached the prompt"


def test_every_turn_is_told_where_it_stands_against_its_budget(cfg, mem):
    """Nothing used to say, on any of the 105 decide prompts in ``runs/``.

    SYSTEM tells the model not to search "indefinitely" while giving it no
    measurement of how long it has been going. Over nine runs `fail` was chosen
    zero times and `ask_user` zero times, and the run that never terminated had
    512 steps of budget left when a human killed it.
    """
    cfg.run.max_steps = 25
    dev = fake.FakeDevice(cfg)
    _, _, llm = run(dev, mem, cfg, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    assert llm.budgets, "no turn was shown a budget"
    assert all("BUDGET:" in b for b in llm.budgets), llm.budgets
    assert "step 1 of 25" in llm.budgets[0]
    assert "step 2 of 25" in llm.budgets[1]


def test_the_budget_line_says_nothing_before_the_first_step():
    from adbagent import prompts

    assert prompts.budget_line(0, 60, 0.0) == ""
    assert "step 3 of 60" in prompts.budget_line(3, 60, 0.0)
    # No ceiling configured: still says where it is, claims no limit it lacks.
    assert "of" not in prompts.budget_line(3, 0, 0.0).split(".")[0]
    assert "2m 5s elapsed" in prompts.budget_line(3, 60, 125.0)


# ---------------------------------------------------------------------------
# The goal check
# ---------------------------------------------------------------------------
#
# The ladder above measures whether the run is getting anywhere. Nothing
# measured whether it was already finished: the completion judge is reachable
# only through a terminal action the model volunteers, and `Oracle` needs a
# condition supplied at launch. `runs/963a4f4ae96c` graded 26 of 27 actions
# `success`, tripped no guard, answered its goal at step 14, and then ran 24 more
# steps and 471s before being killed by hand.


def _never_stops_policy(dev):
    """A run that works perfectly and never volunteers `done`."""
    def policy(screen, llm):
        if dev.state == "home":
            el = next(e for e in screen.elements
                      if e.best_text == "Wi-Fi" and e.interactive)
            return AgentAction(observation="the settings list",
                               reasoning="open Wi-Fi", action="tap",
                               target={"index": el.index},
                               notes=[{"key": f"reading {llm.calls}",
                                       "value": "something new"}])
        return AgentAction(observation="the Wi-Fi screen",
                           reasoning="go back", action="press_key", key="back")
    return policy


def test_a_run_that_has_already_met_its_goal_is_stopped(cfg, mem, tmp_path):
    cfg.run.max_steps = 60
    cfg.run.goal_check_every = 2
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _never_stops_policy(dev))
    llm.goal_check_result = True

    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "success"
    assert state.step < 12, f"ran {state.step} steps past being finished"
    checks = [e for e in _events(tmp_path, state.run_id) if e["kind"] == "goal_check"]
    assert checks and checks[-1]["satisfied"] is True
    # The answer survives: this path has no `done` text and never will, so the
    # collected data is what the caller is given.
    assert state.result, "the run ended with no answer"
    assert state.evidence


def test_one_satisfied_verdict_is_not_enough_to_stop_a_run(cfg, mem, tmp_path):
    """The only guard that ends a run on a model's say-so without being asked to.

    A single sample of anything is how a run that still had work to do gets cut
    off, so the verdicts have to be consecutive.
    """
    cfg.run.max_steps = 30
    cfg.run.goal_check_every = 1
    cfg.run.goal_check_hits = 2
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, _never_stops_policy(dev))
    # Satisfied once, then it changes its mind, then never again.
    llm.goal_check_result = lambda step: step == 2

    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome != "success" or state.step > 3, \
        "a single satisfied verdict ended the run"
    assert llm.goal_checks > 3


def test_the_goal_check_is_off_by_default_for_a_run_that_is_working(cfg, mem):
    """It must not change a run that ends on its own."""
    cfg.run.goal_check_every = 2
    dev = fake.FakeDevice(cfg)
    outcome, state, llm = run(dev, mem, cfg,
                              fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    assert outcome == "success"
    assert llm.goal_check_result is False
    assert state.goal_check_hits == 0


def test_a_goal_check_that_fails_never_loses_the_run(cfg, mem):
    """It is an optimisation on a loop that already terminates on its budgets."""
    cfg.run.max_steps = 20
    cfg.run.goal_check_every = 1
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))

    def boom(**kwargs):
        raise RuntimeError("the checker fell over")

    llm.goal_check = boom
    outcome, _ = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "success"


def test_a_run_that_keeps_learning_is_never_nudged(cfg, mem):
    """The ladder must stay invisible to a run that is working."""
    dev = fake.FakeDevice(cfg)
    outcome, state, llm = run(dev, mem, cfg,
                              fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    assert outcome == "success"
    assert state.steps_since_progress == 0
    assert llm.replans == 0
    assert not any("NO PROGRESS" in n for n in llm.notes)


def test_collecting_data_on_one_screen_is_progress(cfg, mem):
    """A read-only goal never leaves its screen and must not read as a stall."""
    cfg.run.max_steps = 12
    dev = fake.FakeDevice(cfg)
    seen = {"n": 0}

    def collector(screen, llm):
        seen["n"] += 1
        if seen["n"] > 8:
            return AgentAction(observation="done", reasoning="collected",
                               action="done", text="all of it")
        return AgentAction(observation="the settings list",
                           reasoning="record what is on it",
                           action="wait", duration=0.05,
                           notes=[{"key": f"row {seen['n']}",
                                   "value": f"value {seen['n']}"}])

    llm = fake.FakeLLM(dev, collector)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)
    assert outcome == "success"
    assert llm.replans == 0
    assert not any("NO PROGRESS" in n for n in llm.notes)


def test_a_terminal_action_is_never_refused_by_the_stall_guard(cfg, mem, tmp_path):
    """`done` and `fail` are the exits a stall is trying to push the agent to.

    The `fail` here is issued on a turn where the harness is already refusing
    the two-cycle's own actions, so a guard that did not exempt terminals would
    swallow it and the run would carry on to the give-up tier instead.
    """
    dev = fake.FakeDevice(cfg)
    turns = {"n": 0}
    cycle = _two_cycle_policy(dev)

    def policy(screen, llm):
        turns["n"] += 1
        # Late enough that turns 7 and 8 are refused first, so the `fail` on
        # turn 9 is issued while the guard is live rather than before it.
        if turns["n"] > 8:
            return AgentAction(observation="stuck", reasoning="stop",
                               action="fail", text="cannot do it")
        return cycle(screen, llm)

    llm = fake.FakeLLM(dev, policy)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    events = _events(tmp_path, state.run_id)
    kinds = [e["kind"] for e in events]
    assert "stall_block" in kinds, "the fixture never reached the refusing tier"
    gave_up = [e for e in events if e["kind"] == "gave_up"]
    assert gave_up and gave_up[-1]["reason"] == "cannot do it"
    assert outcome == "failed"
    assert "stalled_out" not in kinds, "the fail was swallowed by the guard"


def test_a_long_scroll_that_keeps_revealing_content_is_not_a_stall(cfg, mem):
    """The main false positive to guard against.

    A model searching a long feed writes no records and never leaves the
    screen, so the two loudest progress signals are both silent. What keeps it
    off the ladder is the third: the content moved. If that did not count, every
    feed search would be refused at step 5.
    """
    cfg.run.max_steps = 20
    dev = fake.FakeDevice(cfg)
    row = {"n": 0}

    def scroll(direction, **kw):
        dev.actions.append(f"scroll({direction})")
        row["n"] += 1          # each scroll really does bring new rows into view

    dev.scroll = scroll                                   # type: ignore[assignment]
    dev._xml = lambda: X.settings_screen(                 # type: ignore[assignment]
        rows=3, labels=[f"row {row['n'] * 3 + i}" for i in range(3)])

    def searcher(screen, llm):
        if row["n"] >= 10:
            return AgentAction(observation="found the end", reasoning="stop",
                               action="done", text="searched the whole feed")
        return AgentAction(observation="a long feed", reasoning="keep looking",
                           action="scroll", direction="down")

    llm = fake.FakeLLM(dev, searcher)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "success"
    assert llm.replans == 0, "a working search was sent to the replan tier"
    assert not any("NO PROGRESS" in n for n in llm.notes)


def test_a_scroll_that_reveals_nothing_still_stalls(cfg, mem, tmp_path):
    """The other side of it: scrolling a wall is not progress just because it
    is a scroll. `verify` answers "probably moved" for a gesture it has no
    image to check, and taking that at face value here would make the ladder
    unreachable for any run whose model only ever scrolls.

    `max_consecutive_failures` is lifted out of the way so the ladder is what
    is being tested. In a default configuration that counter gets there first,
    because a scroll against a wall grades `no_change` and so is a failure as
    well as a non-advance -- which is fine, and faster."""
    cfg.run.max_steps = 40
    cfg.run.max_consecutive_failures = 99
    dev = fake.FakeDevice(cfg)

    def searcher(screen, llm):
        return AgentAction(observation="a wall", reasoning="keep looking",
                           action="scroll", direction="down")

    llm = fake.FakeLLM(dev, searcher)
    outcome, state = Agent(dev, mem, llm, cfg).run(GOAL)

    assert outcome == "failed"
    assert state.step < cfg.run.max_steps, f"ran {state.step} steps against a wall"
    assert "stalled_out" in [e["kind"] for e in _events(tmp_path, state.run_id)]


# ---------------------------------------------------------------------------
# The image the step thought it had
# ---------------------------------------------------------------------------

def test_a_failed_vision_pass_withdraws_the_instruction_to_use_the_image(cfg, mem):
    """`needs_screenshot` only asks for an image when the tree cannot answer the
    question, and the note it returns says so: "rely on the screenshot", "look at
    the screen itself". When the vision call fails that note is an instruction to
    consult evidence the decider does not have -- and a model told to read pixels
    it never received describes pixels it never received."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_fails = True
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.notes, "no decide turn happened"
    first = llm.notes[0]
    assert "could NOT be read" in first
    assert "element list" in first


def test_a_working_vision_pass_adds_no_such_warning(cfg, mem):
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "Wi-Fi is on"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert llm.notes
    assert not any("could NOT be read" in note for note in llm.notes)


def test_an_unreadable_value_buys_one_sharper_look(cfg, mem):
    """Both vision prompts ask the model to say "unreadable" by name, and nothing
    used to act on it -- so a value the capture had thrown away read the same as a
    value the screen never held. The everyday frame is downscaled to a 1280 long
    edge; the app drew the digits at more than that."""
    from adbagent.agent import SHARP_LONG_EDGE

    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "unreadable, the figure is too blurry to make out"
    llm.vision_reading_after_reread = "Wi-Fi password: hunter2"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert SHARP_LONG_EDGE in dev.shot_edges
    assert SHARP_LONG_EDGE > 1280
    # The re-read must be given different pixels, not the same frame again.
    assert len(llm.frames_seen) >= 2
    assert llm.frames_seen[0] != llm.frames_seen[1]
    # And the answer it bought is the one the decider acts on -- a re-read whose
    # result is discarded is a round trip spent on nothing.
    assert "hunter2" in llm.analyses_seen[0]
    assert "unreadable" not in llm.analyses_seen[0]


def test_a_readable_value_is_never_re_read(cfg, mem):
    """An empty or answered `reading` is not a blurry one, and re-reading it at
    four times the pixels buys nothing."""
    from adbagent.agent import SHARP_LONG_EDGE

    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "Wi-Fi is on, 3 networks in range"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert SHARP_LONG_EDGE not in dev.shot_edges


def test_one_sharper_look_per_screen_not_per_turn(cfg, mem):
    """The second look answers "is the blur mine or the app's". Asking that twice
    of the same pixels cannot change the answer, and a screen the run sits on for
    twenty turns would otherwise buy twenty full-resolution captures."""
    from adbagent.agent import SHARP_LONG_EDGE

    cfg.run.always_screenshot = True
    cfg.run.max_steps = 6
    dev = fake.FakeDevice(cfg)

    def stay(screen, llm):
        return AgentAction(observation="still here", reasoning="waiting",
                           action="wait", duration=0.05, confidence="high")

    llm = fake.FakeLLM(dev, stay)
    llm.vision_reading = "unreadable, glare on the display"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert dev.shot_edges.count(SHARP_LONG_EDGE) == 1


def test_a_failed_re_read_leaves_the_first_answer_standing(cfg, mem):
    """A re-read is an improvement, never a way to lose what was already read."""
    dev = fake.FakeDevice(cfg)
    cfg.run.always_screenshot = True

    class Flaky(fake.FakeLLM):
        """Answers the everyday frame and drops every sharper one."""

        def analyze_image(self, screenshot, **kw):
            analysis = super().analyze_image(screenshot, **kw)
            if b"2400" in screenshot:             # the re-read, and only it
                failed = type(analysis)()
                failed._failed = True
                return failed
            return analysis

    llm = Flaky(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "unreadable, too small to read"
    Agent(dev, mem, llm, cfg).run(GOAL)

    # The failed re-read must not be mistaken for "the image could not be read
    # this turn" -- the first pass answered, and its answer is what stands.
    assert not any("could NOT be read" in note for note in llm.notes)


def test_a_sharper_look_that_reads_no_better_does_not_erase_the_first_answer(cfg, mem):
    """More pixels are not monotonically more answer. Against a real phone the
    full-resolution frame came back with all four fields empty where the
    downscaled one had at least said what it could not read -- and
    "unreadable, glare on the display" tells the run more than nothing does."""
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "unreadable, glare on the display"
    llm.vision_reading_after_reread = ""            # the sharper frame says nothing
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert "glare on the display" in llm.analyses_seen[0]


def test_a_sharper_look_that_is_still_unreadable_is_not_taken_as_progress(cfg, mem):
    cfg.run.always_screenshot = True
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = "unreadable, the meter is behind glass"
    llm.vision_reading_after_reread = "unreadable, still cannot make out the digits"
    Agent(dev, mem, llm, cfg).run(GOAL)

    assert "behind glass" in llm.analyses_seen[0]


def test_a_screenshot_turn_costs_one_vision_call_even_when_it_reads_nothing(cfg, mem):
    """The agent runs the vision pass itself so it can keep the structured fields,
    and hands the rendered result to `decide`. An analysis with all four fields
    empty renders to "" -- which used to read as "no analysis was done" and buy a
    second look at the same frame, on every turn the vision model correctly had
    nothing to add."""
    cfg.run.always_screenshot = True
    cfg.run.max_steps = 3
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    llm.vision_reading = ""            # nothing to report, and nothing wrong
    llm.vision_label = ""
    Agent(dev, mem, llm, cfg).run(GOAL)

    # One pass per decide turn that was given a frame, plus the judge's own pass
    # on the final screen -- which nobody analysed for it. Two per decide turn is
    # the regression.
    assert llm.analyses <= llm.seen_screenshots + llm.judges, (
        f"{llm.analyses} vision calls for {llm.seen_screenshots} screenshot "
        f"turn(s) and {llm.judges} judge(s)")


def test_keyboard_interrupt_raises(cfg, mem, monkeypatch):
    """KeyboardInterrupt inside agent.run loop must be re-raised so callers
    and watch loops know the run was stopped by user request."""
    dev = fake.FakeDevice(cfg)
    llm = fake.FakeLLM(dev, fake.reach_state(dev, "wifi", ["Wi-Fi"]))
    agent = Agent(dev, mem, llm, cfg)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(dev, "observe", raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        agent.run(GOAL)

