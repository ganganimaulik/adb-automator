"""Repeating a gesture that works.

This file used to test a carousel model: captions as item identity, a ledger of
a set, its size, its two ends, whether it was complete. That model is gone --
see the module docstring in ``adbagent/pager.py`` for the measurements that
retired it -- and so are the tests that pinned it.

What is left tests two claims, which are the only two the module still makes:

* the app's content changed between these two frames, or it did not, or that
  cannot be told;
* a gesture seen to change it may be repeated until it stops changing it.

Nothing here knows what a gallery is, and that is the point: a vertical video
feed and a horizontal photo album exercise the same code paths because neither
is being classified.
"""

from __future__ import annotations

import io

from PIL import Image

from adbagent import pager
from adbagent.actions import AgentAction, Target, execute, verify
from adbagent.fingerprint import attach
from adbagent.screen import parse
from tests import xmlgen as X
from tests.fake import FakeDevice


def _png(seed: int, size: int = 64) -> bytes:
    """A distinct image per seed, so the perceptual hash has something to see."""
    image = Image.new("L", (size, size))
    image.putdata([(seed * 37 + x * 5 + y * 11) % 256
                   for y in range(size) for x in range(size)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def viewer(timestamp: str = "9:33 am", chrome: bool = True, shot: int = None):
    screen = attach(parse(X.media_viewer(timestamp, chrome=chrome),
                          width=X.W, height=X.H, activity=X.MEDIA_ACTIVITY))
    if shot is not None:
        screen.screenshot = _png(shot)
    return screen


def act(**kw) -> AgentAction:
    kw.setdefault("observation", "o")
    kw.setdefault("reasoning", "r")
    return AgentAction(**kw)


# ---------------------------------------------------------------------------
# Did the content change?
# ---------------------------------------------------------------------------

def test_a_different_frame_reads_as_moved():
    assert pager.content_moved(viewer(shot=1), viewer(shot=9)) is True


def test_the_same_frame_reads_as_not_moved():
    assert pager.content_moved(viewer(shot=4), viewer(shot=4)) is False


def test_without_a_screenshot_the_answer_is_not_no():
    """``None`` means "no evidence", and callers must not read it as "no".

    A swipe with no image behind it used to be graded "probably worked", which
    is a fair default for grading one action and a terrible one for handing the
    harness authority to repeat it thirty more times.
    """
    assert pager.content_moved(viewer(), viewer(shot=2)) is None
    assert pager.content_moved(viewer(shot=2), viewer()) is None
    assert pager.content_moved(viewer(), viewer()) is None


def test_chrome_fading_out_is_not_the_content_changing():
    """The overlay lives in bands at the top and bottom, which are cropped."""
    with_chrome = viewer(chrome=True, shot=7)
    without = viewer(chrome=False, shot=7)
    assert pager.content_moved(with_chrome, without) is False


def test_the_status_bar_cannot_move_the_content_hash():
    """The crop is of the frame, so nothing the OS draws at the edges counts."""
    box = pager.content_box(viewer())
    assert box is not None
    left, top, right, bottom = box
    assert top > 0.0 and bottom < 1.0        # the bands are excluded
    assert (left, right) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Verification uses it
# ---------------------------------------------------------------------------

def test_a_swipe_that_did_not_move_the_content_is_no_change():
    before, after = viewer(shot=3), viewer(shot=3)
    action = act(action="swipe", direction="left", target=Target(index=1))
    outcome = verify(action, before, after, None, None)
    assert outcome.grade == "no_change"


def test_a_swipe_that_moved_the_content_is_a_success():
    before, after = viewer(shot=3), viewer(shot=8)
    action = act(action="swipe", direction="left", target=Target(index=1))
    assert verify(action, before, after, None, None).grade != "no_change"


def test_the_models_swipe_target_is_honoured_not_retargeted():
    """A horizontal swipe used to be snapped onto "the pager" -- the largest
    full-bleed horizontal scroller. On an app that nests a tab strip above a
    feed that picked the tab strip, so "next photo" became "next tab" and the
    model could not tell, because its stated target was silently replaced."""
    dev = FakeDevice()
    screen = viewer()
    target = screen.elements[-1]
    execute(dev, act(action="swipe", direction="left",
                     target=Target(index=target.index)), screen)
    assert dev.actions, "the swipe should have been issued"


# ---------------------------------------------------------------------------
# May the gesture be repeated?
# ---------------------------------------------------------------------------

def test_a_gesture_seen_to_move_content_authorises_repeating_it():
    assert pager.can_repeat(action="swipe", direction="left", moved=True)
    assert pager.can_repeat(action="scroll", direction="down", moved=True)


def test_a_vertical_feed_pages_exactly_like_a_horizontal_album():
    """The old gate accepted only left/right, which is why the sweep never
    engaged on the short-video feeds it would have helped on most."""
    for direction in ("up", "down", "left", "right"):
        assert pager.can_repeat(action="swipe", direction=direction, moved=True)


def test_a_gesture_that_moved_nothing_never_authorises_a_second_one():
    assert not pager.can_repeat(action="swipe", direction="left", moved=False)


def test_no_evidence_is_not_permission():
    assert not pager.can_repeat(action="swipe", direction="left", moved=None)


def test_only_a_directional_gesture_qualifies():
    assert not pager.can_repeat(action="tap", direction="left", moved=True)
    assert not pager.can_repeat(action="swipe", direction="", moved=True)


# ---------------------------------------------------------------------------
# When to hand back
# ---------------------------------------------------------------------------

def test_the_repeat_stops_when_the_content_stops_changing():
    reason = pager.stop_repeating(viewer(), package=X.WA, moved=False)
    assert "stopped changing" in reason


def test_the_repeat_stops_when_the_app_changes():
    other = viewer()
    other.package = "com.other.app"
    assert "foreground app changed" in pager.stop_repeating(
        other, package=X.WA, moved=True)


def test_the_repeat_stops_when_it_cannot_be_told_whether_it_advanced():
    assert pager.stop_repeating(viewer(), package=X.WA, moved=None)


def test_the_repeat_continues_while_the_gesture_keeps_working():
    assert pager.stop_repeating(viewer(), package=X.WA, moved=True) == ""


def test_nothing_stops_a_repeat_for_reaching_an_end_that_cannot_be_counted():
    """Four of the six old stop reasons were verdicts about a set -- "every item
    has been read", "the left end of the set has been reached". A feed has no
    end, so those could only ever be wrong on one."""
    keep_going = pager.stop_repeating(viewer(), package=X.WA, moved=True)
    assert keep_going == ""


# ---------------------------------------------------------------------------
# What the sweep reports back
# ---------------------------------------------------------------------------

def test_the_log_lists_readings_in_the_order_they_were_read():
    log = pager.SweepLog()
    log.start("swipe left")
    log.add("banana 120 g")
    log.add("almonds 6 g")
    log.repeats = 2
    rendered = log.render()
    assert "1. banana 120 g" in rendered
    assert "2. almonds 6 g" in rendered
    assert "swipe left" in rendered


def test_the_log_claims_nothing_about_what_it_did_not_see():
    """The old ledger closed with "every item in this set has been read" and
    "you have reached the LAST item". It could not know either."""
    log = pager.SweepLog()
    log.start("swipe left")
    log.add("one reading")
    rendered = log.render(reason="the content stopped changing")
    for claim in ("LAST item", "every item", "complete", "of 15", "STILL NOT READ"):
        assert claim not in rendered
    assert "stopped changing" in rendered


def test_the_log_tells_the_model_the_list_is_not_kept_for_it():
    """It is handed back once and dropped, so anything worth keeping has to go
    into `notes`, which is the memory that actually survives."""
    log = pager.SweepLog()
    log.start("swipe up")
    log.add("a reading")
    assert "`notes`" in log.render()


def test_an_empty_sweep_renders_nothing():
    assert pager.SweepLog().render() == ""


def test_the_log_is_bounded():
    log = pager.SweepLog()
    log.start("swipe left")
    for i in range(pager.MAX_SWEEP_RENDER + 25):
        log.add(f"reading {i}")
    rendered = log.render()
    assert "omitted" in rendered
    assert len(rendered.splitlines()) < pager.MAX_SWEEP_RENDER + 12


def test_a_long_reading_is_truncated():
    log = pager.SweepLog()
    log.start("swipe left")
    log.add("x" * 400)
    assert len(log.readings[0]) <= pager.MAX_DETAIL_CHARS


def test_starting_a_sweep_clears_the_previous_ones_readings():
    log = pager.SweepLog()
    log.start("swipe left")
    log.add("old")
    log.start("swipe up")
    assert log.readings == []
    assert log.gesture == "swipe up"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def test_a_sweep_is_summarised_as_one_history_line():
    line = pager.sweep_summary(4, 12, "swipe left", swept=8, read=8,
                               reason="the content stopped changing")
    assert "steps 4-12" in line
    assert "swipe left" in line
    assert "8" in line


def test_a_one_step_sweep_does_not_render_a_range():
    line = pager.sweep_summary(4, 4, "swipe up", swept=1, read=1, reason="x")
    assert "step 4" in line and "4-4" not in line
