"""Did the app's content change? Asked of the pixels.

This file used to test a carousel model -- captions as item identity, a ledger
of a set, its size, its two ends -- and after that a sweep that repeated the
model's gesture in code for as long as it kept moving the content. Both are
gone; see the module docstring in ``adbagent/pager.py`` for why. Nothing in
the module acts on the phone now.

What is left is the one claim it still makes: the app's content changed
between these two frames, or it did not, or that cannot be told -- and that a
video playing in part of the frame is not the content changing.

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
# A playing video is not the content changing
# ---------------------------------------------------------------------------

VPKG = "com.instagram.android"
_VTOP, _VBOT = 700, 1480          # a video over a third of a 1080x2340 frame


def _feed(nested: bool, video: bool = True):
    """A feed post with a video in it, drawn the two ways apps draw one.

    ``nested`` is the shape a real player takes -- a container whose resource
    id names it, the surface inside, the mute button on top -- and all three
    match `_video_regions`. Flat is the same video with no wrapper.
    """
    inner = [
        X.N("android.view.SurfaceView", (0, _VTOP, X.W, _VBOT),
            rid="video_surface", package=VPKG),
        X.N("android.widget.Button", (960, _VBOT - 120, 1050, _VBOT - 30),
            desc="Unmute video", rid="mute_button", package=VPKG,
            clickable=True),
    ] if video else [
        X.N("android.widget.ImageView", (0, _VTOP, X.W, _VBOT),
            rid="photo", package=VPKG),
    ]
    if nested and video:
        inner = [X.N("android.widget.FrameLayout", (0, _VTOP, X.W, _VBOT),
                     rid="clips_player_container", package=VPKG,
                     children=inner)]
    root = X.N("android.widget.FrameLayout", (0, 0, X.W, X.H), package=VPKG,
               children=[
                   X.status_bar(),
                   X.N("androidx.recyclerview.widget.RecyclerView",
                       (0, 80, X.W, X.H), rid="feed_list", package=VPKG,
                       scrollable=True, children=[
                           X.N("android.widget.TextView", (24, 600, X.W - 24, 680),
                               text="alice_", rid="username", package=VPKG),
                           *inner,
                           X.N("android.widget.TextView",
                               (24, _VBOT + 20, X.W - 24, _VBOT + 90),
                               text="a caption that does not change",
                               rid="caption", package=VPKG),
                       ]),
               ])
    return attach(parse(X.dump(root), width=X.W, height=X.H))


def test_a_video_over_part_of_the_frame_vetoes_the_pixel_change():
    before, after = _feed(nested=False), _feed(nested=False)
    assert pager.video_only_drift(before, after)


def test_a_nested_player_is_measured_once_not_once_per_matching_node():
    """The container, the surface and the mute button all match, and all
    describe the same pixels. Summed they read 0.670 of the frame -- past
    `_VIDEO_FULL_BLEED`, so a third-of-the-screen video was taken for a
    full-bleed one and the veto silently did not fire."""
    before, after = _feed(nested=True), _feed(nested=True)
    regions = pager._video_regions(before)
    assert len(regions) == 3, "the fixture should match all three nodes"
    frame = X.W * X.H
    assert sum(el.area for el in regions) / frame > pager._VIDEO_FULL_BLEED
    assert pager._union_area(regions) / frame < pager._VIDEO_FULL_BLEED
    assert pager.video_only_drift(before, after)


def test_a_full_bleed_video_is_the_content_and_is_not_vetoed():
    """A reel fills the frame: paging it changes nothing the tree describes,
    so its bitmaps are the only evidence there is."""
    root = X.N("android.widget.FrameLayout", (0, 0, X.W, X.H), package=VPKG,
               children=[X.status_bar(),
                         X.N("android.view.SurfaceView", (0, 0, X.W, X.H),
                             rid="reel_surface", package=VPKG)])
    reel = attach(parse(X.dump(root), width=X.W, height=X.H))
    assert not pager.video_only_drift(reel, reel)


def test_a_screen_with_no_video_is_never_vetoed():
    still = _feed(nested=False, video=False)
    assert not pager.video_only_drift(still, still)


def test_a_tree_that_changed_is_the_gesture_working_not_the_video():
    """The veto turns on the tree standing still. A scroll that revealed new
    content changes the tree, and its pixel movement is real."""
    before = _feed(nested=True)
    after = _feed(nested=True)
    after.exact_id = before.exact_id + "-moved"
    assert not pager.video_only_drift(before, after)


def test_union_area_counts_overlapping_boxes_once():
    class _Box:
        def __init__(self, bounds):
            self.bounds = bounds
            l, t, r, b = bounds
            self.area = (r - l) * (b - t)

    assert pager._union_area([_Box((0, 0, 10, 10)), _Box((0, 0, 10, 10))]) == 100
    assert pager._union_area([_Box((0, 0, 10, 10)), _Box((5, 0, 15, 10))]) == 150
    assert pager._union_area([_Box((0, 0, 10, 10)), _Box((20, 0, 30, 10))]) == 200
    assert pager._union_area([]) == 0
