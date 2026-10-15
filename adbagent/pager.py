"""Did the app's content change? Asked of the pixels, never acted on.

A swipe through a photo album or a feed often leaves the accessibility tree
exactly as it was -- the same viewer, the same controls -- while the bitmap
changes completely. So whether a gesture moved anything is answered here, from
a perceptual hash of the content band of two frames, and `verify` and the loop
read the answer.

What this module deliberately does NOT do
-----------------------------------------

It no longer repeats a gesture. It used to: once the model had swiped and the
content moved, a sweep went on swiping in code -- up to twelve times, with a
vision read of each item in place of a decision -- on the theory that the
model's choice was merely being repeated while the situation stayed the one it
had decided about. On a card deck it is not the same situation twice: each
swipe is a decision about a different card, a like or a pass, and every repeat
was one the model never made. ``runs/3c94258b5fc6`` showed what an unreviewed
action on a deck costs. Every gesture is the model's now, one turn each.

Before that it modelled carousels: it classified a screen as a pager from the
largest full-bleed horizontal scroller, minted a per-item identity from the
app's caption, and kept a ledger of a *set*. Each of those was an assumption
dressed as a fact -- the largest scroller was Instagram's tab strip, ViewPager2
pages vertically on every short-video feed, a feed has no ends, and the caption
search matched the status-bar clock -- so the set model went first.

What is left is the one thing that was ever provable: *did the app's content
change*. Remembering which items were seen is the model's job, in ``notes`` and
``progress``.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from .screen import Element, Screen

log = logging.getLogger("adbagent.pager")


# ---------------------------------------------------------------------------
# Did the app's content change?
# ---------------------------------------------------------------------------

#: Overlay chrome -- a media viewer's toolbar, a feed's action rail -- fades in
#: and out in horizontal bands at the top and bottom of the content. Cropping
#: them away is what stops "the toolbar faded out" reading as "the item changed".
_CHROME_BAND_FRAC = 0.18

#: How far the cropped hash must move to call it a real change. A re-render of
#: the same content drifts by a bit or two; an actual page replaces the bitmap.
_PIXEL_DISTANCE = 6


def content_box(screen: Screen) -> Optional[Tuple[float, float, float, float]]:
    """The fraction of the frame holding the app's content, chrome cropped.

    Fractions rather than pixels because the screenshot need not share the
    accessibility tree's coordinate space.

    This used to crop to the bounds of the detected pager element. It crops the
    frame instead: which element is "the pager" was the guess that went wrong,
    and the bands being cropped are where overlay chrome lives on any full-screen
    viewer regardless of which container owns it.
    """
    if not screen.width or not screen.height:
        return None
    inset = _CHROME_BAND_FRAC
    if 1.0 - 2 * inset < 0.05:
        return None
    return (0.0, inset, 1.0, 1.0 - inset)


def content_hash(screen: Screen) -> Optional[int]:
    """Perceptual hash of the app's content. ``None`` without a screenshot.

    Memoised on the frame. The answer is asked for several times per step --
    grading the action, and deciding whether the step was progress -- and it is
    a JPEG decode each time on pixels that are fixed once captured. The memo is only taken when there *is* a
    screenshot, because one can be attached to a frame after the first ask.
    """
    if not screen.screenshot:
        return None
    if screen.content_dhash_done:
        return screen.content_dhash
    box = content_box(screen)
    if box is None:
        value = None
    else:
        from .fingerprint import compute_dhash
        value = compute_dhash(screen.screenshot, box_frac=box)
    screen.content_dhash = value
    screen.content_dhash_done = True
    return value


def content_moved(before: Screen, after: Screen) -> Optional[bool]:
    """Whether the app's content changed between two frames.

    ``None`` when unknowable -- no screenshot on one side, or a hash that could
    not be computed. Callers must treat ``None`` as "no evidence", never as
    "no". This is the whole of what this module claims to know.
    """
    before_hash = content_hash(before)
    after_hash = content_hash(after)
    if before_hash is None or after_hash is None:
        return None
    from .fingerprint import dhash_distance
    distance = dhash_distance(before_hash, after_hash)
    if distance is None:
        return None
    return distance >= _PIXEL_DISTANCE


# ---------------------------------------------------------------------------
# A playing video is not the content changing
# ---------------------------------------------------------------------------

#: Surfaces that repaint themselves: video players and GL views. The dump
#: records the surface, never the frames on it -- so a tree standing still
#: while the pixels churn is what "a video is playing" looks like from here.
_VIDEO_SURFACE_SUFFIXES = (
    "SurfaceView", "TextureView", "VideoView", "GLSurfaceView", "TvView",
    "PlayerView", "ExoPlayerView", "SimpleExoPlayerView", "StyledPlayerView",
)

#: A video *control* says the same thing when the surface itself never reaches
#: the tree (a Compose player draws one): "Unmute video", "Play video".
_VIDEO_CONTROL = re.compile(r"\b(?:unmute|mute|play|pause)\b[\w ]*\bvideo\b",
                            re.I)

#: Past this share of the frame the video IS the content -- a reel, the media
#: viewer -- and its pixels are the only evidence there is, veto or no veto.
_VIDEO_FULL_BLEED = 0.65


def _video_regions(screen: Screen) -> List[Element]:
    """Elements that animate on their own, read off the *full* tree (`nodes`).

    The surface itself when the dump carries one, and otherwise the control
    that names it ("Unmute video"). Resource ids naming a player or a video
    count too -- custom players do not all extend a recognisable base class.
    """
    out: List[Element] = []
    for el in screen.nodes:
        if el.is_system_chrome:
            continue
        short = el.cls.rsplit("$", 1)[-1].rsplit(".", 1)[-1]
        if short.endswith(_VIDEO_SURFACE_SUFFIXES):
            out.append(el)
            continue
        rid = el.resource_id.lower()
        if "exoplayer" in el.cls.lower() or "player" in rid or "video" in rid:
            out.append(el)
            continue
        if _VIDEO_CONTROL.search(el.best_text or ""):
            out.append(el)
    return out


def _union_area(elements: List[Element]) -> int:
    """Area covered by these boxes counted once, however they overlap.

    Summing `el.area` instead double-counts, and a real player is nested: the
    container whose resource id names it, the surface inside it, and the
    "Unmute video" button on top all match `_video_regions` and all describe
    the same pixels. Measured on a 1080x2340 frame with a video over a third
    of it, the sum reported 0.670 -- past `_VIDEO_FULL_BLEED`, so a video that
    was plainly not full-bleed was treated as one and the veto never fired.

    Coordinate compression rather than a sweep line: the region count here is
    a handful, and the cell grid is easier to be sure of.
    """
    boxes = [el.bounds for el in elements if el.area > 0]
    if not boxes:
        return 0
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    ys = sorted({b[1] for b in boxes} | {b[3] for b in boxes})
    total = 0
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            for left, top, right, bottom in boxes:
                if left <= x0 and right >= x1 and top <= y0 and bottom >= y1:
                    total += (x1 - x0) * (y1 - y0)
                    break
    return total


def video_only_drift(before: Screen, after: Screen) -> bool:
    """True when a playing video explains a pixel change the tree never felt.

    `content_moved` compares bitmaps, so a video playing inside an otherwise
    static screen reads as "the content moved" on every gesture -- including
    the scroll that hit the end of the list and moved nothing. That kept the
    dead-scroll ledger empty for as long as the video played. The tree is the tiebreak: a gesture that reveals content the tree
    describes changes the tree, so a byte-identical tree with a video on it
    means the motion is the video, not the gesture.

    A video that fills the frame -- a reel, the media viewer -- is the content
    itself, not noise over it: paging it changes nothing the tree describes,
    so its bitmaps stay the only evidence and are not vetoed.
    """
    if not before.exact_id or after.exact_id != before.exact_id:
        return False
    regions = _video_regions(before)
    if not regions or not _video_regions(after):
        return False
    total = before.width * before.height
    if total <= 0:
        return True
    covered = min(1.0, _union_area(regions) / total)
    return covered < _VIDEO_FULL_BLEED


# A `loop_id` used to live here: `exact_id` plus the content hash, so that the
# fifteen photos of an album -- whose timestamps `mask_text` folds together, and
# which therefore share one `exact_id` -- did not read to the loop detector as
# one screen visited fifteen times. It is gone because the detector no longer
# asks that question. It keys on the (screen, action) pair and on `skeleton_id`,
# which answers the album case without needing pixels: paging an album repeats
# one pair, and `safety.LoopDetector` treats a repeated pair as a cycle only at
# period 2 or 3, never at period 1. See `safety.LoopDetector._counts_as_repetition`.
#
# Sharpening the identity was also the wrong direction for the failure that
# actually costs runs. `exact_id` hashes element bounds, so it already differs on
# nearly every visit to a live screen; adding pixels to it could only make two
# visits look more distinct, and the two-cycle in `runs/2521862d7a23` needed them
# to look the same.
