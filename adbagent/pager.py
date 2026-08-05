"""Repeating a gesture that has been observed to work.

Walking a set of items is the one thing an agent does that is genuinely
mechanical. Every turn of it asks the same question -- is there another item,
and what does it say -- and answers it with the same gesture. In the run this
module was first written for, 71 of 127 steps were the single action
``swipe #4 left``, each paid for with a full reasoning turn at 26s median.

So once the model has chosen to page and the content verifiably moved, the loop
keeps paging in code. The model is not being second-guessed; its decision is
being *repeated* for as long as the situation stays the one it decided about.

What this module deliberately does NOT do
-----------------------------------------

It used to model carousels. It classified a screen as a pager from the largest
full-bleed horizontal scroller, minted a per-item identity from the app's
caption, and kept a ledger of a *set* -- how many items it held, which had been
read, which ends had been reached, whether it was complete.

Every one of those was an assumption dressed as a fact, and each one held for
the WhatsApp media album it was written against and broke elsewhere:

* **The largest scroller is the content pager.** On Instagram it picked
  ``swipeable_tab_view_pager`` -- the Home/Search/Reels/Profile tab strip -- on
  every screen of a 45-step run, never the reel. Flinging it changes tab.
* **A ViewPager pages horizontally.** ``ViewPager2`` also pages vertically, and
  Reels, Shorts and TikTok all use it that way. The orientation is not in the
  accessibility dump, so the class name was the only evidence, and it is wrong
  for every short-video feed.
* **A set is finite and has two ends.** A feed has neither, so "you have reached
  the LAST item of this set" could only ever be false on one.
* **Items carry captions.** Feeds do not, and the caption search fell through to
  a bare clock pattern -- which matched the *status bar*, renaming every item
  once a minute and filling the ledger with phantoms.

Measured over ``runs/``: the machinery engaged in 13 runs, saved steps in 3, and
in one cost 45 steps and the goal. So the set model is gone. What is left is the
one thing that was ever provable -- *did the app's content change* -- and a
sweep built on nothing but that. No captions, no totals, no ends, no axis: a
vertical feed repeats exactly like a horizontal album, because neither is being
classified.

Remembering *which* items were seen is now the model's job, in ``notes`` and
``progress``, which it is already instructed to keep. A harness that cannot tell
one photo from another should say so by staying quiet, not by inventing names
for them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .screen import Screen

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
    """Perceptual hash of the app's content. ``None`` without a screenshot."""
    if not screen.screenshot:
        return None
    box = content_box(screen)
    if box is None:
        return None
    from .fingerprint import compute_dhash
    return compute_dhash(screen.screenshot, box_frac=box)


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
# The sweep
# ---------------------------------------------------------------------------
#
# The gesture the sweep issues is the one the model just issued, repeated. It
# never taps, types, navigates or presses a key, so there is no action it can
# take that the model did not already authorise.

MAX_SWEEP_RENDER = 40
MAX_DETAIL_CHARS = 110


@dataclass
class SweepLog:
    """What repeating a gesture turned up, in the order it turned up.

    Deliberately a list and not a set model. It has no notion of how many items
    exist, which ones remain, or whether the end has been reached, because none
    of that is observable -- and stating it anyway is what made the old ledger
    tell the agent, with total confidence, that a set it had never seen was
    fully read.

    Positions are the order things were read in, not identities. A reading is
    "the third thing this sweep saw", which is true, rather than "item 3 of 15",
    which was usually not.
    """

    #: The gesture being repeated, e.g. "swipe left" -- for the render only.
    gesture: str = ""
    #: One entry per item read, in sweep order.
    readings: List[str] = field(default_factory=list)
    #: Gestures issued, which is >= len(readings) when a frame was not read.
    repeats: int = 0
    #: Why the repeat handed back, for the block the model reads next turn.
    reason: str = ""

    def start(self, gesture: str) -> None:
        """Begin a fresh sweep. Prior readings belong to a finished one."""
        self.gesture = gesture
        self.readings = []
        self.repeats = 0
        self.reason = ""

    def add(self, reading: str) -> None:
        text = " ".join((reading or "").split())[:MAX_DETAIL_CHARS]
        if text:
            self.readings.append(text)

    @property
    def read_count(self) -> int:
        return len(self.readings)

    def render(self, reason: str = "") -> str:
        """The block handed to the model after a sweep hands back."""
        if not self.readings:
            return ""
        lines = [f"YOU REPEATED `{self.gesture}` {self.repeats} time(s) and read, "
                 f"in order:"]
        shown = self.readings
        if len(shown) > MAX_SWEEP_RENDER:
            dropped = len(shown) - MAX_SWEEP_RENDER
            lines.append(f"  (... {dropped} earlier reading(s) omitted)")
            shown = shown[-MAX_SWEEP_RENDER:]
        start = len(self.readings) - len(shown) + 1
        for position, reading in enumerate(shown, start=start):
            lines.append(f"  {position}. {reading}")
        why = reason or self.reason
        if why:
            lines.append(f"The repeat stopped because {why}.")
        # No claim about what remains: nothing here knows.
        lines.append("Record anything you need from these in `notes` -- this "
                     "list is not kept for you.")
        return "\n".join(lines)


def can_repeat(*, action: str, direction: str, moved: Optional[bool]) -> bool:
    """Whether the gesture that just ran may be repeated mechanically.

    Requires a directional gesture that *demonstrably* moved the content. That
    last condition is the whole safety argument on an unfamiliar app: a screen
    where the swipe does nothing, or where the direction means something other
    than "next", never earns a second automatic gesture.

    Note ``moved`` must be ``True``. ``None`` means no screenshot was taken and
    nothing is known, which is not permission.
    """
    if action not in ("swipe", "scroll"):
        return False
    if not direction:
        return False
    return moved is True


def stop_repeating(after: Screen, *, package: str = "",
                   moved: Optional[bool] = None) -> str:
    """Why the sweep should hand back, or ``""`` to keep going.

    Short by design. The old version had six reasons, four of which were about a
    set model that no longer exists ("every item has been read", "the left end
    of the set has been reached", "this is no longer a carousel", "the caption is
    hidden"). What is left is what can actually be seen: the app changed, or the
    gesture stopped moving anything. Budgets are the caller's, since only it
    knows the step and clock it is spending.
    """
    if package and after.package and after.package != package:
        return f"the foreground app changed to {after.package}"
    if moved is False:
        return "the content stopped changing, so the gesture no longer advances"
    if moved is None:
        return "there is no screenshot, so it cannot be told whether it advanced"
    return ""


def sweep_summary(first_step: int, last_step: int, gesture: str,
                  swept: int, read: int, reason: str) -> str:
    """One history line for a whole sweep, so it costs one entry not N."""
    span = f"steps {first_step}-{last_step}" if last_step > first_step \
        else f"step {first_step}"
    return (f"{span}: repeated `{gesture}` {swept} time(s), read {read} "
            f"-> stopped because {reason}")


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
