"""Item identity for pagers: galleries, media viewers, carousels, card stacks.

A pager is a screen whose *identity* never changes while the *item* on it does.
Every fingerprint in :mod:`fingerprint` is deliberately blind to that:
``skeleton_id`` is content-free by design, and ``exact_id`` runs all text through
``mask_text``, which rewrites ``"Today, 9:33 am"`` as ``"<time>"``. So photo 7
and photo 8 of an album hash identically, and three separate mechanisms break at
once:

* ``verify`` cannot tell an advancing swipe from a swipe that did nothing, so
  ``_scroll_changed`` used to answer "changed" for *every* swipe -- meaning a
  gesture the ViewPager dropped was reported to the model as a success.
* the loop detector sees the same ``exact_id`` on every photo and forces a back
  press, which throws the agent out of the carousel it was halfway through.
* nothing records which items have actually been looked at, so the model has to
  keep that ledger by hand in a free-text field it rewrites every turn -- and a
  single omission loses an item permanently.

This module supplies the missing signal: a per-item identity read from the label
the app itself puts on screen (``"Today, 9:33 am"``, ``"3 of 15"``, a filename),
plus a perceptual hash of the item's own pixels for when the app hides its
chrome. That identity is what makes a swipe verifiable and a ledger possible.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .screen import Element, Screen

log = logging.getLogger("adbagent.pager")

#: Activities that are a pager even when the tree is too sparse to prove it.
#: Matched case-insensitively as a substring of the activity name.
PAGER_ACTIVITY_HINTS = (
    "mediaview", "photoview", "imageview", "gallery", "viewpager",
    "lightbox", "slideshow", "storyview", "storiesview", "reels",
)

#: A horizontal scroller must fill this fraction of the screen to count as the
#: pager rather than a chip row or a thumbnail strip.
_MIN_WIDTH_FRAC = 0.8
_MIN_HEIGHT_FRAC = 0.35


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def pager_element(screen: Screen) -> Optional[Element]:
    """The full-bleed horizontal scroller that pages between items, if any.

    Returns the largest qualifying scroller: a media viewer occasionally nests a
    thumbnail strip inside the pager, and the outer one is the one to fling.
    """
    if not screen.width or not screen.height:
        return None
    best: Optional[Element] = None
    for el in screen.elements:
        if not el.scrollable or not el.is_horizontal:
            continue
        if el.width < screen.width * _MIN_WIDTH_FRAC:
            continue
        if el.height < screen.height * _MIN_HEIGHT_FRAC:
            continue
        if best is None or el.area > best.area:
            best = el
    return best


def is_pager_screen(screen: Screen) -> bool:
    """True when this screen shows one item out of a sequence."""
    if pager_element(screen) is not None:
        return True
    activity = (screen.activity or "").lower()
    return any(h in activity for h in PAGER_ACTIVITY_HINTS)


# ---------------------------------------------------------------------------
# Item labels
# ---------------------------------------------------------------------------

#: Resource-id fragments that name the element carrying the item's caption.
_TITLE_RIDS = ("title", "subtitle", "caption", "header", "date", "timestamp",
               "counter", "position", "indicator")

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_WEEKDAY = r"(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?"
_CLOCK = r"\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?"
#: A date part must be a real date. An earlier version allowed
#: ``\d{1,2}\s+\w{3,9}`` for "12 May", which happily read the "91 93275" of a
#: phone number as a date and swallowed the sender into the caption.
_DATE = (r"(?:today|yesterday"
         rf"|{_WEEKDAY}"
         rf"|\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}})?"
         rf"|{_MONTH}\s+\d{{1,2}}(?:,?\s+\d{{4}})?"
         r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)")

#: Ordered by how specific the match is. The first pattern that hits a chrome
#: element wins, so "3 of 15" beats a bare time and both beat a plain date.
_LABEL_PATTERNS: Tuple[re.Pattern, ...] = (
    # "3 of 15", "3/15", "photo 3 of 15"
    re.compile(r"\b(\d{1,4})\s*(?:of|/)\s*(\d{1,4})\b", re.I),
    # "Today, 9:33 am", "Yesterday, 21:04", "12 May 2025, 9:33 am"
    re.compile(rf"\b({_DATE}\s*,\s*{_CLOCK})", re.I),
    # A bare wall-clock time -- the weakest useful identity.
    re.compile(rf"\b({_CLOCK})\b", re.I),
)

_ORDINAL_RE = _LABEL_PATTERNS[0]
#: "15 photos", "15 items", "15 media"
_TOTAL_RE = re.compile(r"\b(\d{1,4})\s+(?:photos?|videos?|items?|images?|media|files?)\b",
                       re.I)

#: Chrome labels that appear on every screen and so name no particular set.
_NAV_LABELS = frozenset((
    "back", "home", "overview", "up", "close", "cancel", "done", "save",
    "share", "forward", "reply", "delete", "more options", "menu", "search",
))


def _chrome_elements(screen: Screen) -> List[Element]:
    """Elements outside any scroller -- the toolbar, title bar and captions.

    Text *inside* the pager belongs to whichever item happens to be rendered and
    is not a reliable caption, so it is excluded. The scrollers themselves are
    excluded too: a container's own label ("Image") is a description of the
    widget, not of the item in it, and once the app fades its toolbar out the
    pager is the *only* thing left -- which is precisely when mistaking it for a
    caption does the most damage.
    """
    return [el for el in screen.elements
            if el.scroller() is None and not el.scrollable]


def item_label(screen: Screen) -> str:
    """The app's own caption for the item on screen, verbatim and unmasked.

    Prefers an element whose resource-id names it as a title, then falls back to
    any chrome element carrying a counter or a timestamp.
    """
    chrome = _chrome_elements(screen)
    if not chrome:
        return ""

    def titled_first(els: List[Element]) -> List[Element]:
        rid_match = [e for e in els
                     if any(f in e.resource_id.lower() for f in _TITLE_RIDS)]
        return rid_match + [e for e in els if e not in rid_match]

    for pattern in _LABEL_PATTERNS:
        for el in titled_first(chrome):
            text = " ".join((el.best_text or "").split())
            if not text:
                continue
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    return ""


def item_ordinal(screen: Screen) -> Optional[Tuple[int, int]]:
    """``(position, total)`` when the app states it, e.g. "View photo, 3 of 15"."""
    for el in _chrome_elements(screen):
        match = _ORDINAL_RE.search(" ".join((el.best_text or "").split()))
        if match:
            pos, total = int(match.group(1)), int(match.group(2))
            if 0 < pos <= total <= 9999:
                return pos, total
    return None


def item_total(screen: Screen) -> int:
    """How many items the set holds, when the screen says so. 0 when unknown."""
    ordinal = item_ordinal(screen)
    if ordinal is not None:
        return ordinal[1]
    for el in screen.elements:
        match = _TOTAL_RE.search(" ".join((el.best_text or "").split()))
        if match:
            return int(match.group(1))
    return 0


def normalise_label(label: str) -> str:
    return " ".join(label.split()).lower()


# ---------------------------------------------------------------------------
# Pixel identity
# ---------------------------------------------------------------------------

#: The overlay chrome a media viewer fades in and out lives in horizontal bands
#: at the top and bottom of the item. Cropping them away is what stops "the
#: toolbar faded out" from reading as "the photo changed".
_CHROME_BAND_FRAC = 0.18


def item_pixel_hash(screen: Screen) -> Optional[int]:
    """Perceptual hash of the item's own pixels, chrome bands cropped away."""
    if not screen.screenshot or not screen.width or not screen.height:
        return None
    element = pager_element(screen)
    if element is not None:
        left, top, right, bottom = element.bounds
    else:
        left, top, right, bottom = 0, 0, screen.width, screen.height
    height = bottom - top
    if height <= 0 or right - left <= 0:
        return None
    inset = height * _CHROME_BAND_FRAC
    # Expressed as fractions of the screen, because the screenshot need not share
    # the accessibility tree's pixel space.
    frac = (left / screen.width, (top + inset) / screen.height,
            right / screen.width, (bottom - inset) / screen.height)
    if frac[3] - frac[1] < 0.05 or frac[2] - frac[0] < 0.05:
        return None
    from .fingerprint import compute_dhash
    return compute_dhash(screen.screenshot, box_frac=frac)


def item_key(screen: Screen) -> str:
    """A stable key for the item on screen. Empty when nothing identifies it.

    A label is authoritative and survives a screenshot the agent did not take.
    The pixel hash is a fallback for the seconds when the app has hidden its
    chrome; it is prefixed so a caller can tell the two apart, because a pixel
    key and a label key for the *same* item do not compare equal.
    """
    label = item_label(screen)
    if label:
        return f"label:{normalise_label(label)}"
    pixels = item_pixel_hash(screen)
    if pixels is not None:
        return f"px:{pixels:016x}"
    return ""


def is_provisional(key: str) -> bool:
    """True for a pixel-derived key -- good enough to verify, not to ledger."""
    return key.startswith("px:")


def attach_item(screen: Screen) -> Screen:
    """Populate the item fields on a screen. Called from ``fingerprint.attach``."""
    screen.is_pager = is_pager_screen(screen)
    if not screen.is_pager:
        return screen
    screen.item_label = item_label(screen)
    screen.item_key = item_key(screen)
    ordinal = item_ordinal(screen)
    screen.item_position = ordinal[0] if ordinal else 0
    screen.item_total = item_total(screen)
    return screen


def set_id(screen: Screen) -> str:
    """Identity of the *set* being paged through, not of the item in it.

    Used to discard the ledger when the agent opens a different album. Built from
    the screen's identity, the set's size and whatever chrome names the set (a
    sender, an album title) with the item's own caption masked out of it.

    Returns ``""`` when the screen names no set at all -- which is exactly what a
    media viewer looks like once it has faded its toolbar out. A ledger must
    never be discarded on that: the overlay hiding itself is not the agent
    opening a different album, and treating it as one wipes the record of every
    item read so far, every few turns.
    """
    label = normalise_label(getattr(screen, "item_label", "") or "")
    candidates: List[str] = []
    for el in _chrome_elements(screen):
        text = normalise_label(el.best_text or "")
        if not text or len(text) > 60:
            continue
        if label and label in text:
            text = text.replace(label, "").strip(" ,-")
        if text and not _TOTAL_RE.search(text) and text not in _NAV_LABELS:
            candidates.append(text)
    # The longest surviving line, not the first: the first is usually a back
    # button's "Back", while a sender or an album title is the one that
    # distinguishes this set from the next one.
    owner = max(candidates, key=len) if candidates else ""
    # Computed rather than read off the screen, so this works on a set's index
    # screen too -- an album grid is not itself a pager and never had the
    # cached fields filled in.
    total = item_total(screen)
    if not owner and not total:
        return ""
    return f"{screen.package}/{screen.activity}/{total}/{owner}"


def loop_id(screen: Screen) -> str:
    """``exact_id`` refined by item identity.

    The loop detector counts repeat visits to one ``exact_id``. On a pager every
    item hashes to the same ``exact_id``, so browsing an album of fifteen photos
    looks exactly like being stuck on one screen fifteen times -- and the
    detector's remedy, a forced back press, ejects the agent from the album.
    """
    key = getattr(screen, "item_key", "")
    return f"{screen.exact_id}/{key}" if key else screen.exact_id


#: How far the cropped item hash must move to call it a different item. A
#: re-render of the same photo drifts by a bit or two; a ViewPager settling on
#: the next photo replaces the whole bitmap.
_PIXEL_DISTANCE = 6


def same_item(before: Screen, after: Screen) -> Optional[bool]:
    """Whether two pager screens show the same item. ``None`` when unknowable.

    A caption proves *difference* but never *sameness*: captions are usually
    minute-resolution timestamps, and two photos sent in the same minute carry
    the same one. (The album that motivated this module has two items labelled
    "9:33 am".) So equal captions fall through to the pixels, and only when
    neither signal is conclusive does this answer ``None``.
    """
    if not (getattr(before, "is_pager", False) or getattr(after, "is_pager", False)):
        return None

    before_label = normalise_label(getattr(before, "item_label", "") or "")
    after_label = normalise_label(getattr(after, "item_label", "") or "")

    # An ordinal ("3 of 15") is the one caption that *is* an exact identity.
    before_ord, after_ord = item_ordinal(before), item_ordinal(after)
    if before_ord is not None and after_ord is not None:
        return before_ord[0] == after_ord[0]

    if before_label and after_label and before_label != after_label:
        return False

    before_px = item_pixel_hash(before)
    after_px = item_pixel_hash(after)
    if before_px is not None and after_px is not None:
        from .fingerprint import dhash_distance
        distance = dhash_distance(before_px, after_px)
        if distance is not None:
            return distance < _PIXEL_DISTANCE
    return None


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

MAX_LEDGER_RENDER = 40
MAX_DETAIL_CHARS = 110


@dataclass
class ItemRecord:
    #: The raw `item_key` this record was minted from, before any ``#2``
    #: disambiguation suffix. Two items sharing a caption share this.
    key: str
    label: str
    first_step: int
    last_step: int
    #: True once a screenshot was taken while this item was the one on screen.
    #: Seeing an item's *caption* is not the same as having looked at the item.
    read: bool = False
    detail: str = ""
    visits: int = 1


@dataclass
class ItemLedger:
    """Which items of a set have been looked at, maintained by code not by the model.

    Keyed by ``item_key`` and ordered by first sighting. Provisional
    (pixel-derived) keys are never recorded: the same photo yields a different
    pixel key each time the chrome fades, which would inflate the count.
    """

    items: "OrderedDict[str, ItemRecord]" = field(default_factory=OrderedDict)
    #: Best known size of the set, from "15 photos" or "3 of 15".
    total: int = 0
    #: Set identity, so opening a different album starts a fresh ledger.
    set_id: str = ""
    #: Key of the item believed to be on screen right now.
    cursor: str = ""
    #: Directions in which a confirmed gesture failed to move the item. Most
    #: apps never publish how many items a set holds, so "swiping forward twice
    #: changed nothing" is the only reliable end-of-set signal there is.
    edges: set = field(default_factory=set)

    def rebase(self, set_id: str) -> None:
        if set_id and set_id != self.set_id:
            if self.items:
                log.info("pager ledger reset: new item set %s", set_id)
            self.items = OrderedDict()
            self.total = 0
            self.cursor = ""
            self.edges = set()
            self.set_id = set_id

    @property
    def complete(self) -> bool:
        """Every item accounted for: a stated total reached, or both ends hit."""
        if self.total and self.read_count >= self.total:
            return True
        return {"left", "right"} <= self.edges or (
            "left" in self.edges and self.read_count == len(self.items)
            and bool(self.items))

    def resolve(self, screen: Screen, *, moved: Optional[bool] = None) -> str:
        """The ledger key for the item on `screen`, minting one when needed.

        `moved` is what verification concluded about the last gesture: ``True``
        the item changed, ``False`` it did not, ``None`` unknown. It is the only
        way to separate two items that share a caption -- when the pixels say we
        moved but the caption is the one we already hold, this is a *second*
        item with that caption, not the same one revisited, and it earns its own
        suffixed key.
        """
        raw = item_key(screen)
        if not raw or is_provisional(raw):
            # Chrome hidden: nothing addressable. Stay where we were.
            return self.cursor
        if moved is False and self.cursor:
            return self.cursor
        current = self.items.get(self.cursor) if self.cursor else None
        if moved is True and current is not None and current.key == raw:
            base = raw
            suffix = 2
            while f"{base}#{suffix}" in self.items:
                suffix += 1
            return f"{base}#{suffix}"
        return raw

    def note(self, key: str, screen: Screen, step: int, *,
             detail: str = "", read: bool = False, label: str = "") -> bool:
        """Record a sighting and move the cursor. True when the item is new.

        `label` is a fallback for when the app's own caption is not in the tree --
        the image model can often still see one. It never overrides a caption the
        app did supply, and it is never used for identity: `key` decides that.
        """
        if not key:
            return False
        self.cursor = key
        if screen.item_total:
            self.total = max(self.total, screen.item_total)
        label = screen.item_label or " ".join(str(label or "").split())[:80] or key
        record = self.items.get(key)
        if record is None:
            occurrence = key.rsplit("#", 1)[-1] if "#" in key else ""
            shown = f"{label} (#{occurrence})" if occurrence.isdigit() else label
            self.items[key] = ItemRecord(key=item_key(screen) or key, label=shown,
                                         first_step=step, last_step=step, read=read,
                                         detail=detail.strip()[:MAX_DETAIL_CHARS])
            return True
        record.last_step = step
        record.visits += 1
        record.read = record.read or read
        if detail.strip():
            record.detail = detail.strip()[:MAX_DETAIL_CHARS]
        return False

    def was_read(self, key: str) -> bool:
        record = self.items.get(key)
        return bool(record and record.read)

    def seen(self, key: str) -> bool:
        return key in self.items

    @property
    def read_count(self) -> int:
        return sum(1 for r in self.items.values() if r.read)

    def render(self, current_key: str = "", current_label: str = "") -> str:
        """The block handed to the model. Its durable memory of the set."""
        if not self.items and not current_label:
            return ""

        counted = len(self.items)
        headline = f"ITEMS INSPECTED IN THIS SET: {self.read_count} read"
        if counted != self.read_count:
            headline += f", {counted} seen"
        if self.total:
            headline += f", out of {self.total} total"
        lines = [headline + "."]

        records = list(self.items.items())
        if len(records) > MAX_LEDGER_RENDER:
            dropped = len(records) - MAX_LEDGER_RENDER
            records = records[-MAX_LEDGER_RENDER:]
            lines.append(f"  (... {dropped} earlier item(s) omitted)")
        for position, (key, record) in enumerate(records, start=1):
            mark = "read" if record.read else "NOT READ"
            here = "  <- you are here" if key == current_key else ""
            detail = f" -- {record.detail}" if record.detail else ""
            lines.append(f"  {position}. {record.label} [{mark}]{detail}{here}")

        if current_key and current_key not in self.items:
            label = current_label or "(caption hidden)"
            lines.append(f"  -> CURRENT ITEM {label} is NEW and NOT yet read.")

        unread = [r.label for r in self.items.values() if not r.read]
        if unread:
            lines.append("STILL NOT READ: " + ", ".join(unread[:12])
                         + (" ..." if len(unread) > 12 else ""))
        if "left" in self.edges:
            lines.append("You have reached the LAST item of this set — swiping "
                         "forward again will not move.")
        if self.complete:
            lines.append("Every item in this set has been read. Do not swipe "
                         "through it again — report what you found.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweeping
# ---------------------------------------------------------------------------
#
# Walking a set is the one thing an agent does that is genuinely mechanical.
# Every turn of it asks the same question -- is there another item, and what does
# it say -- and answers it with the same gesture. In the run this module was
# written for, 71 of 127 steps were the single action `swipe #4 left`, each one
# paid for with a full reasoning turn at 26s median and 96s at the ninetieth
# percentile.
#
# So once the model has chosen to page forward and the item verifiably moved, the
# loop keeps paging in code: swipe, read the item, ledger it, repeat. The model is
# not being second-guessed -- its decision is being *repeated* for as long as the
# situation stays the one it decided about. The moment anything is not mechanical
# any more, control goes back.
#
# The gesture the sweep issues is the narrowest one available: a horizontal fling
# on the pager element, in the direction the model asked for. It never taps, types,
# navigates or presses a key, so there is no action it can take that the model did
# not already authorise.

#: Directions that page between items rather than scrolling within one.
PAGING_DIRECTIONS = frozenset({"left", "right"})


def can_sweep(screen: Screen, ledger: ItemLedger, *, action: str,
              direction: str, moved: bool) -> bool:
    """Whether a model-issued gesture authorises continuing mechanically.

    Requires all of: the gesture was a horizontal page, on a screen that is a
    pager with a flingable element, and it demonstrably moved the item. That last
    condition is what makes this safe on an unfamiliar app -- a screen where the
    swipe does nothing, or where "left" means something other than "next", never
    gets a second automatic gesture.
    """
    if action not in ("swipe", "scroll") or direction not in PAGING_DIRECTIONS:
        return False
    if not moved:
        return False
    if not getattr(screen, "is_pager", False):
        return False
    if pager_element(screen) is None:
        return False
    return not stop_sweeping(screen, ledger, direction=direction)


def stop_sweeping(screen: Screen, ledger: ItemLedger, *, direction: str,
                  package: str = "") -> str:
    """Why the sweep should hand back, or ``""`` to keep going.

    Ordered by how far the situation has departed from "paging through a set":
    left the set entirely, set finished, or merely lost the caption -- the last of
    which is a pause rather than an ending, since the model's own instructions
    tell it to tap the item to bring the title bar back.
    """
    if package and screen.package and screen.package != package:
        return f"the foreground app changed to {screen.package}"
    if not getattr(screen, "is_pager", False):
        return "this is no longer a carousel"
    if pager_element(screen) is None:
        return "the pager element is no longer on screen"
    if ledger.complete:
        return "every item in the set has been read"
    if direction in ledger.edges:
        return f"the {direction} end of the set has been reached"
    if not screen.item_label:
        # A pixel-derived key is not ledgerable -- the crop moves with the
        # chrome, so the same photo hashes differently once the toolbar goes.
        # Sweeping on would advance through items the ledger cannot name.
        return "the item caption is hidden, so items cannot be told apart"
    return ""


def sweep_summary(first_step: int, last_step: int, direction: str,
                  swept: int, read: int, reason: str) -> str:
    """One history line for a whole sweep.

    A line per gesture would be twelve near-identical entries pushing the rest of
    the history out of the prompt, to say something the ledger block already says
    per item and in more detail.
    """
    span = f"{first_step}" if first_step == last_step else f"{first_step}-{last_step}"
    return (f"{span}. swept {swept} item(s) {direction} through the carousel, "
            f"read {read} — stopped because {reason}")

