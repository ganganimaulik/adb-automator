"""Screen fingerprinting: the ladder that decides what "the same screen" means.

Four levels, all computed from a single dump with no extra device round trips:

    app_key      package (+ activity when known)          coarse bucket
    skeleton_id  content-free, geometry-quantised hash    BUCKET KEY (exact match)
    simhash64    weighted SimHash incl. masked chrome     intra-bucket distance
    exact_id     everything, including all text           change + loop detection

Getting the normalisation right is the whole game. Too strict and the cache
never hits, because a clock tick or one new list row changes the hash. Too loose
and it fires on the wrong screen and taps the wrong thing.

Three rules do most of the work:

* ``checked``/``selected``/``focused`` are excluded from identity. They are
  *state* that flips within one logical screen -- if ``checked`` were part of
  the hash you could never cache "flip this toggle", because the post-tap screen
  would be a different screen.
* Vertical position is quantised only for nodes *outside* a scroller. Inside a
  scroller the position is replaced by a first/middle/last ordinal, so a
  half-scrolled list hashes the same as an unscrolled one while the toolbar,
  tabs and bottom nav -- the chrome that actually identifies the screen -- stay
  precisely placed.
* Identical tokens are capped at three copies, so a 7-row list and a 30-row list
  hash identically. This is the single largest contributor to hit rate.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .screen import Element, Screen, zone

log = logging.getLogger("adbagent.fingerprint")

HASH_BYTES = 8  # 64-bit, hex-encoded

# ---------------------------------------------------------------------------
# Class equivalence
# ---------------------------------------------------------------------------

_CLASS_PREFIXES = (
    "android.widget.", "android.view.", "android.webkit.", "android.app.",
    "androidx.appcompat.widget.", "androidx.recyclerview.widget.",
    "androidx.viewpager.widget.", "androidx.viewpager2.widget.",
    "androidx.cardview.widget.", "androidx.core.widget.",
    "androidx.constraintlayout.widget.",
    "com.google.android.material.",
)

#: suffix -> canonical kind. Checked longest-suffix-first.
_CLASS_EQUIV: Tuple[Tuple[str, str], ...] = (
    ("TextInputEditText", "Input"),
    ("AutoCompleteTextView", "Input"),
    ("EditText", "Input"),
    ("SearchView", "Input"),
    ("ToggleButton", "Toggle"),
    ("SwitchCompat", "Toggle"),
    ("SwitchMaterial", "Toggle"),
    ("Switch", "Toggle"),
    ("CheckBox", "Toggle"),
    ("RadioButton", "Toggle"),
    ("CheckedTextView", "Toggle"),
    ("ImageButton", "ImageButton"),
    ("ImageView", "Image"),
    ("MaterialButton", "Button"),
    ("Button", "Button"),
    ("RecyclerView", "Scroller"),
    ("NestedScrollView", "Scroller"),
    ("HorizontalScrollView", "Scroller"),
    ("ScrollView", "Scroller"),
    ("ListView", "Scroller"),
    ("GridView", "Scroller"),
    ("ViewPager2", "Scroller"),
    ("ViewPager", "Scroller"),
    ("TextView", "TextView"),
    ("WebView", "WebView"),
)


def class_eq(cls: str) -> str:
    """Map a concrete widget class onto a stable equivalence class.

    OEM themes and library upgrades swap ``AppCompatButton`` for
    ``MaterialButton`` without changing the screen, so the raw class name is too
    brittle to hash.
    """
    if not cls:
        return ""
    name = cls
    for prefix in _CLASS_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix, canonical in _CLASS_EQUIV:
        if name.endswith(suffix):
            return canonical
    # Inner classes: Foo$Bar -> Bar. Otherwise keep the last dotted segment.
    name = name.rsplit("$", 1)[-1]
    return name.rsplit(".", 1)[-1]


_RID_DIGIT_TAIL = re.compile(r"(?:_+\d+|\d+)$")


def rid_norm(resource_id: str) -> str:
    """``row_item_3`` -> ``row_item_#``. Generated ids differ per instance."""
    if not resource_id:
        return ""
    stripped = _RID_DIGIT_TAIL.sub("", resource_id)
    return f"{stripped}#" if stripped != resource_id else resource_id


# ---------------------------------------------------------------------------
# Text masking
# ---------------------------------------------------------------------------

_MASKS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "<email>"),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?", re.I), "<time>"),
    (re.compile(r"[$€£¥₹]\s?\d[\d.,]*"), "<money>"),
    (re.compile(r"\b\d+(?:[.,]\d+)?\s*%"), "<pct>"),
    (re.compile(r"\b\d+\s*(?:s|m|h|d|w|mo|y|sec|min|hour|day|week|month|year)s?"
                r"(?:\s+ago)?\b", re.I), "<rel>"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"), "<date>"),
    (re.compile(r"^\(?\d{1,4}\)?$"), "<n>"),
    (re.compile(r"\b\d{4,}\b"), "<num>"),
)

TEXT_LIMIT = 32


def mask_text(raw: str) -> str:
    """Strip the parts of a string that change without the screen changing.

    Clocks, battery percentages, unread badges, relative timestamps and prices
    all move on their own. Masking them is what keeps a screen recognisable a
    minute later.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).strip()
    for pattern, token in _MASKS:
        s = pattern.sub(token, s)
    s = " ".join(s.split()).lower()
    return s[:TEXT_LIMIT]


def mask_goal(raw: str) -> str:
    """Like mask_text but without the 32-char truncation.

    Designed for normalising goal strings, which are full sentences and
    must not be truncated -- otherwise two goals that differ only in an
    entity name at position > 32 will collide.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).strip()
    for pattern, token in _MASKS:
        s = pattern.sub(token, s)
    s = " ".join(s.split()).lower()
    return s


def normalize_verb_polarity(goal: str) -> str:
    """Extract and normalize the primary verb polarity from a goal string.

    Returns only the polarity prefix (e.g. ``[+]``, ``[-]``) so that opposite
    actions hash differently, or an empty string when no polarity is detected.
    """
    if not goal:
        return ""

    rules = [
        (r"\b(turn on|enable|activate|switch on|set)\b", "[+]"),
        (r"\b(turn off|disable|deactivate|switch off|unset)\b", "[-]"),
        (r"\b(open|go to|navigate to|launch|start)\b", "[open]"),
        (r"\b(close|exit|leave|quit)\b", "[close]"),
        (r"\b(increase|raise|higher|up)\b", "[+adj]"),
        (r"\b(decrease|lower|reduce|down)\b", "[-adj]"),
    ]

    for pattern, prefix in rules:
        if re.search(pattern, goal, re.IGNORECASE):
            return prefix

    return ""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

GRID_X = 16
GRID_Y = 16
GRID_W = 8
#: Elements nested deeper than this contribute nothing to `skeleton_id` or
#: `simhash`. It must track `DeviceConfig.max_depth`, which is what the dumper
#: was told to keep, and it did not: this was 12 against a dumper keeping 40, so
#: there was a 28-level band of elements that were dumped, pruned, rendered to
#: the model and tappable while contributing nothing to screen identity.
#:
#: On a tree deeper than the old limit the filter did not merely coarsen the
#: hash, it emptied it. `skeleton_tokens` returned the empty tuple and
#: `skeleton_id` degenerated to a digest of (package, rotation) alone -- one
#: value for the whole app. Reproduced against this module: wrap a lone button
#: and a six-row scroller-plus-toggle in 13 layers of `FrameLayout` and the two
#: hash the same. Compose, React Native and Flutter all nest past 12 routinely,
#: and in ``runs/2ca3fe0c2e62`` (Zepto) one skeleton covered a blank loading
#: frame, the loaded home screen, the home screen under an overlay, and a
#: 108-element search grid -- which is why step 14's prompt recited "previous
#: actions on this screen" naming elements that were never on it.
#:
#: Everything keyed on `skeleton_id` inherited that: `state.visits` and the
#: novelty signal, the per-screen ban list, `LoopDetector.tried_on`,
#: `element_history_hint`, and the 24-hour cross-run `dead_end` rows.
#:
#: Depth is not what bounds the hash -- `REPEAT_CAP` is. Raising this does not
#: make the token list unbounded.
MAX_DEPTH = 40
REPEAT_CAP = 3


def _quantise(el: Element, width: int, height: int) -> Tuple[int, int]:
    cx, cy = el.center
    gx = round(cx / width * GRID_X) if width else 0
    gy = round(cy / height * GRID_Y) if height else 0
    return gx, gy


def _scroller_ordinals(elements: Sequence[Element]) -> Dict[int, Tuple[str, bool]]:
    """For each element inside a scroller, an ordinal and axis flag.

    Returns ``{id(el): (ordinal, is_horizontal)}`` where ordinal is one of
    ``f`` / ``m`` / ``l``.  Position inside a scrolling list is meaningless
    for identity -- the list scrolls -- but "is it the first or last item" is
    stable enough to keep some signal.  Horizontal scrollers sort by X,
    vertical by Y.
    """
    groups: Dict[int, List[Element]] = {}
    scroller_map: Dict[int, Element] = {}
    for el in elements:
        scroller = el.scroller()
        if scroller is None:
            continue
        sid = id(scroller)
        groups.setdefault(sid, []).append(el)
        scroller_map[sid] = scroller

    out: Dict[int, Tuple[str, bool]] = {}
    for sid, members in groups.items():
        scroller = scroller_map[sid]
        horiz = scroller.is_horizontal
        if horiz:
            ordered = sorted(members, key=lambda e: (e.bounds[0], e.bounds[1]))
        else:
            ordered = sorted(members, key=lambda e: (e.bounds[1], e.bounds[0]))
        for i, el in enumerate(ordered):
            if i == 0:
                out[id(el)] = ("f", horiz)
            elif i == len(ordered) - 1:
                out[id(el)] = ("l", horiz)
            else:
                out[id(el)] = ("m", horiz)
    return out


def _flags(el: Element) -> str:
    """Four structural bits. Deliberately excludes every stateful flag."""
    return "".join((
        "c" if el.clickable or el.long_clickable else "-",
        "e" if el.editable else "-",
        "s" if el.scrollable else "-",
        "k" if el.checkable else "-",
    ))


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def skeleton_tokens(screen: Screen) -> Tuple[str, ...]:
    """The content-free token multiset, capped and sorted."""
    width = screen.width or 1
    height = screen.height or 1
    ordinals = _scroller_ordinals(screen.elements)

    counts: Dict[str, int] = {}
    ordered: List[str] = []
    for el in screen.elements:
        if el.depth > MAX_DEPTH:
            continue
        gx, gy = _quantise(el, width, height)
        gw = round(el.width / width * GRID_W) if width else 0
        in_scroller = id(el) in ordinals
        if in_scroller:
            ordinal, is_horiz = ordinals[id(el)]
            if is_horiz:
                # Horizontal scroller: X drifts, Y is stable.
                xpos = "s" + ordinal
                ypos = str(gy)
            else:
                # Vertical scroller: Y drifts, X is stable.
                xpos = str(gx)
                ypos = "s" + ordinal
        else:
            xpos = str(gx)
            ypos = str(gy)
        # `selected` on chrome -- a tab, a segmented control -- IS the screen's
        # identity: tab A and tab B are otherwise byte-identical structures.
        # Inside a scroller it is mere row highlighting, so it is ignored there.
        flags = _flags(el) + ("S" if el.selected and not in_scroller else "-")
        token = f"{class_eq(el.cls)}|{rid_norm(el.resource_id)}|{flags}|{xpos}|{ypos}|{gw}"
        # A collapsed run of identical siblings still contributes several copies,
        # bounded by the same cap as everything else.
        for _ in range(min(el.repeat, REPEAT_CAP)):
            n = counts.get(token, 0)
            if n >= REPEAT_CAP:
                break
            counts[token] = n + 1
            ordered.append(token)
    return tuple(sorted(ordered))


#: Characters of the element key shown to the model and stored in a signature.
#: Four hex characters is 65,536 buckets against a screen of ~30 elements, and a
#: collision inside one screen is broken by the ordinal folded in below -- so the
#: length only has to make two *different* screens' keys unlikely to coincide.
KEY_CHARS = 4


def element_keys(screen: Screen) -> None:
    """Stamp `Element.key`: what an element *is*, not where it landed in the list.

    Everything the loop remembers about an element is keyed on
    `AgentAction.signature()`, and that resolved to the bare ordinal -- `tap/#13`
    -- because `Target.describe()` prefers `index` and, measured across
    ``runs/``, all 72 targets the model ever produced were index-only. An ordinal
    is a position in one dump's `prune` output, and `prune` walks the tree in
    window order, which is not stable between dumps of the same screen.

    Measured over the 105 decide prompts in ``runs/``: of the 405 resource-ids
    seen more than once within a single run, 192 (47%) appeared under more than
    one `#N`. `id=back` -- the Android back button, which never moves -- took 13
    different ordinals in ``runs/c1d57cc79d9c``, and `id=navigationView` was #1
    at step 7 and #4 at every step after it in ``runs/963a4f4ae96c``.

    So a ban earned by `tap/#4` was not applied when the same control was next
    listed as #1, and *was* applied to whatever else landed on #4. The same key
    backs `LoopDetector.attempts`, the per-screen ban list, the stall-tier refusal
    set, the pager exemption, and the 24-hour cross-run `dead_end` rows.

    The key is built from four things that already exist and none of which is app
    or language specific: the element's kind, its normalised resource-id, its
    masked text, and its `@zone`. Elements that agree on all four -- the rows of a
    list, a strip of unlabelled icons -- are then separated by their ordinal
    *among that group*, which is stable as long as the group is, and is the best
    that can be done for elements the harness genuinely cannot tell apart.
    """
    groups: Dict[str, int] = {}
    for el in screen.elements:
        base = "|".join((
            el.kind(),
            rid_norm(el.resource_id),
            mask_text(el.best_text),
            zone(el, screen.width, screen.height),
        ))
        n = groups.get(base, 0)
        groups[base] = n + 1
        el.key = _digest(base, str(n))[:KEY_CHARS]


def _digest(*parts: str) -> str:
    h = hashlib.blake2b(digest_size=HASH_BYTES)
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def app_key(screen: Screen, include_activity: bool = False) -> str:
    if include_activity and screen.activity:
        return f"{screen.package}/{screen.activity}"
    return screen.package or "?"


def skeleton_id(screen: Screen, tokens: Optional[Sequence[str]] = None) -> str:
    toks = tokens if tokens is not None else skeleton_tokens(screen)
    if not toks and screen.content_elements:
        # A skeleton with no tokens is not a screen identity -- it is a digest of
        # the package name, and every screen of that app collides on it. It must
        # never pass silently: everything keyed on `skeleton_id` (visits, bans,
        # `tried_on`, `dead_end` rows) would then be pooling unrelated screens.
        # `MAX_DEPTH` tracking the dumper's own depth is what makes this
        # unreachable in practice; this says so out loud if it ever is not.
        log.warning("screen identity is degenerate: %d content element(s) and no "
                    "skeleton tokens (deeper than MAX_DEPTH=%d?)",
                    len(screen.content_elements), MAX_DEPTH)
    # Rotation belongs in identity: a landscape layout is a different layout, and
    # an anchor learned in portrait would resolve to the wrong place.
    return _digest(app_key(screen), str(screen.rotation), "\x1f".join(toks))


def exact_id(screen: Screen) -> str:
    """Includes all text. Used for change detection.

    Reads `content_elements`, so the status bar cannot move it.
    `check_postcondition("screen_changed")` grades an action purely on this hash
    differing, so a status bar that drifts on its own -- a signal icon dropping
    a bar -- graded a tap that did nothing as a success.

    "All text" has to mean `label` as well as `text` and `content_desc`. A
    tappable list row is usually a clickable ``LinearLayout`` whose text lives
    in child ``TextView``s; `screen._absorb_labels` hoists that text onto the
    row and `prune` then drops the children, so for the commonest list in
    Android *none* of the visible text reached this hash. Two pages of a list
    could differ in every row and hash identically -- which `_scroll_changed`
    reads as "scrolling did not reveal new content", i.e. as the end of the
    list. An agent told that stops scrolling and reports the thing it was
    looking for as absent.

    Loop detection deliberately no longer reads this. It counts repeats, and a
    hash carrying every element's bounds differs on nearly every visit to a
    live screen; see `safety.LoopDetector`.
    """
    return _exact_parts(screen, grid=0)


#: How far a bound may drift between two dumps and still count as the same
#: frame, in pixels. Chosen against what the two things being told apart
#: actually do. Residual layout jitter -- a ripple finishing, a row settling, an
#: image swapping in at a rounded height -- moves an element by one or two
#: pixels. A list still under the finger moves by hundreds between samples taken
#: `settle_interval_s` (0.18s) apart, and even a slow fling covers far more than
#: this. So the gap between "noise" and "still moving" is two orders of
#: magnitude wide, and any threshold inside it works; 16 sits in the middle and
#: is smaller than the smallest tappable target Android allows (48dp).
SETTLE_GRID = 16


def settle_id(screen: Screen) -> str:
    """`exact_id` with the bounds quantised. The "has the screen stopped moving" hash.

    `exact_id` hashes `str(el.bounds)` verbatim, which is right for its own job --
    change detection wants to notice everything -- and fatal for this one. A
    single pixel of drift anywhere on the screen mints a fresh `exact_id`, and
    `Device.observe(settle=True)` returns only when two consecutive dumps agree,
    so on any screen with residual animation the comparison could never succeed.

    Measured across the nine runs in ``runs/``: 95 of roughly 100 settling
    observations logged ``screen never settled``, i.e. the loop essentially never
    converged. It exhausted its whole budget on every step, spent an extra dump
    per step doing it -- a dump is ~1.2s over wireless adb -- and then handed
    back whatever the last sample happened to be, which on a moving screen is a
    frame whose element bounds the next tap is aimed at.

    Quantising fixes both halves. Two frames that differ only by jitter now hash
    the same, so a settled screen is recognised on the second sample; two frames
    of a list still in motion still differ, because a scroll moves content by far
    more than `SETTLE_GRID`.
    """
    return _exact_parts(screen, grid=SETTLE_GRID)


def _exact_parts(screen: Screen, grid: int) -> str:
    """The element-by-element hash behind `exact_id` and `settle_id`.

    One function, because the two hashes must agree on *what* they look at.
    They differ only in how precisely a bound is read, and a copy of this loop
    that drifted from the other would make a screen settle that had not.
    """
    parts: List[str] = [app_key(screen), str(screen.rotation)]
    for el in screen.content_elements:
        bounds = (tuple(v // grid for v in el.bounds) if grid > 0
                  else el.bounds)
        parts.append("|".join((
            class_eq(el.cls),
            el.resource_id,
            mask_text(el.text),
            mask_text(el.content_desc),
            mask_text(el.label),
            "1" if el.checked else "0",
            "1" if el.selected else "0",
            "1" if el.enabled else "0",
            str(bounds),
        )))
    return _digest(*parts)


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------

SIMHASH_BITS = 64

#: weight -> which features contribute. Resource ids dominate because they are
#: the most stable thing on a screen; list *content* is excluded entirely.
_W_RID = 8
_W_ACT = 5
_W_SEL = 4
_W_TXT = 3
_W_DESC = 3
_W_GEO = 2
_W_CLS = 1


def simhash_features(screen: Screen) -> List[Tuple[str, int]]:
    width = screen.width or 1
    height = screen.height or 1
    feats: List[Tuple[str, int]] = []
    for el in screen.content_elements:
        if el.depth > MAX_DEPTH:
            continue
        ce = class_eq(el.cls)
        rid = rid_norm(el.resource_id)
        in_scroller = el.scroller() is not None

        feats.append((f"cls:{ce}", _W_CLS))
        if rid:
            feats.append((f"rid:{rid}", _W_RID))
        if el.interactive:
            feats.append((f"act:{ce}:{rid}", _W_ACT))
            gx, gy = _quantise(el, width, height)
            if not in_scroller:
                feats.append((f"geo:{gx}:{gy}:{ce}", _W_GEO))
        # `selected` IS identity -- it is what tells tab A from tab B, which are
        # otherwise structurally identical. `checked` deliberately is not: it is
        # the thing a "flip this toggle" action is about to change.
        if el.selected and not in_scroller:
            feats.append((f"sel:{rid or ce}:{mask_text(el.best_text)}", _W_SEL))
        # Chrome text only. Text inside a scroller is list *content* and must
        # never affect screen identity.
        if not in_scroller:
            t = mask_text(el.text)
            if t:
                feats.append((f"txt:{t}", _W_TXT))
            d = mask_text(el.content_desc)
            if d:
                feats.append((f"desc:{d}", _W_DESC))
    return feats


def simhash(screen: Screen) -> int:
    vector = [0] * SIMHASH_BITS
    for feature, weight in simhash_features(screen):
        h = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8", "replace"), digest_size=8).digest(),
            "big",
        )
        for bit in range(SIMHASH_BITS):
            if h >> bit & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight
    out = 0
    for bit, value in enumerate(vector):
        if value > 0:
            out |= 1 << bit
    if out >= (1 << 63):
        out -= 1 << 64
    return out


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << SIMHASH_BITS) - 1)).count("1")


# ---------------------------------------------------------------------------
# Discriminative tokens
# ---------------------------------------------------------------------------

#: Actions that are truly destructive and cannot be undone in any context.
_HARD_DESTRUCTIVE = (
    r"\b(delete|remove|uninstall|erase|wipe|factory reset|deactivate|"
    r"close account|unsubscribe|pay|buy|purchase|place order|checkout|"
    r"confirm order|"
    r"forget|sign out|log out|revoke|discard|unfollow|unfriend|block|"
    r"leave|archive|clear (?:data|history|all))\b"
)

#: Chat send actions — destructive by default, but whitelisted in chat_mode.
CHAT_SEND_TEXT = re.compile(
    r"\b(send|post|publish|share)\b", re.I,
)

#: Tokens that must never appear on a screen we replay blind, whatever the
#: statistics say. Matched against masked element text, not against the skeleton.
#: This is the union of hard-destructive + chat-send patterns.
DESTRUCTIVE_TEXT = re.compile(
    _HARD_DESTRUCTIVE + r"|" + r"\b(send|post|publish|share)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Perceptual Image Fingerprinting (dHash)
# ---------------------------------------------------------------------------

def compute_dhash(image_bytes: bytes, hash_size: int = 8,
                  box: Optional[Sequence[int]] = None,
                  box_frac: Optional[Sequence[float]] = None) -> Optional[int]:
    """Compute 64-bit difference hash (dHash) for an image.

    Fast perceptual hash for detecting visual screen changes. Pass ``box`` as
    ``(left, top, right, bottom)`` in image pixels, or ``box_frac`` as the same
    four values expressed as fractions of the image, to hash one region -- a
    gallery's image area without the toolbar that fades over it, say -- so a
    chrome animation is not mistaken for a change of content. Prefer
    ``box_frac`` when the coordinates come from the accessibility tree: a
    screenshot is not guaranteed to be in the same pixel space as the tree's
    bounds, and a pixel box computed in the wrong space silently crops to
    nothing.
    """
    if not image_bytes:
        return None
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            if box_frac is not None:
                box = (box_frac[0] * img.width, box_frac[1] * img.height,
                       box_frac[2] * img.width, box_frac[3] * img.height)
            if box is not None:
                left, top, right, bottom = (int(v) for v in box)
                left, top = max(0, left), max(0, top)
                right, bottom = min(img.width, right), min(img.height, bottom)
                if right - left < 2 or bottom - top < 2:
                    return None
                img = img.crop((left, top, right, bottom))
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            diff_bits = 0
            bit_index = 0
            for y in range(hash_size):
                row_start = y * (hash_size + 1)
                for x in range(hash_size):
                    left = pixels[row_start + x]
                    right = pixels[row_start + x + 1]
                    if left > right:
                        diff_bits |= (1 << bit_index)
                    bit_index += 1
            return diff_bits
    except Exception:
        return None


#: How much a frame's brightness may vary before it counts as having something
#: on it. Standard deviation over a 64x64 grayscale downscale.
#:
#: Measured, on the frames of ``runs/a7ef4e0e45e9`` and on drawn cases either
#: side of them: 1.07, 1.12 and 1.16 for the three frames the app had not painted
#: yet; 3.7 for a white screen carrying one low-contrast pill and its label;
#: 51.8 for the emptiest real screen in the run, a mostly-white conversation
#: list; 92.7 for an ordinary populated one. This sits between the first two
#: groups, nearer neither.
#:
#: Deviation and not a min-to-max range, because a blank frame is not uniform --
#: it still carries the status-bar clock, 46 levels of full range on those three
#: -- and not a percentile band either, which ignores small features by design
#: and so scores the lone pill exactly as it scores an empty screen. Deviation is
#: the measure that separates the two, and it needs no help: a feature covering
#: even 0.05% of the frame at full contrast lifts it past 5.
FEATURELESS_STDEV = 2.0


def frame_is_featureless(image_bytes: bytes, *,
                         stdev: float = FEATURELESS_STDEV) -> bool:
    """True when there is nothing in these pixels to read.

    A frame the app has not painted yet -- the white flash after a send, a
    launch that has not drawn -- has no content for a vision model to describe,
    and asking one anyway buys a paragraph saying so. In that run it bought
    three: 10 seconds and ~1,700 output tokens each to report "blank white
    screen; app content not rendered".

    Deliberately shy, and used only where being wrong is cheap. A false "blank"
    costs a screen description the decider would rather have had on a turn where
    it still has the element list. It must never be the answer to "where is this
    control": that is the one question whose wrong answer gets tapped, and the
    threshold above is the reason -- one control on an empty screen is the case
    this cannot see. So `analyze_image` consults it and `locate` does not.

    False when the frame cannot be decoded. Not knowing is a reason to let the
    model look, never a reason to skip it.
    """
    if not image_bytes:
        return False
    try:
        import io
        import statistics
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            small = img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            pixels = list(small.getdata())
    except Exception:
        return False
    if len(pixels) < 20:
        return False
    return statistics.pstdev(pixels) <= stdev


def dhash_distance(h1: Optional[int], h2: Optional[int]) -> Optional[int]:
    """Compute Hamming distance between two 64-bit dHash integers."""
    if h1 is None or h2 is None:
        return None
    return bin(h1 ^ h2).count("1")


def crop_frac(image_bytes: bytes, box_frac: Sequence[float],
              quality: int = 88) -> Optional[bytes]:
    """Re-encode `image_bytes` keeping only the region `box_frac` names.

    Fractions for the same reason `compute_dhash` takes them: the screenshot need
    not share the accessibility tree's pixel space, and a pixel box computed in
    the wrong one crops to nothing.

    Returns None rather than raising, and never returns the uncropped frame as a
    consolation -- a caller that cropped to exclude something must not be handed
    it back silently. The caller decides what to do with nothing.
    """
    if not image_bytes or box_frac is None:
        return None
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            left, top, right, bottom = (
                int(box_frac[0] * img.width), int(box_frac[1] * img.height),
                int(box_frac[2] * img.width), int(box_frac[3] * img.height))
            left, top = max(0, left), max(0, top)
            right, bottom = min(img.width, right), min(img.height, bottom)
            if right - left < 2 or bottom - top < 2:
                return None
            out = io.BytesIO()
            img.crop((left, top, right, bottom)).convert("RGB").save(
                out, "JPEG", quality=quality, optimize=True)
            return out.getvalue()
    except Exception:  # noqa: BLE001 - a crop is an improvement, never a step
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def attach(screen: Screen) -> Screen:
    """Compute and attach every fingerprint level. Mutates and returns."""
    element_keys(screen)
    tokens = skeleton_tokens(screen)
    screen.tokens = tokens
    screen.skeleton_id = skeleton_id(screen, tokens)
    screen.simhash = simhash(screen)
    screen.exact_id = exact_id(screen)
    screen.settle_id = settle_id(screen)
    if screen.screenshot and screen.dhash is None:
        screen.dhash = compute_dhash(screen.screenshot)
    # `pager.attach_item` used to run here, stamping a per-item identity onto the
    # screen. Nothing stamps one now: whether two frames differ is asked of the
    # pair, by `pager.content_moved`, not answered in advance about one of them.
    return screen

