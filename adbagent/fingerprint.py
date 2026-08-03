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
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .screen import Element, Screen

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
MAX_DEPTH = 12
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
    # Rotation belongs in identity: a landscape layout is a different layout, and
    # an anchor learned in portrait would resolve to the wrong place.
    return _digest(app_key(screen), str(screen.rotation), "\x1f".join(toks))


def exact_id(screen: Screen) -> str:
    """Includes all text. Used for change detection and loop detection only."""
    parts: List[str] = [app_key(screen), str(screen.rotation)]
    for el in screen.elements:
        parts.append("|".join((
            class_eq(el.cls),
            el.resource_id,
            mask_text(el.text),
            mask_text(el.content_desc),
            "1" if el.checked else "0",
            "1" if el.selected else "0",
            "1" if el.enabled else "0",
            str(el.bounds),
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
    for el in screen.elements:
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

def compute_dhash(image_bytes: bytes, hash_size: int = 8) -> Optional[int]:
    """Compute 64-bit difference hash (dHash) for an image.

    Fast perceptual hash for detecting visual screen changes.
    """
    if not image_bytes:
        return None
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
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


def dhash_distance(h1: Optional[int], h2: Optional[int]) -> Optional[int]:
    """Compute Hamming distance between two 64-bit dHash integers."""
    if h1 is None or h2 is None:
        return None
    return bin(h1 ^ h2).count("1")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def attach(screen: Screen) -> Screen:
    """Compute and attach every fingerprint level. Mutates and returns."""
    tokens = skeleton_tokens(screen)
    screen.tokens = tokens
    screen.skeleton_id = skeleton_id(screen, tokens)
    screen.simhash = simhash(screen)
    screen.exact_id = exact_id(screen)
    if screen.screenshot and screen.dhash is None:
        screen.dhash = compute_dhash(screen.screenshot)
    return screen

