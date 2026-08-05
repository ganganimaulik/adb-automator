"""Screen representation: XML -> Element[] -> prune -> indexed render.

Pure functions only. Nothing here touches a device or the network, so the whole
module is testable from fixture XML.

We parse the *raw* dump with lxml rather than going through
``uiautomator2.xpath.PageSource``: that class renames every ``<node>`` to its
class name and deletes the ``class`` attribute, which loses information we need
for anchoring.

Attribute set emitted by the u2 3.7.0 on-device dumper, in emission order:
    NAF index text resource-id class package content-desc
    checkable checked clickable enabled focusable focused scrollable
    long-clickable password selected visible-to-user
    bounds drawing-order(API>=24) hint(API>=26) display-id(API>=30)

Note there is no ``displayed`` attribute -- that belongs to other dumpers. Use
``visible-to-user``. And ``<hierarchy>`` carries only ``rotation``; the screen
size must come from the device.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lxml import etree

BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

_TRUE = {"true", "True", "1"}

# Classes we treat as text entry. There is no "editable" attribute in the dump.
_EDIT_SUFFIXES = ("EditText", "AutoCompleteTextView", "SearchView", "TextInputEditText")

# Known horizontal scroller class suffixes.
_HORIZONTAL_SCROLLER_SUFFIXES = (
    "HorizontalScrollView", "ViewPager2", "ViewPager",
)

# The IME renders as its own window root inside the same <hierarchy>. Its
# presence must not change a screen's identity, so it is dropped during parse.
IME_PACKAGE_HINTS = ("inputmethod", "latin", "keyboard", "swiftkey", "gboard")

#: Packages that draw the status bar, the navigation bar and system dialogs.
#: Public because `pager` needs it too: the status bar carries a clock, and a
#: clock read as an item caption renames every item once a minute.
SYSTEM_UI_PACKAGES = {
    "com.android.systemui",
    "android",
}


def _b(v: Optional[str]) -> bool:
    return v in _TRUE


def parse_bounds(raw: Optional[str]) -> Tuple[int, int, int, int]:
    if not raw:
        return (0, 0, 0, 0)
    m = BOUNDS_RE.search(raw)
    if not m:
        return (0, 0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def short_rid(raw: str) -> str:
    """``com.android.settings:id/switch_widget`` -> ``switch_widget``."""
    if not raw:
        return ""
    return raw.rsplit("/", 1)[-1]


@dataclass
class Element:
    """One node of the accessibility tree."""

    cls: str = ""
    package: str = ""
    text: str = ""
    content_desc: str = ""
    hint: str = ""
    resource_id_raw: str = ""
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    node_index: int = 0
    drawing_order: int = 0
    depth: int = 0

    checkable: bool = False
    checked: bool = False
    clickable: bool = False
    enabled: bool = True
    focusable: bool = False
    focused: bool = False
    scrollable: bool = False
    long_clickable: bool = False
    password: bool = False
    selected: bool = False
    visible: bool = True

    parent: Optional["Element"] = field(default=None, repr=False, compare=False)
    children: List["Element"] = field(default_factory=list, repr=False, compare=False)

    #: Index in the rendered list shown to the model (1-based). 0 = not rendered.
    index: int = 0
    #: Text absorbed from non-interactive descendants (see `_absorb_labels`).
    label: str = ""
    #: When this element collapses N identical siblings, how many it stands for.
    repeat: int = 1
    #: Path of child positions from the window root -- a last-resort anchor.
    path: str = ""

    # -- derived -----------------------------------------------------------

    @property
    def resource_id(self) -> str:
        return short_rid(self.resource_id_raw)

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        l, t, r, b = self.bounds
        return ((l + r) // 2, (t + b) // 2)

    @property
    def is_horizontal(self) -> bool:
        """True when this scrollable container scrolls horizontally."""
        if not self.scrollable:
            return False
        if self.cls.endswith(_HORIZONTAL_SCROLLER_SUFFIXES):
            return True
        # Heuristic: a scroller much wider than tall is probably horizontal.
        if self.width > 0 and self.height > 0 and self.width > self.height * 1.5:
            return True
        return False

    @property
    def editable(self) -> bool:
        return self.cls.endswith(_EDIT_SUFFIXES) or self.password

    @property
    def interactive(self) -> bool:
        return (self.clickable or self.long_clickable or self.checkable
                or self.scrollable or self.editable)

    @property
    def best_text(self) -> str:
        """Whatever a human would call this element."""
        return self.text or self.content_desc or self.label or self.hint

    @property
    def is_system_chrome(self) -> bool:
        """Drawn by the OS, not by the app: the status bar and the nav bar."""
        return self.package in SYSTEM_UI_PACKAGES

    def ancestors(self) -> Iterable["Element"]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def scroller(self) -> Optional["Element"]:
        """Nearest scrollable ancestor, if any."""
        for anc in self.ancestors():
            if anc.scrollable:
                return anc
        return None

    def kind(self) -> str:
        """Coarse label used both in the render and in class equivalence."""
        c = self.cls
        if self.editable:
            return "Input"
        if self.checkable or c.endswith(("Switch", "CheckBox", "ToggleButton",
                                         "RadioButton", "SwitchCompat",
                                         "SwitchMaterial")):
            return "Toggle"
        if self.scrollable:
            return "Scroller"
        if c.endswith(("ImageButton",)) or (c.endswith("ImageView") and self.clickable):
            return "ImageButton"
        if c.endswith("ImageView"):
            return "Image"
        if c.endswith("Button") or (self.clickable and self.best_text):
            return "Button"
        if self.clickable or self.long_clickable:
            return "Tappable"
        return "Text"


@dataclass
class Screen:
    """A parsed, pruned snapshot of the device screen."""

    xml: str = ""
    width: int = 0
    height: int = 0
    rotation: int = 0
    package: str = ""
    activity: str = ""
    #: Every parsed node, depth-first, IME windows excluded.
    nodes: List[Element] = field(default_factory=list)
    #: The pruned, indexed subset shown to the model.
    elements: List[Element] = field(default_factory=list)
    packages: Set[str] = field(default_factory=set)
    keyboard_open: bool = False
    #: Populated by fingerprint.attach().
    skeleton_id: str = ""
    simhash: int = 0
    exact_id: str = ""
    tokens: Tuple[str, ...] = ()
    screenshot: Optional[bytes] = None
    dhash: Optional[int] = None
    # A screen used to carry `is_pager`, `item_label`, `item_key`,
    # `item_position` and `item_total`, filled in by the pager module. They are
    # gone because none of them were properties of a screen: whether a gesture
    # pages is a property of the *gesture*, learned by trying it, and the
    # captions and totals were guesses. See `pager.py`.

    def by_index(self, i: int) -> Optional[Element]:
        for el in self.elements:
            if el.index == i:
                return el
        return None

    @property
    def actionable(self) -> List[Element]:
        return [e for e in self.elements if e.interactive]

    @property
    def content_elements(self) -> List[Element]:
        """The elements the *app* drew -- the screen minus the system chrome.

        Every heuristic that asks a question about the app's content should ask
        it of these, not of `elements`. The status bar is the reason: it sits
        outside every scroller and every app, and it changes on its own. Read as
        content it has, in this codebase alone, named a carousel item after the
        clock and graded a tap that did nothing as a success because the signal
        icon moved. Neither bug is about clocks or signal icons specifically --
        both are about asking the app a question and letting the OS answer.

        When the system UI *is* the content -- the notification shade pulled
        down, the volume panel, a system dialog with the screen to itself --
        nothing is excluded, because then it is exactly what the agent is
        working on. Same reasoning as `_dominant_package`, which is what decides
        whose screen this is.
        """
        if not self.package or self.package in SYSTEM_UI_PACKAGES:
            return list(self.elements)
        return [e for e in self.elements if not e.is_system_chrome]

    @property
    def ambiguous(self) -> bool:
        """Two or more actionable elements the model cannot tell apart."""
        seen: Set[Tuple[str, str]] = set()
        for el in self.actionable:
            if el.resource_id:
                continue
            key = (el.kind(), el.best_text.strip().lower())
            if not key[1]:
                continue
            if key in seen:
                return True
            seen.add(key)
        return False

    @property
    def degenerate(self) -> bool:
        """Too few actionable nodes to drive from XML -- WebView/Canvas/game."""
        return len(self.actionable) < 3

    @property
    def chrome_only(self) -> bool:
        """Nothing but the status bar and the nav bar: a frame of nothing.

        A dump contains the windows that exist at that instant, so one taken
        between an app's window going away and the next one's being added holds
        only the two windows that are always there. The danger is that it does
        not read as an error -- it reads as a perfectly plausible screen, with a
        clock, a battery and three nav buttons. In ``runs/71295f360ea5`` WhatsApp
        was in front and in the screenshot, the tree was this, and the model
        dutifully reported "the Android home screen is visible with the clock,
        status bar icons, and navigation buttons".

        Two conditions, so that a real system screen never matches: no app owns
        the dump, and nothing at all is drawn between the two bars. A launcher
        with icons, the notification shade and the volume panel all fill the
        middle of the screen. A mid-transition dump leaves it empty.
        """
        # An empty dump is `degenerate` -- a different signal with its own answer.
        if not self.elements or self.height <= 0:
            return False
        if self.package and self.package not in SYSTEM_UI_PACKAGES:
            return False
        top, bottom = int(self.height * 0.12), int(self.height * 0.88)
        return not any(el.bounds[3] > top and el.bounds[1] < bottom
                       for el in self.elements)

    def has_system_dialog(self) -> bool:
        target = self.package
        return any(p != target and p not in SYSTEM_UI_PACKAGES
                   for p in self.packages)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_ime_root(node) -> bool:
    pkg = (node.get("package") or "").lower()
    return any(h in pkg for h in IME_PACKAGE_HINTS)


def _element_from(node, depth: int, parent: Optional[Element], path: str) -> Element:
    return Element(
        cls=node.get("class") or "",
        package=node.get("package") or "",
        text=node.get("text") or "",
        content_desc=node.get("content-desc") or "",
        hint=node.get("hint") or "",
        resource_id_raw=node.get("resource-id") or "",
        bounds=parse_bounds(node.get("bounds")),
        node_index=int(node.get("index") or 0),
        drawing_order=int(node.get("drawing-order") or 0),
        depth=depth,
        checkable=_b(node.get("checkable")),
        checked=_b(node.get("checked")),
        clickable=_b(node.get("clickable")),
        enabled=node.get("enabled") is None or _b(node.get("enabled")),
        focusable=_b(node.get("focusable")),
        focused=_b(node.get("focused")),
        scrollable=_b(node.get("scrollable")),
        long_clickable=_b(node.get("long-clickable")),
        password=_b(node.get("password")),
        selected=_b(node.get("selected")),
        visible=node.get("visible-to-user") is None or _b(node.get("visible-to-user")),
        parent=parent,
        path=path,
    )


def parse(xml: str, width: int = 0, height: int = 0,
          activity: str = "") -> Screen:
    """Parse a raw ``dump_hierarchy()`` string into a Screen.

    ``width``/``height`` should come from ``d.window_size()``; if omitted we fall
    back to the widest/tallest bounds seen, which is right often enough for tests.
    """
    screen = Screen(xml=xml, activity=activity)
    if not xml or not xml.strip():
        return screen

    try:
        root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    except etree.XMLSyntaxError:
        # A truncated dump is better handled as "empty screen" than as a crash;
        # the caller re-observes.
        return screen

    screen.rotation = int(root.get("rotation") or 0)

    nodes: List[Element] = []

    def walk(node, parent: Optional[Element], depth: int, path: str) -> None:
        el = _element_from(node, depth, parent, path)
        if parent is not None:
            parent.children.append(el)
        nodes.append(el)
        for i, child in enumerate(node):
            if child.tag != "node":
                continue
            walk(child, el, depth + 1, f"{path}.{i}" if path else str(i))

    for i, window_root in enumerate(root):
        if window_root.tag != "node":
            continue
        if _is_ime_root(window_root):
            screen.keyboard_open = True
            continue
        walk(window_root, None, 0, f"w{i}")

    screen.nodes = nodes
    screen.packages = {n.package for n in nodes if n.package}
    screen.width = width or max((n.bounds[2] for n in nodes), default=0)
    screen.height = height or max((n.bounds[3] for n in nodes), default=0)
    screen.package = _dominant_package(nodes)

    _absorb_labels(nodes)
    screen.elements = prune(screen)
    return screen


def _dominant_package(nodes: Sequence[Element]) -> str:
    """The package owning the most screen area, ignoring system chrome."""
    area: Dict[str, int] = {}
    for n in nodes:
        if not n.package:
            continue
        area[n.package] = area.get(n.package, 0) + n.area
    if not area:
        return ""
    non_system = {p: a for p, a in area.items() if p not in SYSTEM_UI_PACKAGES}
    pool = non_system or area
    return max(pool.items(), key=lambda kv: kv[1])[0]


def _absorb_labels(nodes: Sequence[Element]) -> None:
    """Give interactive containers the text of their non-interactive children.

    A tappable row is very often a clickable ``LinearLayout`` whose only text
    lives in a child ``TextView``. Without this the model sees an unlabelled
    tappable box next to a piece of text it cannot tap.
    """
    for el in nodes:
        if not el.interactive or el.text or el.content_desc:
            continue
        parts: List[str] = []
        stack = list(el.children)
        while stack:
            child = stack.pop(0)
            if child.interactive:
                continue  # belongs to that child, not to us
            piece = (child.text or child.content_desc).strip()
            if piece:
                parts.append(piece)
            stack.extend(child.children)
        if parts:
            el.label = " ".join(parts)


def _absorbed_by_ancestor(el: Element) -> bool:
    """True when an interactive ancestor already presents this node's text."""
    piece = (el.text or el.content_desc).strip()
    if not piece:
        return False
    for anc in el.ancestors():
        if anc.interactive and piece in anc.label:
            return True
    return False


def prune(screen: Screen) -> List[Element]:
    """Reduce the tree to what the model needs, then index it."""
    kept: List[Element] = []
    for el in screen.nodes:
        if not el.visible or el.area <= 0:
            continue
        if el.interactive:
            kept.append(el)
            continue
        if (el.text or el.content_desc) and not _absorbed_by_ancestor(el):
            kept.append(el)
    kept = _collapse_identical_siblings(kept)
    for i, el in enumerate(kept, start=1):
        el.index = i
    return kept


def _sibling_key(el: Element) -> Tuple:
    return (el.cls, el.resource_id, el.text, el.content_desc,
            el.clickable, el.checkable, el.checked, el.scrollable,
            el.width, el.height)


def _collapse_identical_siblings(elements: List[Element]) -> List[Element]:
    """Fold runs of identical *unlabelled* siblings into one entry with a count.

    This catches decorative repeats -- rating stars, carousel dots, empty grid
    cells -- which otherwise flood the element list.

    Interactive elements are NEVER collapsed, even when identical. Two buttons
    both reading "Open" are indistinguishable by text, but they still need
    separate indices so the model can address the second one -- and the cost of
    getting this wrong is asymmetric: an over-long list wastes a few tokens,
    whereas a collapsed target is simply unreachable. `Screen.ambiguous` flags
    the look-alike case so a screenshot gets attached instead.
    """
    out: List[Element] = []
    for el in elements:
        prev = out[-1] if out else None
        if (prev is not None
                and not prev.interactive and not el.interactive
                and prev.parent is not None and prev.parent is el.parent
                and _sibling_key(prev) == _sibling_key(el)):
            prev.repeat += 1
            continue
        out.append(el)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _quote(s: str) -> str:
    s = " ".join(s.split())
    return f'"{s}"'


def render_element(el: Element) -> str:
    parts = [f"#{el.index}", f"[{el.kind()}]"]
    label = el.best_text
    if label:
        parts.append(_quote(label))
    if el.resource_id:
        parts.append(f"id={el.resource_id}")
    flags = []
    if el.checkable:
        flags.append(f"checked={'true' if el.checked else 'false'}")
    if el.selected:
        flags.append("selected")
    if not el.enabled:
        flags.append("disabled")
    if el.password:
        flags.append("password")
    if el.scrollable:
        flags.append("scrollable-h" if el.is_horizontal else "scrollable")
    if el.long_clickable and not el.clickable:
        flags.append("long-press-only")
    if el.repeat > 1:
        flags.append(f"x{el.repeat}")
    if flags:
        parts.append(" ".join(flags))
    return "  ".join(parts)


def render(screen: Screen, limit: int = 80) -> str:
    """The exact text handed to the model."""
    header = (f"screen {screen.width}x{screen.height} rot={screen.rotation} "
              f"app={screen.package or '?'}")
    if screen.activity:
        header += f" activity={screen.activity}"
    if screen.keyboard_open:
        header += " keyboard=open"

    lines = [header]
    shown = screen.elements[:limit]
    for el in shown:
        lines.append(render_element(el))
    hidden = len(screen.elements) - len(shown)
    if hidden > 0:
        lines.append(f"... {hidden} more elements not shown (scroll to reach them)")
    if not screen.elements:
        lines.append("(no visible elements -- the screen is a WebView, canvas or "
                     "game, or the app is still loading)")
    return "\n".join(lines)


def element_detail(el: Element) -> str:
    """Full attributes for one element -- the escape hatch from pruning."""
    l, t, r, b = el.bounds
    rows = [
        f"#{el.index} {el.cls}",
        f"  package        {el.package}",
        f"  resource-id    {el.resource_id_raw or '(none)'}",
        f"  text           {el.text!r}",
        f"  content-desc   {el.content_desc!r}",
        f"  hint           {el.hint!r}",
        f"  absorbed label {el.label!r}",
        f"  bounds         [{l},{t}][{r},{b}]  ({el.width}x{el.height})",
        f"  flags          clickable={el.clickable} long={el.long_clickable} "
        f"checkable={el.checkable} checked={el.checked} scrollable={el.scrollable} "
        f"enabled={el.enabled} selected={el.selected} password={el.password}",
        f"  depth/path     {el.depth} / {el.path}",
    ]
    scroller = el.scroller()
    if scroller is not None:
        rows.append(f"  inside scroller {scroller.resource_id or scroller.cls}")
    return "\n".join(rows)
