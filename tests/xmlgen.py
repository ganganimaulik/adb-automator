"""Build realistic uiautomator2 3.7.0 dumps for tests.

The attribute set and emission order match what the on-device dumper actually
produces, so fixtures exercise the same parsing path as a real device.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from xml.sax.saxutils import quoteattr

PKG = "com.android.settings"

_BOOL_ATTRS = (
    "checkable", "checked", "clickable", "enabled", "focusable", "focused",
    "scrollable", "long-clickable", "password", "selected", "visible-to-user",
)

_DEFAULTS: Dict[str, Any] = {
    "checkable": False, "checked": False, "clickable": False, "enabled": True,
    "focusable": False, "focused": False, "scrollable": False,
    "long-clickable": False, "password": False, "selected": False,
    "visible-to-user": True,
}


class N:
    """A node in the fixture tree."""

    def __init__(self, cls: str, bounds, text: str = "", rid: str = "",
                 desc: str = "", package: str = PKG, hint: str = "",
                 children: Optional[List["N"]] = None, **flags: bool):
        self.cls = cls
        self.bounds = tuple(bounds)
        self.text = text
        self.rid = rid
        self.desc = desc
        self.package = package
        self.hint = hint
        self.children = children or []
        self.flags = dict(_DEFAULTS)
        for key, value in flags.items():
            self.flags[key.replace("_", "-")] = value

    def copy(self, **changes) -> "N":
        clone = N(self.cls, self.bounds, self.text, self.rid, self.desc,
                  self.package, self.hint,
                  [c.copy() for c in self.children])
        clone.flags = dict(self.flags)
        for key, value in changes.items():
            key = key.replace("_", "-")
            if key in clone.flags:
                clone.flags[key] = value
            elif key == "bounds":
                clone.bounds = tuple(value)
            elif key == "children":
                clone.children = value
            else:
                setattr(clone, key, value)
        return clone

    def scale(self, factor: float) -> "N":
        clone = self.copy()
        clone.bounds = tuple(int(round(v * factor)) for v in self.bounds)
        clone.children = [c.scale(factor) for c in self.children]
        return clone

    def shift(self, dx: int = 0, dy: int = 0) -> "N":
        l, t, r, b = self.bounds
        clone = self.copy()
        clone.bounds = (l + dx, t + dy, r + dx, b + dy)
        clone.children = [c.shift(dx, dy) for c in self.children]
        return clone

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def to_xml(self, index: int = 0, depth: int = 0) -> str:
        l, t, r, b = self.bounds
        attrs = [
            ("index", str(index)),
            ("text", self.text),
            ("resource-id", f"{self.package}:id/{self.rid}" if self.rid else ""),
            ("class", self.cls),
            ("package", self.package),
            ("content-desc", self.desc),
        ]
        attrs += [(name, "true" if self.flags[name] else "false")
                  for name in _BOOL_ATTRS]
        attrs += [
            ("bounds", f"[{l},{t}][{r},{b}]"),
            ("drawing-order", str(index + 1)),
            ("hint", self.hint),
            ("display-id", "0"),
        ]
        rendered = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs)
        pad = "  " * (depth + 1)
        if not self.children:
            return f"{pad}<node {rendered} />"
        inner = "\n".join(c.to_xml(i, depth + 1)
                          for i, c in enumerate(self.children))
        return f"{pad}<node {rendered}>\n{inner}\n{pad}</node>"


def dump(*roots: N, rotation: int = 0) -> str:
    """Wrap window roots in a <hierarchy>, as the real dumper does."""
    body = "\n".join(r.to_xml(i) for i, r in enumerate(roots))
    return ("<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>\n"
            f"<hierarchy rotation=\"{rotation}\">\n{body}\n</hierarchy>")


# ---------------------------------------------------------------------------
# A Settings-like screen: status bar, toolbar, scrolling list of toggle rows
# ---------------------------------------------------------------------------

W, H = 1080, 2340
ROW_H = 160
LIST_TOP = 500


def status_bar(clock: str = "9:41", battery: str = "84%",
               badge: str = "2") -> N:
    return N("android.widget.LinearLayout", (0, 0, W, 80), package="com.android.systemui",
             rid="status_bar", children=[
                 N("android.widget.TextView", (24, 20, 200, 60), text=clock,
                   rid="clock", package="com.android.systemui"),
                 N("android.widget.TextView", (820, 20, 900, 60), text=badge,
                   rid="notification_count", package="com.android.systemui"),
                 N("android.widget.TextView", (940, 20, 1056, 60), text=battery,
                   rid="battery_percent", package="com.android.systemui"),
             ])


def toolbar(title: str = "Network & internet",
            button_cls: str = "androidx.appcompat.widget.AppCompatButton") -> N:
    return N("android.widget.FrameLayout", (0, 80, W, 300), rid="action_bar", children=[
        N("androidx.appcompat.widget.AppCompatImageButton", (24, 130, 144, 250),
          desc="Navigate up", rid="up", clickable=True, focusable=True),
        N("android.widget.TextView", (180, 150, 800, 230), text=title,
          rid="action_bar_title"),
        # `button_cls` is swapped by the "library upgrade" mutation, so it must
        # be a like-for-like text button, not the image button above.
        N(button_cls, (860, 140, 1040, 240), text="Done", rid="done_button",
          clickable=True, focusable=True),
    ])


def tab_bar(selected: str = "network") -> N:
    tabs = [("network", "Network"), ("devices", "Devices"), ("apps", "Apps")]
    kids = []
    for i, (rid, label) in enumerate(tabs):
        left = 40 + i * 340
        kids.append(N("android.widget.TextView", (left, 320, left + 300, 460),
                      text=label, rid=f"tab_{rid}", clickable=True,
                      focusable=True, selected=(rid == selected)))
    return N("android.widget.LinearLayout", (0, 300, W, 480), rid="tab_bar",
             children=kids)


def row(i: int, label: str, checked: bool = False, top: Optional[int] = None) -> N:
    t = LIST_TOP + i * ROW_H if top is None else top
    return N("android.widget.LinearLayout", (0, t, W, t + ROW_H), rid="row_item",
             clickable=True, focusable=True, children=[
                 N("android.widget.TextView", (48, t + 40, 700, t + 120),
                   text=label, rid="title"),
                 N("androidx.appcompat.widget.SwitchCompat",
                   (880, t + 50, 1030, t + 110), rid="switch_widget",
                   checkable=True, checked=checked, clickable=True, focusable=True),
             ])


def settings_screen(rows: int = 7, title: str = "Network & internet",
                    clock: str = "9:41", battery: str = "84%", badge: str = "2",
                    checked_row: int = -1, labels: Optional[List[str]] = None,
                    scroll: int = 0, selected_tab: str = "network",
                    button_cls: str = "androidx.appcompat.widget.AppCompatButton",
                    rotation: int = 0, extra_roots: Optional[List[N]] = None,
                    keyboard: bool = False) -> str:
    names = labels or [f"Option {i + 1}" for i in range(rows)]
    kids = [row(i, names[i % len(names)], checked=(i == checked_row))
            for i in range(rows)]
    if scroll:
        kids = [k.shift(dy=-scroll) for k in kids]
        kids = [k for k in kids if k.bounds[3] > LIST_TOP]
    recycler = N("androidx.recyclerview.widget.RecyclerView", (0, LIST_TOP, W, H),
                 rid="recycler_view", scrollable=True, children=kids)

    roots = [N("android.widget.FrameLayout", (0, 0, W, H), rid="content", children=[
        status_bar(clock, battery, badge),
        toolbar(title, button_cls),
        tab_bar(selected_tab),
        recycler,
    ])]
    if keyboard:
        roots.append(N("android.widget.FrameLayout", (0, 1400, W, H),
                       package="com.google.android.inputmethod.latin",
                       rid="keyboard_view", children=[
                           N("android.widget.Button", (0, 1450, 120, 1560),
                             text="q", package="com.google.android.inputmethod.latin",
                             clickable=True),
                       ]))
    roots.extend(extra_roots or [])
    return dump(*roots, rotation=rotation)


def permission_dialog() -> N:
    pkg = "com.google.android.permissioncontroller"
    return N("android.widget.FrameLayout", (60, 700, 1020, 1500), package=pkg,
             rid="content_container", children=[
                 N("android.widget.TextView", (100, 760, 980, 900), package=pkg,
                   text="Allow Settings to access your location?",
                   rid="permission_message"),
                 N("android.widget.Button", (620, 1300, 980, 1440), package=pkg,
                   text="Allow", rid="permission_allow_button",
                   clickable=True, focusable=True),
                 N("android.widget.Button", (200, 1300, 560, 1440), package=pkg,
                   text="Deny", rid="permission_deny_button",
                   clickable=True, focusable=True),
             ])


def detail_screen() -> str:
    """A structurally different screen in the same app."""
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content", children=[
        status_bar(),
        toolbar("Wi-Fi details"),
        N("android.widget.TextView", (48, 520, 1030, 640), text="Connected",
          rid="connection_state"),
        N("android.widget.TextView", (48, 680, 1030, 800), text="Signal strength",
          rid="signal_label"),
        N("android.widget.Button", (48, 900, 520, 1040), text="Forget network",
          rid="forget_button", clickable=True, focusable=True),
        N("android.widget.Button", (560, 900, 1030, 1040), text="Disconnect",
          rid="disconnect_button", clickable=True, focusable=True),
    ]))


def webview_screen() -> str:
    """A degenerate tree: one opaque WebView node, no children."""
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content", children=[
        status_bar(),
        N("android.webkit.WebView", (0, 80, W, H), rid="web_content",
          package="com.android.chrome"),
    ]))


# ---------------------------------------------------------------------------
# Horizontal scroll screen: a carousel of cards inside an HorizontalScrollView
# ---------------------------------------------------------------------------

CARD_W = 320
CARD_GAP = 24
CAROUSEL_TOP = 500
CAROUSEL_H = 400


def horizontal_card(i: int, label: str, scroll: int = 0) -> N:
    left = CARD_GAP + i * (CARD_W + CARD_GAP) - scroll
    return N("android.widget.FrameLayout", (left, CAROUSEL_TOP + 20,
             left + CARD_W, CAROUSEL_TOP + CAROUSEL_H - 20),
             rid="card_item", clickable=True, focusable=True, children=[
                 N("android.widget.TextView",
                   (left + 20, CAROUSEL_TOP + 40, left + CARD_W - 20,
                    CAROUSEL_TOP + 120),
                   text=label, rid="card_title"),
             ])


def horizontal_scroll_screen(cards: int = 5, scroll: int = 0,
                              labels: Optional[List[str]] = None,
                              clock: str = "9:41") -> str:
    """A screen with a horizontal carousel of cards."""
    names = labels or [f"Card {i + 1}" for i in range(cards)]
    kids = [horizontal_card(i, names[i % len(names)], scroll=scroll)
            for i in range(cards)]
    # Filter out cards that scrolled entirely off the left edge.
    kids = [k for k in kids if k.bounds[2] > 0]
    carousel = N("android.widget.HorizontalScrollView",
                 (0, CAROUSEL_TOP, W, CAROUSEL_TOP + CAROUSEL_H),
                 rid="carousel", scrollable=True, children=kids)

    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  children=[
                      status_bar(clock),
                      toolbar("Gallery"),
                      carousel,
                  ]))


# ---------------------------------------------------------------------------
# A WhatsApp-style media viewer: one photo of an album, full-bleed pager
#
# Traced from a real run (runs/af76720d05c4). Two shapes matter, because the
# app fades its overlay chrome out after a few seconds and the tree collapses:
# with chrome the photo's timestamp is addressable, without it nothing is.
# ---------------------------------------------------------------------------

WA = "com.whatsapp"
MEDIA_ACTIVITY = ".mediaview.MediaViewActivity"
ALBUM_ACTIVITY = ".conversation.conversationrow.album.MediaAlbumActivity"


def nav_bar() -> N:
    return N("android.widget.FrameLayout", (0, H - 120, W, H),
             package="com.android.systemui", children=[
                 N("android.widget.ImageButton", (60, H - 110, 180, H - 10),
                   desc="Back", rid="back", package="com.android.systemui",
                   clickable=True),
                 N("android.widget.ImageButton", (480, H - 110, 600, H - 10),
                   desc="Home", rid="home", package="com.android.systemui",
                   clickable=True),
                 N("android.widget.ImageButton", (900, H - 110, 1020, H - 10),
                   desc="Overview", rid="recent_apps",
                   package="com.android.systemui", clickable=True),
             ])


def mid_launch(clock: str = "7:05") -> str:
    """The dump a phone returns while one app's window replaces another's.

    Only the two windows that are always there. Reproduces the tree that
    ``runs/71295f360ea5`` handed the model while WhatsApp was on screen; see
    `Screen.chrome_only`.
    """
    return dump(status_bar(clock), nav_bar())


def system_home(clock: str = "7:05") -> str:
    """A launcher whose icons the *system UI* draws, as on some OEM builds.

    The mid-launch dump above and this one share a package and an activity, so
    only the app icons in the middle of the screen tell them apart. This fixture
    exists to keep `chrome_only` from swallowing a real home screen.
    """
    icons = [N("android.widget.TextView", (60 + i * 250, 900, 260 + i * 250, 1100),
               text=name, package="com.android.systemui", clickable=True,
               focusable=True)
             for i, name in enumerate(("Phone", "Messages", "Camera", "Chrome"))]
    return dump(status_bar(clock),
                N("android.widget.FrameLayout", (0, 80, W, H - 120),
                  rid="workspace", package="com.android.systemui", children=icons),
                nav_bar())


def media_viewer(timestamp: str = "9:33 am", sender: str = "+91 93275 84664",
                 chrome: bool = True, clock: str = "9:41") -> str:
    """One photo of an album in the full-screen viewer.

    ``chrome=False`` is the state the overlay fades to: the pager is the only
    thing left, and the photo has no caption anywhere in the tree.
    """
    # Real viewers render under a status bar, and its clock is the exact thing
    # that used to be mistaken for the photo's caption -- so it belongs in the
    # fixture. Without it this file could not reproduce the failure that cost
    # ``runs/2521862d7a23`` 45 steps.
    pager = N("androidx.viewpager.widget.ViewPager", (0, 0, W, H - 120),
              desc="Image", rid="pager", package=WA, scrollable=True)
    kids = [status_bar(clock), pager]
    if chrome:
        kids += [
            N("android.widget.LinearLayout", (0, 80, W, 260), rid="action_bar",
              package=WA, children=[
                  N("android.widget.ImageButton", (24, 120, 144, 240),
                    desc="Back", package=WA, clickable=True),
                  N("android.widget.Button", (170, 110, 700, 250),
                    text=f"{sender} Today, {timestamp}", rid="title_holder",
                    package=WA, clickable=True),
                  N("android.widget.Button", (720, 110, 850, 250), text="Save",
                    package=WA, clickable=True),
                  N("android.widget.ImageButton", (900, 110, 1020, 250),
                    desc="More options", rid="menuitem_overflow", package=WA,
                    clickable=True),
              ]),
            N("android.widget.LinearLayout", (0, H - 400, W, H - 140), package=WA,
              children=[
                  N("android.widget.Button", (40, H - 380, 400, H - 260),
                    text="Reply", rid="quick_reactions_reply_container",
                    package=WA, clickable=True),
                  N("android.widget.Button", (440, H - 380, 560, H - 260),
                    text="❤️", rid="quick_reaction_emoji_1",
                    package=WA, clickable=True),
              ]),
        ]
    kids.append(nav_bar())
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  package=WA, children=kids))


def media_album(header: str = "9:30 am", total: str = "15 photos",
                sender: str = "+91 93275 84664") -> str:
    """The album grid, which publishes only its first two tiles to the tree."""
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  package=WA, children=[
                      status_bar(),
                      N("android.widget.LinearLayout", (0, 80, W, 260),
                        rid="action_bar", package=WA, children=[
                            N("android.widget.ImageButton", (24, 120, 144, 240),
                              desc="Back", package=WA, clickable=True),
                            N("android.widget.TextView", (170, 110, 700, 180),
                              text=sender, package=WA),
                            N("android.widget.TextView", (170, 185, 700, 250),
                              text=total, package=WA),
                        ]),
                      N("androidx.recyclerview.widget.RecyclerView",
                        (0, 260, W, H - 120), rid="list", package=WA,
                        scrollable=True, children=[
                            N("android.widget.Button", (40, 280, 300, 340),
                              text=header, package=WA, clickable=True),
                            N("android.widget.Button", (0, 360, W, 1200),
                              desc="Enlarge photo", rid="image", package=WA,
                              clickable=True),
                            N("android.widget.Button", (0, 1220, W, 2060),
                              desc="Enlarge photo", rid="image", package=WA,
                              clickable=True),
                        ]),
                      nav_bar(),
                  ]))


#: A short-video feed: a vertical ViewPager2 nested under a tab strip, no
#: caption anywhere, exactly the shape the old carousel model could not read.
#: The tab pager is *larger* than the content pager, which is how the old
#: "largest full-bleed horizontal scroller" rule picked the navigation.
IG = "com.instagram.android"
REELS_ACTIVITY = ".activity.MainTabActivity"


def video_feed(author: str = "lzlift", liked: bool = False) -> str:
    """One item of an endless vertical feed. No caption, no total, no ends."""
    content = N("androidx.viewpager2.widget.ViewPager2", (0, 80, W, H - 200),
                rid="clips_viewer_view_pager", package=IG, scrollable=True,
                children=[
                    N("android.widget.FrameLayout", (0, 80, W, H - 200),
                      desc=f"Reel by {author}. Double tap to play or pause.",
                      rid="clips_video_container", package=IG, clickable=True),
                    N("android.widget.ImageView", (960, 1400, 1060, 1500),
                      desc="Like", rid="like_button", package=IG,
                      clickable=True, selected=liked),
                ])
    tabs = N("androidx.viewpager.widget.ViewPager", (0, 0, W, H),
             rid="swipeable_tab_view_pager", package=IG, scrollable=True,
             children=[content])
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  package=IG, children=[status_bar(), tabs, nav_bar()]))


# ---------------------------------------------------------------------------
# A direct-message thread: header with the correspondent's name, a scrolling
# list of bubbles, a composer and a Send control.
# ---------------------------------------------------------------------------

def chat_header(title: str = "khushi", title_rid: str = "thread_title") -> N:
    """Back arrow, avatar, name, call buttons -- the usual chat top bar."""
    return N("android.widget.LinearLayout", (0, 80, W, 300), rid="thread_toolbar",
             package=IG, children=[
                 N("android.widget.ImageButton", (16, 130, 136, 250),
                   desc="Back", rid="back", package=IG,
                   clickable=True, focusable=True),
                 N("android.widget.ImageView", (150, 130, 270, 250),
                   desc="Profile picture", rid="avatar", package=IG),
                 N("android.widget.TextView", (290, 150, 700, 230), text=title,
                   rid=title_rid, package=IG),
                 N("android.widget.ImageButton", (820, 130, 940, 250),
                   desc="Audio call", rid="call", package=IG, clickable=True),
                 N("android.widget.ImageButton", (950, 130, 1070, 250),
                   desc="Video call", rid="video_call", package=IG, clickable=True),
             ])


def chat_thread(title: str = "khushi",
                messages: Optional[List[str]] = None,
                draft: str = "",
                stamp: str = "2m",
                composer_focused: bool = False,
                title_rid: str = "thread_title",
                with_send: bool = True,
                with_header: bool = True) -> str:
    """One open conversation.

    `stamp` is a relative timestamp rendered beside the last bubble -- the thing
    that moves on its own and must not read as a new message.
    """
    texts = messages if messages is not None else ["hey", "you around?"]
    bubbles: List[N] = []
    top = 320
    for i, msg in enumerate(texts):
        # Incoming on the left, outgoing on the right -- the extractor must not
        # need to tell them apart, and alternating here proves it does not.
        left = 40 if i % 2 == 0 else 500
        bubbles.append(N("android.widget.TextView",
                         (left, top, left + 520, top + 120), text=msg,
                         rid="message_text", package=IG))
        top += 140
    if stamp:
        bubbles.append(N("android.widget.TextView", (40, top, 300, top + 60),
                         text=stamp, rid="timestamp", package=IG))

    kids: List[N] = [status_bar()]
    if with_header:
        kids.append(chat_header(title, title_rid))
    kids.append(N("androidx.recyclerview.widget.RecyclerView",
                  (0, 300, W, 1900), rid="message_list", package=IG,
                  scrollable=True, children=bubbles))

    composer: List[N] = [
        N("android.widget.EditText", (40, 1960, 860, 2080), text=draft,
          hint="Message...", rid="composer", package=IG,
          clickable=True, focusable=True, focused=composer_focused),
    ]
    if with_send:
        composer.append(N("android.widget.Button", (880, 1960, 1040, 2080),
                          text="Send", rid="send_button", package=IG,
                          clickable=True, focusable=True))
    kids.append(N("android.widget.LinearLayout", (0, 1940, W, 2100),
                  rid="composer_row", package=IG, children=composer))

    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  package=IG, children=kids))


def chat_thread_nested(title: str = "khushi",
                       messages: Optional[List[str]] = None,
                       stamp: str = "2m") -> str:
    """A thread wrapped in a full-screen outer pager, as Instagram draws it.

    Observed live: on `com.instagram.android` the largest scrollable is
    `swipeable_tab_view_pager`, which spans the whole window and contains the
    thread header as well as the message list. A "largest scroller is the message
    list" heuristic reads the header as part of the conversation here and finds no
    title above it, so every send would be refused.
    """
    texts = messages if messages is not None else ["hey", "you around?"]
    bubbles: List[N] = []
    top = 320
    for i, msg in enumerate(texts):
        left = 40 if i % 2 == 0 else 500
        bubbles.append(N("android.widget.TextView",
                         (left, top, left + 520, top + 120), text=msg,
                         rid="message_text", package=IG))
        top += 140
    if stamp:
        bubbles.append(N("android.widget.TextView", (40, top, 300, top + 60),
                         text=stamp, rid="timestamp", package=IG))

    inner = N("androidx.recyclerview.widget.RecyclerView", (0, 300, W, 1900),
              rid="message_list", package=IG, scrollable=True, children=bubbles)
    composer = N("android.widget.LinearLayout", (0, 1940, W, 2100),
                 rid="composer_row", package=IG, children=[
                     N("android.widget.EditText", (40, 1960, 860, 2080),
                       hint="Message...", rid="composer", package=IG,
                       clickable=True, focusable=True),
                     N("android.widget.Button", (880, 1960, 1040, 2080),
                       text="Send", rid="send_button", package=IG,
                       clickable=True, focusable=True),
                 ])
    # The outer pager is taller and larger than the message list, and holds the
    # header inside it.
    pager = N("androidx.viewpager.widget.ViewPager", (0, 0, W, H),
              rid="swipeable_tab_view_pager", package=IG, scrollable=True,
              children=[chat_header(title), inner, composer])
    return dump(N("android.widget.FrameLayout", (0, 0, W, H), rid="content",
                  package=IG, children=[status_bar(), pager]))
