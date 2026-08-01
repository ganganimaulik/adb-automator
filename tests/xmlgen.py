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
