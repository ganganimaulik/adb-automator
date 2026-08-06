"""Parsing, pruning and rendering."""

from __future__ import annotations

import pytest

from adbagent.screen import Element, parse, render, element_detail, parse_bounds

from . import xmlgen as X


def s(xml: str):
    return parse(xml, width=X.W, height=X.H)


def test_parses_geometry_and_flags():
    scr = s(X.settings_screen())
    assert scr.width == X.W and scr.height == X.H
    assert scr.rotation == 0
    assert scr.package == "com.android.settings"
    toggles = [e for e in scr.elements if e.kind() == "Toggle"]
    assert len(toggles) == 7
    assert all(t.checkable and t.clickable for t in toggles)


def test_parse_bounds():
    assert parse_bounds("[0,80][1080,2340]") == (0, 80, 1080, 2340)
    assert parse_bounds("[-4,-8][10,20]") == (-4, -8, 10, 20)
    assert parse_bounds(None) == (0, 0, 0, 0)
    assert parse_bounds("garbage") == (0, 0, 0, 0)


def test_ime_window_is_dropped_but_flagged():
    plain, typed = s(X.settings_screen()), s(X.settings_screen(keyboard=True))
    assert not plain.keyboard_open
    assert typed.keyboard_open
    # The keyboard's own keys must not appear as tappable elements, or the same
    # logical screen would render differently with and without the IME up.
    assert not any("inputmethod" in e.package for e in typed.elements)
    assert len(plain.elements) == len(typed.elements)


def test_invisible_and_zero_area_nodes_are_pruned():
    hidden = X.N("android.widget.Button", (0, 600, 300, 700), text="Ghost",
                 rid="ghost", clickable=True, visible_to_user=False)
    empty = X.N("android.widget.Button", (500, 600, 500, 600), text="Empty",
                rid="empty", clickable=True)
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.FrameLayout", (0, 600, X.W, 700), rid="extra",
            children=[hidden, empty])]))
    labels = {e.resource_id for e in scr.elements}
    assert "ghost" not in labels
    assert "empty" not in labels


def test_label_absorption_gives_tappable_rows_their_text():
    """A clickable row whose text lives in a child TextView must be labelled."""
    scr = s(X.settings_screen(labels=["Wi-Fi"]))
    rows = [e for e in scr.elements if e.resource_id == "row_item"]
    assert rows, "the clickable row container should survive pruning"
    assert rows[0].best_text == "Wi-Fi"
    # ...and the child TextView must NOT be listed separately, or the model sees
    # the same label twice and cannot tell which one to tap.
    titles = [e for e in scr.elements
              if e.resource_id == "title" and e.text == "Wi-Fi"]
    assert titles == []


def test_decorative_repeats_collapse():
    """A display-only rating strip is one control, not five elements."""
    stars = [X.N("android.widget.ImageView", (100 + i * 80, 620, 160 + i * 80, 680),
                 rid="star", desc="star") for i in range(5)]
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.LinearLayout", (0, 600, X.W, 700), rid="rating",
            children=stars)]))
    collapsed = [e for e in scr.elements if e.resource_id == "star"]
    assert len(collapsed) == 1 and collapsed[0].repeat == 5


def test_interactive_look_alikes_are_never_collapsed():
    """An unreachable tap target is far worse than a few wasted tokens."""
    twins = [X.N("android.widget.ImageView", (100 + i * 80, 620, 160 + i * 80, 680),
                 rid="star", desc="star", clickable=True) for i in range(5)]
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.LinearLayout", (0, 600, X.W, 700), rid="rating",
            children=twins)]))
    assert len([e for e in scr.elements if e.resource_id == "star"]) == 5

    # Rows with different labels stay individually addressable too.
    rows = [e for e in s(X.settings_screen(rows=4)).elements
            if e.resource_id == "row_item"]
    assert len(rows) == 4 and all(r.repeat == 1 for r in rows)


def test_indices_are_contiguous_and_addressable():
    scr = s(X.settings_screen())
    assert [e.index for e in scr.elements] == list(range(1, len(scr.elements) + 1))
    assert scr.by_index(1) is scr.elements[0]
    assert scr.by_index(9999) is None


def test_degenerate_webview_is_flagged():
    """A WebView with no children is the case XML cannot drive."""
    scr = s(X.webview_screen())
    assert scr.degenerate
    assert not s(X.settings_screen()).degenerate


def test_a_mid_launch_dump_is_recognised_as_a_frame_of_nothing():
    """The tree that got a launched WhatsApp reported as the home screen.

    Nothing here is wrong on its face -- a clock, a battery, three nav buttons --
    which is why it has to be caught structurally rather than described.
    """
    blank = s(X.mid_launch())
    assert blank.chrome_only
    assert blank.package == "com.android.systemui"

    # The real screens it must not be confused with: an app, and a home screen
    # the system UI itself draws (same package, same activity, icons in between).
    assert not s(X.settings_screen()).chrome_only
    assert not s(X.system_home()).chrome_only
    assert not s(X.media_viewer(chrome=False)).chrome_only
    # An empty dump is `degenerate`; answering "chrome only" as well would send
    # `observe` re-dumping a screen that is not going to change.
    assert not s("").chrome_only


def test_ambiguity_detection():
    twins = [X.N("android.widget.Button", (100, 620 + i * 120, 400, 720 + i * 120),
                 text="Open", clickable=True) for i in range(2)]
    scr = s(X.settings_screen(extra_roots=[
        X.N("android.widget.LinearLayout", (0, 600, X.W, 900), rid="twins",
            children=twins)]))
    assert scr.ambiguous
    assert not s(X.settings_screen()).ambiguous


def test_system_dialog_detection():
    assert s(X.settings_screen(extra_roots=[X.permission_dialog()])).has_system_dialog()
    assert not s(X.settings_screen()).has_system_dialog()


def test_render_is_compact_and_indexed():
    scr = s(X.settings_screen())
    out = render(scr)
    lines = out.splitlines()
    assert lines[0].startswith("screen 1080x2340 rot=0 app=com.android.settings")
    assert any(line.startswith("#1 ") for line in lines)
    assert "checked=false" in out
    # The whole point of pruning: a fraction of the raw dump.
    assert len(out) < len(scr.xml) / 8


def test_render_truncates_and_says_so():
    out = render(s(X.settings_screen(rows=40)), limit=10)
    assert "more elements not shown" in out
    assert len(out.splitlines()) == 12  # header + 10 elements + the notice


def test_render_explains_an_empty_screen():
    out = render(parse("<?xml version='1.0'?>\n<hierarchy rotation=\"0\" />"))
    assert "no visible elements" in out


def test_malformed_xml_yields_an_empty_screen_not_a_crash():
    scr = parse("<hierarchy rotation=\"0\"><node ")
    assert scr.elements == [] and scr.nodes == []


def test_empty_dump_sentinel():
    """u2 returns this valid-but-empty document instead of raising."""
    scr = parse('<?xml version=\'1.0\' encoding=\'UTF-8\' standalone=\'yes\' ?>\r\n'
                '<hierarchy rotation="0" />')
    assert scr.elements == []


def test_element_detail_includes_the_raw_attributes():
    scr = s(X.settings_screen())
    toggle = next(e for e in scr.elements if e.kind() == "Toggle")
    detail = element_detail(toggle)
    assert "com.android.settings:id/switch_widget" in detail
    assert "checkable=True" in detail
    assert "inside scroller" in detail


def test_scroller_lookup():
    scr = s(X.settings_screen())
    toggle = next(e for e in scr.elements if e.kind() == "Toggle")
    scroller = toggle.scroller()
    assert scroller is not None and scroller.resource_id == "recycler_view"
    title = next(e for e in scr.elements if e.resource_id == "action_bar_title")
    assert title.scroller() is None


@pytest.mark.parametrize("cls,expected", [
    ("android.widget.EditText", "Input"),
    ("androidx.appcompat.widget.SwitchCompat", "Toggle"),
    ("androidx.recyclerview.widget.RecyclerView", "Scroller"),
    ("android.widget.ImageButton", "ImageButton"),
    ("android.widget.TextView", "Text"),
])
def test_element_kind(cls, expected):
    el = Element(cls=cls, bounds=(0, 0, 10, 10))
    if expected in ("Toggle",):
        el.checkable = True
    if expected == "Scroller":
        el.scrollable = True
    assert el.kind() == expected


# ---------------------------------------------------------------------------
# Where on the screen an element is
# ---------------------------------------------------------------------------
#
# The element list used to carry no geometry at all, and its order is no
# substitute: `prune` walks the dump in *window* order, and which window comes
# first varies by screen. Measured on a real phone (720x1600), a Settings search
# screen listed the navigation bar at #1-#5 (y=1504) and the status bar at
# #9-#14 (y=0-18) -- the list ran bottom to top. A model asked for "the icon in
# the top-right" or "the bar along the bottom" had nothing correct to reason from.

from adbagent.screen import render_element, zone


def _el(left, top, right, bottom):
    return Element(bounds=(left, top, right, bottom), cls="android.widget.Button",
                   clickable=True)


@pytest.mark.parametrize("bounds,expected", [
    # Corners, using a 900x1800 frame so the thirds are 300 and 600 wide/tall.
    ((0, 0, 100, 100), "top-left"),
    ((800, 0, 900, 100), "top-right"),
    ((400, 0, 500, 100), "top"),
    ((0, 1700, 100, 1800), "bottom-left"),
    ((800, 1700, 900, 1800), "bottom-right"),
    ((400, 1700, 500, 1800), "bottom"),
    ((400, 850, 500, 950), "mid"),
    ((0, 850, 100, 950), "mid-left"),
    ((800, 850, 900, 950), "mid-right"),
])
def test_the_zone_names_which_end_and_which_side(bounds, expected):
    assert zone(_el(*bounds), 900, 1800) == expected


def test_the_middle_column_is_unnamed():
    """"@bottom", not "@bottom-centre" -- the common case, and the one where the
    extra word says nothing."""
    assert zone(_el(400, 1700, 500, 1800), 900, 1800) == "bottom"
    assert zone(_el(400, 850, 500, 950), 900, 1800) == "mid"


def test_an_element_filling_the_frame_is_full_not_mid():
    """Its centre is the middle of the screen, but calling a scroller that spans
    the view "mid" would be a claim about position where there is none."""
    assert zone(_el(0, 0, 900, 1800), 900, 1800) == "full"
    assert zone(_el(0, 100, 900, 1500), 900, 1800) == "full"
    # A wide but shallow bar is a real position, not a container.
    assert zone(_el(0, 1740, 900, 1800), 900, 1800) == "bottom"


def test_a_zone_needs_a_frame_to_be_measured_against():
    """Rendered without the screen size -- as `adbagent dump --detail` does -- the
    line is exactly what it was before zones existed. A zone computed against an
    unknown frame would be a guess dressed as a fact."""
    el = _el(0, 0, 100, 100)
    el.index = 3
    assert zone(el, 0, 0) == ""
    assert zone(el, 900, 0) == ""
    assert "@" not in render_element(el)
    assert "@top-left" in render_element(el, 900, 1800)


def test_a_zero_area_element_has_no_zone():
    assert zone(_el(50, 50, 50, 50), 900, 1800) == ""


def test_the_render_gives_every_element_a_zone():
    scr = s(X.settings_screen())
    lines = render(scr).splitlines()[1:]          # past the header
    tagged = [l for l in lines if l.startswith("#")]
    assert tagged
    assert all("@" in line for line in tagged), [l for l in tagged if "@" not in l]


def test_the_zone_is_right_even_when_the_list_order_is_upside_down():
    """The regression this exists for. Two windows in the dump: the navigation bar
    first, the app's own content second -- which is the order the real device
    produced. The indices run bottom to top; the zones must not."""
    nav_first = X.dump(
        X.N("android.widget.FrameLayout", (0, X.H - 140, X.W, X.H),
            rid="nav_bar", package="com.android.systemui", children=[
                X.N("android.widget.ImageButton", (60, X.H - 130, 200, X.H - 10),
                    text="Back", rid="back", package="com.android.systemui",
                    clickable=True)]),
        X.N("android.widget.FrameLayout", (0, 0, X.W, 200), rid="status",
            package="com.android.systemui", children=[
                X.N("android.widget.TextView", (30, 20, 200, 90), text="4:16 PM",
                    rid="clock", package="com.android.systemui")]))
    scr = parse(nav_first, width=X.W, height=X.H)
    by_label = {e.best_text: (e.index, zone(e, scr.width, scr.height))
                for e in scr.elements}

    back_index, back_zone = by_label["Back"]
    clock_index, clock_zone = by_label["4:16 PM"]

    assert back_index < clock_index, "fixture must reproduce the inverted order"
    assert "bottom" in back_zone, back_zone
    assert "top" in clock_zone, clock_zone


def test_the_system_prompt_explains_the_zone_and_distrusts_the_order():
    """A tag the model is never told about is tokens spent on nothing."""
    from adbagent import prompts
    assert "@zone" in prompts.SYSTEM
    assert "@full" in prompts.SYSTEM
    assert "never as a measurement" in prompts.SYSTEM
    # And the thing that made zones necessary in the first place.
    assert "List order is NOT screen order" in prompts.SYSTEM
