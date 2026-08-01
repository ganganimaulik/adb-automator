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
