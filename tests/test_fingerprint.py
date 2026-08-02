"""The mutation matrix: what counts as "the same screen".

This is the single most important test in the project. A fingerprint that is too
strict never hits the cache; one that is too loose taps the wrong thing. Every
row below is a real way a screen changes without becoming a different screen --
or a real way two screens look alike but must not be confused.
"""

from __future__ import annotations

import pytest

from adbagent import fingerprint as fp
from adbagent.screen import parse

from . import xmlgen as X

T_SIM = 6  # default MemoryConfig.t_sim


def screen(xml: str):
    return fp.attach(parse(xml, width=X.W, height=X.H))


BASE = X.settings_screen()


def same(xml: str, base: str = BASE):
    a, b = screen(base), screen(xml)
    return (a.skeleton_id == b.skeleton_id
            and fp.hamming(a.simhash, b.simhash) <= T_SIM)


def distance(xml: str, base: str = BASE) -> int:
    a, b = screen(base), screen(xml)
    return fp.hamming(a.simhash, b.simhash)


# ---------------------------------------------------------------------------
# Must MATCH -- the screen changed, its identity did not
# ---------------------------------------------------------------------------

MATCHING = {
    "clock ticks": X.settings_screen(clock="9:42"),
    "clock crosses noon": X.settings_screen(clock="12:07 PM"),
    "battery drops": X.settings_screen(battery="83%"),
    "notification badge changes": X.settings_screen(badge="5"),
    "all three chrome values change": X.settings_screen(clock="11:11", battery="9%",
                                                        badge="17"),
    "list scrolled": X.settings_screen(scroll=240),
    "list gains rows": X.settings_screen(rows=10),
    "list loses rows": X.settings_screen(rows=4),
    "every row's text replaced": X.settings_screen(
        labels=["Zebra", "Yak", "Xerus", "Wombat", "Vole", "Uakari", "Tapir"]),
    "a toggle flips checked": X.settings_screen(checked_row=2),
    "keyboard opens": X.settings_screen(keyboard=True),
    "AppCompatButton -> MaterialButton": X.settings_screen(
        button_cls="com.google.android.material.button.MaterialButton"),
    "DPI change": None,  # built below -- needs a different screen size
}


@pytest.mark.parametrize("name", [k for k, v in MATCHING.items() if v is not None])
def test_mutations_that_must_match(name):
    assert same(MATCHING[name]), f"{name}: should have been the same screen"


def test_dpi_change_matches():
    """Same layout at a different density. Geometry is quantised by fraction."""
    from lxml import etree

    big = X.settings_screen()
    root = etree.fromstring(big.encode())
    for node in root.iter("node"):
        b = node.get("bounds")
        nums = [int(v) for v in
                __import__("re").findall(r"-?\d+", b)]
        scaled = [int(round(v * 1.5)) for v in nums]
        node.set("bounds", f"[{scaled[0]},{scaled[1]}][{scaled[2]},{scaled[3]}]")
    scaled_xml = etree.tostring(root, encoding="unicode")

    a = fp.attach(parse(BASE, width=X.W, height=X.H))
    b = fp.attach(parse(scaled_xml, width=int(X.W * 1.5), height=int(X.H * 1.5)))
    assert a.skeleton_id == b.skeleton_id
    assert fp.hamming(a.simhash, b.simhash) <= 1


# ---------------------------------------------------------------------------
# Must NOT match -- these are different screens
# ---------------------------------------------------------------------------

def test_sibling_tab_is_a_different_screen():
    """Same chrome, same structure, different tab selected.

    The classic false positive. Only `selected` distinguishes them, which is why
    it is a SimHash feature even though `checked` is not.
    """
    other = X.settings_screen(selected_tab="apps")
    assert not same(other)
    assert screen(BASE).skeleton_id != screen(other).skeleton_id


def test_detail_screen_is_different():
    assert not same(X.detail_screen())


def test_rotation_is_different():
    rotated = X.settings_screen(rotation=1)
    a, b = screen(BASE), screen(rotated)
    assert a.skeleton_id != b.skeleton_id


def test_permission_dialog_is_different():
    with_dialog = X.settings_screen(extra_roots=[X.permission_dialog()])
    assert not same(with_dialog)


def test_different_app_is_different():
    other_app = X.settings_screen().replace("com.android.settings", "com.example.other")
    assert not same(other_app)


# ---------------------------------------------------------------------------
# Destructive-token gate
# ---------------------------------------------------------------------------

def test_destructive_tokens_detected():
    tokens = fp.destructive_tokens(screen(X.detail_screen()))
    assert any("forget" in t for t in tokens), tokens
    # A plain settings list has nothing irreversible on it.
    assert fp.destructive_tokens(screen(BASE)) == []


@pytest.mark.parametrize("label,destructive", [
    ("Delete account", True),
    ("Place order", True),
    ("Send message", True),
    ("Forget network", True),
    ("Sign out", True),
    ("Unfollow", True),
    ("Clear history", True),
    ("Cancel", False),
    ("Back", False),
    ("Settings", False),
    ("Disconnect", False),   # reversible: reconnecting costs nothing
])
def test_destructive_vocabulary(label, destructive):
    assert bool(fp.DESTRUCTIVE_TEXT.search(label)) is destructive


def test_delete_confirmation_is_hard_blocked():
    dialog = X.N("android.widget.FrameLayout", (60, 800, 1020, 1400),
                 rid="alert_dialog", children=[
                     X.N("android.widget.TextView", (100, 860, 980, 980),
                         text="Delete account?", rid="alertTitle"),
                     X.N("android.widget.Button", (620, 1240, 980, 1380),
                         text="Delete", rid="button1", clickable=True),
                 ])
    s = screen(X.settings_screen(extra_roots=[dialog]))
    assert fp.destructive_tokens(s), "a Delete confirm must produce a forbidden token"
    assert not same(X.settings_screen(extra_roots=[dialog]))


# ---------------------------------------------------------------------------
# Unit-level checks on the normalisation rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("9:41", "<time>"),
    ("12:07 PM", "<time>"),
    ("84%", "<pct>"),
    ("$12.99", "<money>"),
    ("₹1,200", "<money>"),
    ("5 min ago", "<rel>"),
    ("12/03/2025", "<date>"),
    ("42", "<n>"),
    ("1234567", "<num>"),
    ("me@example.com", "<email>"),
    ("Network & internet", "network & internet"),
])
def test_mask_text(raw, expected):
    assert fp.mask_text(raw) == expected


@pytest.mark.parametrize("cls,expected", [
    ("android.widget.Button", "Button"),
    ("androidx.appcompat.widget.AppCompatButton", "Button"),
    ("com.google.android.material.button.MaterialButton", "Button"),
    ("androidx.appcompat.widget.SwitchCompat", "Toggle"),
    ("android.widget.CheckBox", "Toggle"),
    ("android.widget.EditText", "Input"),
    ("androidx.recyclerview.widget.RecyclerView", "Scroller"),
    ("android.widget.ListView", "Scroller"),
    ("android.widget.LinearLayout$Inner", "Inner"),
    ("android.webkit.WebView", "WebView"),
])
def test_class_equivalence(cls, expected):
    assert fp.class_eq(cls) == expected


@pytest.mark.parametrize("rid,expected", [
    ("row_item_3", "row_item#"),
    ("item12", "item#"),
    ("switch_widget", "switch_widget"),
    ("", ""),
])
def test_rid_normalisation(rid, expected):
    assert fp.rid_norm(rid) == expected


def test_flags_exclude_state():
    """checked/selected/focused/enabled must not appear in skeleton tokens."""
    plain = screen(BASE)
    toggled = screen(X.settings_screen(checked_row=0))
    assert plain.tokens == toggled.tokens


def test_exact_id_does_track_state():
    """exact_id is for change detection, so it MUST notice a flipped toggle."""
    plain = screen(BASE)
    toggled = screen(X.settings_screen(checked_row=0))
    assert plain.exact_id != toggled.exact_id


def test_count_capping_makes_list_length_irrelevant():
    a = screen(X.settings_screen(rows=7))
    b = screen(X.settings_screen(rows=30))
    assert a.tokens == b.tokens


# ---------------------------------------------------------------------------
# Horizontal scroller ordinals
# ---------------------------------------------------------------------------

def test_horizontal_scroller_ordinals_sort_by_x():
    """Items in a horizontal scroller should get f/m/l based on X position.

    A horizontally-scrolled carousel should produce the same skeleton_id as
    the unscrolled version, just like vertical scrolling does for lists.
    """
    a = screen(X.horizontal_scroll_screen(scroll=0))
    b = screen(X.horizontal_scroll_screen(scroll=200))
    assert a.skeleton_id == b.skeleton_id


def test_horizontal_scroll_matches_identity():
    """Horizontal scroll should not change simhash significantly."""
    a = screen(X.horizontal_scroll_screen(scroll=0))
    b = screen(X.horizontal_scroll_screen(scroll=200))
    assert fp.hamming(a.simhash, b.simhash) <= T_SIM
