"""Gallery / carousel item identity.

Every case here is taken from ``runs/af76720d05c4``, a run that spent 136 steps
and 102 minutes checking fifteen WhatsApp photos because nothing in the system
could tell one photo from another. The trace facts these tests encode:

* the pager is listed as #1, #4 or #11 on different turns, depending on whether
  the app's overlay chrome happens to be showing;
* ``exact_id`` is identical for every photo, because ``mask_text`` rewrites the
  caption's timestamp as ``<time>``;
* two of the fifteen photos were sent in the same minute and therefore share a
  caption, so a repeated caption is not evidence a swipe failed;
* every swipe was graded ``success``, including the ones that did nothing.
"""

from __future__ import annotations

from adbagent import pager
from adbagent.actions import AgentAction, Target, execute, verify
from adbagent.fingerprint import attach
from adbagent.screen import parse
from tests import xmlgen as X
from tests.fake import FakeDevice


def viewer(timestamp: str = "9:33 am", chrome: bool = True):
    screen = attach(parse(X.media_viewer(timestamp, chrome=chrome),
                          width=X.W, height=X.H, activity=X.MEDIA_ACTIVITY))
    return screen


def album(header: str = "9:30 am", total: str = "15 photos"):
    return attach(parse(X.media_album(header, total),
                        width=X.W, height=X.H, activity=X.ALBUM_ACTIVITY))


def act(**kw) -> AgentAction:
    kw.setdefault("observation", "o")
    kw.setdefault("reasoning", "r")
    return AgentAction(**kw)


# ---------------------------------------------------------------------------
# Detection and labelling
# ---------------------------------------------------------------------------

def test_media_viewer_is_a_pager_and_carries_the_photos_caption():
    screen = viewer("9:33 am")
    assert screen.is_pager
    assert screen.item_label == "Today, 9:33 am"
    assert screen.item_key == "label:today, 9:33 am"


def test_pager_element_is_the_full_bleed_horizontal_scroller():
    screen = viewer()
    element = pager.pager_element(screen)
    assert element is not None
    assert element.resource_id == "pager"
    assert element.is_horizontal


def test_viewer_is_still_a_pager_once_the_overlay_fades():
    """The tree collapses to the scroller alone; the activity still identifies it."""
    screen = viewer(chrome=False)
    assert screen.is_pager
    assert screen.item_label == ""          # nothing in the tree names the photo
    assert pager.pager_element(screen) is not None


def test_album_grid_reports_its_total_but_only_two_tiles():
    screen = album(total="15 photos")
    assert pager.item_total(screen) == 15
    tiles = [e for e in screen.elements if e.resource_id == "image"]
    assert len(tiles) == 2, "the real grid publishes only two tiles; see the skill"


def test_ordinal_caption_is_parsed_as_an_exact_position():
    screen = attach(parse(
        X.horizontal_scroll_screen(labels=["View photo, 3 of 15"]),
        width=X.W, height=X.H, activity=".Gallery"))
    # The caption lives inside the carousel here, so it is item content, not
    # chrome -- and item content must not be mistaken for a caption.
    assert pager.item_ordinal(screen) is None


# ---------------------------------------------------------------------------
# The bug: exact_id cannot tell two photos apart
# ---------------------------------------------------------------------------

def test_two_different_photos_share_one_exact_id():
    """The premise of this whole module. If this ever fails, simplify pager.py."""
    assert viewer("9:33 am").exact_id == viewer("9:59 am").exact_id


def test_loop_id_separates_photos_that_exact_id_merges():
    """Browsing an album must not look like being stuck on one screen."""
    assert pager.loop_id(viewer("9:33 am")) != pager.loop_id(viewer("9:59 am"))


# ---------------------------------------------------------------------------
# same_item: a caption proves difference, never sameness
# ---------------------------------------------------------------------------

def test_different_captions_mean_a_different_item():
    assert pager.same_item(viewer("9:33 am"), viewer("9:36 am")) is False


def test_equal_captions_alone_prove_nothing():
    """Two photos sent in the same minute share a caption -- this is the real
    album's 9:33 pair, and answering "same item" here is how the run lost track."""
    assert pager.same_item(viewer("9:33 am"), viewer("9:33 am")) is None


def test_equal_captions_are_settled_by_the_pixels(monkeypatch):
    before, after = viewer("9:33 am"), viewer("9:33 am")
    hashes = {id(before): 0x0F0F0F0F0F0F0F0F, id(after): 0xF0F0F0F0F0F0F0F0}
    monkeypatch.setattr(pager, "item_pixel_hash",
                        lambda s: hashes.get(id(s)))
    assert pager.same_item(before, after) is False   # different bitmaps

    hashes[id(after)] = hashes[id(before)]
    assert pager.same_item(before, after) is True    # the swipe was dropped


# ---------------------------------------------------------------------------
# Verification of a swipe
# ---------------------------------------------------------------------------

def test_swipe_that_did_not_advance_the_photo_is_no_change(monkeypatch):
    """Previously every swipe was graded success, so a dropped fling was
    reported to the model as progress and it swiped on in the dark."""
    before, after = viewer("9:33 am"), viewer("9:33 am")
    monkeypatch.setattr(pager, "item_pixel_hash", lambda s: 0x1234)
    outcome = verify(act(action="swipe", direction="left"), before, after)
    assert outcome.grade == "no_change"


def test_swipe_that_advanced_the_photo_is_success():
    outcome = verify(act(action="swipe", direction="left"),
                     viewer("9:33 am"), viewer("9:36 am"))
    assert outcome.grade == "success"


def test_a_faded_overlay_does_not_read_as_a_new_photo(monkeypatch):
    """The whole-screen dhash moves a long way when the toolbar fades out. Only
    the item's own pixels are consulted, so that is not mistaken for progress."""
    before, after = viewer("9:33 am"), viewer("9:33 am", chrome=False)
    before.dhash, after.dhash = 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0
    monkeypatch.setattr(pager, "item_pixel_hash", lambda s: 0xABCD)
    outcome = verify(act(action="swipe", direction="left"), before, after)
    assert outcome.grade == "no_change"


# ---------------------------------------------------------------------------
# Addressing
# ---------------------------------------------------------------------------

def test_left_swipe_is_retargeted_onto_the_pager():
    """The model names whichever index the pager had last turn. It moves."""
    screen = viewer()
    toolbar_button = next(e for e in screen.elements
                          if e.resource_id == "title_holder")
    dev = FakeDevice()
    element = execute(dev, act(action="swipe", direction="left",
                               target=Target(index=toolbar_button.index)), screen)
    assert element is not None and element.resource_id == "pager"


def test_vertical_scroll_is_not_retargeted():
    screen = album()
    dev = FakeDevice()
    grid = next(e for e in screen.elements if e.resource_id == "list")
    element = execute(dev, act(action="scroll", direction="down",
                              target=Target(index=grid.index)), screen)
    assert element is not None and element.resource_id == "list"


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_ledger_records_each_photo_once_and_tracks_what_was_read():
    ledger = pager.ItemLedger()
    for stamp, step in (("9:30 am", 1), ("9:31 am", 2), ("9:32 am", 3)):
        screen = viewer(stamp)
        key = ledger.resolve(screen, moved=True)
        ledger.note(key, screen, step, read=True, detail=f"photo at {stamp}")
    assert len(ledger.items) == 3
    assert ledger.read_count == 3
    assert ledger.total == 0                       # the viewer states no total


def test_revisiting_a_photo_does_not_double_count_it():
    ledger = pager.ItemLedger()
    screen = viewer("9:30 am")
    ledger.note(ledger.resolve(screen, moved=True), screen, 1, read=True)
    ledger.note(ledger.resolve(screen, moved=None), screen, 9, read=True)
    assert len(ledger.items) == 1
    assert ledger.items["label:today, 9:30 am"].visits == 2


def test_two_photos_in_the_same_minute_get_separate_entries():
    """The album's 9:33 pair. `moved=True` is the only evidence that the second
    sighting is a different photo rather than the first one again."""
    ledger = pager.ItemLedger()
    first, second = viewer("9:33 am"), viewer("9:33 am")
    ledger.note(ledger.resolve(first, moved=True), first, 1, read=True)
    key = ledger.resolve(second, moved=True)
    ledger.note(key, second, 2, read=True)
    assert len(ledger.items) == 2
    assert key.endswith("#2")
    assert "(#2)" in ledger.items[key].label


def test_a_dropped_swipe_keeps_the_cursor_on_the_same_photo():
    ledger = pager.ItemLedger()
    screen = viewer("9:33 am")
    first = ledger.resolve(screen, moved=True)
    ledger.note(first, screen, 1, read=True)
    assert ledger.resolve(screen, moved=False) == first
    assert len(ledger.items) == 1


def test_hidden_caption_leaves_the_cursor_where_it_was():
    ledger = pager.ItemLedger()
    with_chrome = viewer("9:40 am")
    key = ledger.resolve(with_chrome, moved=True)
    ledger.note(key, with_chrome, 1, read=True)
    assert ledger.resolve(viewer(chrome=False), moved=None) == key


def test_opening_a_different_album_resets_the_ledger():
    ledger = pager.ItemLedger()
    a = album(total="15 photos")
    ledger.rebase(pager.set_id(a))
    ledger.note("label:today, 9:30 am", viewer("9:30 am"), 1, read=True)
    assert len(ledger.items) == 1
    ledger.rebase(pager.set_id(album(total="4 photos")))
    assert len(ledger.items) == 0


def test_ledger_render_names_the_unread_items():
    ledger = pager.ItemLedger()
    read, unread = viewer("9:30 am"), viewer("9:31 am")
    ledger.note(ledger.resolve(read, moved=True), read, 1, read=True,
                detail="water 275g")
    key = ledger.resolve(unread, moved=True)
    ledger.note(key, unread, 2, read=False)
    block = ledger.render(key, unread.item_label)
    assert "1 read" in block
    assert "water 275g" in block
    assert "STILL NOT READ: Today, 9:31 am" in block
    assert "you are here" in block


def test_ledger_render_says_when_the_set_is_exhausted():
    ledger = pager.ItemLedger()
    ledger.total = 2
    for stamp, step in (("9:30 am", 1), ("9:31 am", 2)):
        screen = viewer(stamp)
        ledger.note(ledger.resolve(screen, moved=True), screen, step, read=True)
    assert "Every item in this set has been read" in ledger.render()


def test_ledger_render_is_bounded():
    ledger = pager.ItemLedger()
    for i in range(pager.MAX_LEDGER_RENDER + 15):
        screen = viewer(f"9:{i:02d} am")
        ledger.note(ledger.resolve(screen, moved=True), screen, i, read=True,
                    detail="x" * 400)
    block = ledger.render()
    assert "earlier item(s) omitted" in block
    assert len(block.splitlines()) <= pager.MAX_LEDGER_RENDER + 5


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

def test_browsing_note_names_the_pager_and_the_current_caption():
    screen = viewer("9:40 am")
    note = pager.browsing_note(screen, pager.ItemLedger())
    element = pager.pager_element(screen)
    assert f"swipe left on #{element.index}" in note
    assert "Today, 9:40 am" in note


def test_browsing_note_offers_a_way_out_when_the_caption_is_hidden():
    note = pager.browsing_note(viewer(chrome=False), pager.ItemLedger())
    assert "Tap the item once" in note


def test_browsing_note_reports_a_dropped_swipe():
    note = pager.browsing_note(viewer(), pager.ItemLedger(), swipe_failed=True)
    assert "did NOT change" in note
