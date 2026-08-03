"""The scratchpad loss guard.

`STEP_73` and `STEP_74` are the verbatim scratchpad values from consecutive turns
of ``runs/af76720d05c4`` (inlined because ``runs/`` is not committed). Between
them the model rewrote four measured readings as ``[pending]`` and never restated
any of them across the remaining 59 turns; the run then reported the 10:03 photo
as unreadable, having already read and recorded it.

The hard part is not spotting that -- it is spotting that and *not* also
reporting the reformatting that happens constantly and loses nothing.
"""

from __future__ import annotations

import pytest

from adbagent import scratchpad as sp
from adbagent.actions import AgentAction
from adbagent.agent import Agent
from adbagent.config import Config
from adbagent.memory import Memory
from tests.fake import FakeDevice, FakeLLM

MENU = ("MENU: Oats meal: oats 100g, whey 60g, almonds 5g, cashews 5g, "
        "walnuts 5g, banana 120g, water 275g. Chicken meal: chicken 425g, "
        "rice 290g, potato 400g, tomato 100g, olive oil 9g+9g. No-carb "
        "chicken meal: chicken 102g, tomato 120g, olive oil 7g, salt 2g. "
        "Krishna photos: 9:30 water 275g (OK)")

READINGS = ("9:31 banana 120g (OK); 9:32 almonds 6g (+1g); 9:33 mixed nuts "
            "~5g (OK); 9:36 oats 101g (+1g); 9:39 olive oil 9g (OK); "
            "9:40 whey 60g (OK); 9:43 salt 4g (+2g)")

STEP_73 = (f"{MENU}; {READINGS}; 9:45 chicken 425g (OK); 9:51 chicken 426g "
           "(+1g); 9:52 [pending]; 9:59 potatoes 403g (+3g vs menu 400g); "
           "10:03 tomatoes 120g (matches no-carb tomato 120g).")

STEP_74 = (f"{MENU}; {READINGS}; 9:45 [pending]; 9:51 [pending]; "
           "9:52 [pending]; 9:59 [pending]; 10:03 [pending].")


def keys_of(losses):
    return {loss.key for loss in losses}


# ---------------------------------------------------------------------------
# Record splitting and keying
# ---------------------------------------------------------------------------

def test_records_split_on_semicolons_and_newlines():
    records = sp.split_records("a 1g; b 2g\n- c 3g\n1. d 4g")
    assert records == ["a 1g", "b 2g", "c 3g", "d 4g"]


def test_a_record_is_keyed_by_its_leading_identifier():
    assert sp.record_key("10:03 tomatoes 120g (matches tomato 120g)") == "10:03"
    assert sp.record_key("Item B: $15") == "item"
    assert sp.record_key("OK") == ""


def test_figures_count_as_distinctive_however_short():
    tokens = sp.distinctive("9:32 almonds 6g (OK)")
    assert "9:32" in tokens and "6g" in tokens and "almonds" in tokens
    assert "ok" not in tokens               # too short to identify anything


# ---------------------------------------------------------------------------
# The run's actual loss
# ---------------------------------------------------------------------------

def test_the_four_readings_wiped_at_step_74_are_all_caught():
    guard = sp.ScratchpadGuard()
    assert guard.update(STEP_73, 73) == []
    losses = guard.update(STEP_74, 74)
    assert keys_of(losses) == {"9:45", "9:51", "9:59", "10:03"}


def test_the_lost_reading_is_handed_back_verbatim():
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    block = guard.report(guard.update(STEP_74, 74))
    assert "YOU DROPPED DATA YOU HAD ALREADY COLLECTED" in block
    assert "10:03 tomatoes 120g" in block
    assert "9:59 potatoes 403g" in block
    assert "(step 73)" in block


def test_the_menu_cannot_bridge_a_lost_reading():
    """The note restates "120g" and "tomato" in its menu section, so a note-wide
    token test finds them present and concludes nothing was lost. Comparison is
    per key precisely to stop that."""
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    assert "120g" in sp.distinctive(STEP_74)          # still somewhere in the note
    assert "10:03" in keys_of(guard.update(STEP_74, 74))


def test_a_reading_stays_recoverable_long_after_the_turn_that_dropped_it():
    """The real run never restated these across 59 further turns. The archive is
    append-only so the record survives every one of them."""
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    for step in range(74, 137):
        losses = guard.update(STEP_74, step)
    assert "10:03" in keys_of(losses)
    assert "10:03 tomatoes 120g" in guard.preserved(STEP_74)


def test_restating_the_record_clears_it():
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    assert guard.update(STEP_74, 74)
    assert keys_of(guard.update(STEP_73, 75)) == set()


# ---------------------------------------------------------------------------
# Not crying wolf
# ---------------------------------------------------------------------------

def test_an_unchanged_note_reports_nothing():
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    assert guard.update(STEP_73, 74) == []


def test_growing_the_note_reports_nothing():
    guard = sp.ScratchpadGuard()
    guard.update("9:30 water 275g (OK)", 1)
    assert guard.update("9:30 water 275g (OK); 9:31 banana 120g (OK)", 2) == []


def test_rewording_a_record_is_not_a_loss():
    """"mixed nuts ~5g" to "nuts 5g" drops a word and no information."""
    guard = sp.ScratchpadGuard()
    guard.update("9:33 mixed nuts ~5g (OK); 9:36 oats 101g", 1)
    assert guard.update("9:33 nuts 5g OK; 9:36 oats 101g", 2) == []


def test_consolidating_pending_placeholders_is_not_a_loss():
    """Observed at step 87 of the run: four "[pending]" records collapsed into
    one line. A naive text diff called that four losses."""
    guard = sp.ScratchpadGuard()
    guard.update("9:45 [pending]; 9:52 [pending]; 9:59 [pending]", 1)
    assert guard.update("9:45, 9:52, 9:59 pending", 2) == []


def test_a_corrected_figure_is_surfaced_rather_than_silently_replaced():
    """The run read 9:59 as 403g, lost it, and later re-read it as 413g. Both
    values existing is a conflict worth seeing, not a rewrite to accept."""
    guard = sp.ScratchpadGuard()
    guard.update("9:59 potatoes 403g", 1)
    losses = guard.update("9:59 potatoes 413g", 2)
    assert keys_of(losses) == {"9:59"}
    assert "403g" in next(iter(losses)).lost


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_one_record_is_only_nagged_about_a_few_times():
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    blocks = [guard.report(guard.update(STEP_74, step))
              for step in range(74, 74 + sp.MAX_REPORTS + 3)]
    assert all(blocks[:sp.MAX_REPORTS])
    assert not any(blocks[sp.MAX_REPORTS:])


def test_a_silenced_record_still_reaches_the_judge():
    """The reporting budget bounds the prompt, never the archive."""
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    for step in range(74, 90):
        guard.report(guard.update(STEP_74, step))
    assert "10:03 tomatoes 120g" in guard.preserved(STEP_74)


def test_the_block_is_capped():
    guard = sp.ScratchpadGuard()
    many = "; ".join(f"item{i} value {i * 11}g" for i in range(40))
    guard.update(many, 1)
    block = guard.report(guard.update("nothing at all here now", 2))
    assert block.count("\n  - ") <= sp.MAX_REPORTED + 1
    assert "and 20 more" in block or "more" in block


def test_the_archive_is_bounded():
    guard = sp.ScratchpadGuard()
    for i in range(sp.MAX_KEYS + 50):
        guard.update(f"item{i} value {i}g", i)
    assert len(guard.keys) <= sp.MAX_KEYS


def test_preserved_returns_the_note_untouched_when_nothing_was_dropped():
    guard = sp.ScratchpadGuard()
    guard.update(STEP_73, 73)
    assert guard.preserved(STEP_73) == STEP_73


# ---------------------------------------------------------------------------
# Through the loop
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.memory.db = str(tmp_path / "memory.db")
    c.run.artifacts_dir = str(tmp_path / "runs")
    c.run.max_steps = 6
    c.safety.unattended = True
    return c


@pytest.fixture
def mem(cfg, tmp_path):
    with Memory(cfg, path=tmp_path / "memory.db") as m:
        yield m


def collect_then_drop(notes):
    """Write `notes[i]` on turn i, then finish."""
    def policy(screen, llm):
        turn = llm.calls - 1
        if turn < len(notes):
            # A turn that changes nothing and only writes notes. `duration` is
            # pinned to the minimum so the suite does not actually sleep.
            return AgentAction(observation="collecting", reasoning="record it",
                               action="wait", duration=0.05, notes=notes[turn])
        return AgentAction(observation="finished", reasoning="report",
                           action="done", text="collected the readings")
    return policy


def run_collection(cfg, mem, notes):
    dev = FakeDevice(cfg)
    llm = FakeLLM(dev, collect_then_drop(notes))
    judged = {}
    original = llm.judge
    llm.judge = lambda **kw: (judged.update(kw), original(**kw))[1]
    outcome, state = Agent(dev, mem, llm, cfg).run("record every weight")
    return llm, state, judged, outcome


def test_the_loop_hands_a_dropped_reading_back_on_the_next_turn(cfg, mem):
    llm, _, _, outcome = run_collection(cfg, mem, [STEP_73, STEP_74, STEP_74])
    assert outcome == "success"
    blocks = [n for n in llm.notes if "YOU DROPPED DATA" in n]
    assert blocks, "the loss was never put in front of the model"
    assert "10:03 tomatoes 120g" in blocks[0]
    assert "9:59 potatoes 403g" in blocks[0]


def test_the_judge_grades_on_everything_collected_not_the_last_rewrite(cfg, mem):
    """The run reported the 10:03 photo as unreadable while its own earlier notes
    held the reading. The judge now sees it either way."""
    _, _, judged, _ = run_collection(cfg, mem, [STEP_73, STEP_74, STEP_74])
    assert "10:03 tomatoes 120g" in judged["scratchpad"]


def test_a_clean_collection_run_shows_no_loss_block(cfg, mem):
    llm, _, judged, _ = run_collection(cfg, mem, [STEP_73, STEP_73, STEP_73])
    assert not any("YOU DROPPED DATA" in n for n in llm.notes)
    assert "EARLIER RECORDS" not in judged["scratchpad"]
