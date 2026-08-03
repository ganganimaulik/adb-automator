"""The collected-data ledger.

`STEP_73` and `STEP_74` are the verbatim scratchpad values from consecutive turns
of ``runs/af76720d05c4`` (inlined because ``runs/`` is not committed). Between
them the model rewrote four measured readings as ``[pending]`` and never restated
any of them across the remaining 59 turns; the run then reported the 10:03 photo
as unreadable, having already read and recorded it.

The old design detected that by diffing consecutive rewrites. This one makes it
unrepresentable: the model sends deltas and the union is kept here, so a record
it stops mentioning is a record nobody touched. Most of these tests exist to show
the four readings surviving *without* anything having to notice they went -- and,
because a model will still sometimes write prose into a field that asked for
records, that they survive that too.
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


def recs(*pairs):
    return [{"key": key, "value": value} for key, value in pairs]


# ---------------------------------------------------------------------------
# The record contract
# ---------------------------------------------------------------------------

def test_records_accumulate_across_turns():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:30", "water 275g")), 1)
    ledger.update(recs(("9:31", "banana 120g")), 2)
    assert "9:30: water 275g" in ledger.plain()
    assert "9:31: banana 120g" in ledger.plain()


def test_a_record_the_model_stops_sending_stays():
    """The whole point. Turn 2 mentions only one key; the other is untouched, not
    dropped, because nothing about turn 2 replaces turn 1."""
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:45", "chicken 425g"), ("9:59", "potatoes 403g")), 1)
    for step in range(2, 60):
        ledger.update(recs(("9:45", "chicken 425g")), step)
    assert "9:59: potatoes 403g" in ledger.plain()


def test_an_empty_note_changes_nothing():
    ledger = sp.NoteLedger()
    ledger.update(recs(("total", "12 items")), 1)
    for empty in (None, "", [], {}):
        assert ledger.update(empty, 2) == []
    assert "total: 12 items" in ledger.plain()


def test_reusing_a_key_corrects_the_value():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:59", "potatoes 403g")), 1)
    ledger.update(recs(("9:59", "potatoes 413g")), 2)
    assert "413g" in ledger.plain()
    assert len(ledger) == 1


def test_a_corrected_figure_keeps_the_value_it_replaced():
    """The run read 9:59 as 403g, lost it, and later re-read it as 413g. Both
    values existing is a disagreement worth seeing, not a rewrite to accept
    silently -- the one property the old loss detector had that an upsert does
    not get for free."""
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:59", "potatoes 403g")), 1)
    ledger.update(recs(("9:59", "potatoes 413g")), 2)
    line = ledger.plain()
    assert "413g" in line and "403g" in line
    assert "earlier" in line


def test_re_sending_an_identical_record_is_not_a_correction():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:59", "potatoes 403g")), 1)
    assert ledger.update(recs(("9:59", "potatoes 403g")), 2) == []
    assert "earlier" not in ledger.plain()


def test_keys_are_matched_past_incidental_punctuation():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:45", "chicken 425g")), 1)
    ledger.update(recs(("9:45:", "chicken 426g")), 2)
    assert len(ledger) == 1


def test_a_value_that_repeats_its_own_key_is_not_rendered_twice():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:45", "9:45 chicken 425g")), 1)
    assert ledger.plain() == "9:45: chicken 425g"


# ---------------------------------------------------------------------------
# Prose, which the model will still sometimes send
# ---------------------------------------------------------------------------

def test_prose_is_split_into_records():
    pairs = sp.as_records("a 1g; b 2g\n- c 3g\n1. d 4g")
    assert [key for key, _ in pairs] == ["1g", "2g", "3g", "4g"]


def test_two_prose_records_sharing_a_leading_word_do_not_overwrite():
    """"Item A: $10" and "Item B: $15" both key on "item". Resolving that by
    letting the second win would lose the first."""
    ledger = sp.NoteLedger()
    ledger.update("Item A: $10; Item B: $15", 1)
    assert len(ledger) == 2
    assert "$10" in ledger.plain() and "$15" in ledger.plain()


def test_the_four_readings_wiped_at_step_74_survive_the_rewrite():
    """The motivating failure, as prose, with nothing detecting anything: the
    rewrite corrects five keys to "[pending]" and the readings they held are
    still on the record."""
    ledger = sp.NoteLedger()
    ledger.update(STEP_73, 73)
    ledger.update(STEP_74, 74)
    collected = ledger.plain()
    for reading in ("chicken 425g", "chicken 426g", "potatoes 403g",
                    "tomatoes 120g"):
        assert reading in collected


def test_a_reading_survives_every_later_turn_that_omits_it():
    """The real run never restated these across 59 further turns."""
    ledger = sp.NoteLedger()
    ledger.update(STEP_73, 73)
    for step in range(74, 137):
        ledger.update(STEP_74, step)
    assert "tomatoes 120g" in ledger.plain()


def test_a_prose_record_reworded_is_not_duplicated():
    """"mixed nuts ~5g" to "nuts 5g" drops a word and no information; both hang
    off the same 9:33 key."""
    ledger = sp.NoteLedger()
    ledger.update("9:33 mixed nuts ~5g (OK); 9:36 oats 101g", 1)
    ledger.update("9:33 nuts 5g OK; 9:36 oats 101g", 2)
    assert len(ledger) == 2


def test_a_bare_string_reaches_the_ledger_through_the_action_schema():
    """A model that ignores the record shape must not have its readings rejected
    into a repair round trip, and neither must a recording made before the schema
    changed -- `replay` loads those."""
    action = AgentAction(observation="o", reasoning="r", action="wait",
                         notes="9:45 chicken 425g; 9:51 chicken 426g")
    assert [note.key for note in action.notes] == ["9:45", "9:51"]


def test_notes_absent_stays_absent():
    action = AgentAction(observation="o", reasoning="r", action="wait")
    assert action.notes is None


# ---------------------------------------------------------------------------
# Rendering and bounds
# ---------------------------------------------------------------------------

def test_the_prompt_block_tells_the_model_not_to_restate_it():
    ledger = sp.NoteLedger()
    ledger.update(recs(("9:30", "water 275g")), 1)
    block = ledger.render()
    assert "COLLECTED DATA (1 record(s))" in block
    assert "do NOT restate" in block
    assert "9:30: water 275g" in block


def test_an_empty_ledger_renders_to_nothing():
    assert sp.NoteLedger().render() == ""
    assert not sp.NoteLedger()


def test_the_ledger_is_bounded_and_says_what_it_dropped():
    ledger = sp.NoteLedger()
    for i in range(sp.MAX_KEYS + 50):
        ledger.update(recs((f"item{i}", f"value {i}g")), i)
    assert len(ledger) <= sp.MAX_KEYS
    assert "50 earlier record(s) dropped for space" in ledger.render()


def test_one_key_keeps_a_bounded_number_of_superseded_values():
    ledger = sp.NoteLedger()
    for i in range(10):
        ledger.update(recs(("9:59", f"potatoes {400 + i}g")), i)
    entry = next(iter(ledger.entries.values()))
    assert len(entry.superseded) <= sp.MAX_SUPERSEDED


def test_a_long_value_is_capped():
    ledger = sp.NoteLedger()
    ledger.update(recs(("blob", "x" * (sp.MAX_VALUE_CHARS + 500))), 1)
    entry = next(iter(ledger.entries.values()))
    assert len(entry.value) <= sp.MAX_VALUE_CHARS


def test_replay_rebuilds_a_finished_run_from_its_deltas():
    ledger = sp.replay([
        {"kind": "run_start"},
        {"kind": "decide", "step": 1,
         "action": {"notes": [{"key": "9:30", "value": "water 275g"}]}},
        {"kind": "verify", "step": 1},
        {"kind": "decide", "step": 2,
         "action": {"notes": [{"key": "9:31", "value": "banana 120g"}]}},
        {"kind": "decide", "step": 3, "action": {"action": "done"}},
    ])
    assert len(ledger) == 2
    assert "9:31: banana 120g" in ledger.plain()


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


def test_the_loop_keeps_a_reading_a_later_turn_stops_mentioning(cfg, mem):
    _, state, _, outcome = run_collection(
        cfg, mem, [recs(("9:59", "potatoes 403g")),
                   recs(("10:03", "tomatoes 120g")),
                   recs(("10:07", "rice 290g"))])
    assert outcome == "success"
    collected = state.scratchpad.plain()
    assert "potatoes 403g" in collected      # written on turn 1, never repeated
    assert "tomatoes 120g" in collected
    assert "rice 290g" in collected


def test_the_judge_grades_on_everything_collected_not_the_last_turn(cfg, mem):
    """The run reported the 10:03 photo as unreadable while its own earlier notes
    held the reading. The judge sees every record either way."""
    _, _, judged, _ = run_collection(cfg, mem, [STEP_73, STEP_74, STEP_74])
    assert "tomatoes 120g" in judged["scratchpad"]


def test_the_model_is_shown_the_ledger_it_no_longer_has_to_restate(cfg, mem):
    llm, _, _, _ = run_collection(
        cfg, mem, [recs(("9:59", "potatoes 403g")), recs(("10:03", "tomatoes 120g"))])
    shown = [call for call in llm.scratchpads if "potatoes 403g" in call]
    assert shown, "the collected ledger was never handed back to the model"
    assert "do NOT restate" in shown[-1]


def test_the_rendered_block_respects_a_char_budget_and_says_what_it_cut():
    ledger = sp.NoteLedger()
    for i in range(40):
        ledger.update(recs((f"9:{i:02d}", f"reading number {i} at 42{i}g")), i)
    block = ledger.render(max_chars=200)
    assert len(block) < 500
    assert "record(s) dropped for space" in block
    assert "reading number 39" in block          # the newest survive
    assert "reading number 0 " not in block


def test_no_budget_renders_everything():
    ledger = sp.NoteLedger()
    for i in range(40):
        ledger.update(recs((f"9:{i:02d}", f"reading {i}")), i)
    assert "reading 0" in ledger.render()
    assert "dropped for space" not in ledger.render()


def test_a_trimmed_plain_view_says_so_rather_than_looking_complete():
    ledger = sp.NoteLedger()
    for i in range(40):
        ledger.update(recs((f"9:{i:02d}", f"reading {i}")), i)
    assert "earlier record(s) omitted" in ledger.plain(max_chars=100)
