"""The plan ledger.

Two properties are under test here, and they are not the same property.

The first is the one `scratchpad` already established for collected data: the
model sends deltas, the harness keeps the union, and a sub-step it stops
mentioning survives because nothing replaces it. That is a straight port.

The second is new, and is the reason this ledger is allowed to touch the stall
ladder when the field it replaced was not. `RunState.note_progress` records why
the old one was cut out: `progress` was present on 76 of 103 turns across
``runs/`` and its text changed on 72, so rewording reset the ladder on 70% of
all steps and `stalled` never once exceeded 3 against block=5, replan=8,
give_up=14. A structured field would be no better if a model could still buy a
reset by writing something -- so most of these tests are attempts to buy one:
declaring a step already done, re-finishing a step, oscillating a status,
resuming from a checkpoint and re-sending the same completions.
"""

from __future__ import annotations

import pytest

from adbagent import plan
from adbagent.actions import AgentAction


def steps(*rows):
    """``(id, status)`` or ``(id, text, status)`` rows as the model sends them."""
    out = []
    for row in rows:
        if len(row) == 2:
            out.append({"id": row[0], "status": row[1]})
        else:
            out.append({"id": row[0], "text": row[1], "status": row[2]})
    return out


def declared(ledger, *ids, step=1):
    """Declare `ids` as pending on `step`, so they are creditable afterwards."""
    ledger.update([{"id": i, "text": f"do {i}"} for i in ids], step)
    return ledger


# ---------------------------------------------------------------------------
# The delta contract
# ---------------------------------------------------------------------------

def test_a_step_not_restated_keeps_its_status():
    ledger = declared(plan.TaskLedger(), "a", "b", "c")
    ledger.update(steps(("a", "done")), 2)
    # Three turns saying nothing about b and c at all.
    for step in (3, 4, 5):
        ledger.update(steps(("a", "done")), step)
    assert ledger.plain() == "[x] do a\n[ ] do b\n[ ] do c"


def test_an_empty_delta_is_a_valid_turn():
    ledger = declared(plan.TaskLedger(), "a")
    assert not ledger.update(None, 2)
    assert not ledger.update([], 3)
    assert ledger.plain() == "[ ] do a"


def test_steps_keep_the_order_they_were_declared_in():
    ledger = plan.TaskLedger()
    ledger.update(steps(("3", "third", "pending")), 1)
    ledger.update(steps(("1", "first", "pending")), 2)
    ledger.update(steps(("3", "done")), 3)
    assert ledger.plain() == "[x] third\n[ ] first"


def test_a_status_can_be_corrected_back():
    """A model that finds it was wrong must be able to say so."""
    ledger = declared(plan.TaskLedger(), "a")
    ledger.update(steps(("a", "done")), 2)
    ledger.update(steps(("a", "pending")), 3)
    assert ledger.plain() == "[ ] do a"
    assert ledger.done_count == 0


def test_ids_are_matched_case_and_punctuation_insensitively():
    ledger = plan.TaskLedger()
    ledger.update(steps(("Send", "send the message", "pending")), 1)
    ledger.update(steps(("send:", "done")), 2)
    assert len(ledger) == 1
    assert ledger.plain() == "[x] send the message"


# ---------------------------------------------------------------------------
# Crediting -- every test here is an attempt to buy a stall-ladder reset
# ---------------------------------------------------------------------------

def test_finishing_a_step_declared_earlier_is_credited():
    ledger = declared(plan.TaskLedger(), "a", "b")
    update = ledger.update(steps(("a", "done")), 2)
    assert update.completed == ["a"]


def test_declaring_and_finishing_in_one_turn_buys_nothing():
    """The whole of "rewording resets the ladder", closed off.

    A model that could mint credit by declaring five finished steps in one
    breath would be exactly where the old free-text field was: holding the reset
    switch of the guard that bounds it.
    """
    ledger = plan.TaskLedger()
    update = ledger.update(steps(("a", "do a", "done"), ("b", "do b", "done"),
                                 ("c", "do c", "done")), 1)
    assert update.completed == []
    assert ledger.done_count == 3, "the statuses are still recorded"


def test_a_step_is_credited_at_most_once():
    ledger = declared(plan.TaskLedger(), "a")
    assert ledger.update(steps(("a", "done")), 2).completed == ["a"]
    ledger.update(steps(("a", "pending")), 3)
    assert ledger.update(steps(("a", "done")), 4).completed == []


def test_oscillating_a_same_turn_completion_cannot_collect_later():
    """Reaching `done` consumes the credit whether or not it was paid.

    Without that, the same-turn rule above would only be a delay: declare the
    step done on arrival (unpaid), set it back to pending next turn, then finish
    it again and be paid for a step that was never outstanding.
    """
    ledger = plan.TaskLedger()
    ledger.update(steps(("a", "do a", "done")), 1)      # unpaid
    ledger.update(steps(("a", "pending")), 2)
    assert ledger.update(steps(("a", "done")), 3).completed == []


def test_restating_a_completion_buys_nothing():
    ledger = declared(plan.TaskLedger(), "a")
    ledger.update(steps(("a", "done")), 2)
    for step in (3, 4, 5, 6):
        assert ledger.update(steps(("a", "done")), step).completed == []


def test_rewording_a_step_is_not_a_completion():
    ledger = declared(plan.TaskLedger(), "a")
    for step, text in enumerate(["do a now", "doing a", "a, in progress"], 2):
        update = ledger.update([{"id": "a", "text": text}], step)
        assert update.completed == []
        assert update.changed == ["a"], "the text still updates"


def test_blocked_is_not_done():
    ledger = declared(plan.TaskLedger(), "a")
    assert ledger.update(steps(("a", "blocked")), 2).completed == []
    assert ledger.outstanding() == ["do a"]


# ---------------------------------------------------------------------------
# Prose, and everything else a model actually sends
# ---------------------------------------------------------------------------

def test_prose_lands_in_one_entry_and_is_never_credited():
    """The shape every recording made before the schema changed contains.

    It is not split into steps: "Done: opened app, found contact" has real
    structure and no reliable delimiter, and a step list guessed wrong is worse
    than none now that completions are credited.
    """
    ledger = plan.TaskLedger()
    for step in range(1, 16):
        update = ledger.update(f"Done: nothing. Next: attempt number {step}", step)
        assert update.completed == []
    assert ledger.plain() == "Done: nothing. Next: attempt number 15"
    assert len(ledger) == 0, "prose is not a step"
    assert ledger.credited == set()


def test_prose_and_steps_coexist():
    ledger = declared(plan.TaskLedger(), "a")
    ledger.update("still looking for the contact", 2)
    assert ledger.plain() == "[ ] do a\nstill looking for the contact"


def test_status_aliases_are_folded_rather_than_rejected():
    ledger = plan.TaskLedger()
    ledger.update([{"id": "a", "text": "do a"}, {"id": "b", "text": "do b"}], 1)
    ledger.update([{"id": "a", "status": "completed"},
                   {"id": "b", "status": "in progress"}], 2)
    assert ledger.plain() == "[x] do a\n[>] do b"


def test_an_unreadable_status_leaves_the_step_alone():
    """Demoting a step because its status was misspelt would undo real work."""
    ledger = declared(plan.TaskLedger(), "a")
    ledger.update(steps(("a", "done")), 2)
    ledger.update([{"id": "a", "status": "mostly there"}], 3)
    assert ledger.plain() == "[x] do a"


@pytest.mark.parametrize("sent, expected", [
    (["open the app", "send it"], "[ ] open the app\n[ ] send it"),
    ({"open the app": "done"}, "[x] open the app"),
    ({"id": "a", "text": "do a", "status": "done"}, "[x] do a"),
])
def test_the_shapes_a_model_reaches_for(sent, expected):
    ledger = plan.TaskLedger()
    ledger.update(sent, 1)
    assert ledger.plain() == expected


def test_a_step_with_no_id_falls_back_to_its_own_title():
    """A forgotten id is not a reason to lose a declared sub-step.

    The title is as stable a key as anything the model would have written, and
    it is the key the bare-list shape uses anyway. Only a step with no id *and*
    no title is dropped -- there is nothing left to hang it off.
    """
    ledger = plan.TaskLedger()
    ledger.update([{"text": "nameless"}, {"id": "a", "text": "do a"},
                   {"id": "", "text": ""}, {}], 1)
    assert ledger.plain() == "[ ] nameless\n[ ] do a"
    # And it is updatable under that key, so the fallback is not a dead end.
    ledger.update([{"id": "nameless", "status": "done"}], 2)
    assert ledger.done_count == 1


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_the_plan_is_capped_and_says_so():
    ledger = plan.TaskLedger()
    ledger.update([{"id": str(n), "text": f"step {n}"}
                   for n in range(plan.MAX_STEPS + 12)], 1)
    assert len(ledger) == plan.MAX_STEPS
    assert ledger.refused == 12
    assert "12 further step(s) refused" in ledger.plain()


def test_the_cap_refuses_new_steps_rather_than_evicting_the_live_plan():
    """Eviction would drop the steps the run is actually following."""
    ledger = plan.TaskLedger()
    ledger.update([{"id": "first", "text": "the one that matters"}], 1)
    ledger.update([{"id": str(n), "text": f"step {n}"}
                   for n in range(plan.MAX_STEPS + 5)], 2)
    assert "the one that matters" in ledger.plain()
    assert ledger.update([{"id": "first", "status": "done"}], 3).completed \
        == ["first"], "a surviving step is still updatable"


def test_long_titles_are_truncated_not_dropped():
    ledger = plan.TaskLedger()
    ledger.update([{"id": "a", "text": "x" * 400}], 1)
    assert len(ledger.plain()) <= plan.MAX_TEXT_CHARS + 8


# ---------------------------------------------------------------------------
# Rendering, and what reaches the model
# ---------------------------------------------------------------------------

def test_render_states_the_delta_contract_and_the_count():
    ledger = declared(plan.TaskLedger(), "a", "b")
    ledger.update(steps(("a", "done")), 2)
    rendered = ledger.render()
    assert "1 of 2 step(s) done" in rendered
    assert "do NOT restate" in rendered
    assert "[x] do a" in rendered and "[ ] do b" in rendered


def test_render_of_prose_alone_does_not_explain_a_checklist():
    ledger = plan.TaskLedger()
    ledger.update("Done: opened the app. Next: find the contact.", 1)
    rendered = ledger.render()
    assert "step(s) done" not in rendered
    assert "Done: opened the app" in rendered


def test_an_empty_plan_renders_to_nothing():
    assert plan.TaskLedger().render() == ""
    assert plan.TaskLedger().plain() == ""


# ---------------------------------------------------------------------------
# Through the schema, and back off a recorded run
# ---------------------------------------------------------------------------

def test_the_action_schema_accepts_both_shapes():
    prose = AgentAction(observation="o", reasoning="r", action="wait",
                        progress="Done: opened app. Next: send.")
    assert [s.id for s in prose.progress] == [plan.PROSE_ID]

    records = AgentAction(observation="o", reasoning="r", action="wait",
                          progress=[{"id": "1", "status": "done"}])
    assert records.progress[0].status == "done"
    # Omitted rather than guessed: the ledger reads a missing status as "no
    # change", which is what a text-only correction should mean.
    assert records.progress[0].text == ""


def test_replay_folds_the_deltas_of_a_finished_run():
    """The last delta is one step's change, not the plan -- so replay them all."""
    events = [
        {"kind": "run_start", "goal": "g"},
        {"kind": "decide", "step": 1, "action": {"progress": [
            {"id": "a", "text": "do a"}, {"id": "b", "text": "do b"}]}},
        {"kind": "decide", "step": 2, "action": {"progress": [
            {"id": "a", "status": "done"}]}},
        {"kind": "decide", "step": 3, "action": {"notes": [{"key": "k",
                                                            "value": "v"}]}},
        {"kind": "decide", "step": 4, "action": {"progress": [
            {"id": "b", "status": "blocked"}]}},
    ]
    assert plan.replay(events).plain() == "[x] do a\n[!] do b"


def test_replay_of_a_run_recorded_before_the_schema_changed():
    events = [{"kind": "decide", "step": n,
               "action": {"progress": f"Done: {n} of 5. Next: item {n + 1}."}}
              for n in (1, 2, 3)]
    assert plan.replay(events).plain() == "Done: 3 of 5. Next: item 4."
