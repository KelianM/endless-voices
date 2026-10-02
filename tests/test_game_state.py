"""Protect arithmetic, money and event timing used by authored dialogue."""

from endless_voices.dataset.source import tree
from endless_voices.dataset.state import GameState, apply, condition


def test_random_threshold_uses_multiplication_precedence():
    state = GameState.fixed({"help": 2})
    possible = state.assume(condition(tree("random < 10 + 7 * help * help"), state))
    assert 0 <= possible.snapshot()["draws"]["random:0"] < 38


def test_payment_updates_credits_before_later_condition():
    state = apply(tree("payment 50000"), GameState.fixed({"credits": 100}))
    assert state.snapshot()["current_values"]["credits"] == 50100
    assert state.assume(condition(tree("credits < 50000"), state)) is None
    state = apply(tree("payment -100000"), state)
    assert state.snapshot()["current_values"]["credits"] == 0


def test_default_event_waits_for_next_day_and_explicit_zero_is_immediate():
    state = GameState.fixed({"flag": 0})
    state.events = {"change": tree("event change\n\tset flag")[0]}
    queued = apply(tree("event change"), state)
    assert queued.snapshot()["current_values"]["flag"] == 0
    outcomes = queued.advance()
    assert {r.snapshot()["current_values"]["flag"] for r in outcomes} == {0, 1}
    for outcome in outcomes:
        saved = outcome.snapshot()
        assert (saved["elapsed_days"] >= 1) == (saved["current_values"]["flag"] == 1)
    assert apply(tree("event change 0"), state).snapshot()["current_values"]["flag"] == 1


def test_later_event_overwrites_earlier_effect_in_due_order():
    state = GameState.fixed({"reputation": 0})
    state.events = {
        name: tree(f"event {name}\n\treputation = {value}")[0]
        for name, value in [("start", 5), ("end", -1000)]
    }
    state = apply(tree("event start 1\nevent end 8"), state)
    outcomes = state.advance()
    assert {r.snapshot()["current_values"]["reputation"] for r in outcomes} == {0, 5, -1000}
    for outcome in outcomes:
        if outcome.snapshot()["current_values"]["reputation"] == -1000:
            assert not outcome.pending


def test_outfit_removal_keeps_a_nonnegative_resource_witness():
    state = apply(tree('outfit "Electron Beam" -1'), GameState())
    saved = state.snapshot()
    assert saved["initial_values"]["outfit: Electron Beam"] >= 1
    assert (
        saved["current_values"]["outfit: Electron Beam"]
        == saved["initial_values"]["outfit: Electron Beam"] - 1
    )


def test_shop_updates_remain_world_changes_without_becoming_conditions():
    state = GameState.fixed({"ready": 0})
    changes = tree('event shops\n\toutfitter "Depot"\n\t\t"Ramscoop"\n'
                   '\tshipyard "Yard"\n\t\t"Shuttle"\n\tset ready')[0]
    state.events = {"shops": changes}
    updated = apply(tree('event shops 0'), state)
    saved = updated.snapshot()
    assert saved["world_changes"] == changes["children"][:2]
    assert saved["current_values"]["ready"] == 1
    assert "outfitter" not in saved["current_values"]


def test_journal_entries_inside_dialogue_preserve_following_condition_changes():
    state = apply(tree('log "Met the captain."\nset introduced'), GameState())
    assert state.snapshot()["current_values"]["introduced"] == 1
    assert state.world[0]["tokens"] == ["log", "Met the captain."]


def test_fines_do_not_spend_credits_and_quoted_conditions_remain_literal_names():
    state = GameState.fixed({"credits": 10, "has mission: done": 1, "mission: done": 0})
    state = apply(tree('fine 70000'), state)
    assert state.snapshot()["current_values"]["credits"] == 10
    assert state.world[0]["tokens"] == ["fine", "70000"]
    assert state.assume(condition(tree('"has mission: done"'), state)) is not None
    assert state.assume(condition(tree('has "mission: done"'), state)) is None
