"""Protect temporal boundaries and whole-exchange selection in the context prototype."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "context_prototype", Path("scripts/prepare_context_prototype.py")
)
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)


def record(split, conversation, group=None):
    return {"metadata": {"split": split, "conversation_id": conversation, "scenario_group": group}}


def path(*ids):
    return {"events": [{"sample_id": value, "position": i} for i, value in enumerate(ids)]}


def test_reject_later_training_and_shared_scenario_leakage():
    records = {"a": record("validation", "a"), "b": record("train", "b")}
    with pytest.raises(ValueError, match="earlier split"):
        prototype.validate_paths([path("a", "b")], records)
    records["a"]["metadata"]["scenario_group"] = "shared"
    records["b"]["metadata"]["scenario_group"] = "shared"
    with pytest.raises(ValueError, match="crosses splits"):
        prototype.validate_paths([path("a"), path("b")], records)


def test_earlier_alternatives_allowed_but_current_alternatives_hidden():
    records = {"a": record("validation", "a"), "b": record("validation", "a"),
               "c": record("validation", "c")}
    declared = path("a", "b", "c")
    declared["events"][1]["position"] = 0
    prototype.validate_paths([declared], records)
    assert prototype.earlier_examples(declared["events"], declared["events"][1]) == []
    assert len(prototype.earlier_examples(declared["events"], declared["events"][2])) == 2
    declared["events"][1]["position"] = 1
    with pytest.raises(ValueError, match="share a position"):
        prototype.validate_paths([declared], records)


def test_budget_keeps_whole_recent_exchanges_without_private_fields():
    events = [{"sample_id": x, "speaker": x, "audience": "player"} for x in ("a", "b")]
    records = {x: {"messages": [
        {"role": "system", "content": "Private old profile"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": x},
    ]} for x in ("a", "b")}
    selected = prototype.select_reference(events, records, 1, len)
    assert selected == [events[1]]
    block = prototype.reference_block(selected, records)
    assert set(block[0]) == {"kind", "speaker", "audience", "dialogue"}
    assert [m["role"] for m in block[0]["dialogue"]] == ["user", "assistant"]
    assert prototype.select_reference(events, records, 0, len) == []
