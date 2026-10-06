"""Protect source attribution when missions reuse dialogue definitions."""

import hashlib

import pytest

from endless_voices.dataset.source import GameCorpus
from endless_voices.dataset.state import UnsupportedOperation


def corpus(tmp_path, files):
    entries = []
    for path, text in files.items():
        (tmp_path / path).write_text(text)
        entries.append({"path": path, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return GameCorpus(tmp_path, {"revision": "fixture", "files": entries})


def test_named_conversation_keeps_definition_location_and_callsite(tmp_path):
    source = corpus(tmp_path, {
        "definitions.txt": 'conversation "shared"\n\t`An authored passage.`\n',
        "missions.txt": 'mission "Trip"\n\ton offer\n\t\tconversation "shared"\n',
    })
    mission = source.missions["Trip"]
    reference = mission.node["children"][0]["children"][0]
    resolved = source.resolve_conversation(reference, mission)
    assert resolved["source_path"] == "definitions.txt"
    assert resolved["source_sha256"] == source.files["definitions.txt"]
    assert resolved["line"] == 1
    assert resolved["reference_line"] == 3
    assert resolved["children"][0]["line"] == 2
    assert reference["children"] == []
    reference = {**reference, "tokens": ["conversation", "missing"]}
    with pytest.raises(UnsupportedOperation, match="Undefined conversation"):
        source.resolve_conversation(reference, mission)


def test_dialog_preserves_inline_children_and_named_phrase_sources(tmp_path):
    source = corpus(tmp_path, {
        "phrases.txt": 'phrase "payment"\n\tword\n\t\t`Payment received.`\n',
        "missions.txt": (
            'mission "Trip"\n\ton offer\n\t\tdialog `First paragraph.`\n'
            '\t\t\t`Second paragraph.`\n\ton complete\n\t\tdialog phrase "payment"\n'
        ),
    })
    mission = source.missions["Trip"]
    offer, complete = mission.node["children"]
    paragraphs = source.dialog_passages(offer["children"][0], mission)
    assert [(p["line"], p["text"]) for p in paragraphs] == [
        (3, "First paragraph."), (4, "Second paragraph."),
    ]
    payment = source.dialog_passages(complete["children"][0], mission)
    assert payment == [{"line": 3, "text": "Payment received.", "path": "phrases.txt",
                        "sha256": source.files["phrases.txt"]}]


def test_named_conversation_cannot_cross_assigned_mission_splits(tmp_path):
    source = corpus(tmp_path, {
        "source.txt": (
            'conversation "shared"\n\t`A shared target.`\n'
            'mission "First"\n\ton offer\n\t\tconversation "shared"\n'
            'mission "Second"\n\ton complete\n\t\tconversation "shared"\n'
            'mission "Unselected"\n\ton offer\n\t\tconversation "shared"\n'
        ),
    })
    with pytest.raises(ValueError, match="crosses mission splits"):
        source.conversation_owners({"First": "train", "Second": "validation"})
    assert source.conversation_owners({"First": "validation", "Second": "validation"}) == {
        "shared": "validation",
    }
    assert source.named_conversation_users() == {
        "shared": {"First", "Second", "Unselected"},
    }


def test_counter_ownership_includes_nonmission_writers(tmp_path):
    source = corpus(tmp_path, {
        "source.txt": (
            'mission "Trip"\n\ton offer\n\t\t"local counter" ++\n'
            '\t\tset "shared flag"\n'
            'event "change"\n\tclear "shared flag"\n\t"shared division" /= 2\n'
            '\t"shared remainder" %= 3\n'
            'conversation "shared"\n\taction\n\t\t"shared flag" += 1\n'
        ),
    })
    assert source.condition_writers["local counter"] == {("mission", "Trip")}
    assert source.condition_writers["shared division"] == {("event", "change")}
    assert source.condition_writers["shared remainder"] == {("event", "change")}
    assert source.condition_writers["shared flag"] == {
        ("mission", "Trip"), ("event", "change"), ("conversation", "shared"),
    }


def test_scratch_flags_require_clear_on_every_exit_and_every_writer(tmp_path):
    source = corpus(tmp_path, {
        "source.txt": (
            'conversation "first"\n\taction\n\t\tset "temporary"\n'
            '\t\tset "persistent"\n\t\tset "outside"\n'
            '\tbranch bypass\n\t\thas "condition"\n'
            '\taction\n\t\tclear "persistent"\n\tlabel bypass\n'
            '\taction\n\t\tclear "temporary"\n\t\tclear "outside"\n\taccept\n'
            'conversation "second"\n\taction\n\t\tset "temporary"\n'
            '\tchoice\n\t\t`Continue.`\n\t\t\tgoto finish\n'
            '\tlabel finish\n\taction\n\t\tclear "temporary"\n\taccept\n'
            'event "other writer"\n\tset "outside"\n'
        ),
    })
    assert source.scratch_conditions == {"temporary"}


@pytest.mark.parametrize("jump", ['\tgoto accept\n', '\t`Continue.`\n\t\tgoto accept\n'])
def test_goto_endpoint_named_label_does_not_hide_persistent_flag(tmp_path, jump):
    source = corpus(tmp_path, {
        "source.txt": (
            'conversation "shared"\n\taction\n\t\tclear "persistent"\n'
            + jump
            + '\tlabel accept\n\taction\n\t\tset "persistent"\n\tdecline\n'
        ),
    })
    assert "persistent" not in source.scratch_conditions
