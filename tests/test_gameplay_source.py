"""Protect source attribution when missions reuse dialogue definitions."""

import hashlib

import pytest

from endless_voices.gameplay.source import GameCorpus
from endless_voices.gameplay.state import UnsupportedOperation


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


@pytest.mark.parametrize("alternative", [
    'mission Other\n\ton decline\n\t\tevent wedding 10\n',
    'event calendar\n\tevent wedding 10\n',
    'conversation shared\n\taction\n\t\tevent wedding 10\n',
    'mission Other\n\ton complete\n\t\tset "event: wedding"\n',
    'event wedding\n\tdate 1 1 3014\n',
])
def test_event_with_another_trigger_does_not_imply_completed_mission(tmp_path, alternative):
    source = corpus(tmp_path, {"missions.txt": (
        'mission Earlier\n\ton complete\n\t\tevent wedding 60\n'
        'mission Later\n\tto offer\n\t\thas "event: wedding"\n' + alternative
    )})
    assert source.mission_graph["Later"] == []


def test_optional_or_uncompleted_event_does_not_become_history(tmp_path):
    source = corpus(tmp_path, {"missions.txt": (
        'mission Earlier\n\ton complete\n\t\tevent wedding 60\n'
        'mission Declined\n\ton decline\n\t\tevent rejected 1\n'
        'mission Later\n\tto offer\n\t\tor\n\t\t\thas "event: wedding"\n'
        '\t\t\thas other\n\t\thas "event: rejected"\n'
    )})
    assert source.mission_graph["Later"] == []


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
