"""Protect chronology and target boundaries when assembling source-context drafts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_validation_context import dependency_terms, route_prefix, source_spans  # noqa: E402
from prepare_conversations import tree  # noqa: E402


def test_optional_mission_is_not_mistaken_for_required_earlier_history():
    mission = tree('mission "Now"\n\tto offer\n\t\thas "Earlier: done"\n'
                   '\t\tor\n\t\t\thas "Optional: done"\n\t\t\trandom < 50\n'
                   '\t\tnot "Later: done"')[0]
    assert dependency_terms(mission) == ["Earlier: done"]


def test_source_route_preserves_selected_branch_and_stops_before_answer():
    conversation = tree('conversation\n\t`Opening`\n\tchoice\n\t\t`A`\n'
                        '\t\t\tgoto a\n\t\t`B`\n\t\t\tgoto b\n'
                        '\tlabel a\n\t`Other branch`\n\t\tgoto end\n'
                        '\tlabel b\n\t`Selected branch`\n\tlabel end\n'
                        '\t`Question`\n\t`Target`\n\t`Future`')[0]
    selected = route_prefix(conversation, [6, 12, 14], 15)
    assert 12 in selected
    assert 9 not in selected
    assert 15 not in selected
    assert 16 not in selected


def test_held_out_ranges_use_source_metadata_without_dialogue():
    metadata = [{"sources": [{"source_group": "mission / Held out", "reference":
                "https://example.org/blob/rev/data/a%20b.txt#L12-L23"}]}]
    assert source_spans(metadata, "rev") == [("data/a b.txt", 12, 23)]
