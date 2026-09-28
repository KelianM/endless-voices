"""Protect verbatim context extraction and the current and held-out answer boundaries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_conversations import tree  # noqa: E402
from prepare_source_context import first_opening, prose, protect_test  # noqa: E402


def test_source_examples_retain_narration_and_options_without_refusal_routes():
    node = tree('conversation\n\t`He smiles. "Hello!"`\n\tchoice\n'
                '\t\t`"What happened?"`\n\t\t`"No thanks."`\n\t\t\tdecline\n'
                '\t`"Nothing," he says.\t"Yet."`')[0]
    passages = prose(node)
    assert [p["text"] for p in passages] == [
        'He smiles. "Hello!"', '"What happened?"', '"Nothing," he says.\t"Yet."',
    ]
    assert passages[1]["kind"] == "option"


def test_current_opening_excludes_later_target_and_alternative_choices():
    offer = tree('on offer\n\tconversation\n\t\t`The opening.`\n'
                 '\t\tchoice\n\t\t\t`The question.`\n\t\t`Withheld target.`')[0]
    assert first_opening(offer)["tokens"] == ["The opening."]


def test_held_out_source_overlap_fails_before_context_export():
    metadata = [{"sources": [{"source_group": "mission / Example",
        "reference": "https://example.org/blob/rev/data/a%20b.txt#L10-L20"}]}]
    excerpts = [{"path": "data/a b.txt", "passages": [{"line": 20, "text": "Secret"}]}]
    with pytest.raises(ValueError, match="held-out"):
        protect_test(excerpts, metadata, "rev")
    excerpts[0]["passages"][0]["line"] = 9
    protect_test(excerpts, metadata, "rev")
