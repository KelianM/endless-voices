"""Protect gameplay reconstruction without dataset configuration."""

import hashlib

from endless_voices.gameplay.dialogue import DialogueInterpreter
from endless_voices.gameplay.histories import HistoryReconstructor
from endless_voices.gameplay.source import GameCorpus
from endless_voices.gameplay.state import GameState


def test_encounters_reconstruct_history_without_dataset_metadata(tmp_path):
    text = '''mission Earlier
\ton complete
\t\tdialog `The first job is finished.`
mission Later
\tto offer
\t\thas "Earlier: done"
\ton offer
\t\tconversation
\t\t\t`Welcome back, <first>.`
'''
    (tmp_path / "missions.txt").write_text(text)
    corpus = GameCorpus(tmp_path, {"revision": "fixture", "files": [
        {"path": "missions.txt", "sha256": hashlib.sha256(text.encode()).hexdigest()}]})
    gameplay = HistoryReconstructor(corpus, DialogueInterpreter(), {"<first>": "Morgan"})
    encounters, omitted = gameplay.encounters(corpus.missions["Later"], GameState())
    assert omitted == []
    assert len(encounters) == 1
    encounter = encounters[0]
    assert encounter.phase == "offer"
    assert encounter.route.text == "Welcome back, <first>."
    assert [(b["mission"], b["phase"]) for b in encounter.history.blocks] == [
        ("Earlier", "complete")]
    assert encounter.history.state.snapshot()["current_values"]["Earlier: done"] == 1
    _, omitted = gameplay.encounters(corpus.missions["Later"], GameState(), excluded={"Earlier"})
    assert omitted == ["Earlier"]


def test_equivalent_condition_order_keeps_the_same_history_witness():
    from types import SimpleNamespace

    import z3

    from endless_voices.gameplay.histories import History

    credits, reputation = z3.Ints("credits reputation")
    conditions = [
        z3.Or(credits > 0, z3.And(reputation > 0, credits > 10)),
        z3.Or(z3.And(credits > 10, reputation > 0), credits > 0),
    ]
    histories = [History(GameState(values={"credits": credits, "reputation": reputation},
                                   constraints=(condition,)), [{"choice": choice}])
                 for choice, condition in enumerate(conditions)]
    gameplay = HistoryReconstructor(SimpleNamespace(mission_dependencies={}),
                                    DialogueInterpreter(), {})
    forward = gameplay.bounded(histories)
    backward = gameplay.bounded(list(reversed(histories)))
    assert len(forward) == 1
    assert forward[0] is backward[0]

    opposite = [History(GameState(values={"credits": credits, "reputation": reputation},
                                  constraints=(condition,)), [])
                for condition in (credits > reputation, reputation > credits)]
    assert len(gameplay.bounded(opposite)) == 2
