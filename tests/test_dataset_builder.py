"""Protect consistent histories, mission ownership and stable dataset loading."""

import hashlib
import json
from argparse import Namespace

import pytest
from torch.utils.data import DataLoader

from endless_voices.assessment import load_dataset
from endless_voices.contracts import validate_manifest
from endless_voices.data import load_conversations
from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.dataset.source import GameCorpus
from endless_voices.dataset.storage import SceneDataset
from endless_voices.generate import select_samples


class Counter:
    def messages(self, messages):
        return sum(len(m["content"]) for m in messages)

    def text(self, text):
        return len(text)


def corpus(tmp_path, text):
    source = tmp_path / "source"
    source.mkdir()
    (source / "missions.txt").write_text(text)
    for name in ["license.txt", "copyright", "credits.txt"]:
        (source / name).write_text("Test attribution")
    return GameCorpus(
        source,
        {
            "revision": "a" * 40,
            "files": [
                {"path": "missions.txt", "sha256": hashlib.sha256(text.encode()).hexdigest()}
            ],
        },
    )


def spec(split="train"):
    return {
        "split": split,
        "identity": "human",
        "species": "human",
        "character_role": "Captain",
        "topics": ["travel"],
        "scenario_group": None,
    }


def config(missions):
    return {
        "missions": missions,
        "seed": "fixture",
        "player": {},
        "context": {
            "name": "mission-depth",
            "depth": 4,
            "max_input_tokens": 8000,
            "seed": "fixture",
        },
    }


STORY = """mission Earlier
\ton offer
\t\tconversation
\t\t\tchoice
\t\t\t\t`I am from Earth.`
\t\t\t\t\tgoto earth
\t\t\t\t`I am from Mars.`
\t\t\t\t\tgoto mars
\t\t\tlabel earth
\t\t\taction
\t\t\t\tset earth
\t\t\t`An Earth childhood.`
\t\t\t\taccept
\t\t\tlabel mars
\t\t\taction
\t\t\t\tclear earth
\t\t\t`A Mars childhood.`
\t\t\t\taccept
mission Later
\tto offer
\t\thas "Earlier: done"
\ton offer
\t\tconversation
\t\t\tbranch earth
\t\t\t\thas earth
\t\t\t`You came from Mars.`
\t\t\t\taccept
\t\t\tlabel earth
\t\t\t`You came from Earth.`
\t\t\t\taccept
"""


def test_each_target_gets_a_compatible_history_and_shared_saved_context(tmp_path):
    source = corpus(tmp_path, STORY)
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}), Counter())
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    assert {p.name for p in output.iterdir()} == {
        "train.jsonl", "manifest.json", "config.json", "provenance.json", "licensing"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["build"]["git_revision"]
    assert manifest["build"]["source_hashes"]
    evidence = json.loads((output / "provenance.json").read_text())
    assert set(evidence) == {row["metadata"]["id"] for row in dataset}
    for row in dataset:
        context = evidence[row["metadata"]["id"]]["context"]
        assert context["messages_sha256"]
        assert all({"mission", "path", "lines"} <= set(block)
                   for block in context["selected"] + context["omitted"])
    assert validate_manifest(output / "manifest.json")["train"]["records"] == len(dataset)
    assert load_conversations(output) == [row["messages"] for row in dataset]
    selected, _ = select_samples(Namespace(
        manifest=output / "manifest.json", split="train", sample_ids=None, limit=None))
    assert selected == list(dataset)
    assert list(load_dataset(output / "manifest.json", "train").values()) == list(dataset)
    later = [r for r in dataset if r["metadata"]["sources"][0]["source_group"] == "mission / Later"]
    assert len(later) == 2
    for row in later:
        prompt = json.dumps(row["messages"][:-1])
        target = row["messages"][-1]["content"]
        assert ("An Earth childhood." in prompt) == ("Earth" in target)
        assert ("A Mars childhood." in prompt) == ("Mars" in target)
        assert target not in prompt
    original = dataset[0]
    original["messages"][0]["content"] = "Changed"
    assert dataset[0]["messages"][0]["content"] != "Changed"
    assert len(list(DataLoader(dataset, batch_size=2, collate_fn=lambda rows: rows))) == 2
    builder.build(tmp_path / "again")
    assert (output / "train.jsonl").read_bytes() == (tmp_path / "again/train.jsonl").read_bytes()


@pytest.mark.parametrize("owner", [None, "validation", "test"])
def test_history_eligibility_excludes_heldout_not_unselected_missions(tmp_path, owner):
    source = corpus(tmp_path, STORY)
    missions = {"Later": spec()}
    if owner is not None:
        missions["Earlier"] = spec(owner)
    builder = DatasetBuilder(source, config(missions), Counter())
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 2
    assert all(("childhood" in json.dumps(dataset.prompt(i))) == (owner is None)
               for i in range(len(dataset)))
    notes = json.loads((tmp_path / "prepared/provenance.json").read_text())
    assert all(n["omitted_history_missions"] == ([] if owner is None else ["Earlier"])
               for n in notes.values())


def test_failed_interpretation_does_not_publish_a_partial_dataset(tmp_path):
    source = corpus(tmp_path, STORY.replace("set earth", "ship Sparrow"))
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}), Counter())
    with pytest.raises(ValueError, match="unsupported operation"):
        builder.build(tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_dataset_rejects_modified_target_files(tmp_path):
    source = corpus(tmp_path, STORY)
    DatasetBuilder(source, config({"Earlier": spec()}), Counter()).build(tmp_path / "prepared")
    (tmp_path / "prepared/train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="artifact differs"):
        SceneDataset.load(tmp_path / "prepared", "train")


def test_mission_assignments_apply_between_display_and_first_response(tmp_path):
    text = """mission Example
	to offer
		not flag
	on offer
		set flag
		conversation
			`Before the mission assignment.`
				to display
					not flag
			choice
				`Continue.`
			`After the mission assignment.`
				to display
					has flag
				accept
"""
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Example": spec()}), Counter())
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert {dataset.target(i) for i in range(len(dataset))} == {
        "Before the mission assignment.",
        "After the mission assignment.",
    }


def test_unhandled_requested_conversation_cannot_silently_disappear(tmp_path):
    source = corpus(tmp_path, STORY)
    selection = spec()
    selection["conversations"] = [999]
    builder = DatasetBuilder(source, config({"Earlier": selection}), Counter())
    with pytest.raises(ValueError, match="unreachable or use unsupported"):
        builder.build(tmp_path / "prepared")


def test_nested_literal_dialog_is_preserved_in_prerequisite_history(tmp_path):
    text = '''mission Earlier
\ton complete
\t\tdialog
\t\t\t`She returns with the papers.`
\t\t\t`"We can leave," she says.`
mission Later
\tto offer
\t\thas "Earlier: done"
\ton offer
\t\tconversation
\t\t\t`The next passage.`
\t\t\t\taccept
'''
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Later": spec()}), Counter())
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 1
    history = dataset.prompt(0)[0]["content"]
    assert 'She returns with the papers.\n\n"We can leave," she says.' in history
    assert dataset.target(0) == "The next passage."


def test_context_samples_event_timing_without_collapsing_current_mission_outcomes(tmp_path):
    from endless_voices.dataset.builder import History
    from endless_voices.dataset.source import tree
    from endless_voices.dataset.state import GameState, apply

    builder = DatasetBuilder(corpus(tmp_path, STORY), config({"Later": spec()}), Counter())
    state = GameState.fixed({"ready": 0})
    state.events = {"change": tree('event change\n\tset ready')[0]}
    history = History(apply(tree('event change 1'), state), [])
    all_outcomes = builder.sampler.advance([history])
    assert {h.state.snapshot()["current_values"]["ready"] for h in all_outcomes} == {0, 1}
    sampled = builder.sampler.advance([history], sample=True)
    assert len(sampled) == 1
    witness = sampled[0].state.snapshot()
    assert witness in [h.state.snapshot() for h in all_outcomes]
    assert witness == builder.sampler.advance([history], sample=True)[0].state.snapshot()
    assert (witness["elapsed_days"] >= 1) == (witness["current_values"]["ready"] == 1)


def test_reference_context_excludes_target_mission_and_heldout_and_resolves_one_branch(tmp_path):
    story = '''mission Target
\ton offer
\t\tconversation
\t\t\t`Current answer.`
\ton complete
\t\tconversation
\t\t\t`Own later answer.`
mission Heldout
\ton offer
\t\tconversation
\t\t\t`Held-out answer.`
mission Reference
\ton offer
\t\tconversation
\t\t\tchoice
\t\t\t\t`Earth.`
\t\t\t\t\tgoto earth
\t\t\t\t`Mars.`
\t\t\t\t\tgoto mars
\t\t\tlabel earth
\t\t\taction
\t\t\t\tset earth
\t\t\t`An Earth childhood.`
\t\t\t\taccept
\t\t\tlabel mars
\t\t\taction
\t\t\t\tclear earth
\t\t\t`A Mars childhood.`
\t\t\t\taccept
'''
    builder = DatasetBuilder(corpus(tmp_path, story),
                             config({"Target": spec(), "Heldout": spec("validation")}), Counter())
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    evidence = json.loads((output / "provenance.json").read_text())
    for row in dataset:
        prompt = json.dumps(row["messages"][:-1])
        assert "Own later answer." not in prompt
        if row["messages"][-1]["content"] == "Current answer.":
            assert "Current answer." not in prompt
        assert "Held-out answer." not in prompt
        assert ("An Earth childhood." in prompt) != ("A Mars childhood." in prompt)
        references = [b for b in evidence[row["metadata"]["id"]]["context"]["selected"]
                      if b.get("reference")]
        assert len(references) == 1 and references[0]["mission"] == "Reference"
        assert references[0]["state"]["current_values"]["earth"] == int(
            "An Earth childhood." in prompt)
