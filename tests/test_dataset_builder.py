"""Protect consistent histories, mission ownership and stable dataset loading."""

import hashlib
import json

import pytest
from torch.utils.data import DataLoader

from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.dataset.source import GameCorpus
from endless_voices.dataset.storage import SceneDataset


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


def test_heldout_mission_text_never_enters_training_history(tmp_path):
    source = corpus(tmp_path, STORY)
    builder = DatasetBuilder(source, config({"Earlier": spec("test"), "Later": spec()}), Counter())
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 2
    assert all("childhood" not in json.dumps(dataset.prompt(i)) for i in range(len(dataset)))
    notes = json.loads((tmp_path / "prepared/provenance.json").read_text())
    assert all(n["omitted_history_missions"] == ["Earlier"] for n in notes.values())


def test_failed_interpretation_does_not_publish_a_partial_dataset(tmp_path):
    source = corpus(tmp_path, STORY.replace("set earth", "outfit Laser"))
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
