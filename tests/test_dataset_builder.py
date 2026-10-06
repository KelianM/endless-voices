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
    earlier = [r for r in dataset if r not in later]
    for row in earlier:
        assert row["metadata"]["task"] == "scene_continuation"
        assert row["messages"][0]["content"].startswith("Continue the scene with the next passage,")
        assert row["messages"][1]["content"] in {"I am from Earth.", "I am from Mars."}
    for row in later:
        assert row["metadata"]["task"] == "scene_opening"
        assert row["messages"][0]["content"].startswith("Write the opening passage of a scene,")
        assert row["messages"][1]["content"] == "Write the opening passage of a scene."
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


@pytest.mark.parametrize("owner", [None, "train", "validation", "test"])
def test_history_eligibility_excludes_heldout_not_unselected_missions(tmp_path, owner):
    source = corpus(tmp_path, STORY)
    missions = {"Later": spec()}
    if owner is not None:
        missions["Earlier"] = {**spec(owner), "conversations": []}
    builder = DatasetBuilder(source, config(missions), Counter())
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 2
    assert all(("childhood" in json.dumps(dataset.prompt(i))) == (owner in (None, "train"))
               for i in range(len(dataset)))
    notes = json.loads((tmp_path / "prepared/provenance.json").read_text())
    assert all(n["omitted_history_missions"] == ([] if owner in (None, "train") else ["Earlier"])
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
mission Variant
\ton offer
\t\tconversation
\t\t\t`Current answer.`
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
                             config({
                                 "Target": {**spec(), "scenario_group": "variants"},
                                 "Variant": {**spec(), "scenario_group": "variants",
                                             "conversations": []},
                                 "Heldout": spec("validation"),
                             }), Counter())
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


def test_shared_completion_flag_is_a_condition_not_a_missing_mission(tmp_path):
    story = STORY.replace('has "Earlier: done"',
                          'has "Earlier: done"\n\t\thas "Shared outcome: done"')
    builder = DatasetBuilder(corpus(tmp_path, story), config({"Later": spec()}), Counter())
    output = tmp_path / "prepared"
    builder.build(output)
    assert builder.sampler.graph["Later"] == ["Earlier"]
    evidence = json.loads((output / "provenance.json").read_text())
    assert len(evidence) == 2
    assert all(p["state"]["initial_values"]["Shared outcome: done"] != 0
               for p in evidence.values())


def test_completion_decline_does_not_discard_completed_predecessor(tmp_path):
    text = '''mission Earlier
\ton complete
\t\tconversation
\t\t\t`The earlier task is finished.`
\t\t\t\tdecline
mission Later
\tto offer
\t\thas "Earlier: done"
\ton offer
\t\tconversation
\t\t\t`The next task begins.`
\t\t\t\taccept
'''
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Later": spec()}), Counter())
    builder.build(tmp_path / "prepared")
    rows = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(rows) == 1
    assert "The earlier task is finished." in rows.prompt(0)[0]["content"]


def test_on_enter_applies_effects_before_later_passage(tmp_path):
    text = '''mission Journey
\ton enter Port
\t\tset arrived
\t\tconversation
\t\t\t`The port comes into view.`
\ton complete
\t\tconversation
\t\t\tbranch arrival
\t\t\t\thas arrived
\t\t\t`Unreachable passage.`
\t\t\t\tdecline
\t\t\tlabel arrival
\t\t\t`The journey is complete.`
'''
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Journey": spec()}), Counter())
    builder.build(tmp_path / "prepared")
    targets = {row["messages"][-1]["content"]
               for row in SceneDataset.load(tmp_path / "prepared", "train")}
    assert targets == {"The port comes into view.", "The journey is complete."}


def test_named_failure_runs_cleanup_once_after_current_actions(tmp_path):
    from endless_voices.dataset.builder import History
    from endless_voices.dataset.source import tree
    from endless_voices.dataset.state import GameState

    text = '''mission Earlier
\ton fail
\t\tfail Earlier
\t\tcleanup += 1
mission Current
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Current": spec()}), Counter())
    state = GameState.fixed({"Earlier: active": 1, "cleanup": 0})
    routes = builder.sampler.event(tree('fail Earlier\nset after'),
                                   [History(state, [])], source.missions["Current"], "accept")
    saved = routes[0].state.snapshot()["current_values"]
    assert saved["Earlier: active"] == 0
    assert saved["Earlier: failed"] == 1
    assert saved["cleanup"] == 1
    assert saved["after"] == 1


def test_unique_mission_owned_counter_starts_at_zero_without_replacing_prior_state(tmp_path):
    from endless_voices.dataset.state import GameState

    text = '''mission Counter
\ton offer
\t\tconversation
\t\t\taction
\t\t\t\tquestions ++
\t\t\t`An answer.`
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Counter": spec()}), Counter())
    absent = GameState()
    builder.sampler.enter(absent, source.missions["Counter"])
    assert absent.snapshot()["current_values"]["questions"] == 0
    existing = GameState.fixed({"questions": 2})
    builder.sampler.enter(existing, source.missions["Counter"])
    assert existing.snapshot()["current_values"]["questions"] == 2


def test_unassigned_caller_cannot_leak_validation_owned_named_conversation(tmp_path):
    text = '''conversation shared
\t`Held-out authored passage.`
mission HeldOut
\ton offer
\t\tconversation shared
mission Unassigned
\ton offer
\t\tconversation shared
mission Training
\ton offer
\t\tconversation
\t\t\t`Training continuation.`
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Training": spec(),
                             "HeldOut": spec("validation")}), Counter())
    builder.build(tmp_path / "prepared")
    rows = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(rows) == 1
    assert "Held-out authored passage." not in json.dumps(rows.prompt(0))


def test_shared_conversation_scratch_flags_default_zero_but_scores_remain_external(tmp_path):
    from endless_voices.dataset.state import GameState

    text = '''mission First
\ton complete
\t\tconversation
\t\t\taction
\t\t\t\tset discussed
\t\t\t\tmedicine = 3
\t\t\t`A discussion.`
\t\t\taction
\t\t\t\tclear discussed
\t\t\t`Finished.`
\t\t\t\tdecline
mission Second
\ton complete
\t\tconversation
\t\t\taction
\t\t\t\tset discussed
\t\t\t\tmedicine = 2
\t\t\t`Another discussion.`
\t\t\taction
\t\t\t\tclear discussed
\t\t\t`Finished again.`
\t\t\t\tdecline
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"First": spec(), "Second": spec()}), Counter())
    state = GameState()
    builder.sampler.enter(state, source.missions["Second"])
    assert state.snapshot()["current_values"]["discussed"] == 0
    assert "medicine" not in state.values
    previous = GameState.fixed({"discussed": 1, "medicine": 3})
    builder.sampler.enter(previous, source.missions["Second"])
    saved = previous.snapshot()["current_values"]
    assert saved["discussed"] == 1
    assert saved["medicine"] == 3


def test_history_projection_retains_correlated_future_choices_and_whole_witness(tmp_path):
    import z3

    from endless_voices.dataset.builder import History
    from endless_voices.dataset.state import GameState

    sampler = DatasetBuilder(corpus(tmp_path, "mission Target\n"),
                             config({"Target": spec()}), Counter()).sampler
    sampler.history_variables = {"future"}
    histories = []
    for irrelevant, positive in [(0, True), (1, True), (2, False)]:
        state = GameState()
        value, related = state.value("future"), state.value("related")
        state.values["old flag"] = z3.IntVal(irrelevant)
        state.constraints = (value == related, related > 0 if positive else related <= 0)
        histories.append(History(state, [{"witness": irrelevant}]))
    selected = sampler.bounded(histories)
    assert len(selected) == 2
    assert all(any(history is original for original in histories) for history in selected)
    assert {history.state.snapshot()["current_values"]["future"] > 0
            for history in selected} == {True, False}
    assert all(history.blocks[0]["witness"]
               == history.state.snapshot()["current_values"]["old flag"] for history in selected)


def test_history_dependencies_include_assignments_and_scheduled_event_conditions(tmp_path):
    source = corpus(tmp_path, '''event Later
\tcarry = prior
mission Target
\ton offer
\t\tconversation
\t\t\taction
\t\t\t\tlive = carry
\t\t\tbranch end
\t\t\t\tlive > 0
\t\t\t`Alternate.`
\t\t\tlabel end
\t\t\t`End.`
''')
    sampler = DatasetBuilder(source, config({"Target": spec()}), Counter()).sampler
    assert {"live", "carry", "prior", "Target: active"} <= sampler.future_variables(["Target"])


def test_final_targets_do_not_require_materializing_unused_terminal_histories(tmp_path):
    from endless_voices.dataset.state import GameState

    source = corpus(tmp_path, '''mission Target
\ton complete
\t\tconversation
\t\t\tchoice
\t\t\t\t`Left.`
\t\t\t\t\tgoto left
\t\t\t\t`Right.`
\t\t\t\t\tgoto right
\t\t\tlabel left
\t\t\taction
\t\t\t\tset flag
\t\t\t`Left ending.`
\t\t\t\taccept
\t\t\tlabel right
\t\t\taction
\t\t\t\tclear flag
\t\t\t`Right ending.`
\t\t\t\taccept
''')
    builder = DatasetBuilder(source, config({"Target": spec()}), Counter())
    builder.sampler.max_routes = 1
    rows = builder.examples.build(source.missions["Target"], spec(), Counter(), GameState())
    assert {row[0]["messages"][-1]["content"] for row in rows} == {
        "Left ending.", "Right ending."}
    source.missions["Target"].node["children"][0]["tokens"] = ["on", "offer"]
    with pytest.raises(ValueError, match="History route limit"):
        builder.examples.build(source.missions["Target"], spec(), Counter(), GameState())


def test_repeated_dialog_entries_preserve_all_paragraphs_in_later_context(tmp_path):
    from endless_voices.dataset.state import GameState

    source = corpus(tmp_path, '''mission Target
\ton accept
\t\tdialog `First paragraph.`
\t\tdialog `Second paragraph.`
\ton complete
\t\tconversation
\t\t\t`Final passage.`
\t\t\t\taccept
''')
    builder = DatasetBuilder(source, config({"Target": spec()}), Counter())
    rows = builder.examples.build(source.missions["Target"], spec(), Counter(), GameState())
    assert len(rows) == 1
    prompt = "\n".join(m["content"] for m in rows[0][0]["messages"][:-1])
    assert "First paragraph.\n\nSecond paragraph." in prompt
