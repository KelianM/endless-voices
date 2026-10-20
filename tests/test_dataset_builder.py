"""Protect consistent histories, mission ownership and stable dataset loading."""

import hashlib
import json
from argparse import Namespace

import pytest
from torch.utils.data import DataLoader

from endless_voices.assessment import load_dataset
from endless_voices.context import render
from endless_voices.contracts import validate_manifest
from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.dataset.storage import SceneDataset
from endless_voices.gameplay.source import GameCorpus
from endless_voices.generate import select_samples


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


def test_each_target_gets_a_compatible_history_from_raw_source(tmp_path):
    source = corpus(tmp_path, STORY)
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}))
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
    assert all("requirements" not in row[state] for row in evidence.values()
               for state in ("state", "encounter_state"))
    for row in dataset:
        context = evidence[row["metadata"]["id"]]["context"]
        assert set(context) == {"pool_sha256"}
    assert validate_manifest(output / "manifest.json")["train"]["records"] == len(dataset)
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
        assert "childhood" not in json.dumps(row["messages"][:-1])
        pool = dataset.context_pool(next(i for i, r in enumerate(dataset) if r == row))
        prompt = json.dumps(pool.blocks)
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
def test_chain_splits_reject_heldout_ancestors_but_keep_unselected_history(tmp_path, owner):
    source = corpus(tmp_path, STORY)
    missions = {"Later": spec()}
    if owner is not None:
        missions["Earlier"] = {**spec(owner), "conversations": []}
    if owner in {"validation", "test"}:
        with pytest.raises(ValueError, match="chain crosses splits"):
            DatasetBuilder(source, config(missions))
        return
    builder = DatasetBuilder(source, config(missions))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert all("childhood" in json.dumps(dataset.context_pool(i).blocks)
               for i in range(len(dataset)))


def test_failed_interpretation_does_not_publish_a_partial_dataset(tmp_path):
    source = corpus(tmp_path, STORY.replace("set earth", "ship Sparrow"))
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}))
    with pytest.raises(ValueError, match="unsupported operation"):
        builder.build(tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_delayed_completion_event_keeps_predecessor_as_history(tmp_path):
    source = corpus(tmp_path, '''mission Earlier
	on offer
		conversation
			`You agree to transport Tom.`
				accept
	on complete
		event wedding 60
		conversation
			`Tom reunites with Carl.`
event wedding
mission Later
	to offer
		has "event: wedding"
	on offer
		conversation
			`Tom and Carl send their wedding photos.`
				decline
''')
    assert source.mission_graph["Later"] == ["Earlier"]
    with pytest.raises(ValueError, match="chain crosses splits"):
        DatasetBuilder(source, config({"Earlier": spec(), "Later": spec("test")}))
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}))
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    index = next(i for i, row in enumerate(dataset)
                 if row["metadata"]["sources"][0]["source_group"] == "mission / Later")
    pool = dataset.context_pool(index)
    assert [(block["mission"], block["phase"]) for block in pool.blocks] == [
        ("Earlier", "offer"), ("Earlier", "complete"),
    ]
    assert all(block["mission"] != "Earlier" for block in pool.references)


def test_event_history_keeps_parent_active_during_required_side_mission(tmp_path):
    text = STORY.replace('has "Earlier: done"', 'has "Earlier: active"').replace(
        'mission Later', '\ton complete\n\t\tevent ready 15\n'
        '\t\tdialog `The journey is complete.`\nmission Later')
    text += '''event ready
mission Finale
	to offer
		has "Later: done"
		has "event: ready"
	on offer
		conversation
			`You return after both journeys.`
				decline
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({name: spec() for name in (
        "Earlier", "Later", "Finale")}))
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    indices = [i for i, row in enumerate(dataset)
               if row["metadata"]["sources"][0]["source_group"] == "mission / Finale"]
    assert indices
    for index in indices:
        assert [(b["mission"], b["phase"]) for b in dataset.context_pool(index).blocks] == [
            ("Earlier", "offer"), ("Later", "offer"), ("Earlier", "complete"),
        ]


def test_dataset_rejects_modified_target_files(tmp_path):
    source = corpus(tmp_path, STORY)
    DatasetBuilder(source, config({"Earlier": spec()})).build(tmp_path / "prepared")
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
    builder = DatasetBuilder(source, config({"Example": spec()}))
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
    builder = DatasetBuilder(source, config({"Earlier": selection}))
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
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Later": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 1
    history = "\n\n".join(map(render, dataset.context_pool(0).blocks))
    assert 'She returns with the papers.\n\n"We can leave," she says.' in history
    assert dataset.target(0) == "The next passage."


def test_context_samples_event_timing_without_collapsing_current_mission_outcomes(tmp_path):
    from endless_voices.gameplay.histories import History
    from endless_voices.gameplay.source import tree
    from endless_voices.gameplay.state import GameState, apply

    builder = DatasetBuilder(corpus(tmp_path, STORY), config({"Later": spec()}))
    state = GameState.fixed({"ready": 0})
    state.events = {"change": tree('event change\n\tset ready')[0]}
    history = History(apply(tree('event change 1'), state), [])
    all_outcomes = builder.sampler.gameplay.advance([history])
    assert {h.state.snapshot()["current_values"]["ready"] for h in all_outcomes} == {0, 1}
    sampled = builder.sampler.gameplay.advance([history], sample=True)
    assert len(sampled) == 1
    witness = sampled[0].state.snapshot()
    assert witness in [h.state.snapshot() for h in all_outcomes]
    assert witness == builder.sampler.gameplay.advance([history], sample=True)[0].state.snapshot()
    assert (witness["elapsed_days"] >= 1) == (witness["current_values"]["ready"] == 1)


def test_reference_context_keeps_distinct_branches_without_target_or_heldout(tmp_path):
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
                             }))
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    for index, row in enumerate(dataset):
        pool = dataset.context_pool(index)
        prompt = json.dumps(pool.blocks + pool.references)
        assert "Own later answer." not in prompt
        if row["messages"][-1]["content"] == "Current answer.":
            assert "Current answer." not in prompt
        assert "Held-out answer." not in prompt
        assert "An Earth childhood." in prompt and "A Mars childhood." in prompt
        references = pool.references
        assert len(references) == 2
        for reference in references:
            text = json.dumps(reference)
            assert reference["mission"] == "Reference"
            assert ("An Earth childhood." in text) != ("A Mars childhood." in text)
            assert "state" not in reference


def test_shared_completion_flag_is_a_condition_not_a_missing_mission(tmp_path):
    story = STORY.replace('has "Earlier: done"',
                          'has "Earlier: done"\n\t\thas "Shared outcome: done"')
    builder = DatasetBuilder(corpus(tmp_path, story), config({"Later": spec()}))
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
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Later": spec()}))
    builder.build(tmp_path / "prepared")
    rows = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(rows) == 1
    assert "The earlier task is finished." in json.dumps(rows.context_pool(0).blocks)


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
    builder = DatasetBuilder(corpus(tmp_path, text), config({"Journey": spec()}))
    builder.build(tmp_path / "prepared")
    targets = {row["messages"][-1]["content"]
               for row in SceneDataset.load(tmp_path / "prepared", "train")}
    assert targets == {"The port comes into view.", "The journey is complete."}


def test_named_failure_runs_cleanup_once_after_current_actions(tmp_path):
    from endless_voices.gameplay.histories import History
    from endless_voices.gameplay.source import tree
    from endless_voices.gameplay.state import GameState

    text = '''mission Earlier
\ton fail
\t\tfail Earlier
\t\tcleanup += 1
mission Current
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Current": spec()}))
    state = GameState.fixed({"Earlier: active": 1, "cleanup": 0})
    routes = builder.sampler.gameplay.event(tree('fail Earlier\nset after'),
                                   [History(state, [])], source.missions["Current"], "accept")
    saved = routes[0].state.snapshot()["current_values"]
    assert saved["Earlier: active"] == 0
    assert saved["Earlier: failed"] == 1
    assert saved["cleanup"] == 1
    assert saved["after"] == 1


def test_unique_mission_owned_counter_starts_at_zero_without_replacing_prior_state(tmp_path):
    from endless_voices.gameplay.state import GameState

    text = '''mission Counter
\ton offer
\t\tconversation
\t\t\taction
\t\t\t\tquestions ++
\t\t\t`An answer.`
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Counter": spec()}))
    absent = GameState()
    builder.sampler.gameplay.enter(absent, source.missions["Counter"])
    assert absent.snapshot()["current_values"]["questions"] == 0
    existing = GameState.fixed({"questions": 2})
    builder.sampler.gameplay.enter(existing, source.missions["Counter"])
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
                             "HeldOut": spec("validation")}))
    builder.build(tmp_path / "prepared")
    rows = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(rows) == 1
    assert "Held-out authored passage." not in json.dumps(rows.context_pool(0).__dict__)


def test_shared_conversation_scratch_flags_default_zero_but_scores_remain_external(tmp_path):
    from endless_voices.gameplay.state import GameState

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
    builder = DatasetBuilder(source, config({"First": spec(), "Second": spec()}))
    state = GameState()
    builder.sampler.gameplay.enter(state, source.missions["Second"])
    assert state.snapshot()["current_values"]["discussed"] == 0
    assert "medicine" not in state.values
    previous = GameState.fixed({"discussed": 1, "medicine": 3})
    builder.sampler.gameplay.enter(previous, source.missions["Second"])
    saved = previous.snapshot()["current_values"]
    assert saved["discussed"] == 1
    assert saved["medicine"] == 3


def test_history_projection_retains_correlated_future_choices_and_whole_witness(tmp_path):
    import z3

    from endless_voices.gameplay.histories import History
    from endless_voices.gameplay.state import GameState

    sampler = DatasetBuilder(corpus(tmp_path, "mission Target\n"),
                             config({"Target": spec()})).sampler.gameplay
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
    sampler = DatasetBuilder(source, config({"Target": spec()})).sampler.gameplay
    assert {"live", "carry", "prior", "Target: active"} <= sampler.future_variables(["Target"])


def test_final_targets_do_not_require_materializing_unused_terminal_histories(tmp_path):
    from endless_voices.gameplay.state import GameState

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
    builder = DatasetBuilder(source, config({"Target": spec()}))
    builder.sampler.gameplay.max_routes = 1
    rows = builder.examples.build(source.missions["Target"], spec(), GameState())
    assert {row[0]["messages"][-1]["content"] for row in rows} == {
        "Left ending.", "Right ending."}
    source.missions["Target"].node["children"][0]["tokens"] = ["on", "offer"]
    with pytest.raises(ValueError, match="History route limit"):
        builder.examples.build(source.missions["Target"], spec(), GameState())


def test_repeated_dialog_entries_preserve_all_paragraphs_in_later_context(tmp_path):
    from endless_voices.gameplay.state import GameState

    source = corpus(tmp_path, '''mission Target
\ton accept
\t\tdialog `First paragraph.`
\t\tdialog `Second paragraph.`
\ton complete
\t\tconversation
\t\t\t`Final passage.`
\t\t\t\taccept
''')
    builder = DatasetBuilder(source, config({"Target": spec()}))
    rows = builder.examples.build(source.missions["Target"], spec(), GameState())
    assert len(rows) == 1
    prompt = "\n\n".join(map(render, rows[0][1].blocks))
    assert "First paragraph.\n\nSecond paragraph." in prompt


def test_raw_context_reconstruction_excludes_target_and_shared_mutation(tmp_path):
    from endless_voices.retrieval import SceneDatabase

    class Encoder:
        identity = "fixture"

        def embed(self, texts, *, kind):
            return [[1.0, float(len(text))] for text in texts]


    source = corpus(tmp_path, STORY)
    builder = DatasetBuilder(source, config({"Earlier": spec(), "Later": spec()}))
    output = tmp_path / "prepared"
    builder.build(output)
    dataset = SceneDataset.load(output, "train")
    assert not (output / "retrieval.json").exists()
    for index, record in enumerate(dataset):
        pool = dataset.context_pool(index)
        assert pool.sample_id == record["metadata"]["id"]
        assert record["messages"][-1]["content"] not in json.dumps(pool.__dict__)
        if pool.blocks:
            original = pool.blocks[0]["passages"][0]["text"]
            pool.blocks[0]["passages"][0]["text"] = "Changed by caller"
            assert dataset.context_pool(index).blocks[0]["passages"][0]["text"] == original
    assert all("references" not in row and "graph" not in row
               for row in dataset.evidence.values())
    pools = [dataset.context_pool(i) for i in range(len(dataset))]
    blocks = [block for pool in pools for block in pool.blocks + pool.references]
    with SceneDatabase.create(tmp_path / "context.sqlite", blocks, Encoder()) as database:
        index = next(i for i in range(len(dataset)) if dataset.context_pool(i).blocks)
        session = dataset.retrieval_session(index, database)
        matches = session.call_tool("search_context", {})
        assert matches
        passages = session.call_tool("read_context", {"ids": [m["id"] for m in matches]})
        assert all(dataset.target(index) not in passage["text"] for passage in passages)
    (source.root / "missions.txt").write_text(STORY + "\n")
    with pytest.raises(ValueError, match="Source hash differs"):
        SceneDataset.load(output, "train")


def test_state_provenance_tampering_is_rejected(tmp_path):
    source = corpus(tmp_path, STORY)
    output = tmp_path / "prepared"
    DatasetBuilder(source, config({"Later": spec()})).build(output)
    path = output / "provenance.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Dataset artifact differs: provenance.json"):
        SceneDataset.load(output, "train")


def test_prepared_dataset_requires_hashed_provenance(tmp_path):
    source = corpus(tmp_path, STORY)
    output = tmp_path / "prepared"
    DatasetBuilder(source, config({"Later": spec()})).build(output)
    manifest = json.loads((output / "manifest.json").read_text())
    del manifest["artifacts"]["provenance.json"]
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "provenance.json").unlink()
    with pytest.raises(ValueError, match="require hashed configuration and provenance"):
        SceneDataset.load(output, "train")


def test_saved_prompt_bundles_require_rebuilding(tmp_path):
    source = corpus(tmp_path, STORY)
    output = tmp_path / "prepared"
    DatasetBuilder(source, config({"Later": spec()})).build(output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["format"] = "scene-dataset-v1"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported prepared dataset format"):
        SceneDataset.load(output, "train")


def test_planet_references_keep_names_and_only_applied_descriptions(tmp_path):
    # A reference must identify the planet without exposing a future event's description.
    from endless_voices.gameplay.state import GameState

    source = corpus(tmp_path, """system Wah
	object Cloudfire
planet Cloudfire
	description `A quiet planet.`
event future
	planet Cloudfire
		description `A ruined planet.`
mission Visit
	destination Cloudfire
	on offer
		conversation
			`A landing.`
""")
    builder = DatasetBuilder(source, config({"Visit": spec()}))
    _, pool, _ = builder.examples.build(source.missions["Visit"], spec(), GameState.fixed({}))[0]
    planet = next(b for b in pool.references if b.get("source_kind") == "planet")
    assert planet["heading"] == "Cloudfire — planet, Wah system"
    assert planet["heading"] in pool.system_prefix
    assert "A quiet planet." in str(planet)
    assert "A ruined planet." not in str(pool)
    changed = source.planet_references(source.events["future"]["children"])[0]
    assert changed["passages"][0]["text"] == "A ruined planet."
    assert changed["path"] == "missions.txt"


def test_scene_cue_uses_event_location_without_completion_answer(tmp_path):
    # Completion payments and dialogue outcomes must not enter the scene cue.
    from endless_voices.gameplay.state import GameState

    source = corpus(tmp_path, """mission Journey
	source Origin
	destination Destination
	on offer
		conversation
			`Ready to depart.`
				accept
	on complete
		payment 500000
		conversation
			`The passenger reveals the secret.`
""")
    builder = DatasetBuilder(source, config({"Journey": spec()}))
    rows = builder.examples.build(source.missions["Journey"], spec(), GameState.fixed({}))
    for record, pool, _ in rows:
        prompt = json.dumps(pool.messages())
        assert "Current mission: Journey" in prompt
        if "secret" in record["messages"][-1]["content"]:
            assert "Scene: on complete. Current location: Destination." in prompt
            from endless_voices.retrieval import scene_id

            assert len(pool.blocks) == 1 and pool.blocks[0]["phase"] == "offer"
            assert f"Journey | on offer | {scene_id(pool.blocks[0])}" in prompt
            assert "Journey | on complete |" not in prompt
        else:
            assert "Scene: on offer. Current location: Origin." in prompt
        assert "500000" not in prompt and "reveals the secret" not in prompt


def test_history_keeps_timeline_that_allows_next_mission(tmp_path):
    story = '''event expired
\tset expired
mission Earlier
\ton complete
\t\tevent expired 1
mission Middle
\tto offer
\t\thas "Earlier: done"
\t\tnot expired
mission Later
\tto offer
\t\thas "Middle: done"
\ton offer
\t\tconversation
\t\t\t`You made it in time.`
'''
    from endless_voices.gameplay.state import GameState

    builder = DatasetBuilder(corpus(tmp_path, story), config({"Later": spec()}))
    builder.sampler.gameplay.seed = "0"
    rows = builder.examples.build(builder.corpus.missions["Later"], spec(),
                                  GameState.fixed({"expired": 0}))
    assert rows
    assert all(row[0]["messages"][-1]["content"] == "You made it in time." for row in rows)


@pytest.mark.parametrize("stop", ["decline", "defer", "accept"])
def test_offer_event_history_requires_the_scheduling_branch_without_completion(tmp_path, stop):
    source = corpus(tmp_path, f'''mission Poll
	on offer
		conversation
			choice
				`Ignore the poll.`
					decline
				`Vote in the poll.`
			action
				event results 7
			`Your vote was recorded.`
				{stop}
	on complete
		dialog `This mission was completed.`
event results
mission Followup
	to offer
		has "event: results"
		has "Poll: offered"
		not "Poll: done"
	on offer
		conversation
			`Here are the poll results.`
				decline
''')
    builder = DatasetBuilder(source, config({"Followup": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 1
    pool = dataset.context_pool(0)
    assert [(b["mission"], b["phase"]) for b in pool.blocks] == [("Poll", "offer")]
    assert "Your vote was recorded." in json.dumps(pool.blocks)
    assert "Ignore the poll." not in json.dumps(pool.blocks)
    assert "This mission was completed." not in json.dumps(pool.blocks)


def test_alternative_event_producers_remain_separate_compatible_histories(tmp_path):
    text = STORY[:STORY.index('mission Later')]
    for name, requirement in [("EarthTrip", "has earth"), ("MarsTrip", "not earth")]:
        text += f'''mission {name}
	to offer
		has "Earlier: done"
		{requirement}
	on offer
		conversation
			`You start {name}.`
				accept
	on complete
		event research 14
		dialog `You finish {name}.`
'''
    text += STORY[STORY.index('mission Later'):].replace(
        'has "Earlier: done"', 'has "event: research"') + 'event research\n'
    source = corpus(tmp_path, text)
    assert source.mission_graph["Later"] == ["EarthTrip", "MarsTrip"]
    with pytest.raises(ValueError, match="chain crosses splits"):
        DatasetBuilder(source, config({"EarthTrip": spec("test"), "Later": spec()}))
    builder = DatasetBuilder(source, config({"Later": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 2
    for i, row in enumerate(dataset):
        pool = dataset.context_pool(i)
        earth = "Earth" in row["messages"][-1]["content"]
        chosen, excluded = ("EarthTrip", "MarsTrip") if earth else ("MarsTrip", "EarthTrip")
        names = {b["mission"] for b in pool.blocks}
        assert names == {"Earlier", chosen}
        assert excluded not in {b["mission"] for b in pool.references}
        assert "EarthTrip (on complete) OR MarsTrip (on complete)" in pool.system_prefix


def test_offer_event_history_can_include_travel_while_parent_remains_active(tmp_path):
    source = corpus(tmp_path, '''mission Journey
	on offer
		event reminder 14
		conversation
			`Investigate the missing links.`
				accept
	on enter Somewhere
		set investigated
		dialog `The link is missing.`
	on complete
		dialog `You finished the journey.`
event reminder
mission Return
	to offer
		has "event: reminder"
		has "Journey: active"
		has investigated
	on offer
		conversation
			`Return to the council.`
				decline
''')
    builder = DatasetBuilder(source, config({"Return": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 1
    assert [(b["mission"], b["phase"]) for b in dataset.context_pool(0).blocks] == [
        ("Journey", "offer"), ("Journey", "enter"),
    ]


@pytest.mark.parametrize("mode", ["sequential", "overlapping", "dependent", "exclusive"])
def test_shared_event_allows_multiple_producers_unless_conditions_forbid_them(tmp_path, mode):
    text = ""
    for name, other in [("Alpha", "Beta"), ("Beta", "Alpha")]:
        requirement = ('not claimed' if mode == "exclusive" else
                       'has "Alpha: done"' if mode == "dependent" and name == "Beta" else
                       f'not "{other}: done"' if name == "Alpha" or mode == "overlapping"
                       else 'not impossible')
        text += f'''mission {name}
	to offer
		{requirement}
	on offer
'''
        if mode == "exclusive":
            text += '\t\tset claimed\n'
        text += f'''		conversation
			`You take the {name} job.`
				accept
	on complete
		event ready 1
		dialog `{name} is finished.`
'''
    text += '''event ready
mission Later
	to offer
		has "event: ready"
	on offer
		conversation
			branch both
				has "Alpha: done"
				has "Beta: done"
			`Thanks for the first job.`
				decline
			label both
			`Thanks for both jobs.`
				decline
'''
    source = corpus(tmp_path, text)
    settings = config({"Later": spec()})
    settings["initial_state"] = {"Alpha: done": 0, "Beta: done": 0, "claimed": 0}
    builder = DatasetBuilder(source, settings)
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    both = [i for i, row in enumerate(dataset) if "both jobs" in row["messages"][-1]["content"]]
    assert bool(both) == (mode != "exclusive")
    if both:
        blocks = dataset.context_pool(both[0]).blocks
        assert {(b["mission"], b["phase"]) for b in blocks} == {
            ("Alpha", "offer"), ("Alpha", "complete"),
            ("Beta", "offer"), ("Beta", "complete"),
        }
        assert len(blocks) == 4
        if mode == "overlapping":
            assert [b["phase"] for b in blocks] == ["offer", "offer", "complete", "complete"]


def test_active_side_mission_can_complete_after_parent_event(tmp_path):
    text = STORY.replace('has "Earlier: done"', 'has "Earlier: active"').replace(
        'mission Later', '\ton complete\n\t\tevent ready 15\n'
        '\t\tdialog `The main trip ends.`\nmission Later')
    text += '''	to complete
		has "event: ready"
	on complete
		dialog `The side trip ends after the wait.`
event ready
mission Finale
	to offer
		has "Later: done"
		has "event: ready"
	on offer
		conversation
			`Both trips are over.`
				decline
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Finale": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset)
    for index in range(len(dataset)):
        assert [(b["mission"], b["phase"]) for b in dataset.context_pool(index).blocks] == [
            ("Earlier", "offer"), ("Later", "offer"),
            ("Earlier", "complete"), ("Later", "complete"),
        ]


def test_event_gate_includes_milestone_that_changes_an_earlier_absent_flag(tmp_path):
    text = '''mission Start
	to offer
		not chapter_done
	on offer
		event island_open
		conversation
			`The island opens.`
				accept
mission Finish
	to offer
		has "Start: done"
	on complete
		set chapter_done
		dialog `The chapter ends.`
mission Timer
	to offer
		has "event: island_open"
		has chapter_done
	on offer
		event ready 7
		fail
mission Later
	to offer
		has "event: ready"
	on offer
		conversation
			`The next chapter begins.`
				decline
event island_open
event ready
'''
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Later": spec()}))
    builder.build(tmp_path / "prepared")
    dataset = SceneDataset.load(tmp_path / "prepared", "train")
    assert len(dataset) == 1
    assert [(b["mission"], b["phase"]) for b in dataset.context_pool(0).blocks] == [
        ("Start", "offer"), ("Finish", "complete"),
    ]


@pytest.mark.parametrize("can_clear", [False, True])
def test_producer_is_pruned_only_when_its_flag_permanently_blocks_target(tmp_path, can_clear):
    text = '''mission Ending
	on complete
		set ended
		event ready
mission Ordinary
	on complete
		event ready
event ready
mission Later
	to offer
		has "event: ready"
		not ended
'''
    if can_clear:
        text += 'event reset\n\tclear ended\n'
    source = corpus(tmp_path, text)
    builder = DatasetBuilder(source, config({"Later": spec()}))
    plans = builder.sampler.gameplay.history_plans("Later")
    assert any("Ending" in stages for stages, _ in plans) == can_clear
    assert any("Ordinary" in stages for stages, _ in plans)


def test_named_conversation_cannot_cross_assigned_mission_splits(tmp_path):
    source = corpus(tmp_path, (
        'conversation "shared"\n\t`A shared target.`\n'
        'mission "First"\n\ton offer\n\t\tconversation "shared"\n'
        'mission "Second"\n\ton complete\n\t\tconversation "shared"\n'
        'mission "Unselected"\n\ton offer\n\t\tconversation "shared"\n'
    ))
    with pytest.raises(ValueError, match="crosses mission splits"):
        DatasetBuilder(source, config({"First": spec(), "Second": spec("validation")}))
    builder = DatasetBuilder(source, config({
        "First": spec("validation"), "Second": spec("validation")}))
    assert builder.sampler.conversation_owners == {"shared": "validation"}
    assert source.named_conversation_users() == {"shared": {"First", "Second", "Unselected"}}


def test_sampled_mission_values_survive_history_and_prompt_reconstruction(tmp_path):
    from endless_voices.view_samples import viewer_data

    source = corpus(tmp_path, '''system Start
	government Syndicate
	object Alpha
	object Beta
planet Alpha
	spaceport `Alpha port.`
planet Beta
	spaceport `Beta port.`
ship Argosy
ship Argosy "Argosy (Blaster)"
mission Hijack
	source
		government Syndicate
	npc
		ship "Argosy (Blaster)" "Moonshake Express"
	on offer
		conversation
			`The <npc>, an <npc model>, is on <origin>.`
			choice
				`Help on <origin>.`
			`Return to <planet>.`
	on complete
		dialog `Saved the <npc> on <origin>.`
mission Later
	to offer
		has "Hijack: done"
	on offer
		conversation
			`Welcome back.`
''')
    builder = DatasetBuilder(source, config({'Hijack': spec(), 'Later': spec()}))
    output = tmp_path / 'prepared'
    builder.build(output)
    dataset = SceneDataset.load(output, 'train')
    viewer, _ = viewer_data(output / 'manifest.json')
    assert [s['messages'] for s in viewer['samples']] == [r['messages'] for r in dataset]
    origins = set()
    for i, row in enumerate(dataset):
        pool = dataset.context_pool(i)
        assert pool.messages() == dataset.prompt(i)
        for block in pool.blocks:
            if block['mission'] == 'Hijack':
                origin = block['substitutions']['<origin>']
                origins.add(origin)
                assert all('<' not in p['text'] for p in block['passages'])
        if pool.mission == 'Hijack':
            origin = pool.variables['<origin>']
            origins.add(origin)
            assert f'origin: {origin}' in pool.system_prefix
            assert 'npc: Moonshake Express' in pool.system_prefix
            assert 'npc model: Argosy' in pool.system_prefix
            assert all('<' not in m['content'] for m in row['messages'])
            choices = viewer['samples'][i]['player_choices']
            if choices:
                assert choices['selected'] == f'Help on {origin}.'
    assert len(origins) == 1
