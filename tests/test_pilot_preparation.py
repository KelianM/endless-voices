"""Protect original speech, conversation boundaries and reproducible dataset preparation."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pilot", SCRIPTS / "build_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from prepare_conversations import speech, tree  # noqa: E402


def test_coverage_hash_does_not_depend_on_topic_insertion_order(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    pilot.dump(first, {"train": {"topics": {"history": 3, "culture": 2}}})
    pilot.dump(second, {"train": {"topics": {"culture": 2, "history": 3}}})
    assert pilot.digest(first) == pilot.digest(second)


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "data/example.txt"
    source.parent.mkdir()
    lines = ['mission "Example"', '\ton offer', '\t\tconversation',
             '\t\t\tchoice', '\t\t\t\t`"May I land?"`',
             '\t\t\t`"Wait," says the clerk. "The berth is occupied."`',
             '\t\t\tchoice', '\t\t\t\t`"How long?"`',
             '\t\t\t`"I cannot say."`']
    source.write_text("\n".join(lines) + "\n")
    candidate = {
        "id": "example-l3", "path": "data/example.txt", "lines": [3, 9],
        "owner": ["mission", "Example"], "source_sha256": pilot.digest(source),
        "prose": [{"line": n, "kind": "choice" if n in (5, 8) else "paragraph",
                   "speech": speech(lines[n - 1])} for n in (5, 6, 8, 9)],
    }
    citation = {"path": "data/example.txt", "lines": [1, 9]}
    batch = {
        "identity": "example", "species": "human",
        "profiles": {"clerk": {"text": "You are a port clerk.", "sources": [citation]}},
        "lore": [{"id": "port", "text": "Port clerks manage berths.", "sources": [citation]}],
        "conversations": [{"id": "port-arrival", "catalog_id": "example-l3", "split": "train",
                           "profile": "clerk", "scenario_group": None, "scene": "A ship arrives.",
                           "lore_ids": ["port"], "topics": ["arrival"],
                           "review_notes": "Clerk is the only NPC; straight branch.",
                           "routes": [{"id": "ask", "turns": [
                               {"user": 5, "assistant": [{"line": 6,
                                                          "sentence_breaks_after": [0]}]},
                               {"user": 8, "assistant": [9]},
                           ]}]}],
    }
    return batch, {"revision": "a" * 40, "conversations": [candidate]}, tmp_path


def test_source_speech_survives_narrator_removal_and_history_is_original(example):
    rows, evidence = pilot.assemble(*example)
    assert len(rows) == 2
    assert rows[0]["messages"][-1]["content"] == "Wait. The berth is occupied."
    assert rows[1]["messages"][:-2] == rows[0]["messages"]
    assert evidence[0]["messages"][-1]["spans"][0]["normalization"] == "comma-to-period"
    assert rows[0]["metadata"]["review_status"] == "draft"
    assert rows[0]["evaluation"]["dimensions"] == ["authenticity"]


def test_duplicate_routes_do_not_inflate_sample_count(example):
    batch, _, _ = example
    batch["conversations"][0]["routes"] *= 2
    rows, _ = pilot.assemble(*example)
    assert len(rows) == 2


@pytest.mark.parametrize("selector", [1, -1, {"line": 6, "quotes": [-1]},
                                     {"line": 6, "quotes": [5]},
                                     {"line": 6, "quotes": [1, 0]}])
def test_source_selection_cannot_escape_conversation_or_index_backwards(example, selector):
    candidate = example[1]["conversations"][0]
    with pytest.raises(ValueError):
        pilot.extract(selector, candidate)


def test_player_choices_cannot_become_npc_targets(example):
    candidate = example[1]["conversations"][0]
    with pytest.raises(ValueError, match="player choice"):
        pilot.extract(5, candidate, assistant=True)


def test_runtime_placeholders_block_materialization(example):
    candidate = example[1]["conversations"][0]
    candidate["prose"][1]["speech"][0]["text"] = "Hello, <first>."
    with pytest.raises(ValueError, match="placeholder"):
        pilot.extract(6, candidate)


def test_calibration_conversation_cannot_move_to_test(example, monkeypatch):
    monkeypatch.setattr(pilot, "RESERVED", {"data/example.txt": 6})
    with pytest.raises(ValueError, match="requires validation"):
        pilot.assemble(*example)
    example[0]["conversations"][0]["split"] = "validation"
    assert pilot.assemble(*example)[0]


def test_copied_targets_in_other_split_history_are_detected(example):
    rows, _ = pilot.assemble(*example)
    rows[0]["messages"][-1]["content"] = " ".join(f"word{i}" for i in range(20))
    other = copy.deepcopy(rows[0])
    other["metadata"].update(id="held-out", split="test")
    other["messages"] += [{"role": "user", "content": "Continue."},
                           {"role": "assistant", "content": "A new answer."}]
    with pytest.raises(ValueError, match="crosses splits"):
        pilot.check_overlap([rows[0], other])


def test_prose_indentation_keeps_separate_source_conversations():
    roots = tree('mission "A"\n\ton offer\n\t\tconversation\n\t\t\t`"One."`\n'
                 '\ton complete\n\t\tconversation\n\t\t\t`"Two."`\n')
    assert len(roots[0]["children"]) == 2
    assert roots[0]["children"][1]["children"][0]["line"] == 6


def test_multiline_quote_opening_is_preserved_without_narration():
    assert [s["text"] for s in speech('\t`He says, "This continues`')] == ["This continues"]
    assert speech('\t`This is narrator prose.`') == []


def test_failed_complete_release_never_publishes_partial_files(example, tmp_path, monkeypatch):
    import json

    batch, source_catalog, source_root = example
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "example.json").write_text(json.dumps(batch))
    monkeypatch.setattr(pilot, "BATCHES", ("example.json",))
    monkeypatch.setattr(pilot, "catalog", lambda *_: source_catalog)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="no records"):
        pilot.build(source_root, annotations, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".pilot-*"))


def test_existing_release_is_not_overwritten(tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    (output / "manifest.json").write_text("Original release")
    with pytest.raises(ValueError, match="already exists"):
        pilot.build(tmp_path, tmp_path, output)
    assert (output / "manifest.json").read_text() == "Original release"


def test_declared_game_variables_render_identically_in_context_and_dialogue(example):
    batch, source_catalog, _ = example
    source_catalog["conversations"][0]["prose"][1]["speech"][0]["text"] = "Welcome, <first>."
    conversation = batch["conversations"][0]
    conversation.update(substitutions={"first": "Alex"},
                        substitution_notes="Player-selected name for this scene.")
    rows, ledger = pilot.assemble(*example)
    assert "first = Alex" in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][-1]["content"].startswith("Welcome, Alex.")
    assert ledger[0]["substitutions"] == {"first": "Alex"}
    del conversation["substitution_notes"]
    with pytest.raises(ValueError, match="substitution_notes"):
        pilot.assemble(*example)


def test_source_graph_rejects_pairing_choice_with_an_earlier_reply():
    from prepare_conversations import flow_graph, reachable

    nodes = tree('conversation\n\t`"First reply."`\n\tchoice\n'
                 '\t\t`"Why?"`\n\t\t`"Goodbye."`\n\t\t\tdecline\n'
                 '\t`"Because."`\n\t\tgoto end\n\tlabel end\n\t`"Farewell."`\n')
    flow = flow_graph(nodes[0])
    assert reachable(flow, 4, 7)
    assert reachable(flow, 7, 10)
    assert not reachable(flow, 4, 2)
    assert not reachable(flow, 5, 7)
    assert not reachable(flow, 2, 7)


def test_source_graph_preserves_both_conditional_routes_without_claiming_save_state():
    from prepare_conversations import flow_graph, reachable

    nodes = tree('conversation\n\tchoice\n\t\t`"Tell me."`\n'
                 '\tbranch changed\n\t\thas "story event"\n'
                 '\t`"Earlier state."`\n\t\tdecline\n'
                 '\tlabel changed\n\t`"Later state."`\n')
    flow = flow_graph(nodes[0])
    assert reachable(flow, 3, 6)
    assert reachable(flow, 3, 9)


def test_explicit_accept_label_is_not_confused_with_terminal_accept():
    from prepare_conversations import flow_graph, reachable

    nodes = tree('conversation\n\tchoice\n\t\t`"I agree."`\n\t\t\tgoto accept\n'
                 '\t`"An unused answer."`\n\t\tdecline\n'
                 '\tlabel accept\n\t`"Here are the arrangements."`\n\t\taccept\n')
    assert reachable(flow_graph(nodes[0]), 3, 8)


def test_release_layout_preserves_inputs_and_reconstructs_all_artifact_hashes(
        example, tmp_path, monkeypatch):
    import json

    batch, source_catalog, source_root = example
    dataset = tmp_path / "dataset"
    annotations = dataset / "annotations"
    annotations.mkdir(parents=True)
    (dataset / "evidence").mkdir()
    (dataset / "evidence/findings.md").write_text("Authored review findings.\n")
    pilot.dump(dataset / "release.json", {"dataset_version": "fixture-v1"})
    names = []
    candidates = []
    for split in ("train", "validation", "test"):
        name = f"{split}.json"
        names.append(name)
        candidate = copy.deepcopy(source_catalog["conversations"][0])
        candidate["id"] = f"{split}-conversation"
        candidates.append(candidate)
        selected = copy.deepcopy(batch)
        selected["conversations"][0].update(
            id=f"{split}-arrival", catalog_id=candidate["id"], split=split)
        pilot.dump(annotations / name, selected)
    for name in ("license.txt", "copyright", "credits.txt"):
        (source_root / name).write_text("Fixture attribution.\n")
    source_catalog["conversations"] = candidates
    monkeypatch.setattr(pilot, "BATCHES", tuple(names))
    monkeypatch.setattr(pilot, "catalog", lambda *_: source_catalog)
    first, second = tmp_path / "first-build", tmp_path / "second-build"
    pilot.build(source_root, annotations, first)
    manifest = json.loads((first / "samples/manifest.json").read_text())
    assert manifest["dataset_version"] == "fixture-v1"
    assert {p.name for p in first.iterdir()} == {
        "annotations", "samples", "evidence", "licensing", "release.json"}
    for name in names:
        assert (first / "annotations" / name).read_bytes() == (annotations / name).read_bytes()
    assert (first / "evidence/findings.md").read_bytes() == (
        dataset / "evidence/findings.md").read_bytes()
    frozen = {"artifact_sha256": {str(p.relative_to(first)): pilot.digest(p)
                                  for p in first.rglob("*") if p.is_file()}}
    pilot.build(source_root, annotations, second, expected_release=frozen)
    assert all(pilot.digest(second / name) == expected
               for name, expected in frozen["artifact_sha256"].items())
    frozen["artifact_sha256"]["samples/train.jsonl"] = "0" * 64
    rejected = tmp_path / "rejected-build"
    with pytest.raises(ValueError, match="Frozen release checksum mismatch"):
        pilot.build(source_root, annotations, rejected, expected_release=frozen)
    assert not rejected.exists()
