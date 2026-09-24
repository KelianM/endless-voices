"""Small invented fixtures exercise contracts, never canonical quality or scoring."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from endless_voices.contracts import (
    evaluation_messages,
    read_records,
    validate_manifest,
    validate_record,
)
from endless_voices.data import load_conversations

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def record(split="train"):
    return json.loads((FIXTURES / f"{split}.jsonl").read_text())


@pytest.fixture
def manifest(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "dataset")
    return tmp_path / "dataset" / "manifest.json"


def rewrite(manifest, split, records):
    path = manifest.parent / f"{split}.jsonl"
    path.write_text("\n" + "\n".join(json.dumps(row) for row in records) + "\n")
    contents = json.loads(manifest.read_text())
    contents["files"][split][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(contents))


def test_valid_manifest_and_loader_compatibility():
    counts = validate_manifest(FIXTURES / "manifest.json")
    assert counts["train"]["identities"] == {"invented-archive-guild": 1}
    assert counts["test"]["records"] == 1
    assert load_conversations(str(FIXTURES / "train.jsonl")) == [record()["messages"]]
    assert load_conversations(str(Path(__file__).parents[1] / "data" / "example.jsonl"))


@pytest.mark.parametrize("key", list(record()["metadata"]))
def test_every_metadata_field_required(key):
    row = record()
    del row["metadata"][key]
    with pytest.raises(ValueError, match=key):
        validate_record(row, "train")


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "has spaces"),
        ("identity", ""),
        ("species", None),
        ("character_role", " "),
        ("topics", []),
        ("topics", [None]),
        ("scenario_group", 2),
        ("split", "validation"),
        ("sources", []),
        ("authorship", []),
        ("review_status", "done"),
    ],
)
def test_invalid_metadata(key, value):
    row = record()
    row["metadata"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_record(row, "train")


@pytest.mark.parametrize("key", ["reference", "revision", "source_group"])
def test_source_details_required(key):
    row = record()
    row["metadata"]["sources"][0][key] = ""
    with pytest.raises(ValueError, match=key):
        validate_record(row, "train")


@pytest.mark.parametrize(
    "messages",
    [
        [],
        None,
        [None],
        [{"role": [], "content": "x"}],
        [{"role": "system", "content": "identity"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        [
            {"role": "system", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
        ],
        [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {"role": "system", "content": "z"},
        ],
    ],
)
def test_curated_role_order(messages):
    row = record()
    row["messages"] = messages
    with pytest.raises(ValueError):
        validate_record(row, "train")


def test_complete_multi_turn_and_open_identity():
    row = record()
    row["messages"] += row["messages"][1:]
    row["metadata"]["identity"] = "human-another-faction"
    validate_record(row, "train")


@pytest.mark.parametrize("payload", ["{", "null", "[]", '"text"', '{"messages": []}'])
def test_bad_jsonl_line_locations(tmp_path, payload):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + payload + "\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:2:"):
        list(read_records(path, "train"))


def test_committed_invalid_fixture():
    with pytest.raises(ValueError, match=r"invalid.jsonl:1:.*metadata"):
        list(read_records(FIXTURES / "invalid.jsonl", "train"))


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_duplicate_ids_across_all_files(manifest, split):
    row = record(split)
    row["metadata"]["id"] = record()["metadata"]["id"]
    rewrite(manifest, split, [row, row] if split == "train" else [row])
    with pytest.raises(ValueError, match=r"jsonl:\d+: duplicate ID.*first at"):
        validate_manifest(manifest)


def test_declared_split(manifest):
    rewrite(manifest, "validation", [record()])
    with pytest.raises(ValueError, match=r"validation.jsonl:2:.*declared split"):
        validate_manifest(manifest)


def test_scenario_family_split_unit(manifest):
    row = record("test")
    train = record()
    train["metadata"]["scenario_group"] = "known-variants"
    rewrite(manifest, "train", [train])
    row["metadata"]["scenario_group"] = "known-variants"
    rewrite(manifest, "test", [row])
    with pytest.raises(ValueError, match="scenario_group.*crosses splits"):
        validate_manifest(manifest)
    row = record()
    row["metadata"]["id"] = "another-variant"
    rewrite(manifest, "test", [record("test")])
    rewrite(manifest, "train", [record(), row])
    assert validate_manifest(manifest)["train"]["records"] == 2


def test_physical_separation_and_hash(manifest):
    contents = json.loads(manifest.read_text())
    contents["files"]["validation"] = contents["files"]["train"]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="split file reused"):
        validate_manifest(manifest)
    (manifest.parent / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(inputs={"gold_answer": "never copy"}),
        lambda row: row["messages"][0].update(reference="private evidence"),
        lambda row: row["evaluation"].update(dimensions=[]),
        lambda row: row["evaluation"].pop("dimensions"),
        lambda row: row["evaluation"].update(sources=[]),
    ],
)
def test_reject_mixed_formats_and_incomplete_evaluator_references(change):
    row = record("test")
    change(row)
    with pytest.raises(ValueError):
        validate_record(row, "test")


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_same_sample_format_withholds_only_final_target(split):
    row = record(split)
    original = copy.deepcopy(row)
    prompt = evaluation_messages(row)
    assert prompt == row["messages"][:-1]
    assert prompt[-1]["role"] == "user"
    if split == "test":
        assert prompt[2] == {"role": "assistant", "content": "The archive is closed."}
    assert "Lore (fixture-v1)" in prompt[0]["content"]
    row["messages"][-1]["content"] = "PRIVATE_TARGET"
    row["metadata"]["character_role"] = "PRIVATE_METADATA"
    row["evaluation"]["sources"][0]["reference"] = "PRIVATE_CRITERIA"
    assert evaluation_messages(row) == prompt
    assert "PRIVATE_" not in json.dumps(prompt)
    assert "EVALUATOR_ONLY" not in json.dumps(prompt)
    prompt[0]["content"] = "caller edit"
    assert row["messages"][0] == original["messages"][0]


def test_single_turn_sample_has_no_assistant_history():
    row = record()
    row["messages"] = row["messages"][:3]
    assert [message["role"] for message in evaluation_messages(row)] == ["system", "user"]


def test_conversation_prefix_samples_cannot_cross_splits(manifest):
    row = record("validation")
    row["metadata"]["conversation_id"] = record()["metadata"]["conversation_id"]
    row["messages"] = row["messages"][:3]
    rewrite(manifest, "validation", [row])
    with pytest.raises(ValueError, match="conversation_id.*crosses splits"):
        validate_manifest(manifest)


def test_unknown_scenario_groups_do_not_merge_unrelated_conversations(manifest):
    assert all(
        record(split)["metadata"]["scenario_group"] is None
        for split in ("train", "validation", "test")
    )
    assert validate_manifest(manifest)["test"]["records"] == 1


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_optional_length_check_has_line_error(manifest, split):
    row = record(split)
    row["messages"][-1]["content"] = "LONG_TARGET"
    rewrite(manifest, split, [row])

    class LocalTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": False}
            return list(range(10 if messages[-1]["content"] == "LONG_TARGET" else 3))

    with pytest.raises(ValueError, match=rf"{split}.jsonl:2:.*Shorten the conversation"):
        validate_manifest(manifest, tokenizer=LocalTokenizer(), max_length=5)
    assert validate_manifest(manifest, tokenizer=LocalTokenizer(), max_length=10)


def test_cli_offline_and_actionable_failure(manifest):
    command = [sys.executable, "-m", "endless_voices.contracts", str(manifest)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["validation"]["records"] == 1
    rewrite(manifest, "train", [None])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "train.jsonl:2:" in result.stderr
    assert "Traceback" not in result.stderr


def test_structure_needs_only_standard_library():
    source = Path(__file__).parents[1] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            f"import sys; sys.path.insert(0, {str(source)!r}); "
            "from pathlib import Path; from endless_voices.contracts import validate_manifest; "
            f"validate_manifest(Path({str(FIXTURES / 'manifest.json')!r}))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [True, 0, 2, "1", None])
def test_unsupported_schema_versions(manifest, value):
    row = record()
    row["schema_version"] = value
    with pytest.raises(ValueError, match="schema_version"):
        validate_record(row, "train")
    contents = json.loads(manifest.read_text())
    contents["schema_version"] = value
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="manifest.json:.*schema_version"):
        validate_manifest(manifest)


def test_missing_split_and_path_escape(manifest):
    contents = json.loads(manifest.read_text())
    del contents["files"]["test"]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="missing fields.*test"):
        validate_manifest(manifest)
    contents["files"]["test"] = [{"path": "../outside.jsonl", "sha256": "a" * 64}]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="inside the manifest directory"):
        validate_manifest(manifest)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_physical_aliases(manifest, alias_kind):
    alias = manifest.parent / "alias.jsonl"
    target = manifest.parent / "train.jsonl"
    if alias_kind == "symlink":
        alias.symlink_to(target)
    else:
        alias.hardlink_to(target)
    contents = json.loads(manifest.read_text())
    contents["files"]["validation"] = [dict(contents["files"]["train"][0], path=alias.name)]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="split file reused"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "expected_facts",
        "expected_behaviours",
        "expected_style",
        "prohibited_contradictions",
        "uncertainty_expectations",
    ],
)
def test_current_samples_reject_retired_checklist_fields(field):
    row = record()
    row["evaluation"][field] = []
    with pytest.raises(ValueError, match="unknown fields"):
        validate_record(row)
