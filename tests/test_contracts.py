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
    benchmark_messages,
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
    assert counts["benchmark"]["records"] == 1
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
        ("split", "development"),
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


@pytest.mark.parametrize("split", ["train", "development", "benchmark"])
def test_duplicate_ids_across_all_files(manifest, split):
    row = record(split)
    row["metadata"]["id"] = record()["metadata"]["id"]
    rewrite(manifest, split, [row, row] if split == "train" else [row])
    with pytest.raises(ValueError, match=r"jsonl:\d+: duplicate ID.*first at"):
        validate_manifest(manifest)


def test_declared_split(manifest):
    rewrite(manifest, "development", [record()])
    with pytest.raises(ValueError, match=r"development.jsonl:2:.*declared split"):
        validate_manifest(manifest)


def test_scenario_family_split_unit(manifest):
    row = record("benchmark")
    row["metadata"]["scenario_group"] = record()["metadata"]["scenario_group"]
    rewrite(manifest, "benchmark", [row])
    with pytest.raises(ValueError, match="scenario_group.*crosses splits"):
        validate_manifest(manifest)
    row = record()
    row["metadata"]["id"] = "another-variant"
    rewrite(manifest, "benchmark", [record("benchmark")])
    rewrite(manifest, "train", [record(), row])
    assert validate_manifest(manifest)["train"]["records"] == 2


def test_physical_separation_and_hash(manifest):
    contents = json.loads(manifest.read_text())
    contents["files"]["development"] = contents["files"]["train"]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="split file reused"):
        validate_manifest(manifest)
    (manifest.parent / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row["inputs"].update(gold_answer="never copy"),
        lambda row: row["inputs"].update(user_turns=[{"role": "assistant", "content": "gold"}]),
        lambda row: row["inputs"].update(user_turns=[]),
        lambda row: row["evaluation"].update(dimensions=[]),
        lambda row: row["evaluation"].pop("uncertainty_expectations"),
        lambda row: row["evaluation"].update(sources=[]),
    ],
)
def test_benchmark_invalid_boundaries(change):
    row = record("benchmark")
    change(row)
    with pytest.raises(ValueError):
        validate_record(row, "benchmark")


def test_benchmark_generated_history_and_reference_exclusion():
    row = record("benchmark")
    original = copy.deepcopy(row)
    first = benchmark_messages(row, [])
    assert first == [
        {"role": "system", "content": row["inputs"]["system"]},
        {"role": "user", "content": "May I enter?"},
    ]
    second = benchmark_messages(row, ["Actual model output"])
    assert second == first + [
        {"role": "assistant", "content": "Actual model output"},
        {"role": "user", "content": "When should I return?"},
    ]
    row["metadata"]["character_role"] = "EVALUATOR_ONLY metadata"
    row["evaluation"]["expected_facts"] = ["changed secret"]
    assert benchmark_messages(row, ["Actual model output"]) == second
    assert "EVALUATOR_ONLY" not in json.dumps(second)
    assert original["inputs"] == row["inputs"]
    second[0]["content"] = "caller edit"
    assert row["inputs"]["system"] == original["inputs"]["system"]
    for history in (["one", "two"], [""], [{"role": "assistant", "content": "x"}]):
        with pytest.raises(ValueError):
            benchmark_messages(row, history)


def test_optional_length_check_has_line_error(manifest):
    class LocalTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": False}
            return list(range(10))

    with pytest.raises(ValueError, match=r"train.jsonl:1:.*Shorten the conversation"):
        validate_manifest(manifest, tokenizer=LocalTokenizer(), max_length=5)
    assert validate_manifest(manifest, tokenizer=LocalTokenizer(), max_length=10)


def test_cli_offline_and_actionable_failure(manifest):
    command = [sys.executable, "-m", "endless_voices.contracts", str(manifest)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["development"]["records"] == 1
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
    del contents["files"]["benchmark"]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="missing fields.*benchmark"):
        validate_manifest(manifest)
    contents["files"]["benchmark"] = [{"path": "../outside.jsonl", "sha256": "a" * 64}]
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
    contents["files"]["development"] = [dict(contents["files"]["train"][0], path=alias.name)]
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="split file reused"):
        validate_manifest(manifest)
