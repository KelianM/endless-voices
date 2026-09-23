"""Protect source attribution and blinding in the development calibration pack."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from endless_voices.contracts import read_records

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "calibration", ROOT / "scripts/prepare_authenticity_calibration.py"
)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    recipe = json.loads(calibration.RECIPE.read_text())
    inventory = json.loads(calibration.INVENTORY.read_text())
    passages = {row["id"]: row for row in inventory["passages"]}
    contents = {}
    for index, case in enumerate(recipe["cases"]):
        path = passages[case["passage"]]["source_file"]
        lines = contents.setdefault(path, [""] * 400)
        lines[case["user"]["line"] - 1] = f'\t`"Question {index}?"`'
        line = f'\t`"Answer {index}."'
        if len(case["target"]["quotes"]) == 2:
            line = line[:-2] + '," narrator says. "More speech."'
        lines[case["target"]["line"] - 1] = line + "`"
    for entry in inventory["files"]:
        if entry["path"] in contents:
            path = tmp_path / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(contents[entry["path"]]))
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("license.txt", "copyright", "credits.txt"):
        (tmp_path / name).write_text("Invented fixture attribution")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    monkeypatch.setattr(calibration, "INVENTORY", inventory_path)
    return tmp_path, recipe, inventory


def test_verified_speech_preserves_target_and_removes_narration(sources):
    root, recipe, inventory = sources
    records = calibration.samples_from_sources(recipe, inventory, root)
    assert records[1]["messages"][-1]["content"] == "Answer 1. More speech."
    assert all(row["metadata"]["split"] == "validation" for row in records)
    assert records[2]["metadata"]["scenario_group"] == "fw-recon-chain"
    assert all(row["metadata"]["review_status"] == "draft" for row in records)
    (root / inventory["files"][0]["path"]).write_text("unexpected revision")
    with pytest.raises(ValueError, match="checksum mismatch"):
        calibration.samples_from_sources(recipe, inventory, root)


def test_revision_and_extraction_boundaries(sources):
    root, recipe, inventory = sources
    recipe["revision"] = "wrong"
    with pytest.raises(ValueError, match="revisions differ"):
        calibration.samples_from_sources(recipe, inventory, root)
    with pytest.raises(ValueError, match="outside"):
        calibration.extract(['`"Speech"`'], {"line": 1, "quotes": [0]}, [(2, 3)])
    with pytest.raises(ValueError, match="missing selected"):
        calibration.extract(['`"Speech"`'], {"line": 1, "quotes": [1]}, [(1, 1)])
    with pytest.raises(ValueError, match="placeholder"):
        calibration.extract(['`"Hello <last>"`'], {"line": 1, "quotes": [0]}, [(1, 1)])


def test_blinding_and_position_reversal(sources):
    root, recipe, inventory = sources
    records = calibration.samples_from_sources(recipe, inventory, root)
    first, key_a = calibration.review_pack(records, recipe, "a")
    second, key_b = calibration.review_pack(records, recipe, "b")
    for a, b, ka, kb in zip(first, second, key_a, key_b, strict=True):
        assert set(a) == {"trial_id", "context", "A", "B"}
        assert a["A"] == b["B"] and a["B"] == b["A"]
        assert a["context"] == b["context"]
        assert a["context"][-1]["role"] == "user"
        if ka["kind"] == "identical":
            assert ka["original"] is None and kb["original"] is None
            assert a["A"] == a["B"]
        else:
            assert ka["original"] != kb["original"]
            sample = next(row for row in records if row["metadata"]["id"] == ka["sample_id"])
            assert a[ka["original"]] == sample["messages"][-1]["content"]
    changed = copy.deepcopy(records)
    for record in changed:
        record["evaluation"]["expected_facts"] = ["PRIVATE ASSESSMENT"]
        record["metadata"]["sources"][0]["reference"] = "PRIVATE SOURCE"
    assert calibration.review_pack(changed, recipe, "a")[0] == first


def test_pack_separates_controls_and_does_not_invent_reviews(sources, tmp_path):
    root, _, _ = sources
    output = tmp_path / "output"
    assert calibration.prepare(root, output, "a") == 7
    assert len(list(read_records(output / "samples.jsonl", "validation"))) == 3
    primary = [json.loads(line) for line in (output / "review.jsonl").read_text().splitlines()]
    controls = [json.loads(line) for line in (output / "controls.jsonl").read_text().splitlines()]
    assert len(primary) == 3 and len(controls) == 4
    assert {row["trial_id"] for row in primary}.isdisjoint(row["trial_id"] for row in controls)
    secret = json.loads((output / "answer-key.json").read_text())
    primary_ids = {row["trial_id"] for row in primary}
    assert all(
        row["kind"] == "authored-alternative"
        for row in secret["trials"]
        if row["trial_id"] in primary_ids
    )
    assert secret["human_review_status"] == "pending"
    for stage in ("review", "controls"):
        review = json.loads((output / f"{stage}-template.json").read_text())
        assert review["reviewer_type"] is None
        assert all(row["choice"] is None for row in review["reviews"])
    for name in ("license.txt", "copyright", "credits.txt"):
        assert (output / name).read_bytes() == (root / name).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        calibration.prepare(root, output, "a")
