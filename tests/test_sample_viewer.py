"""Protect split selection, literal sample text and generation-run matching in the reader."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "view_samples", Path(__file__).parents[1] / "scripts/view_samples.py"
)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)
FIXTURE = Path(__file__).parent / "fixtures/contracts/manifest.json"


def payload(output):
    text = output.read_text()
    return json.loads(
        text.split('<script id="samples-data" type="application/json">')[1].split("</script>")[0]
    )


def test_default_export_excludes_held_out_content_and_preserves_history(tmp_path):
    output = tmp_path / "samples.html"
    assert viewer.build_viewer(FIXTURE, output) == 1
    data = payload(output)
    original = json.loads((FIXTURE.parent / "train.jsonl").read_text())
    assert data["samples"] == [original]
    assert data["run"] is None
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        viewer.build_viewer(FIXTURE, output)
    assert output.read_bytes() == before


def test_literal_script_delimiters_cannot_escape_sample_data(tmp_path):
    import shutil

    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURE.parent, dataset)
    manifest = dataset / "manifest.json"
    source = dataset / "train.jsonl"
    record = json.loads(source.read_text())
    hostile = '</script><script>alert("sample text")</script>'
    record["messages"][-1]["content"] = hostile
    source.write_text(json.dumps(record) + "\n")
    declaration = json.loads(manifest.read_text())
    declaration["files"]["train"][0]["sha256"] = viewer.sha256(source)
    manifest.write_text(json.dumps(declaration))
    output = tmp_path / "samples.html"
    viewer.build_viewer(manifest, output)
    assert hostile not in output.read_text()
    assert payload(output)["samples"][0]["messages"][-1]["content"] == hostile


def test_run_join_keeps_failures_and_missing_responses_visible(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "interrupted",
                "dataset": {
                    "manifest_sha256": viewer.sha256(FIXTURE),
                    "sample_ids": ["fixture-train", "fixture-validation"],
                },
            }
        )
    )
    (run / "responses.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "fixture-train",
                "status": "failed",
                "response": "partial text",
                "error": {"stage": "generation", "type": "OutputLimit", "message": "capped"},
            }
        )
        + "\n"
    )
    output = tmp_path / "samples.html"
    assert viewer.build_viewer(FIXTURE, output, "all", run) == 3
    data = payload(output)
    assert data["run"]["responses"]["fixture-train"]["error"]["type"] == "OutputLimit"
    assert "fixture-validation" in data["run"]["selected_ids"]
    assert "fixture-validation" not in data["run"]["responses"]
    assert "fixture-test" not in data["run"]["selected_ids"]


@pytest.mark.parametrize("problem", ["manifest", "duplicate", "hash"])
def test_unrelated_or_modified_responses_are_rejected(tmp_path, problem):
    run = tmp_path / "run"
    run.mkdir()
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "dataset": {
            "manifest_sha256": viewer.sha256(FIXTURE),
            "sample_ids": ["fixture-train"],
        },
    }
    row = json.dumps({"sample_id": "fixture-train", "status": "ok", "response": "Hello"})
    (run / "responses.jsonl").write_text(row + "\n")
    if problem == "manifest":
        metadata["dataset"]["manifest_sha256"] = "wrong"
    elif problem == "duplicate":
        (run / "responses.jsonl").write_text(row + "\n" + row + "\n")
    else:
        metadata["artifacts_sha256"] = {"responses.jsonl": "wrong"}
    (run / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        viewer.build_viewer(FIXTURE, tmp_path / "samples.html", run_directory=run)
