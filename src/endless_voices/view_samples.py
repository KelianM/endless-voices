"""Build a standalone HTML reader for curated samples and optional generated responses."""

import hashlib
import json
from pathlib import Path

from endless_voices.dataset.storage import SceneDataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory: Path, manifest: Path, known_ids: set[str]) -> dict:
    """Return display fields from a generation run belonging to the supplied dataset."""
    if (directory.parent / "selection.json").exists():
        selection = json.loads((directory.parent / "selection.json").read_text())
        if selection["manifest_sha256"] != sha256(manifest):
            raise ValueError("Benchmark belongs to a different dataset manifest")
        if set(selection["ids"]) - known_ids:
            raise ValueError("Benchmark contains unknown sample IDs")
        responses = {}
        for sid in selection["ids"]:
            path = directory / (sid + ".result.json")
            if path.exists():
                row = json.loads(path.read_text())
                responses[sid] = {k: row[k] for k in (
                    "status", "response", "error", "elapsed_seconds") if k in row}
                trace_path = directory / (sid + ".retrieval.json")
                if trace_path.exists():
                    responses[sid]["retrieval"] = json.loads(trace_path.read_text())
        return {"name": directory.name, "selected_ids": selection["ids"],
                "responses": responses}
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if run.get("schema_version") != 1:
        raise ValueError("Unsupported generation run schema")
    if run["dataset"]["manifest_sha256"] != sha256(manifest):
        raise ValueError("Generation run belongs to a different dataset manifest")
    selected = run["dataset"]["sample_ids"]
    if len(selected) != len(set(selected)) or set(selected) - known_ids:
        raise ValueError("Generation run contains duplicate or unknown sample IDs")
    responses_path = directory / "responses.jsonl"
    expected = run.get("artifacts_sha256", {}).get("responses.jsonl")
    if expected and sha256(responses_path) != expected:
        raise ValueError("Generation response file does not match its recorded hash")
    responses = {}
    for line_number, line in enumerate(responses_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row["sample_id"]
        if sample_id in responses or sample_id not in selected:
            raise ValueError(f"responses.jsonl:{line_number}: duplicate or unselected sample ID")
        if row.get("status") not in {"ok", "failed"} or not (
            row.get("response") is None or isinstance(row["response"], str)
        ):
            raise ValueError(f"responses.jsonl:{line_number}: invalid response status or text")
        responses[sample_id] = {
            key: row[key]
            for key in (
                "status",
                "response",
                "error",
                "finish_reason",
                "elapsed_seconds",
                "input_tokens",
                "output_tokens",
                "retrieval",
            )
            if key in row
        }
    return {
        "name": directory.name,
        "status": run["status"],
        "selected_ids": selected,
        "model": run.get("model"),
        "adapter": run.get("adapter"),
        "responses": responses,
    }


def viewer_data(manifest, split="all", run_directories=()):
    """Load verified scenes and saved runs for browsing."""
    datasets = SceneDataset.load_all(manifest.parent)
    records = [row for name, dataset in datasets.items()
               if split == "all" or name == split for row in dataset]
    known = {row["metadata"]["id"] for dataset in datasets.values() for row in dataset}
    runs = [load_run(directory, manifest, known) for directory in run_directories]
    records = [{k: v for k, v in row.items() if k != "evaluation"} for row in records]
    provenance = json.loads((manifest.parent / "provenance.json").read_text())
    for row in records:
        row["player_choices"] = provenance[row["metadata"]["id"]].get("player_choices")
    return {"dataset_version": "scene-dataset-v2", "samples": records,
            "runs": runs}, datasets


def viewer_html(payload):
    """Render literal scene data safely inside the reader."""
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return Path(__file__).with_name("sample_viewer.html").read_text().replace(
        "__SAMPLE_DATA__", serialized)


def build_viewer(manifest: Path, output: Path, split="all", run_directories=()):
    """Write a standalone reader without replacing an existing file."""
    payload, _ = viewer_data(manifest, split, run_directories)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(viewer_html(payload))
    return len(payload["samples"])
