#!/usr/bin/env python3
"""Build a standalone HTML reader for curated samples and optional generated responses."""

import argparse
import hashlib
import json
from pathlib import Path

from endless_voices.contracts import read_records, validate_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory: Path, manifest: Path, known_ids: set[str]) -> dict:
    """Return display fields from a generation run belonging to the supplied dataset."""
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


def build_viewer(manifest: Path, output: Path, split="train", run_directory: Path | None = None):
    """Validate the dataset and write a self-contained viewer without replacing existing files."""
    validate_manifest(manifest)
    declaration = json.loads(manifest.read_text(encoding="utf-8"))
    records = [
        row
        for split_name in ("train", "validation", "test")
        for entry in declaration["files"][split_name]
        for _, row in read_records(manifest.parent / entry["path"], split_name)
    ]
    run = (
        load_run(run_directory, manifest, {r["metadata"]["id"] for r in records})
        if (run_directory)
        else None
    )
    records = [r for r in records if split == "all" or r["metadata"]["split"] == split]
    if run:
        visible_ids = {r["metadata"]["id"] for r in records}
        run["responses"] = {k: v for k, v in run["responses"].items() if k in visible_ids}
        run["selected_ids"] = [s for s in run["selected_ids"] if s in visible_ids]
    records = [{key: value for key, value in row.items() if key != "evaluation"} for row in records]
    payload = {"dataset_version": declaration["dataset_version"], "samples": records, "run": run}
    # Escape script delimiters; sample text is rendered through textContent in the browser.
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("sample_viewer.html").read_text(encoding="utf-8")
    document = template.replace("__SAMPLE_DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/pilot-v1/samples/manifest.json")
    )
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--run", type=Path, help="Optional generation run directory")
    parser.add_argument("--output", type=Path, required=True, help="New standalone HTML file")
    args = parser.parse_args()
    try:
        count = build_viewer(args.manifest, args.output, args.split, args.run)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {count} samples to {args.output.resolve()}")
    print(f"Open in a browser: {args.output.resolve().as_uri()}")


if __name__ == "__main__":
    main()
