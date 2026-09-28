"""Measure bounded prior dialogue on declared train/validation story paths."""

import argparse
import hashlib
import json
from pathlib import Path

SPLITS = {"train": 0, "validation": 1, "test": 2}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_paths(paths, records):
    """Reject split reversals and inconsistent conversation positions or group assignments."""
    assignments = {}
    for path in paths:
        previous = -1
        conversations = {}
        seen = set()
        previous_position = -1
        for event in path["events"]:
            metadata = records[event["sample_id"]]["metadata"]
            split = metadata["split"]
            rank = SPLITS[split]
            conversation = metadata["conversation_id"]
            position = event["position"]
            if position < previous_position:
                raise ValueError("Path positions must be chronological")
            previous_position = position
            if event["sample_id"] in seen:
                raise ValueError("Duplicate dialogue example")
            seen.add(event["sample_id"])
            if conversation in conversations and conversations[conversation] != position:
                raise ValueError("Conversation variants must share a position")
            conversations[conversation] = position
            if rank < previous:
                raise ValueError("Later conversation belongs to an earlier split")
            previous = rank
            for group in (conversation, metadata.get("scenario_group")):
                if group is not None:
                    if group in assignments and assignments[group] != split:
                        raise ValueError("Conversation or scenario crosses splits")
                    assignments[group] = split


def reference_block(events, records):
    """Return earlier dialogue without source links, split labels or answer metadata."""
    return [
        {
            "kind": "earlier dialogue example; optional alternatives need not both have occurred",
            "speaker": event["speaker"],
            "audience": event["audience"],
            "dialogue": records[event["sample_id"]]["messages"][1:],
        }
        for event in events
    ]


def earlier_examples(events, current):
    """Exclude the current encounter, its alternatives and all future encounters."""
    return [event for event in events if event["position"] < current["position"]]


def select_reference(events, records, budget, count):
    """Select whole exchanges by recency and return them in chronological order."""
    chosen = []
    for event in reversed(events):
        candidate = [event, *chosen]
        if count(reference_block(candidate, records)) <= budget:
            chosen = candidate
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    manifest = json.loads(args.manifest.read_text())
    records, hashes = {}, {str(args.manifest): digest(args.manifest)}
    # The prototype never opens the held-out test file.
    for split in ("train", "validation"):
        for entry in manifest["files"][split]:
            file = args.manifest.parent / entry["path"]
            if digest(file) != entry["sha256"]:
                raise ValueError(f"Dataset hash mismatch: {file}")
            hashes[str(file)] = digest(file)
            for line in file.read_text().splitlines():
                record = json.loads(line)
                metadata = record["metadata"]
                if metadata["split"] != split or metadata["id"] in records:
                    raise ValueError("Invalid split or duplicate sample ID")
                records[metadata["id"]] = record
    paths = json.loads(args.paths.read_text())
    validate_paths(paths, records)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def count(value):
        encoded = tokenizer.encode(json.dumps(value, ensure_ascii=False), add_special_tokens=False)
        return len(encoded)

    cases = []
    for path in paths:
        for event in path["events"]:
            current = records[event["sample_id"]]
            if current["metadata"]["split"] != "validation":
                continue
            context = current["messages"][:-1]
            for budget in (128, 512, 1024, 2048):
                eligible = earlier_examples(path["events"], event)
                selected = select_reference(eligible, records, budget, count)
                reference = reference_block(selected, records)
                cases.append({
                    "path": path["id"], "sample_id": event["sample_id"],
                    "reference_budget_tokens": budget,
                    "eligible_exchanges": len(eligible), "selected_exchanges": len(selected),
                    "selected_sample_ids": [item["sample_id"] for item in selected],
                    "reference_tokens": count(reference),
                    "baseline_context_tokens": count(context),
                    "combined_json_tokens": count({"context": context, "reference": reference}),
                    "reference": reference,
                })
    hashes[str(args.paths)] = digest(args.paths)
    hashes[str(Path(__file__))] = digest(Path(__file__))
    tokenizer_hashes = {
        str(file.relative_to(args.tokenizer)): digest(file)
        for file in sorted(args.tokenizer.rglob("*")) if file.is_file()
    }
    args.output.mkdir(parents=True)
    result = {
        "scope": "Declared partial paths, not recorded playthroughs or a new dataset release",
        "token_count": "Qwen tokenizer over JSON, without chat template or output reserve",
        "input_hashes": hashes, "tokenizer_hashes": tokenizer_hashes, "cases": cases,
    }
    (args.output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in case.items() if k != "reference"} for case in cases]))


if __name__ == "__main__":
    main()
