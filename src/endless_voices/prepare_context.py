"""Save reusable context selections from eligible source drafts, without model calls."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from endless_voices.context import (
    Selection,
    digest,
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_selections(output, selections, provenance):
    """Write an immutable organizer bundle and generation prompts with artifact hashes."""
    ids = [s.sample_id for s in selections]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must contain unique sample IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"selections.json": [asdict(s) for s in selections],
                 "prompts.json": [s.generation_prompt() for s in selections]}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "schema_version": 1, "sample_ids": ids, "preparation": provenance,
        "artifacts": {name: file_hash(output / name) for name in artifacts},
    }, indent=2) + "\n")


def load_selections(output):
    """Verify a saved bundle and return selections for dataset and benchmark consumers."""
    output = Path(output)
    manifest = json.loads((output / "provenance.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported context bundle version")
    for name in ("selections.json", "prompts.json"):
        if file_hash(output / name) != manifest["artifacts"][name]:
            raise ValueError(f"Context artifact hash mismatch: {name}")
    selections = [Selection(**r) for r in json.loads((output / "selections.json").read_text())]
    ids = [s.sample_id for s in selections]
    if ids != manifest["sample_ids"] or len(ids) != len(set(ids)):
        raise ValueError("Context sample IDs differ")
    prompts = json.loads((output / "prompts.json").read_text())
    if prompts != [s.generation_prompt() for s in selections]:
        raise ValueError("Generator and saved context differ")
    if any(digest(s.messages) != s.provenance["messages_sha256"] for s in selections):
        raise ValueError("Selected message hash differs")
    return selections
