"""Verify prepared target bundles and render mission-scoped variables."""

import hashlib
import json
from pathlib import Path

from endless_voices.context import substitute_variables


def load_targets(path):
    """Verify and return a source-continuation bundle keyed by sample ID."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    raw = (path / "targets.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["targets_sha256"]:
        raise ValueError("Source continuation hash mismatch")
    rows = json.loads(raw)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate continuation sample ID")
    if [r["sample_id"] for r in rows] != manifest["sample_ids"]:
        raise ValueError("Continuation inventory differs")
    return {r["sample_id"]: r for r in rows}


def target_text(row, messages, variables):
    """Render game variables and reject target paragraphs already exposed in the input."""
    paragraphs = [substitute_variables(p["text"], variables) for p in row["paragraphs"]]
    if any(p.strip() and p.strip() in m["content"] for p in paragraphs for m in messages):
        raise ValueError("Source continuation paragraph appears in input context")
    return "\n\n".join(paragraphs)
