"""Write empty eligible pools for invented prepared-dataset fixtures."""

import json
from dataclasses import asdict

from endless_voices.artifacts import file_hash, save_progress
from endless_voices.context import ContextPool, digest


def write_empty_pools(root):
    """Save per-record empty pools and update the fixture manifest hashes."""
    manifest = json.loads((root / "manifest.json").read_text())
    pools = {}
    for split in manifest["splits"]:
        for line in (root / f"{split}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            meta, messages = record["metadata"], record["messages"]
            pool = ContextPool(
                meta["id"], meta["conversation_id"],
                meta["sources"][0]["source_group"].removeprefix("mission / "),
                messages[0]["content"], messages[1:-1], [], {})
            pools[pool.sample_id] = asdict(pool)
    save_progress(root / "retrieval.json", {
        "format": "scene-retrieval-pools-v1", "scenes": {}, "pools": pools})
    save_progress(root / "provenance.json", {
        key: {"context": {"pool_sha256": digest(pool)}} for key, pool in pools.items()})
    for name in ("retrieval.json", "provenance.json"):
        manifest["artifacts"][name] = file_hash(root / name)
    save_progress(root / "manifest.json", manifest)
