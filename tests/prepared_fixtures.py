"""Write raw-source context for invented prepared-dataset fixtures."""

import json
from dataclasses import asdict

from endless_voices.artifacts import file_hash, save_progress
from endless_voices.context import digest
from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.dataset.source import GameCorpus
from endless_voices.dataset.state import GameState


def write_raw_context(root):
    """Save minimal raw missions and reproducibility evidence for fixture records."""
    manifest = json.loads((root / "manifest.json").read_text())
    records = [json.loads(line) for split in manifest["splits"]
               for line in (root / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    missions = {r["metadata"]["sources"][0]["source_group"].removeprefix("mission / "):
                {k: r["metadata"][k] for k in
                 ("split", "identity", "species", "character_role", "topics", "scenario_group")}
                for r in records}
    source = root / "raw-source"
    source.mkdir(exist_ok=True)
    path = source / "missions.txt"
    path.write_text("".join("mission " + json.dumps(name) + "\n" for name in missions))
    config = {"missions": missions, "seed": "fixture", "player": {}}
    inventory = {"revision": "a" * 40,
                 "files": [{"path": "missions.txt", "sha256": file_hash(path)}]}
    builder = DatasetBuilder(GameCorpus(source, inventory), config)
    evidence = {}
    for record in records:
        saved = {"state": GameState().snapshot(), "encounter_state": GameState().snapshot(),
                 "history": []}
        pool = builder.sampler.restore_pool(record, saved)
        saved["context"] = {"pool_sha256": digest(asdict(pool))}
        evidence[record["metadata"]["id"]] = saved
    save_progress(root / "provenance.json", evidence)
    save_progress(root / "config.json", config)
    manifest.update(format="scene-dataset-v3", source_root="raw-source",
                    source_revision=inventory["revision"],
                    source_files={"missions.txt": file_hash(path)})
    for name in ("provenance.json", "config.json", "raw-source/missions.txt"):
        manifest["artifacts"][name] = file_hash(root / name)
    save_progress(root / "manifest.json", manifest)
