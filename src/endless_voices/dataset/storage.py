"""Publish prepared examples and verify them for training or benchmarking."""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from endless_voices.artifacts import file_hash
from endless_voices.artifacts import write_json as save
from endless_voices.context import digest
from endless_voices.contracts import SPLITS, read_records, validate_record
from endless_voices.splits import mission_key


class SceneDataset(Sequence):
    """Provide prepared encounters, targets and eligible context pools."""

    def __init__(self, records, context, evidence):
        self.records = deepcopy(records)
        self._context, self.evidence = context, evidence
        ids = [r["metadata"]["id"] for r in self.records if "metadata" in r]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scene IDs")
        for record in self.records:
            validate_record(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return deepcopy(self.records[index])

    @classmethod
    def load_all(cls, root):
        """Verify artifacts and cross-split ownership, returning each published split."""
        root = Path(root).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest.get("format") != "scene-dataset-v3"
                or manifest.get("split_unit") != "mission_chain"):
            raise ValueError("Unsupported prepared dataset format")
        splits = manifest.get("splits")
        if (not isinstance(splits, list) or not splits
                or any(not isinstance(s, str) or s not in SPLITS for s in splits)
                or len(set(splits)) != len(splits)):
            raise ValueError("Expected unique published dataset splits")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or any(f"{s}.jsonl" not in artifacts for s in splits):
            raise ValueError("Every published split requires an artifact hash")
        if not {"config.json", "provenance.json"} <= artifacts.keys():
            raise ValueError("Prepared datasets require hashed configuration and provenance")
        for name, expected in artifacts.items():
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root):
                raise ValueError("Artifact must be inside the manifest directory")
            if file_hash(path) != expected:
                raise ValueError(f"Dataset artifact differs: {name}")
        from .source import GameCorpus

        corpus = GameCorpus(root / manifest["source_root"], {
            "revision": manifest["source_revision"],
            "files": [{"path": path, "sha256": value}
                      for path, value in manifest["source_files"].items()]})
        config = json.loads((root / "config.json").read_text())
        from endless_voices.splits import check_chain_splits

        check_chain_splits({name: spec["split"] for name, spec in config["missions"].items()},
                           corpus.mission_graph)
        context = {"corpus": corpus, "config": config}
        evidence = json.loads((root / "provenance.json").read_text())
        datasets, ids, groups, physical = {}, {}, {}, set()
        for split in splits:
            path = root / f"{split}.jsonl"
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in physical:
                raise ValueError("Split file reused")
            physical.add(identity)
            rows = []
            for line, record in read_records(path, split):
                location = f"{path}:{line}"
                meta = record["metadata"]
                if meta["id"] in ids:
                    raise ValueError(f"{location}: duplicate ID; first at {ids[meta['id']]}")
                ids[meta["id"]] = location
                owners = [("mission", mission_key(meta)),
                          ("conversation_id", meta["conversation_id"]),
                          ("scenario_group", meta["scenario_group"])]
                for kind, owner in owners:
                    if owner is None:
                        continue
                    key = (kind, owner)
                    if key in groups and groups[key] != split:
                        raise ValueError(f"{location}: {kind} {owner!r} crosses splits")
                    groups[key] = split
                rows.append(record)
            for record in rows:
                meta = record["metadata"]
                mission = meta["sources"][0]["source_group"].removeprefix("mission / ")
                if config["missions"][mission]["split"] != split:
                    raise ValueError("Scene split differs from its configured mission")
            datasets[split] = cls(rows, context, evidence)
        if set(evidence) != set(ids):
            raise ValueError("Provenance must cover exactly the published scenes")
        return datasets

    @classmethod
    def load(cls, root, split):
        datasets = cls.load_all(root)
        if split not in datasets:
            raise ValueError(f"Split was not prepared: {split}")
        return datasets[split]

    def prompt(self, index):
        return self[index]["messages"][:-1]

    def target(self, index):
        return self[index]["messages"][-1]["content"]

    def context_pool(self, index):
        """Reconstruct eligible context from raw source and verify its preparation fingerprint."""
        record = self[index]
        evidence = self.evidence[record["metadata"]["id"]]
        if "builder" not in self._context:
            from .builder import DatasetBuilder

            self._context["builder"] = DatasetBuilder(self._context["corpus"],
                                                      self._context["config"])
        pool = self._context["builder"].sampler.restore_pool(record, evidence)
        if digest(asdict(pool)) != evidence["context"]["pool_sha256"]:
            raise ValueError("Reconstructed context differs from its preparation evidence")
        return deepcopy(pool)

    def retrieval_session(self, index, database):
        """Bind retrieval tools to one scene's reconstructed eligible context."""
        from endless_voices.retrieval import RetrievalSession

        return RetrievalSession(database, self.context_pool(index))


def implementation():
    """Identify the preparation source by Git revision and actual Python file hashes."""
    source = Path(__file__).resolve().parent.parent
    repo = source.parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--", str(source)], cwd=repo,
                            capture_output=True, text=True)
    return {"git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "working_tree_modified": bool(status.stdout) if status.returncode == 0 else None,
            "source_hashes": {str(p.relative_to(repo)): file_hash(p)
                              for p in sorted(source.rglob("*.py"))}}


def write_dataset(output, built, config, corpus):
    """Atomically save split records, source provenance, configuration and licensing."""
    import z3

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".scene-build-") as temporary:
        root = Path(temporary) / "dataset"
        root.mkdir()
        splits = sorted({row[0]["metadata"]["split"] for row in built})
        for split in splits:
            rows = [row for row in built if row[0]["metadata"]["split"] == split]
            (root / f"{split}.jsonl").write_text(
                "".join(json.dumps(r[0], ensure_ascii=False) + "\n" for r in rows)
            )
        save(root / "provenance.json", {
            record["metadata"]["id"]: {
                **provenance, "context": {"pool_sha256": digest(asdict(pool))}}
            for record, pool, provenance in built})
        save(root / "config.json", config)
        licensing = root / "licensing"
        licensing.mkdir()
        for name in ["license.txt", "copyright", "credits.txt"]:
            shutil.copyfile(corpus.root / name, licensing / name)
        (licensing / "ATTRIBUTION.md").write_text(
            "# Source attribution\n\nAuthored passages from Endless Sky at revision "
            + corpus.revision
            + ". Preserve the accompanying upstream notices.\n"
        )
        save(
            root / "manifest.json",
            {
                "format": "scene-dataset-v3",
                "split_unit": "mission_chain",
                "splits": splits,
                "solver_version": z3.get_version_string(),
                "build": implementation(),
                "source_root": os.path.relpath(corpus.root.resolve(), output.resolve()),
                "source_revision": corpus.revision,
                "source_files": corpus.files,
                "artifacts": {
                    str(p.relative_to(root)): file_hash(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
            },
        )
        root.rename(output)
