"""Publish prepared examples and verify them for training or benchmarking."""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from endless_voices.artifacts import file_hash
from endless_voices.artifacts import write_json as save
from endless_voices.context import ContextPool, digest
from endless_voices.contracts import SPLITS, read_records, validate_record
from endless_voices.splits import mission_key


class SceneDataset(Sequence):
    """Provide prepared encounters, targets and eligible context pools."""

    def __init__(self, records, pools):
        self.records = deepcopy(records)
        self._pools = deepcopy(pools)
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
        if (manifest.get("format") != "scene-dataset-v2"
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
        if "retrieval.json" not in artifacts:
            raise ValueError("Prepared datasets require a retrieval.json artifact hash")
        for name, expected in artifacts.items():
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root):
                raise ValueError("Artifact must be inside the manifest directory")
            if file_hash(path) != expected:
                raise ValueError(f"Dataset artifact differs: {name}")
        pools = load_pools(root)
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
            split_pools = {}
            for record in rows:
                meta = record["metadata"]
                pool = pools.get(meta["id"])
                if (pool is None or pool.conversation_id != meta["conversation_id"]
                        or "mission / " + pool.mission != meta["sources"][0]["source_group"]):
                    raise ValueError("Retrieval pool does not match its scene record")
                split_pools[meta["id"]] = pool
            datasets[split] = cls(rows, split_pools)
        if set(pools) != set(ids):
            raise ValueError("Retrieval pools must cover exactly the published scenes")
        from endless_voices.splits import check_chain_splits

        assignments = {pool.mission: split for split, dataset in datasets.items()
                       for pool in dataset._pools.values()}
        graph = {name: parents for pool in pools.values() for name, parents in pool.graph.items()}
        check_chain_splits(assignments, graph)
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
        """Return the saved eligible context for one scene without its continuation."""
        return deepcopy(self._pools[self.records[index]["metadata"]["id"]])

    def retrieval_session(self, index, database):
        """Bind retrieval tools to one scene's saved eligible context."""
        from endless_voices.retrieval import RetrievalSession

        return RetrievalSession(database, self.context_pool(index))


def load_pools(root):
    """Restore deduplicated context pools and verify their preparation fingerprints."""
    from endless_voices.retrieval import scene_id

    saved = json.loads((root / "retrieval.json").read_text())
    if saved.get("format") != "scene-retrieval-pools-v1":
        raise ValueError("Unsupported retrieval pool format")
    scenes = saved["scenes"]
    if any(scene_id(block) != key for key, block in scenes.items()):
        raise ValueError("Retrieval scene differs from its identifier")
    evidence = json.loads((root / "provenance.json").read_text())
    pools = {}
    for key, fields in saved["pools"].items():
        fields = dict(fields)
        for kind in ("blocks", "references"):
            fields[kind] = [scenes[sid] for sid in fields[kind]]
        pool = ContextPool(**fields)
        if (pool.sample_id != key
                or digest(pool.__dict__) != evidence[key]["context"]["pool_sha256"]):
            raise ValueError("Retrieval pool differs from its preparation evidence")
        pools[key] = pool
    return pools


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
        from endless_voices.retrieval import scene_id

        scenes, pools = {}, {}
        for record, pool, _ in built:
            fields = asdict(pool)
            for kind in ("blocks", "references"):
                for block in fields[kind]:
                    scenes[scene_id(block)] = block
                fields[kind] = [scene_id(block) for block in fields[kind]]
            pools[record["metadata"]["id"]] = fields
        save(root / "retrieval.json", {
            "format": "scene-retrieval-pools-v1", "scenes": scenes, "pools": pools,
        })
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
                "format": "scene-dataset-v2",
                "split_unit": "mission_chain",
                "splits": splits,
                "solver_version": z3.get_version_string(),
                "build": implementation(),
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
