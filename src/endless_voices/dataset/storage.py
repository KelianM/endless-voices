"""Publish prepared examples and verify them for training or benchmarking."""

import json
import shutil
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from endless_voices.artifacts import file_hash
from endless_voices.artifacts import write_json as save
from endless_voices.contracts import SPLITS, read_records, validate_record
from endless_voices.prepare_context import save_selections
from endless_voices.splits import mission_key


class SceneDataset(Sequence):
    """Provide stable indexed records without resampling their contexts."""

    def __init__(self, records):
        self.records = deepcopy(records)
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
        if manifest.get("format") != "scene-dataset-v1" or manifest.get("split_unit") != "mission":
            raise ValueError("Unsupported prepared dataset format")
        splits = manifest.get("splits")
        if (not isinstance(splits, list) or not splits
                or any(not isinstance(s, str) or s not in SPLITS for s in splits)
                or len(set(splits)) != len(splits)):
            raise ValueError("Expected unique published dataset splits")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or any(f"{s}.jsonl" not in artifacts for s in splits):
            raise ValueError("Every published split requires an artifact hash")
        for name, expected in artifacts.items():
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root):
                raise ValueError("Artifact must be inside the manifest directory")
            if file_hash(path) != expected:
                raise ValueError(f"Dataset artifact differs: {name}")
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
            datasets[split] = cls(rows)
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


def write_dataset(output, built, config, corpus):
    """Atomically save records, shared contexts, targets and execution provenance."""
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
            save_selections(
                root / split / "context",
                [r[1] for r in rows],
                {"builder": "DatasetBuilder", "config": config},
            )
            targets = root / split / "targets"
            targets.mkdir()
            save(
                targets / "targets.json",
                [
                    {
                        "sample_id": r[0]["metadata"]["id"],
                        **r[2],
                        "paragraphs": [{"line": p.line, "text": p.text} for p in r[3]],
                    }
                    for r in rows
                ],
            )
            save(
                targets / "manifest.json",
                {
                    "sample_ids": [r[0]["metadata"]["id"] for r in rows],
                    "targets_sha256": file_hash(targets / "targets.json"),
                },
            )
        save(root / "provenance.json", {r[0]["metadata"]["id"]: r[2] for r in built})
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
        code = root / "code"
        code.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, code / path.name)
        shared = code / "shared"
        shared.mkdir()
        for name in [
            "artifacts.py",
            "context.py",
            "splits.py",
            "game_variables.py",
            "instructions.py",
            "contracts.py",
            "messages.py",
            "prepare_context.py",
        ]:
            shutil.copyfile(Path(__file__).parent.parent / name, shared / name)
        save(
            root / "manifest.json",
            {
                "format": "scene-dataset-v1",
                "split_unit": "mission",
                "splits": splits,
                "solver_version": z3.get_version_string(),
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
