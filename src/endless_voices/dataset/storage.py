"""Publish prepared examples and verify them for training or benchmarking."""

import json
import shutil
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from endless_voices.messages import validate_messages
from endless_voices.prepare_context import file_hash, save_selections


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class SceneDataset(Sequence):
    """Provide stable indexed records without resampling their contexts."""

    def __init__(self, records):
        self.records = deepcopy(records)
        ids = [r["metadata"]["id"] for r in self.records if "metadata" in r]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scene IDs")
        for record in self.records:
            validate_messages(record["messages"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return deepcopy(self.records[index])

    @classmethod
    def from_jsonl(cls, path):
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not records:
            raise ValueError("Dataset has no examples")
        return cls(records)

    @classmethod
    def load(cls, root, split):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["format"] != "scene-dataset-v1" or manifest["split_unit"] != "mission":
            raise ValueError("Unsupported prepared dataset format")
        for name, expected in manifest["artifacts"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
                raise ValueError(f"Dataset artifact differs: {name}")
        if split not in manifest["splits"]:
            raise ValueError(f"Split was not prepared: {split}")
        result = cls.from_jsonl(root / f"{split}.jsonl")
        if any(r["metadata"]["split"] != split for r in result.records):
            raise ValueError("Record split differs from dataset split")
        return result

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
