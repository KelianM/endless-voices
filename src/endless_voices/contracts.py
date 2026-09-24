"""Validate conversation samples and split manifests offline."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from endless_voices.messages import validate_messages

SPLITS = {"train", "validation", "test"}
SLUG = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")


def fields(value: object, required: set[str], field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    missing, extra = required - value.keys(), value.keys() - required
    if missing or extra:
        raise ValueError(
            f"{field}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )


def slug(value: object, field: str) -> None:
    text(value, field)
    if not SLUG.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase ID using letters, digits, '-' or '_'")


def texts(value: object, field: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field} must be {'a' if allow_empty else 'a nonempty'} list of text")
    for item in value:
        text(item, field)


def version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be integer 1")


def sources(value: object, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    for source in value:
        fields(source, {"reference", "revision", "source_group"}, field)
        for key in source:
            text(source[key], f"{field}.{key}")


def metadata(value: object, split: str) -> None:
    fields(
        value,
        {
            "id",
            "identity",
            "species",
            "character_role",
            "topics",
            "scenario_group",
            "conversation_id",
            "split",
            "sources",
            "authorship",
            "review_status",
        },
        "metadata",
    )
    for key in ("id", "identity", "species", "conversation_id"):
        slug(value[key], f"metadata.{key}")
    if value["scenario_group"] is not None:
        slug(value["scenario_group"], "metadata.scenario_group")
    text(value["character_role"], "metadata.character_role")
    texts(value["topics"], "metadata.topics")
    for topic in value["topics"]:
        slug(topic, "metadata.topics")
    if value["split"] != split:
        raise ValueError(f"metadata.split must match declared split {split!r}")
    sources(value["sources"], "metadata.sources")
    if value["authorship"] not in ("human", "agent", "mixed"):
        raise ValueError("metadata.authorship must be human, agent or mixed")
    if value["review_status"] not in ("draft", "reviewed", "approved", "rejected"):
        raise ValueError("metadata.review_status must be draft, reviewed, approved or rejected")


def validate_record(record: object, split: str | None = None) -> None:
    """Validate one conversation sample and optionally its declared file split."""
    fields(record, {"schema_version", "metadata", "messages", "evaluation"}, "record")
    version(record["schema_version"])
    if split is None:
        if not isinstance(record["metadata"], dict):
            raise ValueError("metadata must be an object")
        split = record["metadata"].get("split")
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected train, validation or test")
    metadata(record["metadata"], split)
    validate_messages(record["messages"])
    if record["messages"][0]["role"] != "system":
        raise ValueError("curated messages require a leading identity-setting system message")
    for message in record["messages"]:
        fields(message, {"role", "content"}, "message")
    evaluation = record["evaluation"]
    fields(evaluation, {"dimensions", "sources"}, "evaluation")
    if evaluation["dimensions"] != ["authenticity"]:
        raise ValueError("evaluation.dimensions must be ['authenticity']")
    sources(evaluation["sources"], "evaluation.sources")


def evaluation_messages(record: dict) -> list[dict[str, str]]:
    """Return a copy of the authored context with the final assistant target withheld."""
    validate_record(record)
    return [dict(message) for message in record["messages"][:-1]]


def read_records(path: Path, split: str):
    """Yield physical line numbers and validated JSONL objects; blank lines are ignored."""
    found = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, split)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            found = True
            yield line_number, record
    if not found:
        raise ValueError(f"{path}: no records found")


def validate_manifest(path: Path, *, tokenizer=None, max_length: int | None = None) -> dict:
    """Validate all split files together, returning structural coverage counts."""
    if (tokenizer is None) != (max_length is None):
        raise ValueError("tokenizer and max_length must be supplied together")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        fields(manifest, {"schema_version", "dataset_version", "files"}, "manifest")
        version(manifest["schema_version"])
        text(manifest["dataset_version"], "dataset_version")
        fields(manifest["files"], SPLITS, "manifest.files")
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    ids, groups, physical_files = {}, {}, set()
    coverage = {split: {"records": 0, "identities": {}, "topics": {}} for split in sorted(SPLITS)}
    for split, entries in manifest["files"].items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: files.{split} must be a nonempty list")
        for entry in entries:
            try:
                fields(entry, {"path", "sha256"}, f"files.{split}")
                text(entry["path"], "path")
                text(entry["sha256"], "sha256")
                if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                    raise ValueError("sha256 must be 64 lowercase hex digits")
                relative = Path(entry["path"])
                file = (path.parent / relative).resolve()
                if relative.is_absolute() or not file.is_relative_to(path.parent.resolve()):
                    raise ValueError("split file must be inside the manifest directory")
                # Resolve symlinks and detect hard links as well as reused path strings.
                stat = file.stat()
                physical = (stat.st_dev, stat.st_ino)
                if physical in physical_files:
                    raise ValueError(f"split file reused: {file}")
                physical_files.add(physical)
                if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"{file}: sha256 mismatch; review changes and update manifest")
            except (ValueError, OSError) as error:
                raise ValueError(f"{path}: files.{split}: {error}") from error
            for line, record in read_records(file, split):
                location = f"{file}:{line}"
                meta = record["metadata"]
                if meta["id"] in ids:
                    raise ValueError(
                        f"{location}: duplicate ID {meta['id']!r}; first at {ids[meta['id']]}"
                    )
                ids[meta["id"]] = location
                for field in ("conversation_id", "scenario_group"):
                    group = meta[field]
                    if group is None:
                        continue
                    key = (field, group)
                    if key in groups and groups[key][0] != split:
                        raise ValueError(
                            f"{location}: {field} {group!r} crosses splits; "
                            f"first at {groups[key][1]}"
                        )
                    groups.setdefault(key, (split, location))
                if tokenizer is not None:
                    from endless_voices.data import tokenize_messages

                    try:
                        tokenize_messages(record["messages"], tokenizer, max_length)
                    except ValueError as error:
                        raise ValueError(f"{location}: {error}") from error
                stats = coverage[split]
                stats["records"] += 1
                for key, values in (("identities", [meta["identity"]]), ("topics", meta["topics"])):
                    for value in set(values):
                        stats[key][value] = stats[key].get(value, 0) + 1
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="optional local tokenizer directory only")
    parser.add_argument(
        "--max-length", type=int, help="complete-sample token limit for all splits; no truncation"
    )
    args = parser.parse_args()
    if (args.tokenizer is None) != (args.max_length is None):
        parser.error("--tokenizer and --max-length must be supplied together")
    try:
        tokenizer = None
        if args.tokenizer is not None:
            if not args.tokenizer.is_dir():
                raise ValueError("--tokenizer must be an existing local directory")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
            )
        counts = validate_manifest(args.manifest, tokenizer=tokenizer, max_length=args.max_length)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
