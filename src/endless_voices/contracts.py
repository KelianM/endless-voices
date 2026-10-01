"""Validate conversation samples and split manifests offline."""

import argparse
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
    """Verify the prepared dataset and return coverage with optional token checks."""
    from endless_voices.dataset.storage import SceneDataset

    if (tokenizer is None) != (max_length is None):
        raise ValueError("tokenizer and max_length must be supplied together")
    datasets = SceneDataset.load_all(path.parent)
    coverage = {}
    for split, dataset in datasets.items():
        stats = {"records": len(dataset), "identities": {}, "topics": {}}
        for record in dataset:
            meta = record["metadata"]
            if tokenizer is not None:
                from endless_voices.data import tokenize_messages

                tokenize_messages(record["messages"], tokenizer, max_length,
                                  label=f"{split}/{meta['id']}")
            for key, values in (("identities", [meta["identity"]]), ("topics", meta["topics"])):
                for value in set(values):
                    stats[key][value] = stats[key].get(value, 0) + 1
        coverage[split] = stats
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
