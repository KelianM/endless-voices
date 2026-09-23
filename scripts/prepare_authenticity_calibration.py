"""Prepare a blinded authenticity review pack from pinned local source passages."""

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from endless_voices.contracts import evaluation_messages, validate_record

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "data/evaluation/calibration.json"
INVENTORY = ROOT / "data/source-review/source-inventory.json"


def extract(lines, selector, ranges):
    """Return selected direct-speech spans from one inventoried source line."""
    number = selector["line"]
    if not any(start <= number <= end for start, end in ranges):
        raise ValueError(f"Line {number} is outside the declared evidence passages")
    line = lines[number - 1].strip()
    if not (line.startswith("`") and line.endswith("`")):
        raise ValueError(f"Line {number} is not a backtick-delimited prose line")
    quotes = re.findall(r'"([^"\n]+)"', line[1:-1])
    try:
        selected = [quotes[index] for index in selector["quotes"]]
        for index in selector.get("sentence_breaks_after", []):
            if not selected[index].endswith(","):
                raise ValueError(f"Line {number}: expected narrator-introducing comma")
            selected[index] = selected[index][:-1] + "."
        result = " ".join(selected)
    except IndexError as error:
        raise ValueError(f"Line {number}: missing selected speech span") from error
    if not result or re.search(r"<[^>]+>", result):
        raise ValueError(f"Line {number}: empty speech or unresolved runtime placeholder")
    return result


def samples_from_sources(recipe, inventory, source_root):
    """Build validation samples after verifying the selected source file hashes."""
    if recipe["revision"] != inventory["upstream"]["revision"]:
        raise ValueError("Recipe and source inventory revisions differ")
    passages = {row["id"]: row for row in inventory["passages"]}
    files = {row["path"]: row for row in inventory["files"]}
    records = []
    for case in recipe["cases"]:
        passage = passages[case["passage"]]
        name = passage["source_file"]
        raw = (source_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != files[name]["sha256"]:
            raise ValueError(f"{name}: source checksum mismatch")
        evidence = [passage, *(passages[key] for key in case["context_passages"])]
        ranges = [row["lines"] for row in evidence if row["source_file"] == name]
        lines = raw.decode("utf-8").splitlines()
        sources = [
            {
                "reference": row["source_url"],
                "revision": recipe["revision"],
                "source_group": row["source_group"],
            }
            for row in evidence
        ]
        record = {
            "schema_version": 1,
            "messages": [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": extract(lines, case["user"], ranges)},
                {"role": "assistant", "content": extract(lines, case["target"], ranges)},
            ],
            "metadata": {
                key: case[key]
                for key in (
                    "id",
                    "identity",
                    "species",
                    "character_role",
                    "topics",
                    "conversation_id",
                    "scenario_group",
                )
            }
            | {
                "split": "validation",
                "authorship": "mixed",
                "review_status": "draft",
                "sources": sources,
            },
            "evaluation": {
                "dimensions": ["authenticity"],
                **case["expectations"],
                "sources": sources,
            },
        }
        validate_record(record, "validation")
        records.append(record)
    return records


def review_pack(records, recipe, form):
    """Return blinded trials and a separate provenance key for a calibration form."""
    trials = []
    for index, (record, case) in enumerate(zip(records, recipe["cases"], strict=True)):
        original = record["messages"][-1]["content"]
        donor = records[(index + 1) % len(records)]
        for kind, other, donor_id in (
            ("authored-alternative", case["alternative"], None),
            ("wrong-context", donor["messages"][-1]["content"], donor["metadata"]["id"]),
        ):
            trials.append((record, kind, [original, other], donor_id))
    record = records[1]
    original = record["messages"][-1]["content"]
    trials.append((record, "identical", [original, original], record["metadata"]["id"]))
    random.Random(6).shuffle(trials)
    positions = [False, True] * (len(trials) // 2) + [False] * (len(trials) % 2)
    random.Random(111).shuffle(positions)
    public, key = [], []
    for index, (record, kind, choices, donor_id) in enumerate(trials):
        swap = positions[index] != (form == "b")
        trial_id = f"{form}-{index + 1:02}"
        public.append(
            {
                "trial_id": trial_id,
                "context": evaluation_messages(record),
                "A": choices[int(swap)],
                "B": choices[1 - int(swap)],
            }
        )
        key.append(
            {
                "trial_id": trial_id,
                "sample_id": record["metadata"]["id"],
                "kind": kind,
                "original": None if kind == "identical" else "B" if swap else "A",
                "replacement_sample_id": donor_id,
                "replacement_origin": "agent-authored-with-source-access"
                if kind == "authored-alternative"
                else "upstream-speech",
                "expected_control": "abstain"
                if kind == "identical"
                else "original"
                if kind == "wrong-context"
                else None,
            }
        )
    return public, key


def prepare(source_root, output, form):
    """Write a new local review directory without overwriting earlier reviews."""
    if output.exists():
        raise ValueError(f"{output}: already exists; choose a new review directory")
    recipe = json.loads(RECIPE.read_text())
    inventory = json.loads(INVENTORY.read_text())
    records = samples_from_sources(recipe, inventory, source_root)
    public, key = review_pack(records, recipe, form)
    for name in ("license.txt", "copyright", "credits.txt"):
        if not (source_root / name).is_file():
            raise ValueError(f"Missing upstream attribution: {name}")
    output.mkdir(parents=True)
    for name in ("license.txt", "copyright", "credits.txt"):
        shutil.copyfile(source_root / name, output / name)
    stages = {
        "review": [
            row
            for row, secret in zip(public, key, strict=True)
            if secret["kind"] == "authored-alternative"
        ],
        "controls": [
            row
            for row, secret in zip(public, key, strict=True)
            if secret["kind"] != "authored-alternative"
        ],
    }
    for name, rows in {"samples": records, **stages}.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    private = {
        "version": recipe["version"],
        "revision": recipe["revision"],
        "form": form,
        "human_review_status": "pending",
        "trials": key,
    }
    (output / "answer-key.json").write_text(json.dumps(private, indent=2) + "\n")
    for stage, rows in stages.items():
        reviews = {
            "version": recipe["version"],
            "form": form,
            "reviewer_id": None,
            "reviewer_type": None,
            "judge_model_and_prompt": None,
            "reviews": [
                {
                    "trial_id": row["trial_id"],
                    "choice": None,
                    "confidence": None,
                    "reason": None,
                    "recognized_source": None,
                }
                for row in rows
            ],
        }
        (output / f"{stage}-template.json").write_text(json.dumps(reviews, indent=2) + "\n")
        text = [
            "# Authenticity calibration\n",
            (ROOT / "data/evaluation/judge-instructions.md").read_text(),
        ]
        for row in rows:
            text.append(f"\n## Trial {row['trial_id']}\n")
            for message in row["context"]:
                text.append(f"**{message['role'].capitalize()}**\n\n{message['content']}\n")
            text.extend([f"**A**\n\n{row['A']}\n", f"**B**\n\n{row['B']}\n"])
        (output / f"{stage}.md").write_text("\n".join(text))
    (output / "ATTRIBUTION.md").write_text(
        "# Attribution\n\nLocal calibration excerpts from Endless Sky at "
        + recipe["revision"]
        + ". Source text is GPL-3.0-or-later; see license.txt, copyright, and credits.txt. "
        "Selected file headers credit Michael Zahniser (2014 and 2015) and upstream contributors. "
        "Samples retain passage URLs. Dialogue has been reduced to selected speech spans; "
        "system contexts and alternative responses were authored by an agent for this project. "
        "Treat this mixed calibration pack as GPL-3.0-or-later when redistributing. "
        "Do not send the answer key, samples, or attribution links to a blinded judge.\n"
    )
    return len(public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=ROOT / "data/local/endless-sky-7140eb2a29ce"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--form", choices=("a", "b"), default="a")
    args = parser.parse_args()
    try:
        count = prepare(args.sources, args.output, args.form)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Prepared {count} development trials at {args.output}; human calibration is pending.")


if __name__ == "__main__":
    main()
