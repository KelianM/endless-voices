"""Select validation-only judge calibration scenes without inspecting generated outputs."""

import argparse
import random
from pathlib import Path

from endless_voices.assessment import file_hash, load_dataset, read_json, write_json

# These conversations have already appeared in calibration or smoke work.
REUSED = {
    "human-free-worlds-3-reconciliation-l557",
    "human-free-worlds-2-middle-l121",
    "human-free-worlds-0-prologue-l345",
    "hai-hai-missions-l20",
    "hai-hai-culture-conversations-l187",
    "quarg-quarg-missions-l21",
}
LEGACY_TARGETS = {
    "first-contact-hai-l90-236015e6",
    "quarg-first-contact-l46-4f92d6f0",
    "fw-recon-1-anonymous-captain-merchant-scan-l356-bf239e1d",
}


def select(output):
    manifest = Path("data/pilot-v1/samples/manifest.json")
    records = load_dataset(manifest, "validation")
    excluded = set(read_json(Path("data/evaluation/smoke-sample-ids.json"))) | LEGACY_TARGETS
    rng = random.Random(20260924)
    chosen, used = [], set()
    for identity, count in (
        ("human-free-worlds", 5),
        ("republic", 2),
        ("hai-mainstream", 2),
        ("quarg", 1),
    ):
        candidates = [
            r
            for sid, r in records.items()
            if sid not in excluded and r["metadata"]["identity"] == identity
        ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda r: r["metadata"]["conversation_id"] in REUSED)
        for row in candidates:
            conversation = row["metadata"]["conversation_id"]
            if conversation in used:
                continue
            chosen.append(row["metadata"])
            used.add(conversation)
            count -= 1
            if count == 0:
                break
        if count:
            raise ValueError("Insufficient distinct validation conversations")
    remaining = [
        r for r in records.values() if r["metadata"]["conversation_id"] not in used | REUSED
    ]
    remaining.sort(key=lambda r: r["metadata"]["id"])
    controls = []
    for row in remaining:
        c = row["metadata"]["conversation_id"]
        if c in used:
            continue
        controls.append(row["metadata"]["id"])
        used.add(c)
        if len(controls) == 2:
            break
    if len(controls) != 2:
        raise ValueError("Insufficient separate control conversations")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "sample-ids.json", [r["id"] for r in chosen])
    write_json(
        output / "controls.json",
        [
            {"sample_id": controls[0], "kind": "identical"},
            {"sample_id": controls[1], "kind": "wrong-context", "donor_id": controls[0]},
        ],
    )
    write_json(
        output / "selection.json",
        {
            "seed": 20260924,
            "manifest_sha256": file_hash(manifest),
            "selection": "Seeded selection of one sample per conversation; "
            "prefer unused conversations",
            "primary": [
                {
                    "sample_id": r["id"],
                    "identity": r["identity"],
                    "conversation_id": r["conversation_id"],
                    "conversation_previously_used": r["conversation_id"] in REUSED,
                }
                for r in chosen
            ],
            "limitations": "Hai and Quarg reuse conversations; "
            "unreviewed turns do not erase familiarity. "
            "Control donor is reused only within the control stage. "
            "This round does not validate subtle context errors comprehensively.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    select(parser.parse_args().output)
