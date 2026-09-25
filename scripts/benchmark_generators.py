"""Benchmark the selected generators on validation with immutable raw evidence."""

import argparse
import json
import shutil
from pathlib import Path

import screen_generators as screen

from endless_voices import assessment as assess

MODELS = ("gpt-6-sol", "gemma31b", "claude-sonnet-5")
MANIFEST = Path("data/pilot-v1/samples/manifest.json")
SOURCE = Path("outputs/qwen3-4b-full-validation-v1")


def prepare(root):
    records = assess.load_dataset(MANIFEST, "validation")
    run, prompts, _ = assess.load_run(SOURCE, MANIFEST, records, "validation")
    ids = run["dataset"]["sample_ids"]
    if set(ids) != set(records):
        raise ValueError("Expected the complete validation selection")
    root.mkdir(exist_ok=False)
    screen.save(root / "prompts.json", [prompts[sid] for sid in ids])
    screen.save(
        root / "selection.json",
        {
            "split": "validation",
            "ids": ids,
            "manifest_sha256": screen.sha(MANIFEST),
            "source_run_sha256": screen.sha(SOURCE / "run.json"),
            "models": MODELS,
            "generation_budget_usd": 3,
            "judge_budget_usd": 2,
            "selection": "All validation scenes; no test scenes; fresh generation for all models",
        },
    )
    cfg = screen.read("outputs/generator-screen-v1/gemma31b-config-v2.json")
    cfg["label"] = "gemma31b"
    screen.save(root / "gemma31b-config.json", cfg)
    for path in (Path(__file__), Path(screen.__file__)):
        shutil.copy2(path, root / path.name)


def export(root):
    """Normalize recorded completion evidence without inventing provider tokenization."""
    selection = screen.read(root / "selection.json")
    if screen.sha(MANIFEST) != selection["manifest_sha256"]:
        raise ValueError("Dataset changed")
    records = assess.load_dataset(MANIFEST, "validation")
    originals = {p["sample_id"]: p for p in screen.read(root / "prompts.json")}
    ids = selection["ids"]
    runs = {}
    for model in MODELS:
        source = root / model
        settings = screen.read(source / "settings.json")
        if settings["prompts_sha256"] != screen.sha(root / "prompts.json"):
            raise ValueError("Generation input hash changed")
        if settings["code_sha256"] != screen.sha(root / "screen_generators.py"):
            raise ValueError("Generation implementation changed")
        results = list(source.glob("*.result.json"))
        if {p.name.removesuffix(".result.json") for p in results} != set(ids):
            raise ValueError(f"{model}: incomplete generation; inspect saved failures")
        dest = root / "runs" / model
        dest.mkdir(parents=True, exist_ok=False)
        prompts, responses = [], []
        for sid in ids:
            messages = originals[sid]["messages"]
            if messages != assess.evaluation_messages(records[sid]):
                raise ValueError("Authored context differs")
            raw = screen.read(source / f"{sid}.result.json")
            if raw["sample_id"] != sid:
                raise ValueError("Response identity differs")
            prompt = {
                "sample_id": sid,
                "messages": messages,
                "messages_sha256": assess.digest(assess.encoded(messages)),
            }
            if model == "gemma31b":
                actual = screen.read(source / f"{sid}.prompt.json")
                if actual["messages"] != messages:
                    raise ValueError("Local prompt differs")
                prompt.update(
                    input_ids=actual["input_ids"],
                    input_ids_sha256=assess.digest(assess.encoded(actual["input_ids"])),
                    input_tokens=len(actual["input_ids"]),
                )
                complete = raw.get("finish_reason") == "stop"
            else:
                body = screen.read(source / f"{sid}.request.json")["body"]
                actual = body.get("input") or [
                    {"role": "system", "content": body["system"]},
                    *body["messages"],
                ]
                if actual != messages or body["model"] != model:
                    raise ValueError("Hosted request differs")
                # Hosted token IDs are unavailable; preserve that absence explicitly.
                prompt.update(input_ids_sha256=None, input_tokens=None)
                api = raw.get("api_response", {})
                complete = (
                    api.get("status") == "completed"
                    if model == "gpt-6-sol"
                    else api.get("stop_reason") == "end_turn"
                )
            if raw["status"] == "ok" and not complete:
                raise ValueError("Success without provider completion")
            response = {
                **raw,
                **{k: prompt[k] for k in ("messages_sha256", "input_ids_sha256", "input_tokens")},
                "finish_reason": "eos" if raw["status"] == "ok" else raw.get("finish_reason"),
            }
            prompts.append(prompt)
            responses.append(response)
        screen.save(dest / "sample-ids.json", ids)
        for name, rows in [("prompts.jsonl", prompts), ("responses.jsonl", responses)]:
            with (dest / name).open("x") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        run = {
            "schema_version": 1,
            "status": "complete",
            "dataset": {
                "manifest": screen.read(MANIFEST),
                "manifest_sha256": screen.sha(MANIFEST),
                "split": "validation",
                "sample_ids": ids,
            },
            "model": settings,
            "normalization": (
                "eos denotes recorded normal provider completion; raw evidence retained. "
                "Hosted token IDs/counts are unavailable, represented by null."
            ),
            "raw_artifacts_sha256": {str(p): screen.sha(p) for p in source.glob("*.json")},
            "artifacts_sha256": {p.name: screen.sha(p) for p in dest.iterdir()},
        }
        screen.save(dest / "run.json", run)
        assess.load_run(dest, MANIFEST, records, "validation")
        runs[model] = dest
    controls = [
        {"sample_id": ids[0], "kind": "identical"},
        {"sample_id": ids[0], "kind": "wrong-context", "donor_id": ids[-1]},
    ]
    assess.prepare(MANIFEST, runs, root / "assessment", controls=controls, reverse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "hosted", "export"])
    parser.add_argument("--root", type=Path, default=Path("outputs/generator-benchmark-v1"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
    elif args.stage == "hosted":
        screen.hosted(args.root, models=["gpt-6-sol", "claude-sonnet-5"], budget=3)
    else:
        export(args.root)
