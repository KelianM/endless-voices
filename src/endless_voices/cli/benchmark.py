"""Prepare, verify and run a benchmark from one shared context selection."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from endless_voices import assessment as assess
from endless_voices import benchmark_generators as benchmark
from endless_voices import generation_backends as screen
from endless_voices import hosted_judge


def main(argv=None):
    parser = argparse.ArgumentParser(prog="endless-voices benchmark", description=__doc__)
    parser.add_argument("stage", choices=["prepare", "preflight", "local-smoke", "local",
                                          "hosted", "export", "judge", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--model", choices=["gemma31b", "qwen30b"],
                        help="Local model for local and local-smoke stages")
    args = parser.parse_args(argv)
    root = args.root
    if args.stage == "prepare":
        if args.dataset is None:
            parser.error("prepare requires --dataset")
        benchmark.prepare(root, args.dataset, args.config)
        return
    benchmark.checked_inputs(root)
    plan = screen.read(root / "plan.json")
    if args.stage == "preflight":
        from transformers import AutoTokenizer
        tokenizers = {name: AutoTokenizer.from_pretrained(
            screen.read(root / f"{name}-config.json")["path"], local_files_only=True,
            **screen.read(root / f"{name}-config.json").get("tokenizer_options", {}))
            for name in plan["local_configs"]}
        benchmark.preflight(root, tokenizers)
    elif args.stage == "local-smoke":
        if args.model not in plan["local_configs"]:
            parser.error("Choose a selected local model with --model")
        benchmark.local_smoke(root, args.model)
    elif args.stage == "local":
        if args.model not in plan["local_configs"]:
            parser.error("Choose a selected local model with --model")
        screen.local(root, root / f"{args.model}-config.json")
    elif args.stage == "hosted":
        screen.hosted(root, models=[m for m in plan["generators"]
                                    if m not in plan["local_configs"]],
                      budget=plan["generation_budget_usd"],
                      timeout_seconds=plan.get("api_timeout_seconds", 1800))
    elif args.stage == "export":
        benchmark.export(root)
    elif args.stage == "judge":
        if not (root / "trial-verification.json").exists():
            raise ValueError("Verified trials are required")
        benchmark.verify_trials(root, save=False)
        hosted_judge.run(SimpleNamespace(public=root / "assessment/public", output=root / "judge",
                                        env_file=Path(".env"), models=[plan["judge"]],
                                        budget=plan["judge_budget_usd"], limit=1000,
                                        timeout_seconds=plan.get("api_timeout_seconds", 1800)))
    else:
        assess.report(root / "assessment", [root / "judge" / plan["judge"] / "review.json"],
                      root / "report")

