"""Prepare, verify and run a benchmark from one shared context selection."""

import argparse
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from endless_voices import assessment as assess
from endless_voices import generation_backends as screen
from endless_voices import openai_judge
from endless_voices.context import substitute_variables
from endless_voices.prepare_context import load_selections


def checked_inputs(root):
    """Verify immutable preparation artifacts and return the plan and records."""
    selection = screen.read(root / "selection.json")
    for name, expected in selection["artifacts_sha256"].items():
        if screen.sha(root / name) != expected:
            raise ValueError(f"Benchmark preparation changed: {name}")
    contexts = load_selections(root / "context")
    records = screen.read(root / "records.json")
    prompts = screen.read(root / "prompts.json")
    if [s.sample_id for s in contexts] != selection["ids"]:
        raise ValueError("Context selection IDs differ")
    if prompts != [s.generation_prompt() for s in contexts]:
        raise ValueError("Generation prompts differ from saved context")
    for s in contexts:
        if assess.evaluation_messages(records[s.sample_id]) != s.judge_context():
            raise ValueError("Dataset and judge context differ")
    return selection, records, prompts


def prepare(root, context, manifest, config):
    """Join selected inputs to validation targets, preserving the changed context provenance."""
    if root.exists():
        raise ValueError("Output exists; choose a new directory")
    selections = load_selections(context)
    records = assess.load_dataset(manifest, "validation")
    ids = [s.sample_id for s in selections]
    if set(ids) != set(records):
        raise ValueError("Expected all validation samples, without test or extra samples")
    plan = screen.read(config)
    supported = {"gemma31b", "qwen30b", "gpt-6-luna", "claude-sonnet-5"}
    if (not plan["generators"] or set(plan["generators"]) - supported
            or len(set(plan["generators"])) != len(plan["generators"])
            or plan["judge"] != "gpt-6-luna"):
        raise ValueError("Select unique supported generators and Luna as judge")
    if not (0 < plan["generation_budget_usd"] <= 3
            and 0 < plan["judge_budget_usd"] <= 2):
        raise ValueError("Require positive budgets: generation at most $3 and judging at most $2")
    expected_local = set(plan["generators"]) & {"gemma31b", "qwen30b"}
    if set(plan["local_configs"]) != expected_local:
        raise ValueError("Local configuration names differ from selected models")
    locals_ = {name: screen.read(path) for name, path in plan["local_configs"].items()}
    if any(cfg["label"] != name for name, cfg in locals_.items()):
        raise ValueError("Local model label differs from configuration")
    prepared = {}
    for s in selections:
        original = records[s.sample_id]
        values = s.provenance.get("game_variables", {})
        target = substitute_variables(original["messages"][-1]["content"], values)
        if any(target in m["content"] for m in s.messages):
            raise ValueError("Original target appears in generation context")
        if re.search(r"<[^>]+>", json.dumps(s.messages) + target):
            raise ValueError("Unresolved game variable in final input or target")
        prepared[s.sample_id] = {**deepcopy(original), "messages": s.training_messages(target)}
    root.mkdir(parents=True)
    shutil.copytree(context, root / "context")
    licensing = manifest.parent.parent / "licensing"
    if licensing.is_dir():
        shutil.copytree(licensing, root / "attribution")
    screen.save(root / "records.json", prepared)
    screen.save(root / "prompts.json", [s.generation_prompt() for s in selections])
    for name, cfg in locals_.items():
        screen.save(root / f"{name}-config.json", cfg)
    screen.save(root / "plan.json", plan)
    for p in (Path(__file__), Path(screen.__file__)):
        shutil.copy2(p, root / p.name)
    files = [p for p in root.rglob("*") if p.is_file()]
    screen.save(root / "selection.json", {
        "ids": ids, "split": "validation", "manifest_sha256": screen.sha(manifest),
        "context_sha256": screen.sha(root / "context/provenance.json"),
        "target_source": str(manifest), "models": plan["generators"],
        "context": "Selected game-source context replaces the earlier dataset prompts",
        "candidate_order": "one balanced randomized assignment per condition; no reversed trials",
        "artifacts_sha256": {str(p.relative_to(root)): screen.sha(p) for p in files},
    })
    checked_inputs(root)


def preflight(root, tokenizers):
    """Check complete local prompts and judge request bounds before model execution."""
    _, records, prompts = checked_inputs(root)
    plan = screen.read(root / "plan.json")
    locals_ = {name: screen.read(root / f"{name}-config.json")
               for name in plan["local_configs"]}
    instructions = Path("data/evaluation/judge-instructions.md").read_text()
    rows = []
    for prompt in prompts:
        sid = prompt["sample_id"]
        counts = {}
        for name, tokenizer in tokenizers.items():
            rendered = tokenizer.apply_chat_template(
                prompt["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            counts[name] = len(tokenizer.encode(rendered, add_special_tokens=False))
            local = locals_[name]
            if counts[name] + local["max_output_tokens"] > local["context_ceiling"]:
                raise ValueError(f"{name} context exceeds configured limit; no truncation")
        original = records[sid]["messages"][-1]["content"]
        trial = {"trial_id": "0" * 32, "context": prompt["messages"],
                 "A": original, "B": "Preflight placeholder; no judgment requested."}
        body = openai_judge.payload(plan["judge"], trial, instructions)
        bound = len(json.dumps(body).encode()) + 2048 + 4096 + body["max_output_tokens"]
        if bound > plan["judge_context_limit"]:
            raise ValueError("Judge request bound exceeds context limit")
        rows.append({"sample_id": sid, "generation_tokens": counts,
                     "judge_request_bound_with_candidate_and_output_reserve": bound})
    screen.save(root / "preflight.json", {
        "samples": rows, "longest_sample_ids": {
            name: max(rows, key=lambda r: r["generation_tokens"][name])["sample_id"]
            for name in tokenizers},
        "method": "Exact local prompt tokens; Luna UTF-8 input upper bound plus 4096 candidate "
                  "tokens and 4096 judge output tokens. Actual candidates checked at export.",
        "no_api_calls": True, "unresolved_variables": 0,
    })


def local_smoke(root, model):
    """Run only the longest input in a new folder without overwriting full-run outputs."""
    _, _, prompts = checked_inputs(root)
    sid = screen.read(root / "preflight.json")["longest_sample_ids"][model]
    smoke = root / f"smoke-{model}"
    smoke.mkdir(exist_ok=False)
    screen.save(smoke / "prompts.json", [p for p in prompts if p["sample_id"] == sid])
    screen.local(smoke, root / f"{model}-config.json")


def export(root):
    """Verify raw generation evidence and create one blinded primary trial per success."""
    selection, records, original_prompts = checked_inputs(root)
    plan = screen.read(root / "plan.json")
    ids = selection["ids"]
    originals = {p["sample_id"]: p for p in original_prompts}
    loaded, provenance = {}, {}
    for model in plan["generators"]:
        source = root / model
        settings = screen.read(source / "settings.json")
        if settings["prompts_sha256"] != screen.sha(root / "prompts.json"):
            raise ValueError("Generation input hash changed")
        if settings["code_sha256"] != screen.sha(root / "generation_backends.py"):
            raise ValueError("Generation implementation changed")
        raw_results = assess.indexed([screen.read(p) for p in source.glob("*.result.json")], ids)
        prompts, responses = {}, {}
        for sid, raw in raw_results.items():
            messages = originals[sid]["messages"]
            prompt = {"sample_id": sid, "messages": messages,
                      "messages_sha256": assess.digest(assess.encoded(messages))}
            if model in plan["local_configs"]:
                actual = screen.read(source / f"{sid}.prompt.json")
                if actual["messages"] != messages:
                    raise ValueError("Local prompt differs")
                prompt.update(input_ids=actual["input_ids"], input_tokens=len(actual["input_ids"]),
                              input_ids_sha256=assess.digest(assess.encoded(actual["input_ids"])))
                complete = raw.get("finish_reason") == "stop"
            else:
                body = screen.read(source / f"{sid}.request.json")["body"]
                actual = body.get("input") or [{"role": "system", "content": body["system"]},
                                               *body["messages"]]
                if actual != messages or body["model"] != model:
                    raise ValueError("Hosted request differs")
                prompt.update(input_ids_sha256=None, input_tokens=None)
                api = raw.get("api_response", {})
                complete = (api.get("stop_reason") == "end_turn" if model == "claude-sonnet-5"
                            else api.get("status") == "completed")
            if raw["status"] == "ok" and not complete:
                raise ValueError("Success without provider completion")
            prompts[sid] = prompt
            responses[sid] = {**raw, "finish_reason": "eos" if raw["status"] == "ok"
                              else raw.get("finish_reason"), **{k: prompt[k] for k in
                              ("messages_sha256", "input_ids_sha256", "input_tokens")}}
        run = {"dataset": {"sample_ids": ids}, "model": settings}
        loaded[model] = (run, prompts, responses)
        provenance[model] = {"run": run, "verified_hashes": True,
                             "raw_artifacts_sha256": {str(p.relative_to(root)): screen.sha(p)
                                                      for p in source.glob("*.json")}}
    controls = [{"sample_id": ids[0], "kind": "identical"},
                {"sample_id": ids[0], "kind": "wrong-context", "donor_id": ids[-1]}]
    assess.prepare_trials(records, loaded, root / "assessment", provenance, controls=controls,
                          licensing=root / "attribution", input_provenance=selection)
    verify_trials(root)


def verify_trials(root, save=True):
    """Check the public request boundary, answer mapping and actual judge request sizes."""
    selection, records, prompts = checked_inputs(root)
    plan = screen.read(root / "plan.json")
    pack = root / "assessment"
    private = screen.read(pack / "private.json")
    instructions = (pack / "public/instructions.txt").read_text()
    by_id = {p["sample_id"]: p["messages"] for p in prompts}
    if assess.digest(instructions.encode()) != private["instructions_sha256"]:
        raise ValueError("Judge instructions changed")
    expected = {r["path"] for r in private["trials"]}
    actual = {str(p.relative_to(pack)) for p in (pack / "public").glob("*/*.json")}
    if expected != actual or len(expected) != len(private["trials"]):
        raise ValueError("Public trial inventory differs")
    seen, bounds = set(), []
    for row in private["trials"]:
        if screen.sha(pack / row["path"]) != row["trial_sha256"]:
            raise ValueError("Public trial hash differs")
        trial = screen.read(pack / row["path"])
        if list(trial) != ["trial_id", "context", "A", "B"]:
            raise ValueError("Candidate serialization order differs")
        if trial["context"] != by_id[row["sample_id"]]:
            raise ValueError("Judge context differs from generation")
        if row["kind"] == "primary":
            key = (row["condition"], row["sample_id"])
            if key in seen:
                raise ValueError("Duplicate primary judgment")
            seen.add(key)
            if trial[row["original"]] != records[row["sample_id"]]["messages"][-1]["content"]:
                raise ValueError("Answer mapping differs")
        elif row["kind"] not in {"identical", "wrong-context"}:
            raise ValueError("Reversed trials are not part of this benchmark")
        body = openai_judge.payload(plan["judge"], trial, instructions)
        bound = len(json.dumps(body).encode()) + 2048 + body["max_output_tokens"]
        if bound > plan["judge_context_limit"]:
            raise ValueError("Actual judge request exceeds context limit")
        bounds.append(bound)
    verification = {
        "primary_trials": len(seen), "total_trials": len(bounds),
        "maximum_request_bound": max(bounds), "same_context": True, "fixed_A_then_B": True,
        "single_order": True, "context_sha256": selection["context_sha256"],
    }
    if save:
        screen.save(root / "trial-verification.json", verification)
    elif verification != screen.read(root / "trial-verification.json"):
        raise ValueError("Trial verification changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "preflight", "local-smoke", "local",
                                          "hosted", "export", "judge", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/pilot-v1/samples/manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--model", choices=["gemma31b", "qwen30b"],
                        help="Local model for local and local-smoke stages")
    args = parser.parse_args()
    root = args.root
    if args.stage == "prepare":
        if args.context is None:
            parser.error("prepare requires --context")
        prepare(root, args.context, args.manifest, args.config)
        return
    checked_inputs(root)
    plan = screen.read(root / "plan.json")
    if args.stage == "preflight":
        from transformers import AutoTokenizer
        tokenizers = {name: AutoTokenizer.from_pretrained(
            screen.read(root / f"{name}-config.json")["path"], local_files_only=True)
            for name in plan["local_configs"]}
        preflight(root, tokenizers)
    elif args.stage == "local-smoke":
        if args.model not in plan["local_configs"]:
            parser.error("Choose a selected local model with --model")
        local_smoke(root, args.model)
    elif args.stage == "local":
        if args.model not in plan["local_configs"]:
            parser.error("Choose a selected local model with --model")
        screen.local(root, root / f"{args.model}-config.json")
    elif args.stage == "hosted":
        screen.hosted(root, models=[m for m in plan["generators"]
                                    if m not in plan["local_configs"]],
                      budget=plan["generation_budget_usd"])
    elif args.stage == "export":
        export(root)
    elif args.stage == "judge":
        if not (root / "trial-verification.json").exists():
            raise ValueError("Verified trials are required")
        verify_trials(root, save=False)
        openai_judge.run(SimpleNamespace(public=root / "assessment/public", output=root / "judge",
                                        env_file=Path(".env"), models=[plan["judge"]],
                                        budget=plan["judge_budget_usd"], limit=1000))
    else:
        assess.report(root / "assessment", [root / "judge" / plan["judge"] / "review.json"],
                      root / "report")


if __name__ == "__main__":
    main()
