"""Generate one continuation per curated sample and save reproducible run records."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig, set_seed

from endless_voices.common import load_config, load_tokenizer
from endless_voices.contracts import evaluation_messages, read_records, validate_manifest


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded(value) + b"\n")
    temporary.replace(path)


def append_json(handle, value) -> None:
    handle.write(encoded(value).decode("utf-8") + "\n")
    handle.flush()


def artifact(source: str, revision: str | None, args) -> tuple[Path, dict]:
    path = Path(source)
    local = path.is_dir()
    if not local:
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{source}: supply a full immutable Hub commit as revision")
        path = Path(
            snapshot_download(
                source,
                revision=revision,
                cache_dir=args.cache_dir,
                local_files_only=args.offline,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
            )
        )
    hashes = {
        str(file.relative_to(path)): file_hash(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
        and file.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}
    }
    if not hashes:
        raise ValueError(f"{source}: no model/tokenizer artifacts found")
    return path.resolve(), {
        "source": source,
        "revision": None if local else revision,
        "kind": "local" if local else "hub",
        "files_sha256": hashes,
    }


def select_samples(args) -> tuple[list[dict], dict]:
    validate_manifest(args.manifest)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = []
    for entry in manifest["files"][args.split]:
        file = args.manifest.parent / entry["path"]
        if file_hash(file) != entry["sha256"]:
            raise ValueError(f"{file}: dataset changed during validation")
        records.extend(record for _, record in read_records(file, args.split))
    if args.sample_ids:
        ids = json.loads(args.sample_ids.read_text(encoding="utf-8"))
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("sample-ids must contain a nonempty JSON list of unique sample IDs")
        by_id = {record["metadata"]["id"]: record for record in records}
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"sample IDs absent from {args.split}: {sorted(missing)}")
        records = [by_id[item] for item in ids]
    if args.limit:
        records = records[: args.limit]
    return records, {
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": digest(manifest_bytes),
        "manifest": manifest,
        "split": args.split,
        "sample_ids": [record["metadata"]["id"] for record in records],
    }


def execution_info() -> dict:
    def git(*arguments):
        try:
            result = subprocess.run(
                ["git", *arguments], capture_output=True, check=True, timeout=10
            )
            return result.stdout
        except (OSError, subprocess.SubprocessError):
            return None

    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
                "numpy",
                "jinja2",
            )
        },
        "git_commit": head.decode().strip() if head else None,
        "git_dirty": bool(status) if status is not None else None,
        "code_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        "mps_available": torch.backends.mps.is_available(),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "torch_threads": torch.get_num_threads(),
    }


def tokenizer_signature(tokenizer) -> str:
    return digest(
        encoded(
            {
                "vocab": tokenizer.get_vocab(),
                "template": tokenizer.chat_template,
                "special_tokens": tokenizer.special_tokens_map,
                "backend": tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else None,
                "class": type(tokenizer).__name__,
            }
        )
    )


def load_condition(args, config, run) -> tuple:
    source = config["model"]["name_or_path"]
    revision = config["model"].get("revision")
    base_path, run["model"] = artifact(source, revision, args)
    tokenizer = load_tokenizer(str(base_path))
    run["tokenizer"] = {**run["model"], "signature": tokenizer_signature(tokenizer)}
    tokenizer.save_pretrained(args.output / "tokenizer")
    adapter_path = None
    if args.adapter:
        adapter_path, run["adapter"] = artifact(args.adapter, args.adapter_revision, args)
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise ValueError("adapter requires adapter_model.safetensors weights")
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        declared_base = adapter_config.base_model_name_or_path
        declared_revision = adapter_config.revision
        saved_run = adapter_path / "run_config.json"
        if saved_run.exists():
            trained = json.loads(saved_run.read_text())["model"]
            if declared_base and declared_base != trained["name_or_path"]:
                raise ValueError("adapter config and saved training config disagree on base model")
            if (
                declared_revision
                and trained.get("revision")
                and declared_revision != trained["revision"]
            ):
                raise ValueError("adapter config and saved training config disagree on revision")
            declared_base = trained["name_or_path"]
            declared_revision = trained.get("revision", declared_revision)
        if declared_base not in {source, str(base_path)}:
            raise ValueError("adapter declares a different base model")
        if declared_revision and declared_revision != revision:
            raise ValueError("adapter declares a different base revision")
        if adapter_config.peft_type != "LORA" or adapter_config.task_type != "CAUSAL_LM":
            raise ValueError("only causal-language-model LoRA adapters are supported")
        if (adapter_path / "tokenizer_config.json").exists():
            other = load_tokenizer(str(adapter_path))
            if tokenizer_signature(other) != tokenizer_signature(tokenizer):
                raise ValueError("adapter tokenizer differs from the starting checkpoint tokenizer")
        run["adapter"]["declared_base"] = declared_base
        run["adapter"]["declared_revision"] = declared_revision
        run["adapter"]["revision_verified"] = bool(declared_revision)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    run["execution"].update({"device": device, "dtype": args.dtype, "batch_size": 1})
    if device == "mps":
        run["execution"]["recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
    if device == "cuda":
        run["execution"]["gpu"] = torch.cuda.get_device_name()
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=getattr(torch, args.dtype),
        use_safetensors=True,
        local_files_only=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.to(device).eval()
    run["model_config"] = model.config.to_dict()
    limits = [args.max_context_tokens, tokenizer.model_max_length]
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and model_limit > 0:
        limits.append(model_limit)
    context_limit = min(limits)
    if args.max_new_tokens >= context_limit:
        raise ValueError("output budget leaves no input space in the effective context limit")
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        top_k=0 if args.temperature > 0 else 50,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=(
            model.generation_config.eos_token_id
            if model.generation_config.eos_token_id is not None
            else tokenizer.eos_token_id
        ),
        pad_token_id=tokenizer.pad_token_id,
    )
    run["generation_config"] = generation.to_dict()
    run["use_model_defaults"] = False
    run["effective_context_tokens"] = context_limit
    run["execution"]["attention_implementation"] = model.config._attn_implementation
    return model, tokenizer, device, context_limit, generation


def prepare_prompt(record, tokenizer, context_limit, max_new_tokens, prompt=None):
    messages = evaluation_messages(record)
    if prompt is None:
        prompt = {}
    prompt.update(
        {
            "sample_id": record["metadata"]["id"],
            "messages": messages,
            "messages_sha256": digest(encoded(messages)),
        }
    )
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt["rendered_prompt_sha256"] = digest(rendered.encode("utf-8"))
    cursor = 0
    for message in messages:
        position = rendered.find(message["content"], cursor)
        if position < 0:
            raise ValueError("chat template omits or transforms authored message content")
        cursor = position + len(message["content"])
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=False,
    )
    ids = inputs["input_ids"][0].tolist()
    prompt.update(
        {"input_ids": ids, "input_tokens": len(ids), "input_ids_sha256": digest(encoded(ids))}
    )
    if not ids or len(ids) + max_new_tokens > context_limit:
        raise ValueError(
            f"input {len(ids)} + requested output {max_new_tokens} exceeds "
            f"context {context_limit}, or input is empty"
        )
    return prompt, inputs


def generate_response(record, model, tokenizer, device, context_limit, generation, args, prompts):
    sample_id = record["metadata"]["id"]
    started = time.monotonic()
    result = {"sample_id": sample_id, "status": "failed", "response": None}
    stage = "prompt"
    prompt = {}
    try:
        prompt, inputs = prepare_prompt(
            record, tokenizer, context_limit, args.max_new_tokens, prompt
        )
        seed = int(digest(f"{args.seed}:{sample_id}".encode())[:8], 16)
        result.update(
            {
                "seed": seed,
                "input_tokens": prompt["input_tokens"],
                "messages_sha256": prompt["messages_sha256"],
                "input_ids_sha256": prompt["input_ids_sha256"],
            }
        )
        set_seed(seed)
        stage = "generation"
        with torch.inference_mode():
            output = model.generate(
                **inputs.to(device), generation_config=generation, use_model_defaults=False
            )
        ids = output[0, prompt["input_tokens"] :].tolist()
        response = tokenizer.decode(ids, skip_special_tokens=True).strip()
        eos = generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        finish = "eos" if ids and ids[-1] in (eos or []) else "length"
        result.update(
            {
                "response": response,
                "output_ids": ids,
                "output_tokens": len(ids),
                "finish_reason": finish,
            }
        )
        if finish == "length":
            result["error"] = {
                "stage": stage,
                "type": "OutputLimit",
                "message": "response did not reach EOS; partial output retained",
            }
        elif not response:
            result["error"] = {
                "stage": stage,
                "type": "EmptyResponse",
                "message": "response contains no visible text",
            }
        else:
            result["status"] = "ok"
    except Exception as error:
        result["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    finally:
        if prompt:
            append_json(prompts, prompt)
    for key in ("messages_sha256", "input_ids_sha256", "input_tokens"):
        if key in prompt:
            result[key] = prompt[key]
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def run_generation(args) -> int:
    config = load_config(args.config)
    records, dataset = select_samples(args)
    args.output.mkdir(parents=True, exist_ok=False)
    run = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "adapter": None,
        "config": config,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "execution": execution_info(),
    }
    write_json(args.output / "run.json", run)
    write_json(args.output / "sample-ids.json", dataset["sample_ids"])
    counts = {"ok": 0, "failed": 0}
    completed = set()
    exit_code = 0
    with (
        (args.output / "responses.jsonl").open("x", encoding="utf-8") as responses,
        (args.output / "prompts.jsonl").open("x", encoding="utf-8") as prompts,
    ):
        try:
            condition = load_condition(args, config, run)
            write_json(args.output / "run.json", run)
            for record in records:
                result = generate_response(record, *condition, args, prompts)
                append_json(responses, result)
                completed.add(result["sample_id"])
                counts[result["status"]] += 1
                print(f"{result['sample_id']}: {result['status']}", flush=True)
        except (Exception, KeyboardInterrupt) as error:
            exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
            run["error"] = {"type": type(error).__name__, "message": str(error)}
            for sample_id in dataset["sample_ids"]:
                if sample_id not in completed:
                    append_json(
                        responses,
                        {
                            "sample_id": sample_id,
                            "status": "failed",
                            "response": None,
                            "error": {"stage": "run", **run["error"]},
                        },
                    )
                    counts["failed"] += 1
    run.update(
        {
            "status": "interrupted"
            if exit_code == 130
            else (
                "failed"
                if exit_code
                else ("completed_with_failures" if counts["failed"] else "complete")
            ),
            "finished_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "artifacts_sha256": {
                str(path.relative_to(args.output)): file_hash(path)
                for path in args.output.rglob("*")
                if path.is_file() and path.name != "run.json"
            },
        }
    )
    write_json(args.output / "run.json", run)
    print(f"Saved {counts} to {args.output}")
    return exit_code or int(bool(counts["failed"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/generate.toml", help="Model TOML config")
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset split manifest")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Split to generate",
    )
    parser.add_argument("--sample-ids", type=Path, help="JSON list; order is preserved")
    parser.add_argument("--limit", type=int, help="First N selected samples")
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--adapter", help="Existing local LoRA directory or Hub ID")
    parser.add_argument("--adapter-revision", help="Full Hub commit for the adapter")
    parser.add_argument("--cache-dir", default="data/local/hub", help="Model download cache")
    parser.add_argument("--offline", action="store_true", help="Require cached model files")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Execution device"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Weight precision",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=4096, help="Total prompt and output budget"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Output cap; reaching it is a failure"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Zero for greedy decoding; positive for sampling",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus probability for sampling")
    parser.add_argument(
        "--seed", type=int, default=42, help="Run seed used to derive stable per-sample seeds"
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_new_tokens < args.max_context_tokens:
        parser.error("require 0 < max-new-tokens < max-context-tokens")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("require finite temperature >= 0 and 0 < top-p <= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.adapter_revision and not args.adapter:
        parser.error("adapter-revision requires adapter")
    return args


def main() -> None:
    args = parse_args()
    try:
        code = run_generation(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
