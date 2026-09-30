"""Run local or hosted generation from saved benchmark prompts."""

import hashlib
import importlib.metadata
import json
import time
import urllib.error
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def quantize_full_history(caches, full_cache_type, *, bits=8, group_size=64):
    """Convert full-history caches while preserving all other cache objects."""
    return [
        c.to_quantized(bits=bits, group_size=group_size) if type(c) is full_cache_type else c
        for c in caches
    ]


def local(root, config):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    cfg = read(config)
    folder = root / cfg["label"]
    folder.mkdir(exist_ok=False)
    model_path = Path(cfg["path"])
    context_limit = cfg.get("context_ceiling", 16384)
    output_limit = cfg.get("max_output_tokens", 1024)
    prefill_step_size = cfg.get("prefill_step_size", 256)
    save(
        folder / "settings.json",
        {
            **cfg,
            "code_sha256": sha(__file__),
            "prompts_sha256": sha(root / "prompts.json"),
            "max_output_tokens": output_limit,
            "context_ceiling": context_limit,
            "prefill_step_size": prefill_step_size,
            "temperature": 0,
            "enable_thinking": False,
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "model_files_sha256": {
                str(p.relative_to(model_path)): sha(p) for p in model_path.rglob("*") if p.is_file()
            },
        },
    )
    mx.set_memory_limit(20_000_000_000)
    mx.set_cache_limit(256_000_000)
    model, tokenizer = load(
        str(model_path),
        tokenizer_config={"trust_remote_code": False, **cfg.get("tokenizer_options", {})},
    )
    for prompt in read(root / "prompts.json"):
        started = time.monotonic()
        row = {"sample_id": prompt["sample_id"], "status": "failed", "response": ""}
        rendered = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        save(
            folder / (prompt["sample_id"] + ".prompt.json"),
            {"messages": prompt["messages"], "rendered": rendered, "input_ids": ids},
        )
        try:
            if len(ids) + output_limit > context_limit:
                raise ValueError("Context overflow; no truncation")
            mx.random.seed(42)
            last = None
            cache_options = {
                "kv_bits": cfg.get("kv_bits"),
                "kv_group_size": cfg.get("kv_group_size", 64),
                "quantized_kv_start": cfg.get("quantized_kv_start", 0),
            }
            if cfg.get("kv_scope") == "full_history":
                from mlx_lm.models.cache import KVCache, make_prompt_cache

                caches = quantize_full_history(
                    make_prompt_cache(model), KVCache,
                    bits=cfg["kv_bits"], group_size=cfg.get("kv_group_size", 64),
                )
                cache_options = {"prompt_cache": caches}
                row["cache_types"] = [type(c).__name__ for c in caches]
            for token in stream_generate(
                model,
                tokenizer,
                prompt=ids,
                max_tokens=output_limit,
                sampler=make_sampler(temp=0),
                prefill_step_size=prefill_step_size,
                **cache_options,
            ):
                row["response"] += token.text
                last = token
                if mx.get_peak_memory() > 20_000_000_000:
                    raise MemoryError("20 GB experiment ceiling exceeded")
                if time.monotonic() - started > 600:
                    raise TimeoutError("600 second trial ceiling exceeded")
            if last:
                row.update(output_tokens=last.generation_tokens, finish_reason=last.finish_reason)
            if last is None or last.finish_reason != "stop":
                raise ValueError("Output limit or incomplete response")
            row["status"] = "ok"
        except Exception as error:
            row["error"] = str(error)
        row.update(
            elapsed_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory()
        )
        save(folder / (prompt["sample_id"] + ".result.json"), row)
        print(cfg["label"], prompt["sample_id"], row["status"], flush=True)
        if (row.get("peak_memory_bytes", 0) > 20_000_000_000
                or "Insufficient Memory" in row.get("error", "")):
            break


def hosted(root, models=None, budget=5, timeout_seconds=1800):
    from endless_voices import providers

    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")

    for model in providers.RATES:
        if models is not None and model not in models:
            continue
        folder = root / model
        folder.mkdir(exist_ok=False)
        key = providers.load_key(Path(".env"), model)
        save(
            folder / "settings.json",
            {
                "model": model,
                "code_sha256": sha(__file__),
                "provider_code_sha256": sha(providers.__file__),
                "prompts_sha256": sha(root / "prompts.json"),
                "reasoning": "medium",
                "max_output_tokens_including_thinking": 4096,
                "endpoint": providers.endpoint(model),
                "budget_usd": budget,
                "timeout_seconds": timeout_seconds,
                "rates_per_million_usd": providers.RATES[model],
            },
        )
        for prompt in read(root / "prompts.json"):
            body = providers.payload(model, prompt["messages"])
            reserved = 0.0
            for path in root.glob("*/*.request.json"):
                result = path.with_name(path.name.replace(".request.", ".result."))
                cost = read(result).get("estimated_cost_usd") if result.exists() else None
                reserved += read(path)["reservation_usd"] if cost is None else cost
            reserve = providers.reservation(body)
            if reserved + reserve > budget:
                raise ValueError("Hosted screen budget exhausted")
            save(
                folder / (prompt["sample_id"] + ".request.json"),
                {"body": body, "reservation_usd": reserve},
            )
            row = {"sample_id": prompt["sample_id"], "status": "failed", "response": ""}
            started = time.monotonic()
            try:
                response = providers.send(body, key, timeout_seconds)
                row["api_response"] = response
                row["estimated_cost_usd"] = providers.charge(model, response)
                row["response"] = providers.text(model, response)
                providers.validate(model, response)
                row["status"] = "ok"
            except urllib.error.HTTPError as error:
                row["error"] = f"HTTP {error.code}"
            except Exception as error:
                row["error"] = type(error).__name__
            row["elapsed_seconds"] = time.monotonic() - started
            save(folder / (prompt["sample_id"] + ".result.json"), row)
            print(model, prompt["sample_id"], row["status"], flush=True)
            if row["status"] != "ok":
                break
