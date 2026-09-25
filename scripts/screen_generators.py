"""Generate a four-scene validation screen with preserved inputs and outputs."""

import argparse
import hashlib
import importlib.metadata
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare(root):
    source = Path("outputs/qwen3-4b-validation-cleaned-v1")
    run = read(source / "run.json")
    for name, digest in run["artifacts_sha256"].items():
        assert sha(source / name) == digest
    dataset = run["dataset"]
    manifest = Path(dataset["manifest_path"])
    assert sha(manifest) == dataset["manifest_sha256"]
    rows = {}
    for entry in read(manifest)["files"]["validation"]:
        path = manifest.parent / entry["path"]
        assert sha(path) == entry["sha256"]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["metadata"]["id"]] = row
    prompts = [json.loads(s) for s in (source / "prompts.jsonl").read_text().splitlines()]
    ids = read(source / "sample-ids.json")
    assert [p["sample_id"] for p in prompts] == ids and len(ids) == 4
    from endless_voices.contracts import evaluation_messages

    for prompt in prompts:
        assert prompt["messages"] == evaluation_messages(rows[prompt["sample_id"]])
    save(root / "prompts.json", prompts)
    save(
        root / "selection.json",
        {
            "ids": ids,
            "split": "validation",
            "manifest_sha256": sha(manifest),
            "source_run_sha256": sha(source / "run.json"),
            "selection": (
                "Four existing smoke scenes, one per identity; selected before generation"
            ),
            "identities": {sid: rows[sid]["metadata"]["identity"] for sid in ids},
        },
    )
    save(root / "originals.json", {sid: rows[sid] for sid in ids})
    save(
        root / "qwen4b-baseline.json",
        {
            "source_run": str(source),
            "reused": True,
            "responses": [
                json.loads(s) for s in (source / "responses.jsonl").read_text().splitlines()
            ],
        },
    )


def local(root, config):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    cfg = read(config)
    folder = root / cfg["label"]
    folder.mkdir(exist_ok=False)
    model_path = Path(cfg["path"])
    save(
        folder / "settings.json",
        {
            **cfg,
            "code_sha256": sha(__file__),
            "prompts_sha256": sha(root / "prompts.json"),
            "max_output_tokens": 512,
            "context_ceiling": 4096,
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
            if len(ids) + 512 > 4096:
                raise ValueError("Context overflow; no truncation")
            mx.random.seed(42)
            last = None
            for token in stream_generate(
                model,
                tokenizer,
                prompt=ids,
                max_tokens=512,
                sampler=make_sampler(temp=0),
                prefill_step_size=256,
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
        if row.get("peak_memory_bytes", 0) > 20_000_000_000:
            break


def hosted(root, models=None, budget=5):
    from endless_voices import anthropic_judge, openai_judge

    for model, provider in [
        ("gpt-6-luna", openai_judge),
        ("gpt-6-sol", openai_judge),
        ("claude-sonnet-5", anthropic_judge),
    ]:
        if models is not None and model not in models:
            continue
        folder = root / model
        folder.mkdir(exist_ok=False)
        key = provider.load_key(Path(".env"))
        save(
            folder / "settings.json",
            {
                "model": model,
                "code_sha256": sha(__file__),
                "prompts_sha256": sha(root / "prompts.json"),
                "reasoning": "medium",
                "max_output_tokens_including_thinking": 4096,
                "endpoint": provider.API,
                "budget_usd": budget,
                "rates_per_million_usd": provider.RATES[model],
            },
        )
        for prompt in read(root / "prompts.json"):
            if provider is openai_judge:
                body = {
                    "model": model,
                    "input": prompt["messages"],
                    "reasoning": {"effort": "medium"},
                    "max_output_tokens": 4096,
                    "store": False,
                    "service_tier": "default",
                }
                headers = {"Authorization": "Bearer " + key}
            else:
                body = {
                    "model": model,
                    "system": prompt["messages"][0]["content"],
                    "messages": prompt["messages"][1:],
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "medium"},
                    "max_tokens": 4096,
                }
                headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            reserved = 0.0
            for path in root.glob("*/*.request.json"):
                result = path.with_name(path.name.replace(".request.", ".result."))
                cost = read(result).get("estimated_cost_usd") if result.exists() else None
                reserved += read(path)["reservation_usd"] if cost is None else cost
            reserve = ((len(json.dumps(body).encode()) + 2048) * 2 + 4096 * 10) / 1e6
            if reserved + reserve > budget:
                raise ValueError("Hosted screen budget exhausted")
            save(
                folder / (prompt["sample_id"] + ".request.json"),
                {"body": body, "reservation_usd": reserve},
            )
            row = {"sample_id": prompt["sample_id"], "status": "failed", "response": ""}
            started = time.monotonic()
            try:
                request = urllib.request.Request(
                    provider.API,
                    data=json.dumps(body).encode(),
                    headers={**headers, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=180) as h:
                    response = json.load(h)
                row["api_response"] = response
                row["estimated_cost_usd"] = provider.charge(model, response)
                if provider is openai_judge:
                    row["response"] = "".join(
                        p.get("text", "")
                        for item in response.get("output", [])
                        if item.get("type") == "message"
                        for p in item.get("content", [])
                        if p.get("type") == "output_text"
                    )
                    complete = response.get("status") == "completed"
                else:
                    row["response"] = "".join(
                        p.get("text", "")
                        for p in response.get("content", [])
                        if p.get("type") == "text"
                    )
                    complete = response.get("stop_reason") == "end_turn"
                if not complete or not row["response"]:
                    raise ValueError("Incomplete or refused response")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "local", "hosted"])
    p.add_argument("--root", type=Path, default=Path("outputs/generator-screen-v1"))
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.root)
    elif a.mode == "local":
        local(a.root, a.config)
    else:
        hosted(a.root)
