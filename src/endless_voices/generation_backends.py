"""Run local or hosted generation from saved benchmark prompts."""

import importlib.metadata
import json
import time
import urllib.error
from pathlib import Path

from endless_voices import artifacts, context_generation
from endless_voices.artifacts import file_hash as sha
from endless_voices.artifacts import read_json as read
from endless_voices.artifacts import write_json as save
from endless_voices.context_generation import (
    benchmark_sessions,
    dynamic_messages,
    local_tool_response,
    local_tool_results,
    tool_loop,
)


def quantize_full_history(caches, full_cache_type, *, bits=8, group_size=64):
    """Convert full-history caches while preserving all other cache objects."""
    return [
        c.to_quantized(bits=bits, group_size=group_size) if type(c) is full_cache_type else c
        for c in caches
    ]


def local_cache(model, cfg, row):
    """Build the configured KV cache for a single model request."""
    cache_options = {
        "kv_bits": cfg.get("kv_bits"),
        "kv_group_size": cfg.get("kv_group_size", 64),
        "quantized_kv_start": cfg.get("quantized_kv_start", 0),
    }
    if cfg.get("kv_scope") == "full_history":
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        caches = quantize_full_history(
            make_prompt_cache(model),
            KVCache,
            bits=cfg["kv_bits"],
            group_size=cfg.get("kv_group_size", 64),
        )
        cache_options = {"prompt_cache": caches}
        row["cache_types"] = [type(c).__name__ for c in caches]
    return cache_options


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
            "artifact_code_sha256": sha(artifacts.__file__),
            "context_code_sha256": sha(context_generation.__file__),
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
    settings = read(root / "selection.json")["context"]
    database, prepared, indices = (
        benchmark_sessions(root, settings)
        if settings.get("mode") == "dynamic"
        else (None, None, {})
    )
    try:
        settings = {
            **settings,
            "max_input_tokens": min(
                cfg.get("max_input_tokens", context_limit - output_limit),
                context_limit - output_limit,
            ),
        }

        for prompt in read(root / "prompts.json"):
            started = time.monotonic()
            row = {"sample_id": prompt["sample_id"], "status": "failed", "response": ""}
            session = (prepared.retrieval_session(indices[prompt["sample_id"]], database)
                       if indices else None)
            tools = session.tool_schemas if session else None
            trace = {"tools": session.trace} if session else None
            actual = {}

            def render(messages):
                rendered = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                return {"messages": messages, "tools": tools, "rendered": rendered,
                        "input_ids": tokenizer.encode(rendered, add_special_tokens=False)}

            def ask(messages, turn):
                nonlocal actual
                actual = render(messages)
                ids = actual["input_ids"]
                if len(ids) + output_limit > context_limit:
                    raise ValueError("Context overflow; no truncation")
                request_folder = folder / (prompt["sample_id"] + ".retrieval")
                if session:
                    request_folder.mkdir(exist_ok=True)
                    save(request_folder / f"{turn}.prompt.json", actual)
                response, last = "", None
                try:
                    for token in stream_generate(
                        model, tokenizer, prompt=ids, max_tokens=output_limit,
                        sampler=make_sampler(temp=0), prefill_step_size=prefill_step_size,
                        **local_cache(model, cfg, row),
                    ):
                        response += token.text
                        last = token
                        if mx.get_peak_memory() > 20_000_000_000:
                            raise MemoryError("20 GB experiment ceiling exceeded")
                        if time.monotonic() - started > 600:
                            raise TimeoutError("600 second trial ceiling exceeded")
                finally:
                    row.update(response=response,
                               finish_reason=last.finish_reason if last else None,
                               output_tokens=last.generation_tokens if last else 0)
                    if session:
                        save(request_folder / f"{turn}.result.json", {
                            "response": response, "finish_reason": row["finish_reason"],
                            "output_tokens": row["output_tokens"],
                        })
                if last is None or last.finish_reason != "stop":
                    raise ValueError("Output limit or incomplete response")
                return (local_tool_response(tokenizer, response) if session
                        else ({"content": response}, []))

            try:
                mx.random.seed(42)
                if session:
                    local_tool_response(tokenizer, "")
                    _, response = tool_loop(
                        session,
                        dynamic_messages(session.initial_messages(), session.history_count),
                        ask, local_tool_results,
                        lambda messages: len(render(messages)["input_ids"])
                        <= settings["max_input_tokens"], trace,
                    )
                else:
                    response, _ = ask(prompt["messages"], 0)
                row["response"] = response["content"]
                if not row["response"].strip():
                    raise ValueError("Empty continuation")
                row["status"] = "ok"
            except Exception as error:
                row["error"] = str(error)
            if actual and (session is None or row["status"] == "ok"):
                save(folder / (prompt["sample_id"] + ".prompt.json"), actual)
            if session:
                save(folder / (prompt["sample_id"] + ".retrieval.json"), trace)
            row.update(
                elapsed_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory()
            )
            save(folder / (prompt["sample_id"] + ".result.json"), row)
            print(cfg["label"], prompt["sample_id"], row["status"], flush=True)
            if row.get("peak_memory_bytes", 0) > 20_000_000_000 or "Insufficient Memory" in row.get(
                "error", ""
            ):
                break
    finally:
        if database is not None:
            database.close()


def hosted(root, models=None, budget=5, timeout_seconds=1800):
    from endless_voices import providers

    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")

    settings = read(root / "selection.json")["context"]
    selected_models = [model for model in providers.RATES if models is None or model in models]
    if settings["mode"] == "dynamic":
        for model in selected_models:
            cap = settings["dynamic_max_input_tokens"].get(model)
            if type(cap) is not int or cap <= 0:
                raise ValueError(f"Positive dynamic input budget required for {model}")
    database, prepared, indices = (
        benchmark_sessions(root, settings) if settings["mode"] == "dynamic" else (None, None, {})
    )
    try:
        if indices:
            from transformers import AutoTokenizer

            from endless_voices.context import TokenizerCounter

            counter = TokenizerCounter(
                AutoTokenizer.from_pretrained(
                    settings["tokenizer"],
                    local_files_only=True,
                    **settings.get("tokenizer_options", {}),
                )
            )
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
                    "artifact_code_sha256": sha(artifacts.__file__),
                    "context_code_sha256": sha(context_generation.__file__),
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
                sample_started = time.monotonic()
                if indices:
                    row = hosted_dialogue(
                        root, folder / prompt["sample_id"], model,
                        prepared.retrieval_session(indices[prompt["sample_id"]], database),
                        settings["dynamic_max_input_tokens"][model], counter,
                        key, budget, timeout_seconds,
                    )
                    print(model, prompt["sample_id"], row["status"], flush=True)
                    if row["status"] != "ok":
                        break
                    continue
                save(
                    folder / (prompt["sample_id"] + ".prompt.json"),
                    {"messages": prompt["messages"]},
                )
                row = hosted_request(
                    root,
                    folder / prompt["sample_id"],
                    model,
                    prompt["messages"],
                    key,
                    budget,
                    timeout_seconds,
                    started=sample_started,
                )
                print(model, prompt["sample_id"], row["status"], flush=True)
                if row["status"] != "ok":
                    break
    finally:
        if database is not None:
            database.close()


def hosted_dialogue(root, path, model, session, limit, counter, key, budget, timeout_seconds):
    """Generate with native retrieval tools and save the complete provider conversation."""
    from endless_voices import providers

    started = time.monotonic()
    folder = path.parent / (path.name + ".retrieval")
    folder.mkdir()
    body = providers.payload(
        model, dynamic_messages(session.initial_messages(), session.history_count),
        tools=session.tool_schemas,
    )
    rows, trace = [], {"tools": session.trace}

    def fits(candidate):
        # Include schemas, tool results and reasoning in the reference-tokenizer budget.
        return counter.messages([{"role": "system", "content": json.dumps(candidate)}]) <= limit

    def ask(body, turn):
        row = hosted_request(root, folder / str(turn), model, None, key,
                             budget, timeout_seconds, body=body)
        rows.append(row)
        if row["status"] != "ok":
            raise ValueError(row.get("error", "Hosted request failed"))
        response = row["api_response"]
        return response, providers.tool_calls(model, response)

    try:
        body, _ = tool_loop(session, body, ask, providers.with_tool_results, fits, trace)
        row = rows[-1]
    except Exception as error:
        row = {"status": "failed", "response": "", "error": str(error)}
    conversation = body.get("input") or [{"role": "system", "content": body["system"]},
                                         *body["messages"]]
    if row["status"] == "ok":
        save(path.parent / (path.name + ".prompt.json"), {"messages": conversation})
    save(path.parent / (path.name + ".retrieval.json"), trace)
    row = {**row, "sample_id": path.name, "elapsed_seconds": time.monotonic() - started,
           **request_totals(rows), "request_count": len(rows)}
    save(path.parent / (path.name + ".result.json"), row)
    return row


def hosted_request(root, path, model, messages, key, budget, timeout_seconds, *,
                   started=None, body=None):
    """Record one paid generation request and reserve its cost before sending."""
    from endless_voices import providers

    body = providers.payload(model, messages) if body is None else body
    reserved = 0.0
    for request in root.rglob("*.request.json"):
        result = request.with_name(request.name.replace(".request.", ".result."))
        cost = read(result).get("estimated_cost_usd") if result.exists() else None
        reserved += read(request)["reservation_usd"] if cost is None else cost
    reserve = providers.reservation(body)
    if reserved + reserve > budget:
        raise ValueError("Hosted screen budget exhausted")
    save(path.parent / (path.name + ".request.json"), {"body": body, "reservation_usd": reserve})
    row = {"sample_id": path.name, "status": "failed", "response": ""}
    started = time.monotonic() if started is None else started
    try:
        response = providers.send(body, key, timeout_seconds)
        row.update(
            api_response=response,
            estimated_cost_usd=providers.charge(model, response),
            response=providers.text(model, response),
        )
        providers.validate(model, response, allow_tools=bool(body.get("tools")))
        row["status"] = "ok"
    except urllib.error.HTTPError as error:
        row["error"] = f"HTTP {error.code}"
    except Exception as error:
        row["error"] = type(error).__name__
    row["elapsed_seconds"] = time.monotonic() - started
    previous = [read(p) for p in (path.parent / (path.name + ".retrieval")).glob("*.result.json")]
    row.update(request_totals([*previous, row]))
    save(path.parent / (path.name + ".result.json"), row)
    return row


def request_totals(rows):
    """Sum known request charges and usage without treating missing accounting as zero."""
    costs = [row.get("estimated_cost_usd") for row in rows]
    usages = [row.get("api_response", {}).get("usage", {}) for row in rows]
    return {
        "total_estimated_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
        "total_usage": {
            key: sum(u[key] for u in usages) if all(key in u for u in usages) else None
            for key in ("input_tokens", "output_tokens")
        },
    }
