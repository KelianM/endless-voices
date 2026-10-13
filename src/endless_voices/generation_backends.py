"""Run local or hosted generation from saved benchmark prompts."""

import importlib.metadata
import json
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from endless_voices import artifacts
from endless_voices.artifacts import file_hash as sha
from endless_voices.artifacts import read_json as read
from endless_voices.artifacts import write_json as save
from endless_voices.context_generation import benchmark_sessions, select_context


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

        def count(messages):
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return len(tokenizer.encode(rendered, add_special_tokens=False))

        for prompt in read(root / "prompts.json"):
            started = time.monotonic()
            row = {"sample_id": prompt["sample_id"], "status": "failed", "response": ""}
            if indices:
                mx.random.seed(42)

                def ask(messages, turn):
                    rendered = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    tokens = tokenizer.encode(rendered, add_special_tokens=False)
                    request_folder = folder / (prompt["sample_id"] + ".retrieval")
                    request_folder.mkdir(exist_ok=True)
                    save(
                        request_folder / f"{turn}.prompt.json",
                        {"messages": messages, "rendered": rendered, "input_ids": tokens},
                    )
                    response, last = "", None
                    try:
                        for token in stream_generate(
                            model,
                            tokenizer,
                            prompt=tokens,
                            max_tokens=output_limit,
                            sampler=make_sampler(temp=0),
                            prefill_step_size=prefill_step_size,
                            **local_cache(model, cfg, row),
                        ):
                            response += token.text
                            last = token
                            if mx.get_peak_memory() > 20_000_000_000:
                                raise MemoryError("20 GB experiment ceiling exceeded")
                            if time.monotonic() - started > 600:
                                raise TimeoutError("600 second trial ceiling exceeded")
                    finally:
                        save(
                            request_folder / f"{turn}.result.json",
                            {
                                "response": response,
                                "finish_reason": last.finish_reason if last else None,
                            },
                        )
                    if last is None or last.finish_reason != "stop":
                        raise ValueError("Retrieval output limit or incomplete response")
                    return response

                try:
                    messages, trace = select_context(
                        prepared.retrieval_session(indices[prompt["sample_id"]], database),
                        settings,
                        SimpleNamespace(messages=count),
                        ask,
                    )
                    prompt = {**prompt, "messages": messages}
                    save(folder / (prompt["sample_id"] + ".retrieval.json"), trace)
                except Exception as error:
                    row.update(
                        error=str(error),
                        elapsed_seconds=time.monotonic() - started,
                        peak_memory_bytes=mx.get_peak_memory(),
                    )
                    save(folder / (prompt["sample_id"] + ".result.json"), row)
                    if isinstance(error, MemoryError) or "Insufficient Memory" in str(error):
                        break
                    continue
            rendered = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
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
                cache_options = local_cache(model, cfg, row)
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
                    row.update(
                        output_tokens=last.generation_tokens, finish_reason=last.finish_reason
                    )
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
    messages = session.initial_messages()
    messages[0]["content"] += (
        "\n\nRetrieve context as needed while composing the continuation. "
        + (f"{session.history_count} eligible history scenes are available. "
           if session.history_count else "No eligible mission history is available at any depth. ")
        + "Use the mission history map and read_context IDs for continuity. "
        "Search accepts your own queries and excludes scenes already retrieved. "
        "Writing references are independent examples, not events in this playthrough. "
        "Earlier scenes have already happened; write at the current scene and location. "
        "After reading tool results, retrieve more if needed. "
        "Your final response should contain only the finished continuation."
    )
    body = providers.payload(model, messages, tools=session.tool_schemas)
    rows, turns = [], []
    selected = set()

    def fits(candidate):
        # Include schemas, tool results and reasoning in the reference-tokenizer budget.
        return counter.messages([{"role": "system", "content": json.dumps(candidate)}]) <= limit

    row = {"status": "failed", "response": ""}
    try:
        while True:
            if not fits(body):
                raise ValueError("Native conversation exceeds the dynamic input budget")
            row = hosted_request(root, folder / str(len(rows)), model, None, key,
                                 budget, timeout_seconds, body=body)
            rows.append(row)
            if row["status"] != "ok":
                break
            response = row["api_response"]
            calls = providers.tool_calls(model, response)
            if not calls:
                break
            results = [{"error": "Tool result exceeds the remaining input budget"}
                       for _ in calls]
            for index, call in enumerate(calls):
                try:
                    arguments = call["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    session.selected_scene_ids = set(selected)
                    documents = session.call_tool(call["name"], arguments)
                    result = {"documents": [], "omitted_documents": len(documents)}
                    results[index] = result
                    for document in documents:
                        result["documents"].append(document)
                        result["omitted_documents"] -= 1
                        candidate = providers.with_tool_results(body, response, calls, results)
                        if not fits(candidate):
                            result["documents"].pop()
                            result["omitted_documents"] += 1
                    selected.update(d["scene_id"] for d in result["documents"])
                except (ValueError, TypeError, KeyError) as error:
                    results[index] = {"error": str(error)}
            turns.append({"calls": calls, "results": results})
            body = providers.with_tool_results(body, response, calls, results)
    except Exception as error:
        row = {"status": "failed", "response": "", "error": str(error)}
    conversation = body.get("input") or [{"role": "system", "content": body["system"]},
                                         *body["messages"]]
    if row["status"] == "ok":
        save(path.parent / (path.name + ".prompt.json"), {"messages": conversation})
    save(path.parent / (path.name + ".retrieval.json"), {
        "turns": turns, "tools": session.trace, "selected_scene_ids": sorted(selected),
    })
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
