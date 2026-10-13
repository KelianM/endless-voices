"""Select eligible context before generating a continuation."""

import json

from endless_voices.retrieval import DEFAULT_EMBEDDING_MODEL, SceneDatabase, TransformersEmbedder


def open_database(settings):
    """Open a scene index with its cached query encoder on CPU."""
    encoder = TransformersEmbedder.from_pretrained(
        settings.get("model", DEFAULT_EMBEDDING_MODEL),
        device="cpu",
        dtype=settings.get("dtype", "float16"),
        max_length=settings.get("embedding_max_length", 32768),
        cache_dir=settings.get("cache_dir", "data/local/hub"),
    )
    return SceneDatabase(settings["index"], encoder)


def select_context(session, settings, counter, ask=None):
    """Return final context messages and a trace of model-directed retrieval."""
    limit = settings["max_input_tokens"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_input_tokens must be positive")
    if settings["mode"] == "static":
        messages = session.static_messages(counter, limit, settings.get("depth", 4))
        return messages, {
            "tools": session.trace,
            "selected_scene_ids": session.trace[-1]["result_ids"],
        }
    if settings["mode"] != "dynamic" or ask is None:
        raise ValueError("Dynamic context requires a model callback")
    instruction = (
        f"Select context for mission {session.mission!r}. Do not write the passage yet. "
        "This call is only for context selection. Reply only with JSON: "
        '{"name":"search_context","arguments":{...}} to call a tool, '
        'or {"name":"finish","arguments":{}} when ready. '
        "Retrieved scenes that fit are included automatically. "
        "Use mission history for continuity, especially before writing an opening. "
        "Search writing references only for missing facts or voice examples. "
        "Finish once the selected context supports the next passage; exhaustive coverage "
        "is unnecessary. If a search adds no useful evidence, finish unless a specific "
        "unanswered question needs a different tool or query. Available tools: "
        + json.dumps(session.tool_schemas)
    )
    session.trace.clear()
    selected, turns, progress = [], [], []

    def selector(ids, feedback):
        messages = session.messages_for(ids)
        messages[0] = {**messages[0], "content": messages[0]["content"] + "\n\n" + instruction}
        messages[-1] = {
            **messages[-1],
            "content": messages[-1]["content"]
            + "\n\nRetrieval so far: "
            + json.dumps([*progress, feedback] if feedback is not None else progress)
            + '\nDo not write the passage. Reply only with a JSON tool call, or '
            + '{"name":"finish","arguments":{}} if the context is sufficient.',
        }
        return messages

    while True:
        messages = selector(selected, None)
        if counter.messages(messages) > limit:
            raise ValueError("Retrieval instructions and encounter exceed the input budget")
        response = ask(messages, len(turns))
        entry = {"messages": messages, "response": response}
        turns.append(entry)
        try:
            call = json.loads(response)
            if set(call) != {"name", "arguments"}:
                raise ValueError("Expected name and arguments")
            if call["name"] == "finish":
                if call["arguments"] != {}:
                    raise ValueError("finish takes no arguments")
                break
            documents = session.call_tool(call["name"], call["arguments"])
            accepted, omitted = [], 0
            before = set(selected)

            def feedback():
                return {**call, "added_scenes": len(set(accepted) - before),
                        "already_selected_scenes": len(set(accepted) & before),
                        "omitted_scenes": omitted}
            for document in documents:
                sid = document["scene_id"]
                candidate = selected if sid in selected else [*selected, sid]
                accepted.append(sid)
                if counter.messages(selector(candidate, feedback())) <= limit:
                    selected = candidate
                else:
                    accepted.pop()
                    omitted += 1
            while counter.messages(selector(selected, feedback())) > limit and selected:
                removed = selected.pop()
                accepted = [sid for sid in accepted if sid != removed]
                omitted += 1
            entry["result"] = feedback()
            progress.append(entry["result"])
        except (ValueError, TypeError, KeyError) as error:
            entry["result"] = {"response": response, "error": str(error)}
            progress.append(entry["result"])
    return session.messages_for(selected), {
        "turns": turns,
        "tools": session.trace,
        "selected_scene_ids": selected,
    }


def benchmark_sessions(root, settings):
    """Open a benchmark index and locate encounters without loading their vectors."""
    from pathlib import Path

    from endless_voices.artifacts import file_hash, read_json
    from endless_voices.dataset.storage import SceneDataset

    selection = read_json(root / "selection.json")
    source = Path(selection["dataset_source"])
    if file_hash(source / "manifest.json") != selection["manifest_sha256"]:
        raise ValueError("Prepared dataset changed since benchmark preparation")
    if file_hash(settings["index"]) != selection["index_sha256"]:
        raise ValueError("Scene index changed since benchmark preparation")
    dataset = SceneDataset.load(source, selection["split"])
    indices = {
        record["metadata"]["id"]: i
        for i, record in enumerate(dataset)
        if record["metadata"]["id"] in selection["ids"]
    }
    if set(indices) != set(selection["ids"]):
        raise ValueError("Selected encounters are missing from prepared dataset")
    return open_database(settings), dataset, indices
