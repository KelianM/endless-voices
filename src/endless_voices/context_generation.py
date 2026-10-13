"""Select static context or retrieve evidence during generation."""

import json
import re

from endless_voices.context import render
from endless_voices.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    SceneDatabase,
    TransformersEmbedder,
    passage_chunks,
    scene_id,
)


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


def select_static_context(session, settings, counter):
    """Return static context and its selected scene IDs."""
    messages = session.static_messages(
        counter, settings["max_input_tokens"], settings.get("depth", 4))
    return messages, {"tools": session.trace,
                      "selected_scene_ids": session.trace[-1]["result_ids"]}


def dynamic_messages(messages, history_count):
    """Return the encounter and shared instructions for model-directed retrieval."""
    messages[0]["content"] += (
        "\n\nRetrieve context as needed while composing the continuation. "
        + (f"{history_count} eligible history scenes are available. "
           if history_count else "No eligible mission history is available at any depth. ")
        + "Use the mission history map and read_context IDs for continuity. "
        "Search accepts your own queries and excludes scenes already retrieved. "
        "Writing references are independent examples, not events in this playthrough. "
        "Earlier scenes have already happened; write at the current scene and location. "
        "After reading tool results, retrieve more if needed. "
        "Your final response should contain only the finished continuation."
    )
    return messages


def tool_loop(session, conversation, ask, continue_with, fits, trace, *, selected_ids=()):
    """Retrieve and generate until a complete response contains no tool calls."""
    selected = set(selected_ids)
    trace.update(turns=[], selected_scene_ids=sorted(selected))
    while True:
        if not fits(conversation):
            raise ValueError("Native conversation exceeds the dynamic input budget")
        response, calls = ask(conversation, len(trace["turns"]))
        if not calls:
            return conversation, response
        results = [{"error": "Tool result exceeds the remaining input budget"} for _ in calls]
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
                    if not fits(continue_with(conversation, response, calls, results)):
                        result["documents"].pop()
                        result["omitted_documents"] += 1
                selected.update(d["scene_id"] for d in result["documents"])
            except (ValueError, TypeError, KeyError) as error:
                results[index] = {"error": str(error)}
        trace["turns"].append({"calls": calls, "results": results})
        trace["selected_scene_ids"] = sorted(selected)
        conversation = continue_with(conversation, response, calls, results)


def local_tool_response(tokenizer, text):
    """Parse native local tool calls, retaining assistant text between calls."""
    if getattr(tokenizer, "has_tool_calling", False):
        start, end = tokenizer.tool_call_start, tokenizer.tool_call_end
        parse = tokenizer.tool_parser
    elif "<tool_call>" in (tokenizer.chat_template or ""):
        start, end, parse = "<tool_call>", "</tool_call>", json.loads
    else:
        raise ValueError("Local tokenizer has no supported native tool format")
    think_start = getattr(tokenizer, "think_start", None) or "<think>"
    think_end = getattr(tokenizer, "think_end", None) or "</think>"
    thought_pattern = re.escape(think_start) + r"(.*?)" + re.escape(think_end)
    thoughts = re.findall(thought_pattern, text, flags=re.DOTALL)
    if text.count(think_start) != len(thoughts):
        raise ValueError("Incomplete native reasoning block")
    text = re.sub(thought_pattern, "", text, flags=re.DOTALL)
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    segments = re.findall(pattern, text, flags=re.DOTALL)
    if text.count(start) != len(segments):
        raise ValueError("Incomplete native tool call")
    calls = []
    for segment in segments:
        parsed = parse(segment)
        calls.extend(parsed if isinstance(parsed, list) else [parsed])
    content = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    response = {"role": "assistant", "content": content}
    if thoughts:
        response["reasoning_content"] = "\n".join(thoughts).strip()
    return response, calls


def local_tool_results(messages, response, calls, results):
    """Append assistant calls and named tool results for the model's chat template."""
    ids = [f"call-{len(messages)}-{i}" for i in range(len(calls))]
    return [*messages, {**response, "tool_calls": [
        {"id": id_, "type": "function", "function": call}
        for id_, call in zip(ids, calls, strict=True)]}, *[
        {"role": "tool", "tool_call_id": id_, "name": call["name"],
         "content": json.dumps(result)}
        for id_, call, result in zip(ids, calls, results, strict=True)]]


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


def verify_tool_context(pool, messages, selected_ids):
    """Check native tool evidence against the encounter's eligible scenes and passages."""
    initial = dynamic_messages(pool.messages(), len(pool.blocks))
    if messages[:len(initial)] != initial:
        raise ValueError("Native encounter differs from prepared context")
    allowed = {}
    for kind, blocks in (("history", pool.blocks), ("reference", pool.references)):
        for block in blocks:
            sid = scene_id(block)
            allowed[sid] = (sid, kind, render(block))
            for passage in passage_chunks(block):
                allowed[passage["id"]] = (sid, kind, passage["text"])
    selected = set()
    for message in messages[len(initial):]:
        outputs = ([message["output"]] if message.get("type") == "function_call_output"
                   else [message["content"]] if message.get("role") == "tool"
                   else [part["content"] for part in message.get("content", [])
                         if isinstance(part, dict) and part.get("type") == "tool_result"])
        for output in outputs:
            for document in json.loads(output).get("documents", []):
                expected = allowed.get(document["id"])
                actual = (document["scene_id"], document["kind"],
                          document.get("text", document.get("excerpt")))
                if expected != actual:
                    raise ValueError("Tool result differs from eligible scene evidence")
                selected.add(document["scene_id"])
    if selected != set(selected_ids):
        raise ValueError("Retrieved scene selection differs from native tool results")
