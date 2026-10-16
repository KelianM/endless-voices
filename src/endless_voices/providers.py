"""Build native provider requests and account for hosted model usage."""

import json
import urllib.parse
import urllib.request
from copy import deepcopy

RATES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.0, 0.20, 10.0),
    "claude-sonnet-5-5": (2.0, 0.20, 10.0),
}


def is_anthropic(model):
    return model.startswith("claude-")


def endpoint(model):
    return (
        "https://api.anthropic.com/v1/messages"
        if is_anthropic(model)
        else "https://api.openai.com/v1/responses"
    )


def load_key(path, model):
    name = "ANTHROPIC_API_KEY" if is_anthropic(model) else "OPENAI_API_KEY"
    for line in path.read_text().splitlines():
        if line.startswith(name + "="):
            key = line.split("=", 1)[1].strip().strip("\"'")
            if key:
                return key
    raise ValueError(name + " is empty or missing")


def payload(model, messages, schema=None, *, tools=None):
    if is_anthropic(model):
        body = {
            "model": model,
            "system": messages[0]["content"],
            "messages": messages[1:],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
            "max_tokens": 4096,
        }
        if schema:
            body["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    else:
        body = {
            "model": model,
            "input": messages,
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 4096,
            "store": False,
            "service_tier": "default",
        }
        if schema:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "authenticity_judgment",
                    "strict": True,
                    "schema": schema,
                }
            }
    if tools is not None:
        functions = [tool["function"] for tool in tools]
        body["tools"] = ([{"name": f["name"], "description": f["description"],
                          "input_schema": f["parameters"]} for f in functions]
                         if is_anthropic(model) else
                         [{"type": "function", **f, "strict": False} for f in functions])
        if not is_anthropic(model):
            body["include"] = ["reasoning.encrypted_content"]
    if model not in RATES:
        body.pop("thinking", None)
        body.pop("reasoning", None)
        if "output_config" in body:
            body["output_config"].pop("effort", None)
            if not body["output_config"]:
                body.pop("output_config")
    return body


def list_models(provider, key):
    """Return the account's hosted text-model IDs, following Anthropic pagination."""
    anthropic = provider == "anthropic"
    base = "https://api.anthropic.com/v1/models" if anthropic else "https://api.openai.com/v1/models"
    headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"} if anthropic
               else {"Authorization": "Bearer " + key})
    url, models = base, []
    while True:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            page = json.load(r)
        models.extend(row["id"] for row in page["data"])
        if not anthropic or not page.get("has_more"):
            break
        url = base + "?after_id=" + urllib.parse.quote(page["last_id"], safe="")
    excluded = ("audio", "realtime", "transcribe", "tts", "image", "search",
                "deep-research", "instruct", "live")
    return sorted(m for m in models if m.startswith(("claude-", "gpt-", "o1", "o3", "o4"))
                  and not any(kind in m for kind in excluded))


def tool_calls(model, response):
    """Return native tool call IDs, names and unparsed arguments."""
    if is_anthropic(model):
        return [{"id": p["id"], "name": p["name"], "arguments": p["input"]}
                for p in response.get("content", []) if p.get("type") == "tool_use"]
    return [{"id": p["call_id"], "name": p["name"], "arguments": p["arguments"]}
            for p in response.get("output", []) if p.get("type") == "function_call"]


def with_tool_results(body, response, calls, results):
    """Continue the native conversation, preserving assistant and reasoning blocks."""
    body = deepcopy(body)
    if is_anthropic(body["model"]):
        body["messages"].extend([
            {"role": "assistant", "content": response["content"]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call["id"],
                 "content": json.dumps(result), "is_error": "error" in result}
                for call, result in zip(calls, results, strict=True)]},
        ])
    else:
        body["input"].extend(response["output"])
        body["input"].extend(
            {"type": "function_call_output", "call_id": call["id"],
             "output": json.dumps(result)}
            for call, result in zip(calls, results, strict=True))
    return body


def send(body, key, timeout_seconds):
    headers = (
        {"x-api-key": key, "anthropic-version": "2023-06-01"}
        if is_anthropic(body["model"])
        else {"Authorization": "Bearer " + key}
    )
    request = urllib.request.Request(
        endpoint(body["model"]),
        data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
        return json.load(handle)


def text(model, response):
    if is_anthropic(model):
        parts = response.get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return "".join(
        p.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for p in item.get("content", [])
        if p.get("type") == "output_text"
    )


def validate(model, response, *, allow_tools=False):
    calls = tool_calls(model, response) if allow_tools else []
    if is_anthropic(model):
        if response.get("stop_reason") != ("tool_use" if calls else "end_turn"):
            raise ValueError("Response did not end normally")
        if any(
            p.get("type") not in ({"text", "thinking", "redacted_thinking"}
                                  | ({"tool_use"} if allow_tools else set()))
            for p in response.get("content", [])
        ):
            raise ValueError("Unexpected content block")
    else:
        if response.get("status") != "completed":
            raise ValueError("Response was not completed")
        messages = [item for item in response.get("output", []) if item.get("type") == "message"]
        if (not allow_tools and len(messages) != 1
                or any(m.get("status") != "completed" for m in messages)):
            raise ValueError("Expected one completed message")
        if any(p.get("type") != "output_text"
               for m in messages for p in m.get("content", [])):
            raise ValueError("Expected text without refusal")
    if not calls and not text(model, response):
        raise ValueError("Empty response")


def reservation(request):
    # UTF-8 bytes plus overhead conservatively bound text and schema input tokens.
    incoming, _, outgoing = RATES[request["model"]]
    cap = request["max_tokens"] if is_anthropic(request["model"]) else request["max_output_tokens"]
    return ((len(json.dumps(request).encode()) + 2048) * incoming + cap * outgoing) / 1_000_000


def charge(model, response):
    usage = response.get("usage")
    if model not in RATES or not usage or not all(
            k in usage for k in ("input_tokens", "output_tokens")):
        return None
    incoming, cache, outgoing = RATES[model]
    if is_anthropic(model):
        input_cost = (
            usage["input_tokens"] * incoming
            + usage.get("cache_read_input_tokens", 0) * cache
            + usage.get("cache_creation_input_tokens", 0) * 4.0
        )
    else:
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        input_cost = (usage["input_tokens"] - cached) * incoming + cached * cache
    return (input_cost + usage["output_tokens"] * outgoing) / 1_000_000
