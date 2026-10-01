"""Build native provider requests and account for hosted model usage."""

import json
import urllib.request

RATES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.0, 0.20, 10.0),
    "claude-sonnet-5-5": (2.0, 0.20, 10.0),
}


def is_anthropic(model):
    if model not in RATES:
        raise ValueError("Unsupported hosted model")
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


def payload(model, messages, schema=None):
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


def validate(model, response):
    if is_anthropic(model):
        if response.get("stop_reason") != "end_turn":
            raise ValueError("Response did not end normally")
        if any(
            p.get("type") not in {"text", "thinking", "redacted_thinking"}
            for p in response.get("content", [])
        ):
            raise ValueError("Unexpected content block")
    else:
        if response.get("status") != "completed":
            raise ValueError("Response was not completed")
        messages = [item for item in response.get("output", []) if item.get("type") == "message"]
        if len(messages) != 1 or messages[0].get("status") != "completed":
            raise ValueError("Expected one completed message")
        if any(p.get("type") != "output_text" for p in messages[0].get("content", [])):
            raise ValueError("Expected text without refusal")
    if not text(model, response):
        raise ValueError("Empty response")


def reservation(request):
    # UTF-8 bytes plus overhead conservatively bound text and schema input tokens.
    incoming, _, outgoing = RATES[request["model"]]
    cap = request["max_tokens"] if is_anthropic(request["model"]) else request["max_output_tokens"]
    return ((len(json.dumps(request).encode()) + 2048) * incoming + cap * outgoing) / 1_000_000


def charge(model, response):
    usage = response.get("usage")
    if not usage or not all(k in usage for k in ("input_tokens", "output_tokens")):
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
