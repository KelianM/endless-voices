"""Assess validation pairs with Luna then Sol under a shared dollar budget."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from endless_voices.assessment import file_hash, read_json, write_json
from endless_voices.judge import judge_messages, parse_answer, save_progress

API = "https://api.openai.com/v1/responses"
RATES = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-6-sol": (2.0, 0.20, 10.0)}
SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "abstain"]},
        "confidence": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
        "reason": {"type": "string"},
        "recognized_source": {"type": "boolean"},
    },
    "required": ["choice", "confidence", "reason", "recognized_source"],
    "additionalProperties": False,
}


def load_key(path):
    for line in path.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'")
            if key:
                return key
    raise ValueError("OPENAI_API_KEY is empty or missing")


def payload(model, trial, instructions):
    return {
        "model": model,
        "input": judge_messages(trial, instructions),
        "reasoning": {"effort": "medium"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "authenticity_judgment",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        "max_output_tokens": 4096,
        "store": False,
        "service_tier": "default",
    }


def assessment(response):
    if response.get("status") != "completed":
        raise ValueError("Response was not completed")
    messages = [item for item in response.get("output", []) if item.get("type") == "message"]
    if len(messages) != 1 or messages[0].get("status") != "completed":
        raise ValueError("Expected one completed message")
    parts = messages[0].get("content", [])
    if not parts or any(p.get("type") != "output_text" for p in parts):
        raise ValueError("Expected text without refusal")
    raw = "".join(p["text"] for p in parts)
    return {"raw_output": raw, **parse_answer(raw)}


def reservation(request):
    # UTF-8 bytes plus overhead conservatively bound text and schema input tokens.
    input_rate, _, output_rate = RATES[request["model"]]
    return (
        (len(json.dumps(request).encode()) + 2048) * input_rate
        + request["max_output_tokens"] * output_rate
    ) / 1_000_000


def charge(model, response):
    usage = response.get("usage")
    if not usage or not all(k in usage for k in ("input_tokens", "output_tokens")):
        return None
    cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
    incoming, cache, outgoing = RATES[model]
    return (
        (usage["input_tokens"] - cached) * incoming
        + cached * cache
        + usage["output_tokens"] * outgoing
    ) / 1_000_000


def spent(root):
    total = 0.0
    for request_path in root.glob("*/*.request.json"):
        request = read_json(request_path)
        result_path = request_path.with_name(request_path.name.replace(".request.", ".result."))
        cost = read_json(result_path).get("estimated_cost_usd") if result_path.exists() else None
        total += reservation(request) if cost is None else cost
    return total


def run(args):
    models = getattr(args, "models", None) or list(RATES)
    if not models or len(set(models)) != len(models) or set(models) - RATES.keys():
        raise ValueError("Select unique supported judge models")
    key = load_key(args.env_file)
    instructions = (args.public / "instructions.txt").read_text()
    paths = [
        p
        for stage in ("primary", "reversed", "controls")
        for p in sorted((args.public / stage).glob("*.json"))
    ]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Expected unique public trials")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "endpoint": API,
        "models": models,
        "rates_per_million_usd": RATES,
        "budget_usd": args.budget,
        "instructions": instructions,
        "code_sha256": file_hash(Path(__file__)),
        "judge_code_sha256": file_hash(Path(__file__).with_name("judge.py")),
        "trial_hashes": {p.stem: file_hash(p) for p in paths},
        "isolation": "stateless request per trial; no tools or conversation history",
        "model_revision": "returned model recorded per response; aliases may change",
        "timeout_seconds": 180,
    }
    metadata = json.loads(json.dumps(metadata))
    settings = args.output / "settings.json"
    if settings.exists():
        if read_json(settings) != metadata:
            raise ValueError("Resume settings or inputs changed")
    else:
        write_json(settings, metadata)
    attempted = 0
    for model in models:
        folder = args.output / model
        folder.mkdir(exist_ok=True)
        rows = []
        consecutive_errors = 0
        for path in paths:
            tid = path.stem
            body = payload(model, read_json(path), instructions)
            request_path = folder / f"{tid}.request.json"
            result_path = folder / f"{tid}.result.json"
            if request_path.exists() and read_json(request_path) != body:
                raise ValueError("Saved request differs from expected request")
            if result_path.exists():
                row = read_json(result_path)
                if (
                    not request_path.exists()
                    or row["trial_id"] != tid
                    or row["trial_sha256"] != file_hash(path)
                    or row["request_sha256"] != file_hash(request_path)
                ):
                    raise ValueError("Saved result provenance differs")
                rows.append(row)
                continue
            if attempted >= args.limit:
                return
            row = {
                "trial_id": tid,
                "trial_sha256": file_hash(path),
                "status": "failed",
                "choice": None,
            }
            stop = None
            if request_path.exists():
                row["error"] = "Interrupted request; outcome unknown; not retried"
            else:
                if spent(args.output) + reservation(body) > args.budget:
                    save_progress(args.output / "state.json", {"stop": "budget limit"})
                    return
                write_json(request_path, body)
                request = urllib.request.Request(
                    API,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
                )
                row["started_at"] = time.time()
                try:
                    with urllib.request.urlopen(request, timeout=180) as handle:
                        response = json.load(handle)
                    row["api_response"] = response
                    row["estimated_cost_usd"] = charge(model, response)
                    row.update(assessment(response), status="ok")
                except urllib.error.HTTPError as error:
                    row["error"] = f"HTTP {error.code}"
                    row["http_status"] = error.code
                    if error.code in {400, 401, 403, 404, 429}:
                        stop = row["error"]
                except Exception as error:
                    row["error"] = type(error).__name__
                row["finished_at"] = time.time()
                attempted += 1
            row["request_sha256"] = file_hash(request_path)
            write_json(result_path, row)
            rows.append(row)
            consecutive_errors = consecutive_errors + 1 if row["status"] == "failed" else 0
            save_progress(
                folder / "review.json",
                {
                    "reviewer_id": model + "-medium-validation-v1",
                    "reviewer_type": "llm",
                    "judge_model_and_prompt": {
                        **metadata,
                        "model": model,
                        "request_settings": body,
                    },
                    "reviews": rows,
                },
            )
            print(f"{model}: {len(rows)}/{len(paths)} {row['status']}", flush=True)
            if stop or consecutive_errors >= 3:
                save_progress(
                    args.output / "state.json",
                    {
                        "stop": stop or "three consecutive failures",
                        "model": model,
                    },
                )
                return
            time.sleep(2)
    save_progress(
        args.output / "state.json",
        {"complete": True, "estimated_or_reserved_usd": spent(args.output)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--models", nargs="+", choices=list(RATES))
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=196)
    args = parser.parse_args()
    if not 0 < args.budget <= 5 or args.limit < 1:
        parser.error("budget must be positive and at most $5; limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
