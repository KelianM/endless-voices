"""Assess isolated validation pairs with selected models under a dollar budget."""

import argparse
import json
import time
import urllib.error
from pathlib import Path

from endless_voices import artifacts, providers
from endless_voices.artifacts import file_hash, read_json, save_progress, write_json
from endless_voices.judge import judge_messages, parse_answer
from endless_voices.providers import RATES, charge, reservation

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "abstain"]},
        "confidence": {
            "anyOf": [
                {"type": "string", "enum": ["low", "medium", "high"]},
                {"type": "null"},
            ]
        },
        "reason": {"type": "string"},
        "recognized_source": {"type": "boolean"},
    },
    "required": ["choice", "confidence", "reason", "recognized_source"],
    "additionalProperties": False,
}


def payload(model, trial, instructions):
    return providers.payload(model, judge_messages(trial, instructions), SCHEMA)


def assessment(model, response):
    providers.validate(model, response)
    raw = providers.text(model, response)
    return {"raw_output": raw, **parse_answer(raw)}


def spent(root):
    total = 0.0
    for request_path in root.glob("*/*.request.json"):
        request = read_json(request_path)
        result_path = request_path.with_name(request_path.name.replace(".request.", ".result."))
        cost = read_json(result_path).get("estimated_cost_usd") if result_path.exists() else None
        total += reservation(request) if cost is None else cost
    return total


def run(args):
    timeout_seconds = getattr(args, "timeout_seconds", 1800)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    models = getattr(args, "models", None) or ["gpt-6-luna"]
    if not models or len(set(models)) != len(models) or set(models) - RATES.keys():
        raise ValueError("Select unique supported judge models")
    instructions = (args.public / "instructions.txt").read_text()
    paths = [
        p
        for stage in ("primary", "controls")
        for p in sorted((args.public / stage).glob("*.json"))
    ]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Expected unique public trials")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "endpoints": {m: providers.endpoint(m) for m in models},
        "provider_code_sha256": file_hash(Path(providers.__file__)),
        "artifact_code_sha256": file_hash(Path(artifacts.__file__)),
        "models": models,
        "rates_per_million_usd": RATES,
        "budget_usd": args.budget,
        "instructions": instructions,
        "code_sha256": file_hash(Path(__file__)),
        "judge_code_sha256": file_hash(Path(__file__).with_name("judge.py")),
        "trial_hashes": {p.stem: file_hash(p) for p in paths},
        "isolation": "stateless request per trial; no tools or conversation history",
        "model_revision": "returned model recorded per response; aliases may change",
        "timeout_seconds": timeout_seconds,
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
        key = providers.load_key(args.env_file, model)
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
                row["started_at"] = time.time()
                try:
                    response = providers.send(body, key, timeout_seconds)
                    row["api_response"] = response
                    row["estimated_cost_usd"] = charge(model, response)
                    row.update(assessment(model, response), status="ok")
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
    parser.add_argument(
        "--models", nargs="+", choices=list(RATES), help="Judge models; defaults to gpt-6-luna only"
    )
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=196)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Network operation timeout in seconds (default: 1800)",
    )
    args = parser.parse_args()
    if not 0 < args.budget <= 5 or args.limit < 1:
        parser.error("budget must be positive and at most $5; limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
