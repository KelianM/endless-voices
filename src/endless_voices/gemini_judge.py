"""Run resumable, isolated Gemini assessments with immutable per-trial records."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from endless_voices.assessment import file_hash, read_json, write_json
from endless_voices.judge import judge_messages, parse_answer, save_progress

API = "https://generativelanguage.googleapis.com/v1beta/models/"
MODELS = ("gemini-3.8-flash", "gemini-3.1-flash-lite")
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
    """Read only the key assignment without executing the dotenv file."""
    for line in path.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'")
            if key:
                return key
    raise ValueError("GEMINI_API_KEY is empty or missing")


def payload(trial, instructions):
    messages = judge_messages(trial, instructions)
    return {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }


def assessment(response):
    """Validate one completed text candidate without accepting partial or blocked output."""
    candidates = response.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Expected one completed STOP candidate")
    parts = candidates[0].get("content", {}).get("parts", [])
    raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return {"raw_output": raw, **parse_answer(raw)}


def run(args):
    key = load_key(args.env_file)
    instructions = (args.public / "instructions.txt").read_text()
    paths = [
        p
        for stage in ("primary", "reversed", "controls")
        for p in sorted((args.public / stage).glob("*.json"))
    ]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Expected unique public trials")
    requests = {p.stem: payload(read_json(p), instructions) for p in paths}
    metadata = {
        "model": args.model,
        "endpoint": API + args.model + ":generateContent",
        "generation_config": next(iter(requests.values()))["generationConfig"],
        "thinking": "provider default; no override",
        "isolation": "one stateless request per trial",
        "instructions": instructions,
        "code_sha256": file_hash(Path(__file__)),
        "judge_code_sha256": file_hash(Path(__file__).with_name("judge.py")),
        "trial_hashes": {p.stem: file_hash(p) for p in paths},
        "model_revision": "recorded per response as modelVersion when supplied",
        "timeout_seconds": 120,
        "interval_seconds": 15,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "settings.json"
    if config.exists():
        if read_json(config) != metadata:
            raise ValueError("Resume settings or inputs changed")
    else:
        write_json(config, metadata)
    rows = []
    state = args.output / "state.json"
    previous = read_json(state) if state.exists() else {}
    waiting = time.time() < previous.get("resume_after", 0)
    attempted = 0
    for path in paths:
        tid = path.stem
        request_path = args.output / f"{tid}.request.json"
        result_path = args.output / f"{tid}.result.json"
        if result_path.exists():
            rows.append(read_json(result_path))
            continue
        if waiting:
            continue
        row = {
            "trial_id": tid,
            "trial_sha256": metadata["trial_hashes"][tid],
            "status": "failed",
            "choice": None,
        }
        if request_path.exists():
            row["error"] = "Interrupted request; outcome unknown; not retried"
        else:
            write_json(request_path, requests[tid])
            request = urllib.request.Request(
                metadata["endpoint"],
                data=json.dumps(requests[tid]).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
            )
            row["started_at"] = time.time()
            try:
                with urllib.request.urlopen(request, timeout=120) as handle:
                    response = json.load(handle)
                row["api_response"] = response
                row.update(assessment(response), status="ok")
            except urllib.error.HTTPError as error:
                row["error"] = f"HTTP {error.code}"
                row["http_status"] = error.code
                if error.code in {401, 403, 429}:
                    waiting = True
                    save_progress(
                        state, {"resume_after": time.time() + 86400, "reason": row["error"]}
                    )
            except Exception as error:
                row["error"] = type(error).__name__
            row["finished_at"] = time.time()
            attempted += 1
        write_json(result_path, row)
        rows.append(row)
        print(f"{args.model}: {len(rows)}/{len(paths)} {row['status']}", flush=True)
        if attempted >= args.limit:
            waiting = True
        if not waiting:
            time.sleep(15)
    record = {
        "reviewer_id": args.model + "-validation-v1",
        "reviewer_type": "llm",
        "judge_model_and_prompt": metadata,
        "reviews": rows,
    }
    save_progress(args.output / "review.json", record)
    print(f"Saved {len(rows)}/{len(paths)} assessments", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=10, help="Maximum new calls this invocation")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
