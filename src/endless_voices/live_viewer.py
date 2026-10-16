"""Serve the scene reader and hosted conversations on loopback."""

import json
import os
import secrets
import subprocess
import sys
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from endless_voices import providers
from endless_voices.context_generation import (
    dynamic_messages,
    local_tool_response,
    local_tool_results,
    open_database,
    tool_loop,
)
from endless_voices.generation_backends import hosted_request
from endless_voices.view_samples import viewer_data, viewer_html


class LiveViewer:
    """Keep native provider conversations and eligible retrieval sessions in memory."""

    def __init__(self, data, datasets, config, output, budget):
        self.data, self.config, self.output, self.budget = data, config, output, budget
        self.scenes = {row["metadata"]["id"]: (dataset, i)
                       for dataset in datasets.values() for i, row in enumerate(dataset)}
        self.sessions = {}
        self.database = self.worker = None
        self.local_model = None
        self.local_configs = {name: json.loads(Path(path).read_text())
                              for name, path in config.get("local_configs", {}).items()}
        self.models = {m["id"] for m in data.get("live", {}).get("models", [])}

    def reply(self, request):
        """Continue a server-owned conversation, saving every hosted request."""
        settings = self.config["context"]
        model, sid = request["model"], request["sample_id"]
        local = self.local_configs.get(model)
        if model not in self.models:
            raise ValueError("Model is not available in this viewer")
        limit = (min(local["max_input_tokens"],
                     local["context_ceiling"] - local["max_output_tokens"]) if local else None)
        text = request.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Reply must be text")
        session_id = request.get("session_id")
        state = self.sessions.get(session_id)
        if session_id and (state is None or (state["sample_id"], state["model"]) != (sid, model)):
            raise ValueError("Conversation does not belong to this scene and model")
        key = None if local else providers.load_key(Path(".env"), model)
        if self.database is None:
            self.database = open_database(settings)
        if not local:
            self.stop_worker()
        if local and (self.local_model != model or self.worker.poll() is not None):
            self.stop_worker()
            from transformers import AutoTokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                local["path"], local_files_only=True, **local.get("tokenizer_options", {}))
            self.worker = subprocess.Popen([
                self.config.get("local_python", "data/local/screen-env/bin/python"),
                "-u", "-m", "endless_voices.live_viewer", "--worker",
                self.config["local_configs"][model]], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                     "HF_HUB_OFFLINE": "1"})
            self.local_model = model
        if state is None:
            dataset, index = self.scenes[sid]
            session = dataset.retrieval_session(index, self.database)
            messages = dynamic_messages(session.initial_messages(), session.history_count)
            run_index = request.get("run_index")
            choices = next(row["player_choices"] for row in self.data["samples"]
                           if row["metadata"]["id"] == sid)
            if run_index is None and choices and text.strip():
                messages[-1]["content"] = "\n\n".join(
                    part for part in (choices["prefix"], text.strip()) if part)
                text = ""
            if run_index is not None:
                if type(run_index) is not int or not 0 <= run_index < len(self.data["runs"]):
                    raise ValueError("Unknown saved run")
                saved = self.data["runs"][run_index]["responses"].get(sid)
                if not saved or saved["status"] != "ok" or not saved.get("response"):
                    raise ValueError("This run has no completed response for the scene")
                messages.append({"role": "assistant", "content": saved["response"]})
                if not text.strip():
                    raise ValueError("Enter a reply to continue a saved conversation")
            state = {"sample_id": sid, "model": model, "session": session,
                     "body": messages if local else providers.payload(
                         model, messages, tools=session.tool_schemas)}
            session_id = uuid4().hex
        elif not text.strip():
            raise ValueError("Enter a reply")
        body = deepcopy(state["body"])
        if text.strip():
            messages = body if local else (
                body["messages"] if providers.is_anthropic(model) else body["input"])
            # Scene prefixes can end in a user message; avoid consecutive user turns.
            if messages and messages[-1].get("role") == "user" and isinstance(
                    messages[-1].get("content"), str):
                messages[-1]["content"] += "\n\nPlayer response: " + text.strip()
            else:
                messages.append({"role": "user", "content": text.strip()})
        folder = self.output / uuid4().hex
        folder.mkdir(parents=True)
        rows, trace = [], {}

        def ask(candidate, turn):
            if local:
                self.worker.stdin.write(json.dumps(
                    {"messages": candidate, "tools": state["session"].tool_schemas}) + "\n")
                self.worker.stdin.flush()
                line = self.worker.stdout.readline()
                if not line:
                    raise ValueError("Local model worker stopped; restart the conversation")
                result = json.loads(line)
                if "error" in result:
                    raise ValueError(result["error"])
                return result["response"], result["calls"]
            row = hosted_request(self.output, folder / str(turn), candidate, key,
                                 self.budget, self.config.get("api_timeout_seconds", 1800))
            rows.append(row)
            if row["status"] != "ok":
                raise ValueError(row.get("error", "Hosted request failed"))
            response = row["api_response"]
            return response, providers.tool_calls(model, response)

        def fits(candidate):
            if local:
                return len(self.local_tokenizer.apply_chat_template(
                    candidate, tools=state["session"].tool_schemas, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)) <= limit
            return True

        body, response = tool_loop(
            state["session"], body, ask,
            local_tool_results if local else providers.with_tool_results, fits, trace,
            selected_ids=state.get("selected_ids", ()))
        if local:
            body.append(response)
        elif providers.is_anthropic(model):
            body["messages"].append({"role": "assistant", "content": response["content"]})
        else:
            body["input"].extend(response["output"])
        state["body"] = body
        state["selected_ids"] = trace["selected_scene_ids"]
        self.sessions[session_id] = state
        result = {"session_id": session_id,
                  "response": response["content"] if local else providers.text(model, response),
                  "retrieval": trace,
                  "usage": response.get("usage", {}),
                  "tool_calls": sum(len(t["calls"]) for t in trace["turns"]),
                  "cost_usd": (sum(r["estimated_cost_usd"] for r in rows)
                               if all(r.get("estimated_cost_usd") is not None for r in rows)
                               else None)}
        (folder / "conversation.json").write_text(json.dumps(
            {**result, "sample_id": sid, "model": model, "body": body, "retrieval": trace},
            ensure_ascii=False, indent=2))
        return result

    def stop_worker(self):
        """Release the previously loaded local model."""
        if self.worker is not None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            self.worker = None
            self.local_model = None


def available_models(config):
    """Discover provider models and configured local models with downloaded weights."""
    models, errors = [], []
    for provider, representative in (("openai", "gpt-6-luna"), ("anthropic", "claude-sonnet-5-5")):
        try:
            key = providers.load_key(Path(".env"), representative)
            models.extend({"id": model, "provider": provider} for model in
                          providers.list_models(provider, key))
        except (ValueError, OSError) as error:
            errors.append(f"{provider}: {error}")
    for name, path in config.get("local_configs", {}).items():
        cfg = json.loads(Path(path).read_text())
        if (Path(config.get("local_python", "data/local/screen-env/bin/python")).is_file()
                and any(p.is_file() for p in Path(cfg["path"]).glob("*.safetensors"))):
            models.append({"id": name, "provider": "local",
                           "max_input_tokens": cfg["max_input_tokens"]})
    return models, errors


def serve(manifest, split, runs, config, output, budget, port):
    """Serve a local reader with authenticated live requests and server-side keys."""
    if budget is not None and budget <= 0:
        raise ValueError("Spending budget must be positive")
    data, datasets = viewer_data(manifest, split, runs)
    token = secrets.token_urlsafe(32)
    models, errors = available_models(config)
    data["live"] = {"token": token, "models": models, "budget_usd": budget,
                    "errors": errors}
    application = LiveViewer(data, datasets, config, output, budget)
    document = viewer_html(data).encode()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.send(403, b'{}')
            elif self.path == "/":
                self.send(200, document, "text/html")
            else:
                self.send(404, b'{}')

        def do_POST(self):
            if self.path != "/continue" or self.headers.get("X-Viewer-Token") != token:
                self.send(403, b'{}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Invalid request size")
                result = application.reply(json.loads(self.rfile.read(length)))
                self.send(200, json.dumps(result).encode())
            except Exception as error:
                self.send(400, json.dumps({"error": str(error)}).encode())

    try:
        with HTTPServer(("127.0.0.1", port), Handler) as server:
            print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.stop_worker()
        if application.database is not None:
            application.database.close()


def local_worker(config_path):
    """Keep one cached MLX model loaded while serving native assistant turns."""
    protocol = sys.stdout
    sys.stdout = sys.stderr
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from endless_voices.generation_backends import local_cache

    cfg = json.loads(Path(config_path).read_text())
    mx.set_memory_limit(20_000_000_000)
    mx.set_cache_limit(256_000_000)
    model, tokenizer = load(cfg["path"], tokenizer_config={
        "trust_remote_code": False, **cfg.get("tokenizer_options", {})})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                request["messages"], tools=request["tools"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            if len(prompt) > min(cfg["max_input_tokens"],
                                 cfg["context_ceiling"] - cfg["max_output_tokens"]):
                raise ValueError("Local conversation exceeds its input budget")
            text, last = "", None
            for last in stream_generate(
                    model, tokenizer, prompt=prompt, max_tokens=cfg["max_output_tokens"],
                    sampler=make_sampler(temp=0.7),
                    prefill_step_size=cfg.get("prefill_step_size", 64),
                    **local_cache(model, cfg, {})):
                text += last.text
            if last is None or last.finish_reason != "stop":
                raise ValueError("Local model reached its output limit")
            response, calls = local_tool_response(tokenizer, text)
            print(json.dumps({"response": response, "calls": calls}), file=protocol, flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), file=protocol, flush=True)


if __name__ == "__main__":
    local_worker(sys.argv[2])
