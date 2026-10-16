import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from endless_voices import generation_backends as backends
from endless_voices import providers


@pytest.fixture(autouse=True)
def prepared_selection(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    backends.save(tmp_path / "selection.json", {"context": {
        "tokenizer": "fake", "hosted_max_input_tokens": dict.fromkeys(providers.RATES, 16000)}})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(len(json.dumps(messages))))))

    def sessions(root, settings):
        prompts = {p["sample_id"]: p for p in backends.read(root / "prompts.json")}

        def session(sid, db):
            return SimpleNamespace(
                history_count=0, trace=[], tool_schemas=[],
                initial_messages=lambda: [dict(m) for m in prompts[sid]["messages"]]
                or [{"role": "system", "content": "Encounter"}],
            )
        return (SimpleNamespace(close=lambda: None), SimpleNamespace(retrieval_session=session),
                {sid: sid for sid in prompts})
    monkeypatch.setattr(backends, "benchmark_sessions", sessions)


def test_hosted_keeps_partial_outputs_and_never_overwrites(tmp_path, monkeypatch):
    prompts = [
        {
            "sample_id": "opaque",
            "messages": [
                {"role": "system", "content": "Character context"},
                {"role": "user", "content": "Authored history"},
            ],
        }
    ]
    backends.save(tmp_path / "prompts.json", prompts)
    monkeypatch.setattr(providers, "load_key", lambda *_: "fake-secret")
    calls = []

    def send(request, timeout):
        assert timeout == 2400
        body = json.loads(request.data)
        calls.append(body)
        if "input" in body:
            assert body["input"][0]["content"].startswith("Character context")
            result = {
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Partial dialogue"}],
                    }
                ],
            }
        else:
            assert body["system"].startswith("Character context")
            assert body["messages"] == prompts[0]["messages"][1:]
            result = {
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "Partial dialogue"}],
            }
        return io.StringIO(json.dumps(result))

    with patch.object(providers.urllib.request, "urlopen", send):
        backends.hosted(tmp_path, timeout_seconds=2400)
        assert len(calls) == 3
        for p in tmp_path.glob("*/settings.json"):
            assert backends.read(p)["timeout_seconds"] == 2400
        for p in tmp_path.glob("*/*.result.json"):
            result = backends.read(p)
            assert result["status"] == "failed" and result["response"] == "Partial dialogue"
        with pytest.raises(FileExistsError):
            backends.hosted(tmp_path)
        assert len(calls) == 3
    assert all("fake-secret" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_hosted_stops_before_exceeding_shared_budget(tmp_path, monkeypatch):
    backends.save(tmp_path / "prompts.json", [{"sample_id": "x", "messages": []}])
    old = tmp_path / "earlier"
    old.mkdir()
    backends.save(old / "unknown.request.json", {"reservation_usd": 5})
    monkeypatch.setattr(providers, "load_key", lambda *_: "fake-secret")
    with patch.object(
        providers.urllib.request, "urlopen", side_effect=AssertionError("API called")
    ):
        backends.hosted(tmp_path)
    result = backends.read(tmp_path / "gpt-6-luna/x.result.json")
    assert result["status"] == "failed" and "budget exhausted" in result["error"]


def test_hosted_selection_does_not_call_unselected_model(tmp_path, monkeypatch):
    backends.save(
        tmp_path / "prompts.json",
        [{"sample_id": "x", "messages": [{"role": "system", "content": "Context"}]}],
    )
    monkeypatch.setattr(providers, "load_key", lambda *_: "fake-secret")
    calls = []

    def send(request, timeout):
        calls.append(json.loads(request.data)["model"])
        return io.StringIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "A response"}],
                        }
                    ],
                }
            )
        )

    with patch.object(providers.urllib.request, "urlopen", send):
        backends.hosted(tmp_path, models=["gpt-6-sol"], budget=3)
    assert calls == ["gpt-6-sol"]
    assert not (tmp_path / "gpt-6-luna").exists()


@pytest.mark.parametrize("failure", ["memory", "output_limit"])
def test_local_runner_preserves_limits_cache_and_partial_failures(tmp_path, monkeypatch, failure):
    class FullCache:
        def to_quantized(self, *, bits, group_size):
            assert (bits, group_size) == (8, 64)
            return quantized

    class RotatingCache(FullCache):
        max_size = 1024

        def to_quantized(self, **kwargs):
            pytest.fail("Sliding-window cache was quantized")

    quantized, rotating = object(), RotatingCache()
    tokenizer = SimpleNamespace(
        chat_template="<tool_call>",
        apply_chat_template=lambda messages, **kwargs:
            messages[0]["content"].split("\n\nRetrieve")[0],
        encode=lambda text, **kwargs: list(range(len(text))),
    )
    limits, calls = {}, []
    mx = SimpleNamespace(
        set_memory_limit=lambda value: limits.update(memory=value),
        set_cache_limit=lambda value: limits.update(cache=value),
        get_peak_memory=lambda: 1_000,
        random=SimpleNamespace(seed=lambda _: None),
    )

    def stream(model, tokenizer, **kwargs):
        calls.append(kwargs)
        assert kwargs["prompt_cache"] == [quantized, rotating]
        assert rotating.max_size == 1024
        assert kwargs["prefill_step_size"] == 64
        assert kwargs["max_tokens"] == 4
        assert "kv_bits" not in kwargs
        if len(calls) == 2:
            yield SimpleNamespace(text="Partial", generation_tokens=4, finish_reason="length")
            if failure == "memory":
                raise RuntimeError("Insufficient Memory")
        else:
            yield SimpleNamespace(text="Complete", generation_tokens=1, finish_reason="stop")

    modules = {
        "mlx": SimpleNamespace(core=mx),
        "mlx.core": mx,
        "mlx_lm": SimpleNamespace(
            load=lambda *a, **kw: (object(), tokenizer), stream_generate=stream
        ),
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=lambda **kw: object()),
        "mlx_lm.models.cache": SimpleNamespace(
            KVCache=FullCache, make_prompt_cache=lambda _: [FullCache(), rotating]
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backends.importlib.metadata, "version", lambda _: "test-runtime")
    prompts = [
        {"sample_id": sid, "messages": [{"role": "user", "content": text}]}
        for sid, text in [("overflow", "x" * 9), ("ok", "a"), ("partial", "bb"), ("later", "ccc")]
    ]
    backends.save(tmp_path / "prompts.json", prompts)
    config = tmp_path / "local.json"
    backends.save(
        config,
        {
            "label": "local",
            "path": str(tmp_path / "weights"),
            "context_ceiling": 12,
            "max_output_tokens": 4,
            "prefill_step_size": 64,
            "kv_scope": "full_history",
            "kv_bits": 8,
        },
    )
    backends.local(tmp_path, config)
    folder = tmp_path / "local"
    overflow = backends.read(folder / "overflow.result.json")
    assert overflow["status"] == "failed" and "input budget" in overflow["error"]
    assert backends.read(folder / "ok.result.json")["status"] == "ok"
    partial = backends.read(folder / "partial.result.json")
    assert partial["status"] == "failed" and partial["response"] == "Partial"
    assert [c["prompt"] for c in calls] == (
        [[0], [0, 1]] if failure == "memory" else [[0], [0, 1], [0, 1, 2]]
    )
    assert (folder / "later.result.json").exists() == (failure != "memory")
    assert limits["memory"] == 20_000_000_000
    saved = backends.read(folder / "ok.prompt.json")
    assert saved["messages"][0]["content"].startswith(prompts[1]["messages"][0]["content"])


def test_local_native_loop_continues_after_reading_without_a_turn_cap():
    # A ninth retrieval must still work, and oversized results must not reach the model.
    from endless_voices.context_generation import (
        dynamic_messages,
        local_tool_response,
        local_tool_results,
        tool_loop,
    )

    class Session:
        selected_scene_ids = set()

        def call_tool(self, name, arguments):
            assert name == "search_context"
            return [{"scene_id": "large", "text": "x" * 20000},
                    {"scene_id": "small", "text": "useful lore"}]

    tokenizer = SimpleNamespace(chat_template="<tool_call>")
    requests = []

    def ask(messages, turn):
        requests.append(messages)
        if turn:
            assert messages[-1]["role"] == "tool"
            assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]
            assert "useful lore" in messages[-1]["content"]
            assert "x" * 20000 not in json.dumps(messages)
        text = ('<think>Consider <tool_call>not an actual call</tool_call></think>'
                '<tool_call>{"name":"search_context","arguments":{}}</tool_call>'
                if turn < 9 else "<think></think>Written continuation")
        return local_tool_response(tokenizer, text)

    trace = {}
    messages, response = tool_loop(
        Session(), dynamic_messages([{"role": "system", "content": "Encounter"}], 0),
        ask, local_tool_results, lambda ms: len(json.dumps(ms)) <= 15000, trace,
    )
    assert "No eligible mission history" in messages[0]["content"]
    assert len(requests) == 10 and response["content"] == "Written continuation"
    assert trace["selected_scene_ids"] == ["small"]
    assert trace["turns"][0]["results"][0]["omitted_documents"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        local_tool_response(tokenizer, '<tool_call>{"name":"read_context"}')


def test_retrieval_requests_reserve_failed_costs_across_nested_folders(tmp_path, monkeypatch):
    # An unknown charge from a failed selection request must still consume the global budget.
    selection = tmp_path / "model" / "sample.retrieval"
    selection.mkdir(parents=True)
    monkeypatch.setattr(providers, "reservation", lambda body: 0.6)
    calls = []

    def failed(*args):
        calls.append(args)
        raise TimeoutError("provider may have charged")

    monkeypatch.setattr(providers, "send", failed)
    body = providers.payload("gpt-6-sol", [{"role": "system", "content": "Encounter"}])
    result = backends.hosted_request(
        tmp_path, selection / "0", body, "secret", 1.0, 60
    )
    assert result["status"] == "failed"
    with pytest.raises(ValueError, match="budget exhausted"):
        backends.hosted_request(
            tmp_path, tmp_path / "final", body, "secret", 1.0, 60
        )
    assert len(calls) == 1


@pytest.mark.parametrize("model", ["gpt-6-luna", "claude-sonnet-5-5"])
def test_native_loop_reads_then_retrieves_again_with_larger_hosted_budget(
    tmp_path, monkeypatch, model,
):
    # Losing native reasoning or tool results prevents follow-up retrieval after reading evidence.
    from transformers import AutoTokenizer

    class Session:
        trace = []
        tool_schemas = [{"type": "function", "function": {
            "name": "search_context", "description": "Search history",
            "parameters": {"type": "object", "properties": {}}}}]
        history_count = 2

        def initial_messages(self):
            return [{"role": "system", "content": "Lore"},
                    {"role": "user", "content": "Encounter"}]

        def call_tool(self, name, arguments):
            if name == "unknown":
                raise ValueError("Unknown context tool")
            if arguments:
                assert self.selected_scene_ids == {"large"}
                return [{"scene_id": "small", "text": "Follow-up evidence"}]
            return [{"scene_id": "large", "text": "x" * 9000},
                    {"scene_id": "oversized", "text": "y" * 20000}]

    (tmp_path / "selection.json").write_text(json.dumps({"context": {
        "max_input_tokens": 8192,
        "tokenizer": "fake", "hosted_max_input_tokens": {model: 16000}}}))
    backends.save(tmp_path / "prompts.json", [{"sample_id": "one",
                                              "messages": Session().initial_messages()}])
    monkeypatch.setattr(backends, "benchmark_sessions", lambda *a: (
        SimpleNamespace(close=lambda: None),
        SimpleNamespace(retrieval_session=lambda *a: Session()), {"one": 0}))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(
            sum(len(m["content"]) for m in messages)))))
    monkeypatch.setattr(providers, "load_key", lambda *a: "fake-secret")
    bodies, responses = [], []
    anthropic = providers.is_anthropic(model)

    def send(body, *args):
        turn = len(bodies)
        bodies.append(body)
        assert body["tools"][0]["name"] == "search_context"
        if turn:
            encoded = json.dumps(body)
            assert "x" * 9000 in encoded and "y" * 20000 not in encoded
            assert "secret-reasoning" in encoded
            assert "omitted_documents" in encoded
        calls = ([("a", "search_context", {}), ("bad", "unknown", {})] if turn == 0
                 else [("b", "search_context", {"query": "Follow-up"})] if turn == 1 else [])
        if anthropic:
            parts = [{"type": "thinking", "thinking": "reason", "signature": "secret-reasoning"}]
            parts += [{"type": "text", "text": "Reading" if calls else "Written continuation"}]
            parts += [{"type": "tool_use", "id": id_, "name": name, "input": args}
                      for id_, name, args in calls]
            response = {"stop_reason": "tool_use" if calls else "end_turn", "content": parts}
            if turn:
                assert body["messages"][-2]["content"] == responses[-1]["content"]
                results = body["messages"][-1]["content"]
                assert results[0]["tool_use_id"] == ("a" if turn == 1 else "b")
                if turn == 1:
                    assert results[1]["is_error"]
        else:
            parts = [{"type": "reasoning", "id": f"r{turn}", "summary": [],
                      "encrypted_content": "secret-reasoning"},
                     {"type": "message", "status": "completed", "content": [
                         {"type": "output_text", "text": "Reading" if calls
                          else "Written continuation"}]}]
            parts += [{"type": "function_call", "call_id": id_, "name": name,
                       "arguments": json.dumps(args)} for id_, name, args in calls]
            response = {"status": "completed", "output": parts}
            if turn:
                assert all(p in body["input"] for p in responses[-1]["output"])
                assert body["input"][-1]["call_id"] == ("bad" if turn == 1 else "b")
        response["usage"] = {"input_tokens": 100, "output_tokens": 20}
        responses.append(response)
        return response

    monkeypatch.setattr(providers, "send", send)
    backends.hosted(tmp_path, models=[model], budget=1)
    assert len(bodies) == 3
    assert 8192 < len(json.dumps(bodies[-1])) < 16000
    result = backends.read(tmp_path / model / "one.result.json")
    assert result["status"] == "ok" and result["response"] == "Written continuation"
    assert result["total_usage"]["input_tokens"] == 300
    trace = backends.read(tmp_path / model / "one.retrieval.json")
    assert trace["selected_scene_ids"] == ["large", "small"]
