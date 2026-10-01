import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from endless_voices import generation_backends as backends
from endless_voices import providers


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
            assert body["input"] == prompts[0]["messages"]
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
            assert body["system"] == "Character context"
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
        with pytest.raises(ValueError, match="budget exhausted"):
            backends.hosted(tmp_path)


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
        apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
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
        "mlx": SimpleNamespace(core=mx), "mlx.core": mx,
        "mlx_lm": SimpleNamespace(load=lambda *a, **kw: (object(), tokenizer),
                                  stream_generate=stream),
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=lambda **kw: object()),
        "mlx_lm.models.cache": SimpleNamespace(
            KVCache=FullCache, make_prompt_cache=lambda _: [FullCache(), rotating]),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backends.importlib.metadata, "version", lambda _: "test-runtime")
    prompts = [{"sample_id": sid, "messages": [{"role": "user", "content": text}]}
               for sid, text in [("overflow", "x" * 9), ("ok", "a"),
                                 ("partial", "bb"), ("later", "ccc")]]
    backends.save(tmp_path / "prompts.json", prompts)
    config = tmp_path / "local.json"
    backends.save(config, {"label": "local", "path": str(tmp_path / "weights"),
                          "context_ceiling": 12, "max_output_tokens": 4,
                          "prefill_step_size": 64, "kv_scope": "full_history", "kv_bits": 8})
    backends.local(tmp_path, config)
    folder = tmp_path / "local"
    overflow = backends.read(folder / "overflow.result.json")
    assert overflow["status"] == "failed" and "no truncation" in overflow["error"]
    assert backends.read(folder / "ok.result.json")["status"] == "ok"
    partial = backends.read(folder / "partial.result.json")
    assert partial["status"] == "failed" and partial["response"] == "Partial"
    assert [c["prompt"] for c in calls] == (
        [[0], [0, 1]] if failure == "memory" else [[0], [0, 1], [0, 1, 2]])
    assert (folder / "later.result.json").exists() == (failure != "memory")
    assert limits["memory"] == 20_000_000_000
    for prompt in prompts[:3]:
        saved = backends.read(folder / f"{prompt['sample_id']}.prompt.json")
        assert saved["messages"] == prompt["messages"]
