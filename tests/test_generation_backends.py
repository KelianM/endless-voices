import io
import json
from unittest.mock import patch

import pytest

from endless_voices import generation_backends as screen
from endless_voices import providers


def test_hosted_screen_keeps_partial_outputs_and_never_overwrites(tmp_path, monkeypatch):
    prompts = [
        {
            "sample_id": "opaque",
            "messages": [
                {"role": "system", "content": "Character context"},
                {"role": "user", "content": "Authored history"},
            ],
        }
    ]
    screen.save(tmp_path / "prompts.json", prompts)
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
        screen.hosted(tmp_path, timeout_seconds=2400)
        assert len(calls) == 3
        for p in tmp_path.glob("*/settings.json"):
            assert screen.read(p)["timeout_seconds"] == 2400
        for p in tmp_path.glob("*/*.result.json"):
            result = screen.read(p)
            assert result["status"] == "failed" and result["response"] == "Partial dialogue"
        with pytest.raises(FileExistsError):
            screen.hosted(tmp_path)
        assert len(calls) == 3
    assert all("fake-secret" not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_hosted_screen_stops_before_exceeding_shared_budget(tmp_path, monkeypatch):
    screen.save(tmp_path / "prompts.json", [{"sample_id": "x", "messages": []}])
    old = tmp_path / "earlier"
    old.mkdir()
    screen.save(old / "unknown.request.json", {"reservation_usd": 5})
    monkeypatch.setattr(providers, "load_key", lambda *_: "fake-secret")
    with patch.object(
        providers.urllib.request, "urlopen", side_effect=AssertionError("API called")
    ):
        with pytest.raises(ValueError, match="budget exhausted"):
            screen.hosted(tmp_path)


def test_hosted_selection_does_not_call_unselected_model(tmp_path, monkeypatch):
    screen.save(
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
        screen.hosted(tmp_path, models=["gpt-6-sol"], budget=3)
    assert calls == ["gpt-6-sol"]
    assert not (tmp_path / "gpt-6-luna").exists()


def test_selective_quantization_preserves_rotating_cache_and_window():
    class FullCache:
        def to_quantized(self, *, bits, group_size):
            return (bits, group_size)

    class RotatingCache(FullCache):
        max_size = 1024

        def to_quantized(self, **kwargs):
            raise AssertionError("Sliding-window cache must not be quantized")

    rotating = RotatingCache()
    result = screen.quantize_full_history([FullCache(), rotating], FullCache)
    assert result[0] == (8, 64)
    assert result[1] is rotating
    assert result[1].max_size == 1024
