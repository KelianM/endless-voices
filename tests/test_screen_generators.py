import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from endless_voices import anthropic_judge, openai_judge

spec = importlib.util.spec_from_file_location(
    "screen_generators", Path("scripts/screen_generators.py")
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


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
    monkeypatch.setattr(openai_judge, "load_key", lambda _: "fake-secret")
    monkeypatch.setattr(anthropic_judge, "load_key", lambda _: "fake-secret")
    calls = []

    def send(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        if "input" in body:
            assert body["input"] == prompts[0]["messages"]
            result = {
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
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

    with patch.object(screen.urllib.request, "urlopen", send):
        screen.hosted(tmp_path)
        assert len(calls) == 3
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
    monkeypatch.setattr(openai_judge, "load_key", lambda _: "fake-secret")
    with patch.object(screen.urllib.request, "urlopen", side_effect=AssertionError("API called")):
        with pytest.raises(ValueError, match="budget exhausted"):
            screen.hosted(tmp_path)


def test_hosted_selection_does_not_call_unselected_model(tmp_path, monkeypatch):
    screen.save(
        tmp_path / "prompts.json",
        [{"sample_id": "x", "messages": [{"role": "system", "content": "Context"}]}],
    )
    monkeypatch.setattr(openai_judge, "load_key", lambda _: "fake-secret")
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
                            "content": [{"type": "output_text", "text": "A response"}],
                        }
                    ],
                }
            )
        )

    with patch.object(screen.urllib.request, "urlopen", send):
        screen.hosted(tmp_path, models=["gpt-6-sol"], budget=3)
    assert calls == ["gpt-6-sol"]
    assert not (tmp_path / "gpt-6-luna").exists()
