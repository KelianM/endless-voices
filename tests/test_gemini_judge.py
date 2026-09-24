import json

import pytest

from endless_voices.gemini_judge import assessment, load_key, payload


def test_gemini_rejects_partial_candidates_and_invalid_abstention():
    answer = {"choice": "A", "confidence": "low", "reason": "Simulated", "recognized_source": False}
    response = {
        "candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(answer)}]}}
        ]
    }
    assert assessment(response)["choice"] == "A"
    response["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError):
        assessment(response)
    with pytest.raises(ValueError):
        assessment({"candidates": []})


def test_gemini_payload_is_stateless_and_rejects_private_fields():
    trial = {"trial_id": "opaque", "context": [], "A": "Simulated A", "B": "Simulated B"}
    body = payload(trial, "Choose the original")
    assert len(body["contents"]) == 1
    assert "tools" not in body
    assert json.loads(body["contents"][0]["parts"][0]["text"]) == trial
    with pytest.raises(ValueError):
        payload({**trial, "original": "A"}, "Choose")


def test_dotenv_is_read_without_shell_execution(tmp_path):
    path = tmp_path / ".env"
    path.write_text('GEMINI_API_KEY="simulated-key"\nIGNORED=$(not-a-command)\n')
    assert load_key(path) == "simulated-key"


def test_resume_preserves_quota_failure_and_does_not_resend(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from urllib.error import HTTPError

    from endless_voices import gemini_judge as g

    public = tmp_path / "public"
    (public / "primary").mkdir(parents=True)
    (public / "instructions.txt").write_text("Simulated instructions")
    trial = {"trial_id": "opaque", "context": [], "A": "Simulated A", "B": "Simulated B"}
    (public / "primary/opaque.json").write_text(json.dumps(trial))
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=simulated-secret\n")
    calls = []

    def reject(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, 429, "quota", {}, None)

    monkeypatch.setattr(g.urllib.request, "urlopen", reject)
    args = SimpleNamespace(
        public=public, output=tmp_path / "out", env_file=env, model="gemini-3.8-flash", limit=1
    )
    g.run(args)
    original = (args.output / "opaque.result.json").read_bytes()
    g.run(args)
    assert len(calls) == 1
    assert (args.output / "opaque.result.json").read_bytes() == original
    assert b"simulated-secret" not in original
    assert json.loads(original)["http_status"] == 429
    (public / "instructions.txt").write_text("Changed")
    with pytest.raises(ValueError, match="changed"):
        g.run(args)
