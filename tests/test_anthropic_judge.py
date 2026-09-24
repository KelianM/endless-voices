import json
from argparse import Namespace

import pytest

from endless_voices import anthropic_judge as judge


def response(status="end_turn"):
    return {
        "stop_reason": status,
        "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "choice": "A",
                        "confidence": "low",
                        "reason": "Concise voice",
                        "recognized_source": False,
                    }
                ),
            },
        ],
        "usage": {"input_tokens": 1000, "cache_read_input_tokens": 100, "output_tokens": 200},
    }


def test_incomplete_and_refused_responses_are_not_judgments():
    assert judge.assessment(response())["choice"] == "A"
    with pytest.raises(ValueError):
        judge.assessment(response("max_tokens"))
    with pytest.raises(ValueError):
        judge.assessment(response("refusal"))


def test_cost_counts_reasoning_output_and_cached_input():
    assert judge.charge("claude-sonnet-5", response()) == pytest.approx(0.00402)
    assert judge.charge("claude-sonnet-5", {}) is None


def test_budget_and_resume_never_resend_submitted_requests(tmp_path, monkeypatch):
    public = tmp_path / "public"
    (public / "primary").mkdir(parents=True)
    (public / "instructions.txt").write_text("Choose the original")
    trial = {"trial_id": "opaque", "context": [], "A": "First", "B": "Second"}
    (public / "primary" / "opaque.json").write_text(json.dumps(trial))
    key = tmp_path / ".env"
    key.write_text("ANTHROPIC_API_KEY=secret-test-value\n")
    args = Namespace(public=public, output=tmp_path / "run", env_file=key, budget=5, limit=1)
    calls = []

    class Handle:
        def __enter__(self):
            import io

            return io.StringIO(json.dumps(response()))

        def __exit__(self, *args):
            pass

    def send(request, timeout):
        calls.append(json.loads(request.data))
        return Handle()

    monkeypatch.setattr(judge.urllib.request, "urlopen", send)
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    judge.run(args)
    judge.run(args)
    judge.run(args)
    assert [c["model"] for c in calls] == ["claude-sonnet-5"]
    assert all("tools" not in c and len(c["messages"]) == 1 for c in calls)
    assert all("secret-test-value" not in p.read_text() for p in args.output.rglob("*.json"))
    assert judge.spent(args.output) == pytest.approx(0.00402)
    args.output = tmp_path / "tiny-budget"
    args.budget = 0.0000001
    judge.run(args)
    assert len(calls) == 1


def test_unknown_request_reserves_budget_and_is_not_retried(tmp_path, monkeypatch):
    public = tmp_path / "public"
    (public / "primary").mkdir(parents=True)
    (public / "instructions.txt").write_text("Choose")
    trial = {"trial_id": "opaque", "context": [], "A": "First", "B": "Second"}
    (public / "primary" / "opaque.json").write_text(json.dumps(trial))
    folder = tmp_path / "run" / "claude-sonnet-5"
    folder.mkdir(parents=True)
    body = judge.payload("claude-sonnet-5", trial, "Choose")
    (folder / "opaque.request.json").write_text(json.dumps(body))
    assert judge.spent(tmp_path / "run") == judge.reservation(body)
    key = tmp_path / ".env"
    key.write_text("ANTHROPIC_API_KEY=test\n")
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        judge.urllib.request, "urlopen", lambda *a, **k: pytest.fail("sent request")
    )
    args = Namespace(
        public=public,
        output=tmp_path / "run",
        env_file=key,
        budget=judge.reservation(body) + 0.000001,
        limit=1,
    )
    judge.run(args)
    result = json.loads((folder / "opaque.result.json").read_text())
    assert result["status"] == "failed" and "unknown" in result["error"]
