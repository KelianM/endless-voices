import json
from argparse import Namespace

import pytest

from endless_voices import hosted_judge as judge
from endless_voices import providers


def response(status="completed"):
    return {
        "status": status,
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "choice": "A",
                                "confidence": "low",
                                "reason": "Concise voice",
                                "recognized_source": False,
                            }
                        ),
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 100},
            "output_tokens": 200,
        },
    }


def anthropic_response(status="end_turn"):
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
    assert judge.assessment("gpt-6-sol", response())["choice"] == "A"
    with pytest.raises(ValueError):
        judge.assessment("gpt-6-sol", response("incomplete"))
    refusal = response()
    refusal["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    with pytest.raises(ValueError):
        judge.assessment("gpt-6-sol", refusal)


def test_cost_counts_reasoning_output_and_cached_input():
    assert judge.charge("gpt-6-sol", response()) == pytest.approx(0.00382)
    assert judge.charge("gpt-6-sol", {}) is None


@pytest.mark.parametrize("models", [None, ["gpt-6-sol"], ["claude-sonnet-5-5"]])
def test_budget_and_resume_never_resend_submitted_requests(tmp_path, monkeypatch, models):
    public = tmp_path / "public"
    (public / "primary").mkdir(parents=True)
    (public / "instructions.txt").write_text("Choose the original")
    trial = {"trial_id": "opaque", "context": [], "A": "First", "B": "Second"}
    (public / "primary" / "opaque.json").write_text(json.dumps(trial))
    key = tmp_path / ".env"
    key.write_text("OPENAI_API_KEY=secret-test-value\nANTHROPIC_API_KEY=secret-test-value\n")
    args = Namespace(
        public=public, output=tmp_path / "run", env_file=key, budget=5, limit=1, models=models
    )
    calls = []

    class Handle:
        def __enter__(self):
            import io

            return io.StringIO(
                json.dumps(anthropic_response() if models == ["claude-sonnet-5-5"] else response())
            )

        def __exit__(self, *args):
            pass

    def send(request, timeout):
        assert timeout == 1800
        calls.append(json.loads(request.data))
        return Handle()

    monkeypatch.setattr(providers.urllib.request, "urlopen", send)
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    judge.run(args)
    judge.run(args)
    judge.run(args)
    expected = models or ["gpt-6-luna"]
    assert [c["model"] for c in calls] == expected
    assert all(c.get("store", False) is False and "tools" not in c for c in calls)
    assert all("secret-test-value" not in p.read_text() for p in args.output.rglob("*.json"))
    assert judge.spent(args.output) == pytest.approx(
        sum(
            judge.charge(
                model, anthropic_response() if providers.is_anthropic(model) else response()
            )
            for model in expected
        )
    )
    args.output = tmp_path / "tiny-budget"
    args.budget = 0.0000001
    judge.run(args)
    assert len(calls) == len(expected)
    args.output = tmp_path / "run"
    args.budget = 5
    (public / "instructions.txt").write_text("Changed instructions")
    with pytest.raises(ValueError, match="Resume settings or inputs changed"):
        judge.run(args)
    assert len(calls) == len(expected)


def test_unknown_request_reserves_budget_and_is_not_retried(tmp_path, monkeypatch):
    public = tmp_path / "public"
    (public / "primary").mkdir(parents=True)
    (public / "instructions.txt").write_text("Choose")
    trial = {"trial_id": "opaque", "context": [], "A": "First", "B": "Second"}
    (public / "primary" / "opaque.json").write_text(json.dumps(trial))
    folder = tmp_path / "run" / "gpt-6-luna"
    folder.mkdir(parents=True)
    body = judge.payload("gpt-6-luna", trial, "Choose")
    (folder / "opaque.request.json").write_text(json.dumps(body))
    assert judge.spent(tmp_path / "run") == judge.reservation(body)
    key = tmp_path / ".env"
    key.write_text("OPENAI_API_KEY=test\n")
    monkeypatch.setattr(judge.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        providers.urllib.request, "urlopen", lambda *a, **k: pytest.fail("sent request")
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


def test_anthropic_blocks_and_usage():
    assert judge.assessment("claude-sonnet-5-5", anthropic_response())["choice"] == "A"
    for status in ("max_tokens", "refusal"):
        with pytest.raises(ValueError):
            judge.assessment("claude-sonnet-5-5", anthropic_response(status))
    assert providers.charge("claude-sonnet-5-5", anthropic_response()) == pytest.approx(0.00402)
