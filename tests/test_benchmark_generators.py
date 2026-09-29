"""Protect selected-context benchmarks using synthetic generation evidence."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from endless_voices import generation_backends as screen
from endless_voices.context import ContextPool, FullContext
from endless_voices.prepare_context import save_selections

spec = importlib.util.spec_from_file_location(
    "benchmark_generators", "scripts/benchmark_generators.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Counter:
    def text(self, text):
        return len(text)

    def messages(self, messages):
        return sum(len(m["content"]) for m in messages)


@pytest.fixture
def recorded(tmp_path, monkeypatch, request):
    models = getattr(request, "param", ["gemma31b", "gpt-6-luna"])
    manifest = tmp_path / "manifest.json"
    screen.save(manifest, {})
    template = json.loads(
        Path("tests/fixtures/contracts/validation.jsonl").read_text().splitlines()[0]
    )
    records, selections = {}, []
    for sid in ["one", "two", "three"]:
        row = deepcopy(template)
        row["metadata"]["id"] = sid
        row["messages"][-1]["content"] = "Original fixture continuation " + sid
        records[sid] = row
        pool = ContextPool(
            sid,
            "conversation",
            "mission",
            "New source lore\n",
            [{"role": "user", "content": "New source encounter"}],
            [],
            {},
        )
        selections.append(FullContext().select(pool, Counter()))
    context = tmp_path / "context"
    save_selections(context, selections, {"simulated_fixture": True})
    monkeypatch.setattr(benchmark.assess, "load_dataset", lambda *_: records)
    local = tmp_path / "local.json"
    screen.save(local, {"label": "gemma31b"})
    locals_ = {"gemma31b": str(local)}
    if "qwen30b" in models:
        qwen = tmp_path / "qwen.json"
        screen.save(qwen, {"label": "qwen30b"})
        locals_["qwen30b"] = str(qwen)
    config = tmp_path / "config.json"
    screen.save(
        config,
        {
            "generators": models,
            "judge": "gpt-6-luna",
            "local_configs": locals_,
            "judge_context_limit": 1050000,
            "generation_budget_usd": 0.5,
            "judge_budget_usd": 0.5,
        },
    )
    root = tmp_path / "run"
    benchmark.prepare(root, context, manifest, config)
    messages = selections[0].messages
    for model in models:
        folder = root / model
        folder.mkdir()
        screen.save(
            folder / "settings.json",
            {
                "prompts_sha256": screen.sha(root / "prompts.json"),
                "code_sha256": screen.sha(root / "generation_backends.py"),
            },
        )
        for sid in ["one", "two"]:
            if model in locals_:
                screen.save(
                    folder / f"{sid}.prompt.json", {"messages": messages, "input_ids": [1, 2]}
                )
            else:
                body = (
                    {"model": model, "system": messages[0]["content"], "messages": messages[1:]}
                    if model == "claude-sonnet-5-5"
                    else {"model": model, "input": messages}
                )
                screen.save(folder / f"{sid}.request.json", {"body": body})
            screen.save(
                folder / f"{sid}.result.json",
                {
                    "sample_id": sid,
                    "status": "ok" if sid == "one" else "failed",
                    "response": "Simulated complete response" if sid == "one" else "Partial",
                    "finish_reason": "stop" if sid == "one" else "length",
                    "error": None if sid == "one" else "Output limit",
                    "api_response": {
                        "status": "completed" if sid == "one" else "incomplete",
                        "stop_reason": "end_turn" if sid == "one" else "max_tokens",
                    },
                },
            )
    return root


def test_export_uses_saved_context_once_and_reports_failures_and_missing(recorded):
    benchmark.export(recorded)
    pack = recorded / "assessment"
    private = screen.read(pack / "private.json")
    assert len([r for r in private["trials"] if r["kind"] == "primary"]) == 2
    assert not list((pack / "public/reversed").glob("*.json"))
    assert sorted(r["status"] for r in private["coverage"]) == [
        "failed",
        "failed",
        "missing",
        "missing",
        "ok",
        "ok",
    ]
    expected = screen.read(recorded / "prompts.json")[0]["messages"]
    for r in private["trials"]:
        trial = screen.read(pack / r["path"])
        assert list(trial) == ["trial_id", "context", "A", "B"]
        assert trial["context"] == expected
    assert screen.read(recorded / "gpt-6-luna/two.result.json")["response"] == "Partial"
    with pytest.raises(ValueError, match="exists"):
        benchmark.export(recorded)


@pytest.mark.parametrize("tamper", ["context", "completion", "input_hash", "selection"])
def test_export_rejects_mismatched_or_incomplete_evidence(recorded, tamper):
    folder = recorded / "gpt-6-luna"
    if tamper == "context":
        path = folder / "one.request.json"
        value = screen.read(path)
        value["body"]["input"][0]["content"] = "Different context"
    elif tamper == "completion":
        path = folder / "one.result.json"
        value = screen.read(path)
        value["api_response"]["status"] = "incomplete"
    elif tamper == "selection":
        path = recorded / "prompts.json"
        value = []
    else:
        path = folder / "settings.json"
        value = screen.read(path)
        value["prompts_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        benchmark.export(recorded)


def test_judge_rechecks_public_trials_before_sending_requests(recorded):
    benchmark.export(recorded)
    benchmark.verify_trials(recorded, save=False)
    path = next((recorded / "assessment/public/primary").glob("*.json"))
    trial = screen.read(path)
    trial["context"][0]["content"] = "Leaked answer key"
    path.write_text(json.dumps(trial))
    with pytest.raises(ValueError, match="hash differs"):
        benchmark.verify_trials(recorded, save=False)


@pytest.mark.parametrize(
    "recorded", [["gemma31b", "qwen30b", "gpt-6-luna", "claude-sonnet-5-5"]], indirect=True
)
def test_four_model_lineup_accepts_both_provider_completion_formats(recorded):
    benchmark.export(recorded)
    private = screen.read(recorded / "assessment/private.json")
    assert len(private["coverage"]) == 12
    assert len([r for r in private["trials"] if r["kind"] == "primary"]) == 4
    assert set(private["runs"]) == {"gemma31b", "qwen30b", "gpt-6-luna", "claude-sonnet-5-5"}
