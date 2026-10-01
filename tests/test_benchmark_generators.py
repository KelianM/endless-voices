"""Protect selected-context benchmarks using synthetic generation evidence."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from endless_voices import generation_backends as backends
from endless_voices.context import ContextPool, FullContext

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
def recorded(tmp_path, request):
    models = getattr(request, "param", ["gemma31b", "gpt-6-luna"])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    template = json.loads(
        Path("tests/fixtures/contracts/validation.jsonl").read_text().splitlines()[0]
    )
    records, selections = {}, []
    for sid in ["one", "two", "three", "four"]:
        row = deepcopy(template)
        row["metadata"]["id"] = sid
        row["messages"][-1]["content"] = "Original fixture continuation " + sid
        records[sid] = row
        pool = ContextPool(
            sid,
            "conversation",
            "mission",
            f"Source lore for {sid}\n",
            [{"role": "user", "content": f"Encounter for {sid}"}],
            [],
            {},
        )
        selections.append(FullContext().select(pool, Counter()))
    local = tmp_path / "local.json"
    backends.save(local, {"label": "gemma31b"})
    locals_ = {"gemma31b": str(local)}
    if "qwen30b" in models:
        qwen = tmp_path / "qwen.json"
        backends.save(qwen, {"label": "qwen30b"})
        locals_["qwen30b"] = str(qwen)
    config = tmp_path / "config.json"
    backends.save(
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
    for row, selection in zip(records.values(), selections, strict=True):
        row['messages'] = selection.training_messages(
            'She nods. "A complete source reply for ' + row['metadata']['id'] + '."')
    (dataset / 'validation.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in records.values()))
    backends.save(dataset / 'manifest.json', {
        'format': 'scene-dataset-v1', 'split_unit': 'mission', 'splits': ['validation'],
        'artifacts': {str(p.relative_to(dataset)): backends.sha(p)
                      for p in dataset.rglob('*') if p.is_file()}})
    benchmark.prepare(root, dataset, config)
    for model in models:
        folder = root / model
        folder.mkdir()
        backends.save(
            folder / "settings.json",
            {
                "prompts_sha256": backends.sha(root / "prompts.json"),
                "code_sha256": backends.sha(root / "generation_backends.py"),
                "provider_code_sha256": backends.sha(root / "providers.py"),
                "artifact_code_sha256": backends.sha(root / "artifacts.py"),
            },
        )
        for sid in ["one", "two", "four"]:
            messages = records[sid]["messages"][:-1]
            if model in locals_:
                backends.save(
                    folder / f"{sid}.prompt.json", {"messages": messages, "input_ids": [1, 2]}
                )
            else:
                body = (
                    {"model": model, "system": messages[0]["content"], "messages": messages[1:]}
                    if model == "claude-sonnet-5-5"
                    else {"model": model, "input": messages}
                )
                backends.save(folder / f"{sid}.request.json", {"body": body})
            backends.save(
                folder / f"{sid}.result.json",
                {
                    "sample_id": sid,
                    "status": "ok" if sid in {"one", "four"} else "failed",
                    "response": ("Simulated complete response"
                                 if sid in {"one", "four"} else "Partial"),
                    "finish_reason": "stop" if sid in {"one", "four"} else "length",
                    "error": None if sid in {"one", "four"} else "Output limit",
                    "api_response": {
                        "status": "completed" if sid in {"one", "four"} else "incomplete",
                        "stop_reason": "end_turn" if sid in {"one", "four"} else "max_tokens",
                    },
                },
            )
    return root


def test_export_uses_saved_context_once_and_reports_failures_and_missing(recorded):
    benchmark.export(recorded)
    pack = recorded / "assessment"
    private = backends.read(pack / "private.json")
    assert len([r for r in private["trials"] if r["kind"] == "primary"]) == 4
    assert not list((pack / "public/reversed").glob("*.json"))
    assert sorted(r["status"] for r in private["coverage"]) == [
        "failed",
        "failed",
        "missing",
        "missing",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    expected = {p["sample_id"]: p["messages"]
                for p in backends.read(recorded / "prompts.json")}
    for r in private["trials"]:
        trial = backends.read(pack / r["path"])
        assert list(trial) == ["trial_id", "context", "A", "B"]
        assert trial["context"] == expected[r["sample_id"]]
    primary = next(r for r in private["trials"]
                   if r["kind"] == "primary" and r["sample_id"] == "one")
    trial = backends.read(pack / primary["path"])
    assert trial[primary["original"]] == 'She nods. "A complete source reply for one."'
    assert backends.read(recorded / "gpt-6-luna/two.result.json")["response"] == "Partial"
    with pytest.raises(ValueError, match="exists"):
        benchmark.export(recorded)


@pytest.mark.parametrize("tamper", ["context", "completion", "input_hash", "selection"])
def test_export_rejects_mismatched_or_incomplete_evidence(recorded, tamper):
    folder = recorded / "gpt-6-luna"
    if tamper == "context":
        path = folder / "one.request.json"
        value = backends.read(path)
        value["body"]["input"] = backends.read(folder / "four.request.json")["body"]["input"]
    elif tamper == "completion":
        path = folder / "one.result.json"
        value = backends.read(path)
        value["api_response"]["status"] = "incomplete"
    elif tamper == "selection":
        path = recorded / "prompts.json"
        value = []
    else:
        path = folder / "settings.json"
        value = backends.read(path)
        value["prompts_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        benchmark.export(recorded)


def test_judge_rechecks_public_trials_before_sending_requests(recorded):
    benchmark.export(recorded)
    benchmark.verify_trials(recorded, save=False)
    path = next((recorded / "assessment/public/primary").glob("*.json"))
    trial = backends.read(path)
    trial["context"][0]["content"] = "Leaked answer key"
    path.write_text(json.dumps(trial))
    with pytest.raises(ValueError, match="hash differs"):
        benchmark.verify_trials(recorded, save=False)


@pytest.mark.parametrize(
    "recorded", [["gemma31b", "qwen30b", "gpt-6-luna", "claude-sonnet-5-5"]], indirect=True
)
def test_four_model_lineup_accepts_both_provider_completion_formats(recorded):
    benchmark.export(recorded)
    private = backends.read(recorded / "assessment/private.json")
    assert len(private["coverage"]) == 16
    assert len([r for r in private["trials"] if r["kind"] == "primary"]) == 8
    assert set(private["runs"]) == {"gemma31b", "qwen30b", "gpt-6-luna", "claude-sonnet-5-5"}


def test_prepare_cli_passes_the_prepared_dataset(monkeypatch, tmp_path):
    import sys

    dataset = tmp_path / 'dataset'
    output = tmp_path / 'benchmark'
    calls = []
    monkeypatch.setattr(sys, 'argv', ['benchmark_generators.py', 'prepare',
                                     '--root', str(output), '--dataset', str(dataset)])
    monkeypatch.setattr(benchmark, 'prepare', lambda *args: calls.append(args))
    benchmark.main()
    assert calls == [(output, dataset, Path('configs/benchmark.json'))]
