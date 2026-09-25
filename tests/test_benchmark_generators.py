"""Protect normalization of synthetic hosted generation fixtures."""

import importlib.util
import json
import sys

import pytest

spec = importlib.util.spec_from_file_location("screen_generators", "scripts/screen_generators.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
sys.modules["screen_generators"] = screen
spec = importlib.util.spec_from_file_location(
    "benchmark_generators", "scripts/benchmark_generators.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    screen.save(manifest, {})
    root = tmp_path / "run"
    root.mkdir()
    messages = [{"role": "user", "content": "Synthetic authored context"}]
    records = {sid: {"messages": messages} for sid in ["one", "two"]}
    monkeypatch.setattr(benchmark, "MANIFEST", manifest)
    monkeypatch.setattr(benchmark, "MODELS", ["gpt-6-sol"])
    monkeypatch.setattr(benchmark.assess, "load_dataset", lambda *_: records)
    monkeypatch.setattr(benchmark.assess, "evaluation_messages", lambda row: row["messages"])
    monkeypatch.setattr(benchmark.assess, "prepare", lambda *a, **k: None)
    screen.save(
        root / "selection.json", {"ids": list(records), "manifest_sha256": screen.sha(manifest)}
    )
    screen.save(
        root / "prompts.json", [{"sample_id": sid, "messages": messages} for sid in records]
    )
    (root / "screen_generators.py").write_text("# Synthetic fixture implementation\n")
    folder = root / "gpt-6-sol"
    folder.mkdir()
    screen.save(
        folder / "settings.json",
        {
            "prompts_sha256": screen.sha(root / "prompts.json"),
            "code_sha256": screen.sha(root / "screen_generators.py"),
        },
    )
    for sid in records:
        screen.save(
            folder / f"{sid}.request.json",
            {
                "body": {"model": "gpt-6-sol", "input": messages},
            },
        )
        screen.save(
            folder / f"{sid}.result.json",
            {
                "sample_id": sid,
                "status": "ok" if sid == "one" else "failed",
                "response": "Simulated complete response" if sid == "one" else "Partial",
                "api_response": {"status": "completed" if sid == "one" else "incomplete"},
            },
        )
    return root


def test_export_retains_failed_text_and_unknown_hosted_tokens(recorded):
    benchmark.export(recorded)
    rows = [
        json.loads(line)
        for line in (recorded / "runs/gpt-6-sol/responses.jsonl").read_text().splitlines()
    ]
    assert rows[0]["finish_reason"] == "eos"
    assert rows[0]["input_tokens"] is None and rows[0]["input_ids_sha256"] is None
    assert rows[1]["status"] == "failed" and rows[1]["response"] == "Partial"
    assert rows[1]["finish_reason"] is None
    with pytest.raises(FileExistsError):
        benchmark.export(recorded)


@pytest.mark.parametrize("tamper", ["context", "completion", "input_hash"])
def test_export_rejects_mismatched_or_incomplete_evidence(recorded, tamper):
    folder = recorded / "gpt-6-sol"
    if tamper == "context":
        path = folder / "one.request.json"
        value = screen.read(path)
        value["body"]["input"][0]["content"] = "Different context"
    elif tamper == "completion":
        path = folder / "one.result.json"
        value = screen.read(path)
        value["api_response"]["status"] = "incomplete"
    else:
        path = folder / "settings.json"
        value = screen.read(path)
        value["prompts_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        benchmark.export(recorded)
