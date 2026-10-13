"""Protect selected-context benchmarks using synthetic generation evidence."""

import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from prepared_fixtures import write_empty_pools

from endless_voices import benchmark_generators as benchmark
from endless_voices import generation_backends as backends
from endless_voices.context import ContextPool


@pytest.fixture
def recorded(tmp_path, request, monkeypatch):
    models = getattr(request, "param", ["gemma31b", "gpt-6-luna"])
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    template = json.loads(
        Path("tests/fixtures/contracts/validation.jsonl").read_text().splitlines()[0]
    )
    records, pools = {}, []
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
        pools.append(pool)
    local = tmp_path / "local.json"
    backends.save(local, {"label": "gemma31b", "context_ceiling": 20000,
                          "max_output_tokens": 1024, "max_input_tokens": 8192})
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
    index = tmp_path / "context.sqlite"
    index.write_text("synthetic index")
    plan = backends.read(config)
    plan["context"] = {"index": str(index), "tokenizer": "fake",
                       "max_input_tokens": 8192, "depth": 4}
    config.write_text(json.dumps(plan))
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(
        apply_chat_template=lambda messages, **kw: list(range(
            sum(len(m["content"]) for m in messages)))))
    monkeypatch.setattr(benchmark.context_generation, "open_database", lambda _: nullcontext(
        SimpleNamespace(embedder=SimpleNamespace(embed=lambda *a, **kw: [[1, 0]]), dimension=2)))
    root = tmp_path / "run"
    for row, pool in zip(records.values(), pools, strict=True):
        row["messages"] = [*pool.messages(), {"role": "assistant", "content":
            'She nods. "A complete source reply for ' + row['metadata']['id'] + '."'}]
    (dataset / 'validation.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in records.values()))
    backends.save(dataset / 'manifest.json', {
        'format': 'scene-dataset-v2', 'split_unit': 'mission_chain', 'splits': ['validation'],
        'artifacts': {str(p.relative_to(dataset)): backends.sha(p)
                      for p in dataset.rglob('*') if p.is_file()}})
    write_empty_pools(dataset)
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
                "context_code_sha256": backends.sha(root / "context_generation.py"),
            },
        )
        for sid in ["one", "two", "four"]:
            messages = benchmark.context_generation.dynamic_messages(
                deepcopy(records[sid]["messages"][:-1]), 0)
            backends.save(folder / f"{sid}.retrieval.json", {"selected_scene_ids": []})
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
                backends.save(folder / f"{sid}.prompt.json", {"messages": messages})
                requests = folder / f"{sid}.retrieval"
                requests.mkdir()
                backends.save(requests / "0.request.json", {"body": body})
            backends.save(
                folder / f"{sid}.result.json",
                {
                    "sample_id": sid,
                    "request_count": 1,
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
    expected = {sid: row["messages"][:-1]
                for sid, row in backends.read(recorded / "records.json").items()}
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
        path = folder / "one.retrieval/0.request.json"
        value = backends.read(path)
        other = backends.read(folder / "four.retrieval/0.request.json")
        value["body"]["input"] = other["body"]["input"]
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
    from endless_voices.cli import benchmark as cli

    dataset = tmp_path / 'dataset'
    output = tmp_path / 'benchmark'
    calls = []
    monkeypatch.setattr(benchmark, 'prepare', lambda *args: calls.append(args))
    cli.main(['prepare', '--root', str(output), '--dataset', str(dataset)])
    assert calls == [(output, dataset, Path('configs/benchmark.json'))]


def test_preparation_records_cli_source_after_split(recorded):
    from endless_voices.cli import benchmark as cli

    snapshot = recorded / "cli-benchmark.py"
    assert snapshot.read_bytes() == Path(cli.__file__).read_bytes()
    snapshot.write_text("changed CLI orchestration")
    with pytest.raises(ValueError, match="Benchmark preparation changed: cli-benchmark.py"):
        benchmark.checked_inputs(recorded)


def test_dynamic_export_keeps_common_judge_evidence_and_verifies_selected_scenes(recorded):
    # A dynamic generator may use less context, but must not change the judge's reference.
    selection = backends.read(recorded / "selection.json")
    records = backends.read(recorded / "records.json")
    for row in records.values():
        row["messages"][0]["content"] += "\nCommon static evidence for judging."
    (recorded / "records.json").write_text(json.dumps(records))
    selection["artifacts_sha256"]["records.json"] = backends.sha(recorded / "records.json")
    (recorded / "selection.json").write_text(json.dumps(selection))
    benchmark.export(recorded)
    private = backends.read(recorded / "assessment/private.json")
    for row in private["trials"]:
        trial = backends.read(recorded / "assessment" / row["path"])
        assert trial["context"] == records[row["sample_id"]]["messages"][:-1]


@pytest.mark.parametrize("artifact", ["dataset", "index"])
def test_prepared_run_rejects_replaced_retrieval_inputs(recorded, artifact):
    selection = backends.read(recorded / "selection.json")
    path = (Path(selection["dataset_source"]) / "manifest.json" if artifact == "dataset"
            else Path(selection["context"]["index"]))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        benchmark.checked_inputs(recorded)


def test_native_export_rejects_ineligible_tool_text():
    # Saved native transcripts must not bypass the eligible-pool check used for local runs.
    from endless_voices.context import render
    from endless_voices.retrieval import scene_id

    block = {"mission": "Prior", "heading": "Prior", "passages": [
        {"role": "narrator", "text": "Known history", "line": 1}]}
    pool = ContextPool("sample", "conversation", "mission", "Lore", [], [block], {})
    sid = scene_id(block)
    document = {"id": sid, "scene_id": sid, "kind": "history", "text": render(block)}
    messages = [*benchmark.context_generation.dynamic_messages(pool.messages(), len(pool.blocks)),
                {"type": "function_call_output", "call_id": "a",
                                  "output": json.dumps({"documents": [document]})}]
    benchmark.verify_tool_context(pool, messages, [sid])
    document["text"] = "Withheld target"
    messages[-1]["output"] = json.dumps({"documents": [document]})
    with pytest.raises(ValueError, match="eligible scene"):
        benchmark.verify_tool_context(pool, messages, [sid])


def test_preflight_counts_tools_against_local_input_budget(recorded):
    # A prompt can fit the model ceiling yet exceed Gemma's configured input allowance.
    def render(messages, *, tools, **kwargs):
        assert len(tools) == 3 and "Retrieve context" in messages[0]["content"]
        return "rendered"

    tokenizer = SimpleNamespace(apply_chat_template=render,
                                encode=lambda *a, **kw: [0] * 8193)
    with pytest.raises(ValueError, match="context exceeds"):
        benchmark.preflight(recorded, {"gemma31b": tokenizer})
