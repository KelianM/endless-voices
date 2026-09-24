"""Regression checks for comparable continuations and durable failure records."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from endless_voices import generate as gen

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "friend": 3}))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + eos_token }}{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return tokenizer


@pytest.fixture
def base(tmp_path, tokenizer):
    path = tmp_path / "base"
    torch.manual_seed(7)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def args(tmp_path, base):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES, dataset)
    path = dataset / "validation.jsonl"
    record = json.loads(path.read_text())
    record["messages"][1:1] = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Authored earlier reply"},
    ]
    path.write_text(json.dumps(record) + "\n")
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["files"]["validation"][0]["sha256"] = gen.file_hash(path)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    config = tmp_path / "model.toml"
    config.write_text(f'[model]\nname_or_path = "{base}"\n')
    return gen.parse_args(
        [
            "--config",
            str(config),
            "--manifest",
            str(dataset / "manifest.json"),
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--offline",
            "--max-context-tokens",
            "256",
            "--max-new-tokens",
            "4",
        ]
    )


class ReplyModel:
    def __init__(self, output=(2, 1), error=None):
        self.output = output
        self.error = error
        self.inputs = []

    def generate(self, input_ids, **kwargs):
        self.inputs.append(input_ids.tolist())
        if self.error:
            raise self.error
        return torch.cat([input_ids, torch.tensor([self.output])], dim=1)


def use_stub(monkeypatch, model):
    original = gen.load_condition

    def load(*values):
        _, tokenizer, device, limit, settings = original(*values)
        return model, tokenizer, device, limit, settings

    monkeypatch.setattr(gen, "load_condition", load)


def read_run(args):
    run = json.loads((args.output / "run.json").read_text())
    rows = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
    prompts = [
        json.loads(line) for line in (args.output / "prompts.jsonl").read_text().splitlines()
    ]
    return run, rows, prompts


def test_only_final_target_withheld_and_provenance_saved(args, monkeypatch):
    # A batch runner must not leak targets or replace authored assistant history.
    use_stub(monkeypatch, ReplyModel())
    record = json.loads((args.manifest.parent / "validation.jsonl").read_text())
    args.max_context_tokens = 512
    assert gen.run_generation(args) == 0
    run, rows, prompts = read_run(args)
    assert prompts[0]["messages"] == record["messages"][:-1]
    assert any(message["role"] == "assistant" for message in prompts[0]["messages"])
    assert "evaluation" not in prompts[0] and "metadata" not in prompts[0]
    assert rows[0]["response"] == "hello"
    assert rows[0]["finish_reason"] == "eos"
    assert rows[0]["sample_id"] == record["metadata"]["id"]
    assert run["status"] == "complete" and run["counts"] == {"ok": 1, "failed": 0}
    assert run["model"]["files_sha256"]["model.safetensors"]
    assert run["tokenizer"]["signature"]
    assert run["execution"]["packages"]["transformers"]
    assert run["execution"]["code_sha256"]["generate.py"]
    assert run["dataset"]["manifest_sha256"] == gen.file_hash(args.manifest)
    assert run["generation_config"]["do_sample"] is False
    assert run["effective_context_tokens"] == 256
    for name, digest in run["artifacts_sha256"].items():
        assert gen.file_hash(args.output / name) == digest
    assert (args.output / "tokenizer/tokenizer_config.json").exists()
    before = (args.output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        gen.run_generation(args)
    assert (args.output / "run.json").read_bytes() == before


def test_target_and_private_notes_cannot_change_tokenized_input(tokenizer):
    record = json.loads((FIXTURES / "validation.jsonl").read_text())
    before, _ = gen.prepare_prompt(record, tokenizer, 256, 4)
    record["messages"][-1]["content"] = "SECRET TARGET"
    record["evaluation"]["sources"][0]["reference"] = "SECRET CRITERIA"
    after, _ = gen.prepare_prompt(record, tokenizer, 256, 4)
    assert before == after
    assert "SECRET" not in json.dumps(after)


@pytest.mark.parametrize(
    "output,error,kind",
    [
        ((2, 2, 2, 2), None, "OutputLimit"),
        ((1,), None, "EmptyResponse"),
        ((2, 1), RuntimeError("device failure"), "RuntimeError"),
    ],
)
def test_generation_failures_preserve_ids_and_partial_output(
    args, monkeypatch, output, error, kind
):
    use_stub(monkeypatch, ReplyModel(output, error))
    assert gen.run_generation(args) == 1
    run, rows, _ = read_run(args)
    assert run["status"] == "completed_with_failures"
    assert rows[0]["error"]["type"] == kind
    assert rows[0]["sample_id"] == run["dataset"]["sample_ids"][0]
    if kind == "OutputLimit":
        assert rows[0]["response"] == "hello hello hello hello"


def test_overlong_prompt_is_recorded_without_generation(args, monkeypatch):
    model = ReplyModel()
    use_stub(monkeypatch, model)
    args.max_context_tokens = 8
    assert gen.run_generation(args) == 1
    _, rows, prompts = read_run(args)
    assert rows[0]["error"]["stage"] == "prompt"
    assert "exceeds" in rows[0]["error"]["message"]
    assert prompts[0]["input_tokens"] > 8
    assert not model.inputs


def test_template_cannot_silently_drop_system_or_history(args, monkeypatch, base, tokenizer):
    tokenizer.chat_template = "{{ messages[-1]['content'] }}"
    tokenizer.save_pretrained(base)
    model = ReplyModel()
    use_stub(monkeypatch, model)
    assert gen.run_generation(args) == 1
    _, rows, _ = read_run(args)
    assert "omits" in rows[0]["error"]["message"]
    assert not model.inputs


@pytest.mark.parametrize("error", [RuntimeError("load failed"), KeyboardInterrupt()])
def test_setup_failure_accounts_for_every_selected_id(args, monkeypatch, error):
    def fail(*_):
        raise error

    monkeypatch.setattr(gen, "load_condition", fail)
    assert gen.run_generation(args) in (1, 130)
    run, rows, _ = read_run(args)
    assert [row["sample_id"] for row in rows] == run["dataset"]["sample_ids"]
    assert all(row["status"] == "failed" for row in rows)
    assert run["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")


def test_input_plus_output_budget_includes_generation_template(tokenizer):
    record = json.loads((FIXTURES / "validation.jsonl").read_text())
    prompt, _ = gen.prepare_prompt(record, tokenizer, 256, 4)
    exact = prompt["input_tokens"] + 4
    gen.prepare_prompt(record, tokenizer, exact, 4)
    with pytest.raises(ValueError, match="exceeds"):
        gen.prepare_prompt(record, tokenizer, exact - 1, 4)


def test_explicit_selection_rejects_unknown_duplicate_and_wrong_split_ids(args, tmp_path):
    selection = tmp_path / "ids.json"
    args.sample_ids = selection
    for ids in (["absent"], ["fixture-train"], ["fixture-validation"] * 2, []):
        selection.write_text(json.dumps(ids))
        with pytest.raises(ValueError):
            gen.run_generation(args)
        assert not args.output.exists()


def test_changed_dataset_rejected_before_run(args, tmp_path):
    dataset = args.manifest.parent
    with (dataset / "validation.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        gen.run_generation(args)
    assert not args.output.exists()


def make_adapter(base, path):
    model = get_peft_model(
        GPT2LMHeadModel.from_pretrained(base),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            target_modules="all-linear",
        ),
    )
    model.save_pretrained(path)
    return path


def test_real_tiny_model_and_lora_cli_use_identical_context(args, base, tmp_path):
    adapter = make_adapter(base, tmp_path / "adapter")
    saved = []
    for name, extra in (("base-run", []), ("adapter-run", ["--adapter", str(adapter)])):
        args.output = tmp_path / name
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "endless_voices.generate",
                "--config",
                args.config,
                "--manifest",
                str(args.manifest),
                "--output",
                str(args.output),
                "--device",
                "cpu",
                "--offline",
                "--max-context-tokens",
                "256",
                "--max-new-tokens",
                "4",
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        run, rows, prompts = read_run(args)
        assert result.returncode in (0, 1), result.stderr
        assert "error" not in run, result.stderr
        assert rows[0].get("error", {}).get("type") in (None, "OutputLimit", "EmptyResponse")
        assert rows[0]["output_ids"]
        saved.append((run, rows, prompts))
    assert saved[0][2] == saved[1][2]
    assert saved[0][1][0]["output_ids"] == saved[1][1][0]["output_ids"]
    assert saved[0][0]["model"] == saved[1][0]["model"]
    assert saved[1][0]["adapter"]["files_sha256"]["adapter_model.safetensors"]


@pytest.mark.parametrize("mismatch", ["base", "revision", "tokenizer", "conflicting-revision"])
def test_incompatible_adapter_rejected(args, base, tmp_path, tokenizer, mismatch):
    adapter = make_adapter(base, tmp_path / "adapter")
    if mismatch == "conflicting-revision":
        path = adapter / "adapter_config.json"
        config = json.loads(path.read_text())
        config["revision"] = "declared-revision"
        path.write_text(json.dumps(config))
        (adapter / "run_config.json").write_text(
            json.dumps({"model": {"name_or_path": str(base), "revision": "other-revision"}})
        )
    elif mismatch == "tokenizer":
        tokenizer.chat_template += "changed"
        tokenizer.save_pretrained(adapter)
    else:
        path = adapter / "adapter_config.json"
        config = json.loads(path.read_text())
        config["base_model_name_or_path" if mismatch == "base" else "revision"] = "wrong"
        path.write_text(json.dumps(config))
    args.adapter = str(adapter)
    assert gen.run_generation(args) == 1
    run, rows, _ = read_run(args)
    assert any(word in run["error"]["message"] for word in ("different", "differs", "disagree"))
    assert rows[0]["error"]["stage"] == "run"


def test_hub_revisions_must_be_immutable(args):
    with pytest.raises(ValueError, match="immutable"):
        gen.artifact("an-owner/a-model", "main", args)


def test_seed_is_independent_of_selection_order(args, monkeypatch):
    use_stub(monkeypatch, ReplyModel())
    path = args.manifest.parent / "validation.jsonl"
    first_record = json.loads(path.read_text())
    second_record = copy.deepcopy(first_record)
    second_record["metadata"]["id"] = "second-validation"
    path.write_text(json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n")
    manifest = json.loads(args.manifest.read_text())
    manifest["files"]["validation"][0]["sha256"] = gen.file_hash(path)
    args.manifest.write_text(json.dumps(manifest))
    assert gen.run_generation(args) == 0
    first = read_run(args)[1][0]
    other = copy.copy(args)
    other.output = args.output.parent / "repeat"
    other.sample_ids = args.output.parent / "reversed-ids.json"
    ids = json.loads((args.output / "sample-ids.json").read_text())
    other.sample_ids.write_text(json.dumps(ids[::-1]))
    assert gen.run_generation(other) == 0
    second = read_run(other)[1][1]
    assert first["seed"] == second["seed"]
    assert first["input_ids_sha256"] == second["input_ids_sha256"]


@pytest.mark.parametrize("eos", [0, 1])
def test_checkpoint_sampling_defaults_cannot_override_recorded_greedy(args, base, monkeypatch, eos):
    config_path = base / "generation_config.json"
    saved = json.loads(config_path.read_text())
    saved.update(do_sample=True, temperature=0.7, top_k=20, top_p=0.8, eos_token_id=eos)
    config_path.write_text(json.dumps(saved))
    original = GPT2LMHeadModel._prepare_generation_config
    observed = []

    def inspect(self, *values, **kwargs):
        result = original(self, *values, **kwargs)
        observed.append(result[0].do_sample)
        return result

    monkeypatch.setattr(GPT2LMHeadModel, "_prepare_generation_config", inspect)
    gen.run_generation(args)
    run, _, _ = read_run(args)
    assert run["generation_config"]["do_sample"] is False
    assert observed == [False]
    assert run["generation_config"]["eos_token_id"] == eos
