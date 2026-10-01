"""Offline checks of the data boundary and the actual CLI workflow."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from endless_voices.data import (
    ConversationCollator,
    tokenize_conversations,
)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "friend": 3}))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] + eos_token }}{% endfor %}"
    )
    return tokenizer


def test_padding_preserves_eos(tokenizer: PreTrainedTokenizerFast) -> None:
    batch = ConversationCollator(tokenizer)(
        [
            {"input_ids": [2, 1], "attention_mask": [1, 1]},
            {"input_ids": [2, 3, 1], "attention_mask": [1, 1, 1]},
        ]
    )
    assert batch["labels"].tolist() == [[2, 1, -100], [2, 3, 1]]


def prepared(tmp_path, messages):
    root = tmp_path / "dataset"
    root.mkdir()
    record = json.loads((Path(__file__).parent / "fixtures/contracts/train.jsonl").read_text())
    record["messages"] = messages
    path = root / "train.jsonl"
    path.write_text(json.dumps(record) + "\n")
    (root / "manifest.json").write_text(json.dumps({
        "format": "scene-dataset-v1", "split_unit": "mission", "splits": ["train"],
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
    return root


def test_long_conversation_rejected(tmp_path: Path, tokenizer: PreTrainedTokenizerFast) -> None:
    path = prepared(tmp_path, [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "hello hello hello"},
        {"role": "assistant", "content": "friend"}])
    with pytest.raises(ValueError, match="Shorten the conversation"):
        tokenize_conversations(str(path), tokenizer, max_length=2)


def test_train_and_chat_cli(tmp_path: Path, tokenizer: PreTrainedTokenizerFast) -> None:
    base = tmp_path / "base"
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(base)
    tokenizer.save_pretrained(base)
    dataset = prepared(tmp_path, [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "friend"}])
    output = tmp_path / "adapter"
    config = tmp_path / "train.toml"
    config.write_text(f'''
[model]
name_or_path = "{base}"
[data]
path = "{dataset}"
max_length = 32
[lora]
r = 2
lora_alpha = 4
target_modules = "all-linear"
[training]
output_dir = "{output}"
max_steps = 1
per_device_train_batch_size = 1
learning_rate = 0.001
save_strategy = "no"
use_cpu = true
''')
    subprocess.run(
        [sys.executable, "-m", "endless_voices.train", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "adapter_model.safetensors").exists()
    assert (output / "run_config.json").exists()
    model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained(base), output)
    assert any(
        "lora_B" in name and parameter.abs().sum() > 0
        for name, parameter in model.named_parameters()
    )
    for adapter_args in ([], ["--adapter", str(output)]):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "endless_voices.chat",
                "--config",
                str(config),
                "--device",
                "cpu",
                "--max-new-tokens",
                "2",
                "--temperature",
                "0",
                *adapter_args,
            ],
            input="hello\n/reset\n/quit\n",
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Voice:" in result.stdout


def test_contract_cli_local_tokenizer(tmp_path: Path, tokenizer: PreTrainedTokenizerFast):
    local = tmp_path / "tokenizer"
    tokenizer.save_pretrained(local)
    manifest = Path(__file__).parent / "fixtures" / "contracts" / "manifest.json"
    command = [
        sys.executable,
        "-m",
        "endless_voices.contracts",
        str(manifest),
        "--tokenizer",
        str(local),
        "--max-length",
    ]
    result = subprocess.run([*command, "100"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([*command, "2"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "train/" in result.stderr
    assert "Shorten the conversation" in result.stderr


def test_continuation_loss_excludes_lore_and_preserves_target_eos(tmp_path, tokenizer):
    path = prepared(tmp_path, [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "friend"}])
    examples = tokenize_conversations(str(path), tokenizer, 32)
    batch = ConversationCollator(tokenizer)(examples)
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 3, 1]]


def test_mapping_tokenizer_output_keeps_ids_and_loss_boundary(tmp_path, tokenizer, monkeypatch):
    original = tokenizer.apply_chat_template
    monkeypatch.setattr(tokenizer, "apply_chat_template",
                        lambda *a, **kw: {"input_ids": original(*a, **kw)})
    path = prepared(tmp_path, [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "friend"}])
    row = tokenize_conversations(str(path), tokenizer, 16)[0]
    assert row["input_ids"] == [2, 1, 2, 1, 3, 1]
    assert row["labels"] == [-100, -100, -100, -100, 3, 1]
