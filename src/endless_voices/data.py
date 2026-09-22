"""JSONL conversations and causal language modelling batches."""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from endless_voices.messages import validate_messages


def load_conversations(path: str) -> list[list[dict[str, str]]]:
    conversations = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record["messages"]
                validate_messages(messages)
                conversations.append(messages)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not conversations:
        raise ValueError(f"{path}: no conversations found")
    return conversations


def tokenize_conversations(
    path: str, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> list[dict[str, list[int]]]:
    if max_length < 2:
        raise ValueError("data.max_length must be at least 2")
    examples = []
    for index, messages in enumerate(load_conversations(path), 1):
        ids = tokenize_messages(messages, tokenizer, max_length, label=f"Conversation {index}")
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return examples


def tokenize_messages(messages, tokenizer, max_length: int, *, label: str = "Conversation"):
    """Shared length check for the loader and optional offline curated-data validation."""
    if max_length < 2:
        raise ValueError("data.max_length must be at least 2")
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(ids) > max_length or len(ids) < 2:
        raise ValueError(
            f"{label} has {len(ids)} tokens; expected 2..{max_length}. "
            "Shorten the conversation or increase data.max_length."
        )
    return ids


class ConversationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        # Mask padding by position, preserving genuine EOS when PAD == EOS.
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)
