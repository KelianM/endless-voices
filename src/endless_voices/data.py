"""JSONL conversations and causal language modelling batches."""

from copy import deepcopy
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from endless_voices.dataset.storage import SceneDataset


def load_conversations(path: str, split: str = "train") -> list[list[dict[str, str]]]:
    """Load messages from a verified prepared dataset split."""
    return [record["messages"] for record in SceneDataset.load(path, split)]


class TokenizedSceneDataset(torch.utils.data.Dataset):
    """Provide fixed tokenized examples to the trainer's DataLoader."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return deepcopy(self.examples[index])


def tokenize_conversations(
    path: str, tokenizer: PreTrainedTokenizerBase, max_length: int, loss: str = "continuation",
    split: str = "train"
) -> TokenizedSceneDataset:
    if max_length < 2:
        raise ValueError("data.max_length must be at least 2")
    if loss not in {"continuation", "all"}:
        raise ValueError("data.loss must be continuation or all")
    examples = []
    for index, messages in enumerate(load_conversations(path, split), 1):
        ids = tokenize_messages(messages, tokenizer, max_length, label=f"Conversation {index}")
        labels = list(ids)
        if loss == "continuation":
            prefix = chat_ids(messages[:-1], tokenizer)
            if ids[:len(prefix)] != prefix or len(prefix) >= len(ids):
                raise ValueError("Chat template does not preserve the supplied conversation prefix")
            labels[:len(prefix)] = [-100] * len(prefix)
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                         "labels": labels})
    return TokenizedSceneDataset(examples)


def chat_ids(messages, tokenizer):
    """Return flat chat token IDs across tokenizer return formats."""
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized conversation")
        ids = ids[0]
    return ids


def tokenize_messages(messages, tokenizer, max_length: int, *, label: str = "Conversation"):
    """Shared length check for the loader and optional offline curated-data validation."""
    if max_length < 2:
        raise ValueError("data.max_length must be at least 2")
    ids = chat_ids(messages, tokenizer)
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
        supplied = [example.get("labels", example["input_ids"]) for example in examples]
        inputs = [{k: v for k, v in example.items() if k != "labels"} for example in examples]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        padded = []
        for row in supplied:
            pad = [-100] * (width - len(row))
            padded.append(pad + list(row) if self.tokenizer.padding_side == "left"
                          else list(row) + pad)
        labels = torch.tensor(padded, dtype=torch.long)
        # Mask padding by position, preserving genuine EOS when PAD == EOS.
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)
