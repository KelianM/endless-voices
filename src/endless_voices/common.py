"""Shared configuration and tokenizer setup."""

import tomllib
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        config = tomllib.load(file)
    name = config["model"]["name_or_path"]
    if name == "YOUR_HF_MODEL_OR_LOCAL_PATH":
        raise ValueError("Set model.name_or_path in the config to a compatible model first.")
    return config


def load_tokenizer(source: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if not tokenizer.chat_template:
        raise ValueError("Choose a tokenizer with a chat template compatible with your messages.")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
