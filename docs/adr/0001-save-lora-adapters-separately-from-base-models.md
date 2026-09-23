# 1. Training saves LoRA adapters separately from base models

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Initial implementation, f26cb1a](https://github.com/KelianM/endless-voices/commit/f26cb1a2e8f7a4f86cbc085ee723e6d0b179c54e), [PR #11, retrospective record](https://github.com/KelianM/endless-voices/pull/11)

## Context

The training and chat commands use the same configured base checkpoint. Adapted behaviour
needs a saved artifact that the chat command can load. The implementation predates issue
tracking and was committed on 2026-09-22. This record documents that existing code.

## Decision

[train.py](../../src/endless_voices/train.py) wraps the base model with PEFT's `get_peft_model`
and a `LoraConfig`. After training, `model.save_pretrained(output)` saves the adapter. The
output directory also contains the tokenizer and `run_config.json`, but not the base weights.

[chat.py](../../src/endless_voices/chat.py) loads the configured base checkpoint and applies
`PeftModel.from_pretrained` when an adapter is supplied. The
[CPU smoke test](../../tests/test_pipeline.py) trains a tiny random model, checks the saved
adapter, reloads the adapter, and exercises generation.

## Consequences

An adapter directory cannot be used as a standalone base model. Loading an adapter requires
the matching base checkpoint and tokenizer. Each training run rejects a nonempty output
directory, preserving existing artifacts.

The implementation trains and saves adapter parameters instead of updating and saving all
base weights. Adapter training reduces the number of trainable parameters, but the current
float32 implementation still loads the full base model into memory. The smoke test verifies
the mechanism; the repository contains no trained faction model or measured quality result.
