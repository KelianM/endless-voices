# 1. Fine-tuning changes an adapter, not the base model

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Initial implementation, f26cb1a](https://github.com/KelianM/endless-voices/commit/f26cb1a2e8f7a4f86cbc085ee723e6d0b179c54e), [PR #11](https://github.com/KelianM/endless-voices/pull/11)

## Context

The project asks whether fine-tuning improves a model's portrayal of a fictional identity.
The starting checkpoint is both the foundation for training and the baseline for comparison.
Updating every model weight would require training and storing a complete model for each
experiment, even when every experiment starts from the same checkpoint.

## Decision

Fine-tuning trains a LoRA adapter while keeping the base weights fixed. Each run saves the
adapter separately, together with the tokenizer and run configuration. Chat loads the same
base checkpoint with or without an adapter
([train.py](../../src/endless_voices/train.py), [chat.py](../../src/endless_voices/chat.py)).

## Consequences

- Experiments can share a base checkpoint without storing another full copy of its weights
  for every trained variant. Each adapter remains an independent artifact.
- An adapter is not a standalone model. Reproducing a run requires the matching base checkpoint,
  tokenizer, and configuration; keeping only the adapter is insufficient.
- Fewer parameters are trained, but the full base model still occupies memory. The current
  implementation loads the base weights in float32, so a small adapter does not make a large
  base model cheap to run.
- Adaptation is restricted to the changes LoRA can represent. Full-weight fine-tuning is not
  implemented, and the project has not measured whether that restriction affects persona quality.
