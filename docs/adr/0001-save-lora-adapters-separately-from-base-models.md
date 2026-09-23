# 1. Save LoRA adapters separately from base models

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Initial implementation, f26cb1a](https://github.com/KelianM/endless-voices/commit/f26cb1a2e8f7a4f86cbc085ee723e6d0b179c54e), [PR #11, retrospective record](https://github.com/KelianM/endless-voices/pull/11)

## Context

The initial project compares a starting language model with an adapted model for fictional
identity conversations. The training foundation predates the issue roadmap. This record
captures the implementation introduced on 2026-09-22; the repository contains no notebooks
for that decision.

## Decision

`src/endless_voices/train.py` uses PEFT LoRA with a causal language model. Each run saves
adapter weights, a tokenizer, and `run_config.json` in an independent output directory.
`src/endless_voices/chat.py` loads the configured base model and optionally applies an adapter.
Identity labels are supplied through data and prompts, without faction-specific model code.

## Consequences

A saved adapter requires the matching base checkpoint; the output directory is not a standalone
model. Users must preserve the base revision and tokenizer alongside experiment records.
LoRA limits trainable parameters but the current float32 implementation still loads the full
base weights. The implementation uses adapters instead of updating and saving all base weights;
no comparison establishing LoRA's quality advantage has been performed.

One adapter per identity and a shared adapter remain possible. The decision does not select a
model, dataset size, quantization method, or response-only loss. The current trainer computes
loss on every non-padding token; changing that objective requires a separate implementation.
