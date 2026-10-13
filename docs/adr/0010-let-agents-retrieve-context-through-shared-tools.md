# 10. Let agents retrieve context through shared tools

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [Issue 18](https://github.com/KelianM/endless-voices/issues/18), [PR 19](https://github.com/KelianM/endless-voices/pull/19)

## Context

Fixed mission depth and seeded sampling choose a model's context before generation. The model
cannot request missing history or examples of the game's writing. Comparing larger agents
with smaller fine-tuned models needs both fixed context and agent-directed retrieval over
the same permitted material.

## Decision

- **Agents choose context through tools:** retrieve history by prerequisite depth, search
  semantically and read selected passages or scenes. The initial input contains the writing
  instruction and visible encounter.
- **The fixed sampler remains the baseline,** using the same eligible material as retrieval.
  The sampling policy from [ADR 8](0008-infer-scene-writing-from-sampled-game-context.md)
  remains reproducible.
- **Preparation determines eligibility.** Each example saves its permitted scenes using the
  state interpretation from [ADR 9](0009-build-examples-from-consistent-game-state.md).
  Tools enforce branch, split and target boundaries before ranking or reading; excluded
  scenes cannot displace eligible results.
- **History and writing references stay distinct.** A reference demonstrates the game's
  writing without asserting that its events happened in the current playthrough.
- **The caller runs the agent loop and sets its budgets.** The shared tools provide schemas,
  dispatch and a trace of retrieval choices independently of the model provider.

## Consequences

- **Agents can seek evidence the fixed sampler misses.** Retrieval choices become part of
  the model's performance, alongside the model's use of the returned text.
- **Equal access does not mean equal cost.** Models can retrieve different amounts of text
  and make different numbers of calls; the shared pool alone does not equalize a comparison.
- **Prepared bundles retain more text than fixed prompts need.** Saved pools inherit the
  interpreter's limitations and do not represent arbitrary live game saves.
- **Existing generation and benchmark runs still use fixed prompts.** The retrieval tools
  are available, but those runners do not execute an agent loop.
