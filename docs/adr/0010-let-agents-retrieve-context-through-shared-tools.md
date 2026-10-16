# 10. Let agents retrieve context through shared tools

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [Issue 18](https://github.com/KelianM/endless-voices/issues/18), [PR 19](https://github.com/KelianM/endless-voices/pull/19)

## Context

Saved sampled prompts fix the available context before a model can decide what history or
writing examples it needs. Local and hosted models need the same opportunity to retrieve
permitted evidence, with budgets that fit each model.

## Decision

- **Generation retrieves context through one native tool loop.** Local and hosted models
  receive the same instructions and eligible evidence, using their native call formats.
- **Select context beforehand only for training and judging.** Fit nearby prerequisite
  history first, then semantic matches. Benchmark judges receive common evidence.
- **Preparation saves encounters, game states and source traces.** Reconstruct eligible
  pools from shared raw source when needed, and verify their fingerprints. State interpretation
  from [ADR 9](0009-build-examples-from-consistent-game-state.md) determines permitted scenes.
  Retrieval enforces branch, split and target boundaries before ranking or reading.
- **Fit whole scenes within the configured input budget.** Local runs reserve output space
  within the model's context ceiling. Hosted runs may use larger budgets. The model chooses
  when to finish retrieval; hosted calls remain subject to the run’s spend ceiling.
- **Keep history and writing references distinct.** A reference demonstrates the game's
  writing without asserting that its events happened in the current playthrough. Retain
  distinct authored passage branches with compatible prefixes; selecting only the longest
  sampled route discarded alternative writing references.
- **Save conversations and retrieval traces with the run.** Record the evidence the model
  actually received, including tool results.

## Consequences

- **Dynamic conversations retain tool results and reasoning between calls.** The growing
  transcript consumes the input budget. Separate JSON selection was rejected because it
  prevented tool use during composition and gave local models a different retrieval flow.
- **Retrieval choices become part of model performance.** Models decide what evidence to read
  as well as what to write. A fixed-context generation mode was removed to avoid maintaining
  a second flow; controlled writing-only comparisons belong in experiments.
- **Context limits do not prevent repeated calls.** Queries can repeat without adding scenes.
  A fixed turn cap was rejected because it cuts off useful retrieval as well as repetition.
- **Equal access does not mean equal cost.** Models can use different context lengths and
  retrieval turns; a shared pool does not equalize resource use.
- **Old sampled-prompt results need a new baseline.** This replaces the saved selection and
  random-fill policy in [ADR 8](0008-infer-scene-writing-from-sampled-game-context.md), while
  retaining minimal instructions and authored evidence. Reproducing the old sampler was
  rejected because the benchmark will be rerun with the shared retrieval flow.
- **Eligibility remains bounded by the interpreter.** Dynamic retrieval can choose among
  permitted scenes but cannot reconstruct an arbitrary live game save.
- **Judges do not see each generator's particular evidence.** Common static context keeps
  the judging conditions fixed, but may omit a fact a dynamic generator retrieved.
