# 10. Let agents retrieve context through shared tools

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [Issue 18](https://github.com/KelianM/endless-voices/issues/18), [PR 19](https://github.com/KelianM/endless-voices/pull/19)

## Context

Saved sampled prompts fix the available context before a model can decide what history or
writing examples it needs. Comparing fixed selection with agent-directed retrieval requires
two policies over the same permitted material, with budgets that fit each model.

## Decision

- **Select static or dynamic context at runtime.** Both modes retrieve from the same scene
  database. Static selection fits nearby prerequisite history first, then semantic matches.
  Dynamic selection lets the model request history, search and reads before writing.
- **Preparation saves encounters and eligible pools, not sampled prompts.** State interpretation
  from [ADR 9](0009-build-examples-from-consistent-game-state.md) determines permitted scenes.
  Retrieval enforces branch, split and target boundaries before ranking or reading.
- **Fit whole scenes within the configured input budget.** Local runs reserve output space
  within the model's context ceiling. Hosted dynamic runs may use larger budgets. The model chooses
  when to finish retrieval; hosted calls remain subject to the run’s spend ceiling.
- **Keep history and writing references distinct.** A reference demonstrates the game's
  writing without asserting that its events happened in the current playthrough.
- **Save selected messages and retrieval traces with the run.** Training selects static
  context before loading the generator. Judges receive common static evidence regardless
  of the generator's retrieval mode.

## Consequences

- **Retrieval choices become part of model performance.** Dynamic selection can find useful
  scenes the static heuristic misses, but adds model calls and may choose worse context.
- **Context limits do not prevent repeated calls.** Queries can repeat without adding scenes.
  A fixed turn cap was rejected because it cuts off useful retrieval as well as repetition.
- **Equal access does not mean equal cost.** Models can use different context lengths and
  retrieval turns; a shared pool does not equalize resource use.
- **Old sampled-prompt results need a new baseline.** This replaces the saved selection and
  random-fill policy in [ADR 8](0008-infer-scene-writing-from-sampled-game-context.md), while
  retaining minimal instructions and authored evidence. Reproducing the old sampler was
  rejected because the benchmark will be rerun with the shared retrieval flow.
- **Prepared pools remain bounded by the interpreter.** Dynamic retrieval can choose among
  permitted scenes but cannot reconstruct an arbitrary live game save.
- **Judges do not see each generator's particular evidence.** Common static context keeps
  the judging conditions fixed, but may omit a fact a dynamic generator retrieved.
