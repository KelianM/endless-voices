# 14. Build scene examples through stateful dialogue

- **Status:** Proposed
- **Date:** 2026-09-30
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [game conversation rules](https://github.com/endless-sky/endless-sky/wiki/WritingConversations), [game conditions](https://github.com/endless-sky/endless-sky/wiki/Player-Conditions)

## Context

Extracting individual speakers' quotes introduced boundaries that the game does not have. A question and an answer can occupy one authored paragraph. A structural traversal also admits contradictory branches when an earlier action has already determined a condition. Targets and sampled history need the same interpretation of game state.

## Decision

`DatasetBuilder` owns preparation in `endless_voices.dataset`. `ExampleBuilder` and `ContextSampler` share one `DialogueInterpreter`. The parser preserves authored paragraphs and explicit choices. The interpreter follows assignments and conditions, returning continuations through the next visible player choice or endpoint. No speech extraction or punctuation rewriting remains.

Integer state uses Z3 constraints. Unspecified initial conditions represent possible prior states, not a claim that the game starts with arbitrary values. Known assignments override current values while constraints preserve what was true earlier. Contradictory routes are rejected. Each distinct target paragraph sequence remains eligible; a seeded selection chooses one compatible history when several histories lead to the same target.

Whole-mission ownership follows [ADR 13](0013-split-datasets-by-source-mission.md). Context eligibility is established before token-budget selection. Required ancestors are inferred from positive `has "mission: done"` prerequisites outside OR groups. Unassigned and disallowed missions supply no text. Their omitted history is recorded; unknown initial conditions may represent those earlier events. This is not a complete campaign simulator.

Preparation writes fixed records, contexts, authored targets, state witnesses, source coordinates and hashes. `SceneDataset` provides indexed records. The training adapter tokenizes those records; PyTorch's DataLoader batches and shuffles them without resampling story context. The benchmark reads the same prepared validation bundle. The task wording remains the shared instruction in [ADR 11](0011-use-a-shared-scene-continuation-instruction.md).

Unsupported operations, loops beyond the configured bound, excessive route expansion and context overflow fail preparation. A failed build publishes no partial dataset. In particular, scheduled events and resource effects require explicit support before their examples can be published. Old preparation scripts are removed rather than kept as another path.

## Consequences

The interpreter tests consistency within its supported state model, not reachability from every real game start. OR prerequisites and flag-producing events do not yet reconstruct predecessor missions. Configuration records the reviewed mission scope and existing splits; broadening extraction is a separate dataset change.

Branch exploration can grow rapidly. The route limit makes that cost visible instead of silently dropping targets. Multiple earlier routes sharing a target do not create a Cartesian product of training examples. The seed, witness and selected history record the route actually used.

Previously saved datasets remain evidence but do not acquire state-consistency guarantees retroactively. The current source selection must pass the new builder before a replacement dataset is described as ready for training. Tests of the interpreter alone do not establish that every mission operation is supported.
