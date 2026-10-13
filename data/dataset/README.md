# Published scene dataset

This directory contains the previous prepared release, with sampled prompts saved in its
records. Rebuild from [the preparation command](../../README.md#prepare-data) before using
the current retrieval flow. The coverage figures below describe the published release.

The dataset contains authored passages from cached Human, Hai and Quarg game sources. Whole missions and known text-sharing variants stay in one split. Training context excludes held-out missions; every input excludes its target source passages and reference scenes from sibling variants.

| Split | Examples | Source missions | Supervised tokens |
|---|---:|---:|---:|
| Train | 1,847 | 347 | 256,605 |
| Validation | 250 | 57 | 36,411 |
| Test | 360 | 58 | 54,544 |

Training contains 1,834 distinct target texts. Supervised token counts include the final
assistant framing and end tokens, measured with the cached Qwen tokenizer. Runtime context
lengths depend on the selected policy and the model's tokenizer.

| Source region | Train | Validation | Test |
|---|---:|---:|---:|
| Human | 1,642 | 181 | 311 |
| Hai | 166 | 54 | 45 |
| Quarg | 39 | 15 | 4 |

Identity labels group source regions, not individual speakers. Species is marked unspecified because complete scenes can contain multiple characters and were not annotated per speaker. No character-specific speaking instructions or generated targets are included. Human material dominates the release; Quarg test coverage is small.

`metadata.task` is derived from the current conversation prefix. Training has 480 scene openings and 1,367 continuations; validation has 76 openings and 174 continuations; test has 74 openings and 286 continuations. Each task selects its shared minimal instruction before context budgeting.

Eligible context contains authored location lore, compatible mission history and independent
writing references. References are separate scenes, not one continuous player history.
Unknown starting conditions and travel durations are constrained possibilities rather than
a reconstructed game save.

All 462 selected missions produced targets, yielding 2,457 examples across the three splits. No selected mission is excluded for an extraction failure. The selection does not cover every mission in the three regions.

Unresolved runtime substitutions remain literal markers under the game-variable policy: 200 training targets, 29 validation targets and 71 test targets contain markers. Player-name and ship-name substitutions use the configured neutral values. No missing game facts are filled with model-generated text.

The published `train.jsonl`, `validation.jsonl` and `test.jsonl` retain the old sampled inputs.
New builds save only instructions, visible encounters and authored targets in these records;
`retrieval.json` stores eligible scenes once and records each example's pool. Provenance,
configuration, manifest hashes and upstream licensing accompany both formats.

Verification checked split separation, source-span exclusion, task/instruction agreement, token budgets, training mask boundaries and 2,025 distinct state witnesses. Two test examples contain an earlier, separately authored occurrence of their target paragraph in the same conversation. Source-span exclusion preserves these valid repeated passages. No complete target text is shared across splits. Existing development material remains part of these source regions; this release is not a claim of a pristine research test set or measured fine-tuning improvement.

## How this dataset is produced

[`DatasetBuilder`](../../src/endless_voices/dataset/builder.py) uses the same interpreter for
targets, history and references. Each reachable target gets one reproducible compatible
history; mutually exclusive histories are never combined into one transcript.

Training context excludes held-out missions. Validation can use training context, and test
can use training and validation context. Unselected prerequisite missions remain eligible
unless assigned to a held-out split. References exclude the target mission and known sibling
variants. The prerequisite graph follows positive completion requirements, not arbitrary OR
relationships.

The interpreter handles supported dialogue conditions, assignments and mission effects.
Shop, fleet and map changes are recorded effects, not a full game simulation.
[ADR 9](../../docs/adr/0009-build-examples-from-consistent-game-state.md) records those limits.
The [design overview](../../docs/design.md) explains runtime context selection and consumers.
