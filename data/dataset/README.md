# Prepared scene dataset

The dataset contains authored passages from cached Human, Hai and Quarg game sources. Whole missions and known text-sharing variants stay in one split. Training context excludes held-out missions; every input excludes its target source passages and reference scenes from sibling variants.

| Split | Examples | Source missions | Supervised tokens |
|---|---:|---:|---:|
| Train | 1,847 | 347 | 256,605 |
| Validation | 250 | 57 | 36,411 |
| Test | 360 | 58 | 54,544 |

Training contains 1,834 distinct target texts. The cached Qwen tokenizer measures median input lengths of 8,188 training tokens, 8,187 validation tokens and 8,189 test tokens, with an 8,192-token input ceiling. Training inputs range from 3,784 to 8,192 tokens. The longest complete sequence across all splits contains 8,818 tokens; the training configuration allows 9,216. Supervised counts include the final assistant framing and end tokens; prompt and reference tokens are masked. Recheck lengths with the training model's tokenizer.

| Source region | Train | Validation | Test |
|---|---:|---:|---:|
| Human | 1,642 | 181 | 311 |
| Hai | 166 | 54 | 45 |
| Quarg | 39 | 15 | 4 |

Identity labels group source regions, not individual speakers. Species is marked unspecified because complete scenes can contain multiple characters and were not annotated per speaker. No character-specific speaking instructions or generated targets are included. Human material dominates the release; Quarg test coverage is small.

`metadata.task` is derived from the current conversation prefix. Training has 480 scene openings and 1,367 continuations; validation has 76 openings and 174 continuations; test has 74 openings and 286 continuations. Each task selects its shared minimal instruction before context budgeting.

Context preserves the current encounter, authored location lore and nearby eligible mission history. Older missions and independent writing references fill the remaining input budget. References are separate internally consistent scenes, not one continuous player history. Training has a median preserved core of 839 tokens; independent references supply much of the remaining input. Unknown starting conditions and travel durations are constrained possibilities rather than a reconstructed game save.

All 462 selected missions produced targets, yielding 2,457 examples across the three splits. No selected mission is excluded for an extraction failure. The selection does not cover every mission in the three regions.

Unresolved runtime substitutions remain literal markers under the game-variable policy: 200 training targets, 29 validation targets and 71 test targets contain markers. Player-name and ship-name substitutions use the configured neutral values. No missing game facts are filled with model-generated text.

`train.jsonl`, `validation.jsonl` and `test.jsonl` contain the saved inputs and authored targets. `provenance.json` retains source coordinates, selected context and state witnesses. `config.json` records mission ownership, target selection and sampling settings. `manifest.json` records source and implementation hashes; `licensing/` preserves upstream notices. Load through `SceneDataset.load(root, split)` to verify all published artifacts.

Verification checked split separation, source-span exclusion, task/instruction agreement, token budgets, training mask boundaries and 2,025 distinct state witnesses. Two test examples contain an earlier, separately authored occurrence of their target paragraph in the same conversation. Source-span exclusion preserves these valid repeated passages. No complete target text is shared across splits. Existing development material remains part of these source regions; this release is not a claim of a pristine research test set or measured fine-tuning improvement.
