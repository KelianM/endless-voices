# Prepared scene training dataset

This release contains 163 training examples from all 32 selected missions and 33 conversations. The shared stateful builder produced every example. Each record contains authored context and a complete authored continuation; no model-generated targets are included.

| Identity | Training examples |
|---|---:|
| Free Worlds | 73 |
| Republic | 19 |
| Hai | 59 |
| Quarg | 12 |

The scope matches the selected training missions. Additional examples represent reachable target branches within those conversations. Validation and test data were not regenerated, expanded or used as training history.

Use `train.jsonl` with continuation loss. The cached Qwen tokenizer produces 92–1,559 tokens per complete training sequence, with a median of 492 tokens. A 2,048-token training limit covers this release with that tokenizer. Recheck length with the actual training model's tokenizer; no adapter training or GPU memory test is implied.

The artifact includes shared contexts, target paragraphs, source coordinates, compatible state witnesses, code snapshots, configuration, hashes and upstream licensing. Loading with `SceneDataset.load(root, 'train')` verifies the manifest. The [verification report](../evaluation/scene-builder-v2/README.md) checks every state witness and training loss mask.

Unknown travel duration, starting conditions and runtime payment factors are constrained possibilities, not a reconstructed player save. Unrelated missions are inactive unless the modeled history activated them. Scheduled events update conditions and retain world changes; applicable planet descriptions update lore. Fleet simulation and arbitrary mission failure handlers are outside this release's scope. Unknown game text substitutions remain explicit markers under the existing variable policy.
