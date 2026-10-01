# Prepared scene training dataset

This release contains 163 training examples from all 32 selected missions and 33 conversations. The shared stateful builder produced every example. Each record contains authored context and a complete authored continuation; no model-generated targets are included.

| Identity | Training examples |
|---|---:|
| Free Worlds | 73 |
| Republic | 19 |
| Hai | 59 |
| Quarg | 12 |

The scope matches the selected training missions. Additional examples represent reachable target branches within those conversations. Only the training split is currently published. Validation and test mission assignments remain in the configuration, but those splits have not been built with the current interpreter. Held-out missions are not used as training history.

Load this directory with `SceneDataset.load(root, "train")` and use continuation loss. The cached Qwen tokenizer produces 39–8,192 input tokens per example (median 433) and 88–8,336 tokens per complete training sequence (median 538). The example training configuration allows 9,216 tokens per complete sequence. Recheck length with the actual training model's tokenizer; no adapter training or GPU memory test is implied.

Earlier dialogue from 35 supporting missions appears in the selected context. Supporting missions do not need their own target examples, but held-out missions remain excluded. Of the 163 examples, 60 contain earlier mission dialogue; 14 have at least 4,096 input tokens. The 8,192-token ceiling is a maximum, not a minimum: this sampler follows explicit required mission predecessors and does not fill short examples with unrelated campaign text.

`train.jsonl` contains each selected context and final authored target once. `provenance.json` records source coordinates, compatible state witnesses and context-selection evidence without copying dialogue. `config.json` records the build configuration. `manifest.json` records hashes and the implementation Git revision; `licensing/` contains upstream notices. Loading with `SceneDataset.load(root, 'train')` verifies the manifest. All state witnesses and target-only training masks were checked when this dataset was prepared.

Unknown travel duration, starting conditions and runtime payment factors are constrained possibilities, not a reconstructed player save. Prerequisite context samples one compatible event timeline per route from the dataset seed; target extraction still enumerates reachable outcomes within the current mission. Unrelated missions are inactive unless the modeled history activated them. Scheduled events update conditions and retain world changes; applicable planet descriptions update lore. Fleet simulation and arbitrary mission failure handlers are outside this release's scope. Unknown game text substitutions remain explicit markers under the existing variable policy.
