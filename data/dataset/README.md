# Prepared scene training dataset

This release contains 163 training examples from all 32 selected missions and 33 conversations. The shared stateful builder produced every example. Each record contains authored context and a complete authored continuation; no model-generated targets are included.

| Identity | Training examples |
|---|---:|
| Free Worlds | 73 |
| Republic | 19 |
| Hai | 59 |
| Quarg | 12 |

The scope matches the selected training missions. Additional examples represent reachable target branches within those conversations. Only the training split is currently published. Validation and test mission assignments remain in the configuration, but those splits have not been built with the current interpreter. Held-out missions are not used as training history.

Load this directory with `SceneDataset.load(root, "train")` and use continuation loss. The cached Qwen tokenizer measures 5,311–8,192 input tokens per example, with a median of 8,187 tokens. Complete training sequences contain 5,449–8,674 tokens, with a median of 8,275 tokens. The example training configuration allows 9,216 tokens per complete sequence. Recheck lengths with the actual training model's tokenizer before training.

Context preserves the current encounter, authored location lore and nearby eligible mission history. Remaining space contains independently resolved conversations from the source files associated with the identity, followed by authored ship, outfit and government descriptions. References are labelled separately from preceding events. The current mission and held-out missions cannot supply reference scenes. Human and Hai inputs reach 8,175–8,192 tokens; Quarg inputs contain 5,311–5,817 tokens from the available eligible dialogue and lore.

`train.jsonl` contains each selected context and final authored target once. `provenance.json` records source coordinates, compatible state witnesses and context-selection evidence without copying dialogue. `config.json` records the build configuration. `manifest.json` records hashes and the implementation Git revision; `licensing/` contains upstream notices. Loading with `SceneDataset.load(root, 'train')` verifies the manifest. All state witnesses and target-only training masks were checked when this dataset was prepared.

Unknown travel duration, starting conditions and runtime payment factors are constrained possibilities, not a reconstructed player save. Prerequisite context samples one compatible event timeline per route from the dataset seed; target extraction still enumerates reachable outcomes within the current mission. Unrelated missions are inactive unless the modeled history activated them. Scheduled events update conditions and retain world changes; applicable planet descriptions update lore. Fleet simulation and arbitrary mission failure handlers are outside this release's scope. Unknown game text substitutions remain explicit markers under the existing variable policy.
