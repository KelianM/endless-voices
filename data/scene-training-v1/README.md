# Preliminary scene-continuation training data

This release contains 92 training examples with source-derived context and complete authored continuations. The shared task is: “Continue the scene with the next passage, keeping the characters and events authentic to the supplied context.” No tone, cadence, invention or speech-only directions are added.

The release starts from 101 reviewed training examples. Four ambiguous continuation boundaries and five examples whose missions overlap held-out splits are excluded in `exclusions.json`. Training references exclude entire missions represented in validation or test. Later training dialogue remains eligible under the writing objective; this is not a chronological knowledge evaluation. The validation set has already been used for development.

Player identity uses Alex Morgan and the ship Copper Finch. Each passage uses its owning mission's static location values. Unknown runtime variables remain literal markers. Forty-one examples contain at least one unresolved marker in their input or target: `<cargo>`, `<npc>`, `<origin>`, `<payment>` or `<stopovers>`. No generic destination or invented payment replaces a missing value.

Gemma's cached tokenizer measures a maximum of 8,396 tokens for a complete training conversation. The longest labelled assistant turn is 405 tokens; the median is 86 tokens. `measurements.json` counts the supplied conversation separately from the final assistant turn, including its chat delimiters. The context selector independently enforces the 8,192-token generation-input budget.

Train on `train.jsonl` with final-assistant-turn loss, the repository default. Set `data.max_length` to at least 8,396 for this tokenizer and remeasure for another tokenizer. The loader rejects overlength examples instead of truncating targets. Model weights and training memory have not been tested by this preparation; the existing float32 Transformers example is not a demonstrated Gemma 31B training configuration.

The manifest records artifact and upstream hashes. Preparation evidence is archived in [training-preparation-v1](../evaluation/training-preparation-v1/README.md). Preserve the accompanying Endless Sky licensing. No dialogue was synthesized, no model was trained, and historical benchmark prompts were not changed.
