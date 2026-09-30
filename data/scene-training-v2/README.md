# Mission-separated scene training data

This release contains 101 training examples from 33 conversations. The 92 valid examples from scene-training-v1 retain identical model inputs. Nine reviewed, unused source continuations replace four ambiguous targets and five examples from missions represented in held-out splits. No broader corpus expansion was performed.

The dataset's 48 validation and 27 test records are unchanged copies of the original records. `split-manifest.json` declares mission-level splitting and validates all three sets together. No primary source mission crosses splits. Validation remains development data; the unchanged evaluation records retain their historical presentation and are not a new benchmark run.

Training targets preserve narration and full authored paragraphs. All examples use the shared minimal next-passage instruction and the same depth-four, 8,192-token context strategy. Held-out missions are removed before context sampling. The nine new targets have no source-paragraph overlap with previous training targets. No target paragraph appears in its own input.

Gemma's tokenizer measures 10,559 labelled assistant-turn tokens across the training set. The maximum complete conversation is 8,396 tokens, and the longest assistant turn is 405 tokens. Runtime markers remain explicit in 44 examples, counting inputs and targets. Mission-scoped variables and player identity follow the previous release.

Use `train.jsonl` with final-assistant-turn loss. Tokenize again for another model and set the complete-conversation limit accordingly; training-memory fit has not been tested. The dataset remains suitable for a preliminary experiment rather than broad conclusions about fine-tuning.

[Replacement annotations and verification](../training-replacements-v1/README.md) record the source review, exclusions and preserved split boundaries. The manifest and split manifest carry hashes; preserve the accompanying licensing. No API generation, local inference or training was run for this release.
