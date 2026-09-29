# Complete source continuations

The 48 validation targets retain authored narration, actions, dialogue and quotation marks from the cached Endless Sky source. The benchmark uses this bundle instead of the speech-only targets in the historical dataset. Target text is not summarized, paraphrased or stripped down to one speaker's quotations.

Each continuation begins at the first annotated target paragraph. Source control flow selects the route through the remaining annotated target anchors. The continuation includes intervening and trailing prose until the next player-choice node, an existing recorded response boundary, or the end of the route. Recorded response boundaries can be another character's reply, as in Tomek's parole exchange. Unknown branches that produce different passages are rejected rather than merged. Boundaries are inherited from existing scene annotations; this change does not claim a newly reviewed segmentation of the whole game.

Source syntax delimiters are removed. Authored paragraph contents, indentation, punctuation and enclosing speech quotes are retained. Paragraphs are joined with a blank line, as in the supplied game context. Runtime variables use the same configured dummy values as the context. The raw target bundle retains the original variable markers.

`targets.json` contains 79 source paragraphs across 48 targets. `manifest.json` hashes the target file and records the source annotation provenance. Each target records its source file hash, revision, paragraph lines and stopping boundary. Sample IDs remain stable scene identifiers inherited from the previous dataset; their historical suffix is not the new target hash.

Benchmark preparation verifies the bundle and copies it into the run. Each restored paragraph is checked against the model input after variable substitution; a target paragraph already present in the context is rejected. The saved reference records can supply the same complete target to training through `Selection.training_messages`, while generation and judging receive only the saved context.

Preparation used only cached source files, without downloads or model calls. The existing 8,192-token context selections passed the overlap checks for all 48 new targets. Historical runs and their targets remain unchanged. No new test-set evaluation or training release is implied.

Rebuild into a new directory with `scripts/prepare_continuations.py --output PATH`. The default benchmark target location is configured in `configs/benchmark.json`.
