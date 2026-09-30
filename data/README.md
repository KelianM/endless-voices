# Dataset preparation

The current preliminary dataset is [scene-training-v2](scene-training-v2/README.md), with 101 training examples, 48 validation examples and 27 test examples. Training uses full authored scene continuations and saved source context. Evaluation records remain unchanged from the existing development selection.

Whole source missions are the split unit. Known conversation and scenario variants also stay together. The current validator accepts only mission-level manifests:

```sh
python -m endless_voices.contracts data/scene-training-v2/split-manifest.json
```

Mission eligibility is checked before context sampling. Training histories exclude held-out missions. The context strategy preserves lore, the current encounter and nearby prerequisite missions, then fills the remaining input budget with whole older missions. Targets must not appear in their own input.

## Rebuild the replacement selection

[Replacement annotations](training-replacements-v1/annotations.json) identify nine unused continuations in existing training-only missions. The replacement preparation reads the original reviewed annotations and removes the nine recorded exclusions. This is a data migration input, not a supported conversation-level split mode.

```sh
python scripts/replace_training_examples.py --output outputs/replacement-preparation
python scripts/prepare_continuations.py \
  --manifest outputs/replacement-preparation/split-manifest.json \
  --provenance outputs/replacement-preparation/provenance.json \
  --split train --output outputs/replacement-targets
```

Run `assemble_validation_context.py --split train` with that manifest, provenance and target bundle, the cached game source and a cached tokenizer. The assembler writes source-context drafts and a prerequisite graph. Pass those drafts to `python -m endless_voices.prepare_context` with `configs/context-mission-depth.json`, then export the selected contexts and targets with `scripts/prepare_training_release.py`. Each stage requires a new output directory. No model generation is involved.

The [data contract](../docs/contracts.md) defines record validation. [ADR 13](../docs/adr/0013-split-datasets-by-source-mission.md) records mission splitting; [ADR 11](../docs/adr/0011-use-a-shared-scene-continuation-instruction.md) records the shared task; [ADR 12](../docs/adr/0012-resolve-game-variables-within-their-mission.md) records game variables.

## Scope and evidence

The source selection covers Free Worlds representatives, Republic Navy, mainstream Hai and Quarg. The replacement release does not expand into the broader corpus. Source inventories describe available text, not reviewed training yield.

Preserve source hashes, licensing, reviewed annotations, replacement records and selected-context provenance. Agent review does not imply human approval. Validation is development data, and no adapter training has been performed by dataset preparation.

Historical data and reports remain evidence. The current tools do not reconstruct historical split behavior; use the corresponding Git revision if that becomes necessary.
