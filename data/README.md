# Dataset preparation

The maintained preparation entry point is `endless_voices.dataset`. `DatasetBuilder` parses verified game source, constructs state-consistent targets and history, applies mission ownership and the context budget, and publishes one prepared bundle.

```sh
python -m endless_voices.dataset \
  --source data/local/endless-sky-7140eb2a29ce \
  --inventory data/overview/source-statistics.json \
  --config configs/scene-dataset.json \
  --tokenizer /path/to/cached/tokenizer \
  --splits train \
  --output outputs/prepared-scenes
```

`configs/scene-dataset.json` records the existing reviewed mission/conversation scope and split assignments. The configuration contains no quote selectors or per-branch routes. Omitting a mission's conversation list selects all supported inline conversations in that mission. Broader corpus extraction is not enabled by default.

A successful build writes split JSONL files, per-split context and target bundles, state provenance, configuration, code snapshots, hashes and licensing. `SceneDataset.load(output, split)` verifies the saved artifacts and provides stable indexed records. No API calls or model weights are involved; the tokenizer is loaded locally.

Preparation fails instead of publishing a partial dataset when source operations are unsupported, branch exploration exceeds its limit or required context exceeds its budget. The current reviewed scope still contains unsupported random expressions and event/resource operations. The [source audit](evaluation/scene-builder-v1/README.md) currently supports 11 of 32 selected training missions. No complete replacement release has been published.

The existing [101-example release](scene-training-v2/README.md) and earlier reports remain historical evidence. They are not retroactively state-consistent. The old speech extraction and separate prototype preparation scripts have been removed; Git retains their history.

See [the ownership diagram and training objectives](../docs/story-context.md), [ADR 13](../docs/adr/0013-split-datasets-by-source-mission.md) for mission ownership and [ADR 14](../docs/adr/0014-build-scene-examples-through-stateful-dialogue.md) for the stateful builder.
