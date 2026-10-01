# Dataset preparation

`src/endless_voices/dataset/` contains the implementation. `data/dataset/` is its built output.
`data/sources/source-statistics.json` pins the game-source revision and file hashes used by
`scripts/fetch_sources.py`; raw sources and model caches remain in ignored `data/local/`.
Experiment outputs belong in ignored `outputs/`.

The maintained preparation entry point is `endless_voices.dataset`. `DatasetBuilder` parses verified game source, constructs state-consistent targets and history, applies mission ownership and the context budget, and publishes one prepared bundle.

```sh
python -m endless_voices.dataset \
  --source data/local/endless-sky-7140eb2a29ce \
  --inventory data/sources/source-statistics.json \
  --config data/dataset/config.json \
  --tokenizer /path/to/cached/tokenizer \
  --splits train \
  --output outputs/prepared-scenes
```

`data/dataset/config.json` records the existing reviewed mission/conversation scope and split assignments. The configuration contains no quote selectors or per-branch routes. Omitting a mission's conversation list selects all supported inline conversations in that mission. Broader corpus extraction is not enabled by default.

A successful build writes one JSONL file per split, source and state provenance, configuration, hashes and licensing. The manifest records the implementation Git revision and file hashes; it does not copy Python code. `SceneDataset.load(output, split)` verifies the saved artifacts and provides stable indexed records. No API calls or model weights are involved; the tokenizer is loaded locally.

The [current training release](dataset/README.md) contains 163 examples from all 32 selected training missions. The build verifies state witnesses, split boundaries and training loss masks. Preparation still refuses unsupported operations or context overflow rather than publishing a partial dataset.


See [the ownership diagram and training objectives](../docs/story-context.md), [ADR 10](../docs/adr/0010-split-datasets-by-source-mission.md) for mission ownership and [ADR 11](../docs/adr/0011-build-scene-examples-through-stateful-dialogue.md) for the stateful builder.
