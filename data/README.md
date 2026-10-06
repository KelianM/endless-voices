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
  --splits train validation test \
  --output outputs/prepared-scenes
```

`data/dataset/config.json` records the selected mission/conversation scope and split assignments. The configuration contains no quote selectors or per-branch routes. Omitting a mission's conversation list selects supported inline conversations, named conversations and dialog passages in that mission. An empty conversation list excludes targets while retaining split ownership. Exclusion reasons remain in the configuration.

A successful build writes one JSONL file per split, source and state provenance, configuration, hashes and licensing. The manifest records the implementation Git revision and file hashes; it does not copy Python code. `SceneDataset.load(output, split)` verifies the saved artifacts and provides stable indexed records. No API calls or model weights are involved; the tokenizer is loaded locally.

The [current dataset](dataset/README.md) publishes all three splits. Preparation refuses unsupported operations or context overflow for selected targets rather than publishing partial examples.


See [how the dataset is produced](dataset/README.md#how-this-dataset-is-produced), [ADR 7](../docs/adr/0007-split-datasets-by-source-mission.md) for mission ownership and [ADR 9](../docs/adr/0009-build-examples-from-consistent-game-state.md) for the stateful builder.
