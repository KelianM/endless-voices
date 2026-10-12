# Compare retrieval embeddings

This experiment compares Qwen3-Embedding-0.6B and Qwen3-Embedding-4B for writing context,
using passage retrieval and passage retrieval with scene expansion. The experiment is a
consumer of the [retrieval system](../../docs/retrieval.md). Run commands from the repository
root after installing the project.

`experiments/retrieval-embeddings/run.py` runs the production `RetrievalSession` against
frozen queries and prepares blinded review packets. The script adds experiment setup,
timing and token counts; encoding, eligibility, ranking and scene expansion remain in
`src/endless_voices/retrieval.py`. The query plan in `experiments/retrieval-embeddings/queries.json` covers
Hai and Quarg references plus Acorn Delights history. Freeze the queries and eligible pools
once, before running either model:

```sh
python experiments/retrieval-embeddings/run.py prepare \
  --plan experiments/retrieval-embeddings/queries.json \
  --config data/dataset/config.json \
  --inventory data/sources/source-statistics.json \
  --source data/local/endless-sky-7140eb2a29ce \
  --output outputs/retrieval-comparison/frozen.json
```

Every run requires an explicit `--mode`: `passage` returns matching passages; `scene` returns
distinct full parent scenes. `--limit` counts the chosen result type. Supply the same cached
`--tokenizer` directory for both models so returned-text counts use a common tokenizer.
The encoder's own tokenizer still determines its embedding inputs.

```sh
python experiments/retrieval-embeddings/run.py run \
  --input outputs/retrieval-comparison/frozen.json \
  --model /path/to/cached/model-one --revision exact-model-revision \
  --tokenizer /path/to/cached/Qwen3-Embedding-0.6B \
  --mode passage --limit 3 --device mps --dtype float16 --batch-size 4 \
  --output outputs/retrieval-comparison/model-one-passages

python experiments/retrieval-embeddings/run.py run \
  --input outputs/retrieval-comparison/frozen.json \
  --model /path/to/cached/model-one --revision exact-model-revision \
  --tokenizer /path/to/cached/Qwen3-Embedding-0.6B \
  --mode scene --limit 3 --device mps --dtype float16 --batch-size 4 \
  --index outputs/retrieval-comparison/model-one-passages/context.sqlite \
  --output outputs/retrieval-comparison/model-one-scenes
```

Without `--index`, the run builds an index from the frozen pools. With `--index`, the run
opens an existing index without re-embedding; the encoder fingerprint must match. Repeat
both commands for the second cached model with separate output directories. Model files and
the common counting tokenizer are loaded locally without downloads.

```sh
python experiments/retrieval-embeddings/run.py blind \
  --input outputs/retrieval-comparison/frozen.json \
  --first outputs/retrieval-comparison/model-one-scenes/results.json \
  --second outputs/retrieval-comparison/model-two-scenes/results.json \
  --output outputs/retrieval-comparison/scene-review
```

The blinding command requires matching modes, frozen inputs, source hashes, encoding settings,
result limits and counting-tokenizer identities. Results missing the explicit format or mode
are rejected. The `review/` directory contains combined and per-query packets with complete
text and source evidence; scores, timing and model identities are omitted. Reviewers score
relevance and choose a result set before opening `key.json` in the parent directory.

Timing, exact model revisions, index hashes and returned-text counts accompany every run.
A scene comparison can return substantially more text than a passage comparison with the
same result limit. A small handwritten query set measures usefulness on those encounters;
the results do not establish retrieval accuracy across the game.

## Recorded comparison

The comparison used ten source-informed queries over 664 passages from 163 interpreted
scenes. Each run returned three results per query. Fresh blinded AI reviews produced:

| Retrieval mode | 0.6B preferred | 4B preferred | Ties |
| --- | ---: | ---: | ---: |
| Passage | 2 queries | 4 queries | 4 queries |
| Scene expansion | 3 queries | 0 queries | 7 queries |

Scene expansion returned identical scenes in identical order for six queries. The expanded
0.6B results contained 13,017 tokens; the 4B results contained 14,387 tokens, measured with
the same tokenizer. These results supported retaining 0.6B with scene expansion in
[PR #19](https://github.com/KelianM/endless-voices/pull/19).

Different reviewers assessed the two modes. The preference counts do not measure a controlled
improvement, general model quality or generated dialogue quality. Original run artifacts
remain under ignored `outputs/retrieval-embedding-comparison/`; the commands above reproduce
retrieval, while another qualitative review may reach different judgments.
