# Context retrieval

Prepared datasets save the full eligible context pool for each example alongside the fixed
sampled prompt. The existing mission-depth sampler remains the baseline for training and
benchmarking. Retrieval tools let an agent choose passages from the same eligible material.

Rebuild a dataset with the [preparation command](../data/README.md), then create a static
passage index with a locally cached embedding model:

```sh
python -m endless_voices.dataset.index_context \
  --dataset outputs/prepared-scenes \
  --output outputs/context.sqlite
```

The default encoder is `Qwen/Qwen3-Embedding-0.6B`, loaded from `data/local/hub`. Supply
`--model` to use a local model directory or another cached Qwen3 embedding model. Indexing
does not download models or call hosted services. Automatic device selection prefers CUDA,
then MPS, then CPU. The command
accepts explicit device, precision, batch size and token-limit settings.

The SQLite index stores every authored passage separately, including player choices, and
retains each passage's complete interpreted parent scene. Repeated scenes share one entry;
different interpretations remain separate entries. Search computes exact cosine similarity
over passages whose parent scenes are eligible for the current encounter. Search results
contain the complete passage and a parent scene ID for reading the surrounding scene.

The encoder follows the [Qwen3 embedding model's pooling and input format](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage):
documents contain the scene heading and authored passage; queries receive an instruction.
The encoder selects the last non-padding token and normalizes the resulting vector. The
default query instruction is:

> Given a scene-writing query, retrieve relevant fictional mission history and writing passages that support continuity, lore, character voice, and style.

Queries use `Instruct: {instruction}\nQuery:{query}`. The Python constructor accepts a
different `query_instruction`. Use the same model files, precision, instruction and token
limit when building and opening an index; the encoder identity includes these settings.

No passage or query is silently truncated. The default limit is 32,768 tokens per passage
or instructed query, within the model's configured capacity. An oversized input fails before
model inference. A smaller `--max-length` can enforce a lower limit; a passage above that
limit needs explicit splitting, and an oversized query needs explicit shortening. Rebuild
indexes produced by the earlier whole-scene format before using passage retrieval.

```python
from endless_voices.dataset.storage import SceneDataset
from endless_voices.retrieval import SceneDatabase, TransformersEmbedder

dataset = SceneDataset.load("outputs/prepared-scenes", "validation")
embedder = TransformersEmbedder.from_pretrained()

with SceneDatabase("outputs/context.sqlite", embedder) as database:
    session = dataset.retrieval_session(0, database)
    messages = session.initial_messages()
    tools = session.tool_schemas
    history = session.call_tool("get_history", {"depth": 4})
    matches = session.call_tool("search_context", {"query": "departure and farewell"})
    if matches:
        scene = session.call_tool("read_context", {"ids": [matches[0]["scene_id"]]})
    trace = session.trace
```

`get_history` returns eligible mission history within a prerequisite distance.
`search_context` searches eligible history and independent writing references; omitting
the query uses the visible encounter. `read_context` accepts a passage `id` or a parent
`scene_id` returned by retrieval and returns the complete passage or scene.
Independent references demonstrate authored writing and do not assert events in the current
playthrough. The session supplies tool schemas and dispatch; the caller runs the model loop,
appends tool results, sets spending limits and saves the retrieval trace.

Eligibility comes from the existing builder and interpreter. Each session can read and rank
only the exact scene variants in its saved pool, even when the static database contains other
splits or branches. Target source paragraphs are rejected throughout the eligible pool before
publication. The index contains only pooled history and references. A passage targeted by
another example can appear in the index, but each session excludes its own target paragraphs
from scoring and reads. Existing interpreter limitations still apply, including the
state used to resolve lore during preparation.

`retrieval.json` deduplicates scene text and records each sample's permitted scene IDs.
Dataset loading verifies the sidecar hash and pool fingerprints. Older prepared datasets
continue to supply fixed prompts; retrieval requires rebuilding those datasets.

The current benchmark still consumes fixed saved prompts. Integrating a model's tool loop
and comparing retrieval budgets are separate runner work. Adaptive retrieval should retain
common judging evidence across generators and report total tokens, latency and cost alongside
answer quality.

## Compare embedding models

`scripts/compare_retrieval_embeddings.py` freezes real interpreter pools and the query plan
in `configs/retrieval-comparison.json` before running either model. The current plan covers
Hai and Quarg writing references plus Acorn Delights mission history. Both models search
every eligible passage in the selected pools.

```sh
python scripts/compare_retrieval_embeddings.py prepare \
  --plan configs/retrieval-comparison.json \
  --config data/dataset/config.json \
  --inventory data/sources/source-statistics.json \
  --source data/local/endless-sky-7140eb2a29ce \
  --output outputs/retrieval-comparison/frozen.json

python scripts/compare_retrieval_embeddings.py run \
  --input outputs/retrieval-comparison/frozen.json \
  --model /path/to/cached/model-snapshot --revision exact-model-revision \
  --device mps --dtype float16 --batch-size 4 \
  --output outputs/retrieval-comparison/model-one
```

Repeat `run` with the second cached model and a different output directory, preserving the
frozen input and encoding settings. `blind --help` describes creating review packets from
the two result files. Reviewers score each of the first three passages for relevance and
choose a preferred result set before opening the separate model key. Timing, source hashes
and exact model revisions accompany the saved results. A small handwritten query set measures
usefulness on those encounters; the results do not establish retrieval accuracy across the game.
