# Context retrieval

Prepared datasets save the full eligible context pool for each example alongside the fixed
sampled prompt. The existing mission-depth sampler remains the baseline for training and
benchmarking. Retrieval tools let an agent choose passages from the same eligible material.

Rebuild a dataset with the [preparation command](../data/README.md), then create a static
scene index with a locally cached embedding model:

```sh
python -m endless_voices.dataset.index_context \
  --dataset outputs/prepared-scenes \
  --model /path/to/cached/embedding-model \
  --output outputs/context.sqlite
```

The index stores interpreted scenes and their vectors in SQLite. Repeated scenes share one
entry; different interpreted passages remain separate entries. Indexing does not download
models or call hosted services. Use the same encoder when building and opening the index.
Search computes exact cosine similarity over eligible scene vectors. The supplied adapter
mean-pools the first 512 encoder tokens of each scene, so later passages do not affect ranking;
reads return the full original scene. The Python constructor accepts a different `max_length`;
the indexing command uses the default. Custom embedders can implement another encoding policy.

```python
from endless_voices.dataset.storage import SceneDataset
from endless_voices.retrieval import SceneDatabase, TransformersEmbedder

dataset = SceneDataset.load("outputs/prepared-scenes", "validation")
embedder = TransformersEmbedder.from_pretrained("/path/to/cached/embedding-model")

with SceneDatabase("outputs/context.sqlite", embedder) as database:
    session = dataset.retrieval_session(0, database)
    messages = session.initial_messages()
    tools = session.tool_schemas
    history = session.call_tool("get_history", {"depth": 4})
    matches = session.call_tool("search_context", {"query": "departure and farewell"})
    trace = session.trace
```

`get_history` returns eligible mission history within a prerequisite distance.
`search_context` searches eligible history and independent writing references; omitting
the query uses the visible encounter. `read_context` expands scene IDs returned by retrieval.
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
