# Endless Voices

Dataset preparation, LoRA training and blinded response evaluation for authored
[Endless Sky](https://endless-sky.github.io/) dialogue.

## Setup

Use Python 3.11–3.13 and Git LFS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
git lfs install --local
git lfs pull
```

Use `endless-voices --help` to list supported commands and
`endless-voices <command> --help` for command options. `python -m endless_voices` uses
the same CLI. Relative configuration and data paths resolve from the working directory;
the examples below run from the repository root. Dataset payloads, source inventory and upstream
licensing are stored in LFS. Raw source caches, model weights, secrets and run outputs
are ignored. Normal Git contains code, configuration, tests, code documentation,
dataset descriptions and architecture decisions.

## Prepare data

[Dataset preparation](data/README.md) describes the cached-source inputs and build command.
The [current dataset](data/dataset/README.md) contains training, validation and test splits.
Dataset sizes, provenance and limitations belong in that README.

Validate a prepared bundle with:

```sh
endless-voices validate outputs/prepared-scenes/manifest.json
```

Add `--tokenizer /path/to/local/tokenizer --max-length 9216` to check complete sequence
lengths against a 9,216-token limit without downloading a tokenizer or model weights.

## Retrieve context

Build a passage index from a prepared dataset:

```sh
endless-voices index-context \
  --dataset outputs/prepared-scenes \
  --output outputs/context.sqlite
```

The default encoder is `Qwen/Qwen3-Embedding-0.6B`, which must already be cached in
`data/local/hub`. Use `--model` for another cached Qwen3 embedding model or local model
directory. Indexing does not download models. Passages and queries exceeding the encoder's
token limit fail without truncation. See `index-context --help` for device and encoding options.

Create a retrieval session for an example:

```python
from endless_voices.dataset.storage import SceneDataset
from endless_voices.retrieval import SceneDatabase, TransformersEmbedder

dataset = SceneDataset.load("outputs/prepared-scenes", "validation")
embedder = TransformersEmbedder.from_pretrained()

with SceneDatabase("outputs/context.sqlite", embedder) as database:
    session = dataset.retrieval_session(0, database)
    messages = session.initial_messages()
    tools = session.tool_schemas
    matches = session.call_tool("search_context", {"query": "departure and farewell"})
    trace = session.trace
```

`search_context` returns distinct full scenes by default; `expand_scenes=False` returns
passages. The result limit counts the chosen result type. Omit the query to search using
the visible encounter. `get_history` takes a prerequisite `depth`; `read_context` takes
passage or scene `ids`. Results distinguish mission history from independent writing references.

Open an index with the same model files and encoding settings used to build it. The caller
runs the model's tool loop, appends results, enforces token and spending budgets, and saves
the trace. Generation and benchmarking currently use fixed saved prompts.

## Train and chat

Edit `configs/train.toml` with a model path, prepared dataset directory and split and sequence limit, then run:

```sh
endless-voices train --config configs/train.toml
endless-voices chat --config configs/train.toml --adapter outputs/example-adapter
```

The trainer uses Transformers and PEFT LoRA with float32 base weights. It is not a quantized
31B training recipe. Adapter files are saved separately from base weights. Pin the model revision
and record dependencies with each run.

Training uses the tokenizer's chat template. `data.loss = "continuation"` trains the final
assistant passage; `"all"` trains all non-padding tokens. Inputs exceeding `data.max_length`
fail rather than being truncated. The model's tokenizer determines the required sequence limit.

`endless-voices generate --help` describes saved-response generation. `endless-voices view-samples --help`
describes the existing sample explorer.

## Benchmark

The benchmark reads a prepared validation dataset so generators and judges share selected context:

```sh
endless-voices benchmark prepare \
  --dataset outputs/prepared-scenes --root outputs/benchmark \
  --config configs/benchmark.json
```

The same command exposes `preflight`, `local-smoke`, `local`, `hosted`, `export`, `judge` and
`report` stages. Inspect `--help` and the configuration before running model stages. Local MLX
inference uses a separately installed MLX runtime; ordinary offline tests do not require it.
Hosted runners read provider keys from ignored `.env`, preserve requests and
results, enforce configured budgets and retain failures without silently retrying them.

[Evaluation behavior](docs/evaluation.md) describes blinding and reporting. The judge instruction
is maintained in `configs/judge-instructions.md`. Experiment outputs stay in ignored `outputs/`;
recorded investigations live in [experiments/](experiments/README.md).

## Development

```sh
pytest -q
ruff check src tests
```

Command parsing lives in `src/endless_voices/cli/`; reusable behavior lives in the rest
of the package. Ad-hoc investigation runners, inputs and findings live in `experiments/`.

The [design overview](docs/design.md) maps the system and its boundaries to the code.
Lasting decisions are recorded in [ADRs](docs/adr/README.md). See [NOTICE.md](NOTICE.md) for
source-data licensing; full upstream notices travel with the LFS datasets.
