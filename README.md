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

Prepare encounters and eligible context from cached game sources, without loading a model
or tokenizer:

```sh
endless-voices prepare-dataset \
  --source data/local/endless-sky-7140eb2a29ce \
  --inventory data/sources/source-statistics.json \
  --config data/dataset/config.json \
  --splits train validation test \
  --output outputs/prepared-scenes
```

[Dataset inputs](data/README.md) explains source and mission selection. The
[published dataset](data/dataset/README.md) records release coverage and limitations;
rebuild that older bundle before using the current retrieval flow.

Validate a prepared bundle with:

```sh
endless-voices validate outputs/prepared-scenes/manifest.json
```

Add `--tokenizer /path/to/local/tokenizer --max-length 9216` to check complete sequence
lengths against a 9,216-token limit without downloading a tokenizer or model weights.
Validation checks the saved encounter and target; runtime selection checks retrieved context.

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

Set `context.index` in the generation, training or benchmark configuration to this database.
Open the database with the same model and encoding settings used to build it.

`context.mode` selects the context policy:

- `static` fits complete scenes into `context.max_input_tokens`: nearby prerequisite history
  first, then scenes ranked by similarity to the visible encounter. `context.depth` controls
  nearby history. The default input budget is 8,192 tokens.
- `dynamic` lets the model request history, semantic search and scene reads before writing.
  The model emits JSON tool requests until it chooses to finish, then receives the selected
  scenes for the final continuation. There is no retrieval-turn cap; context budgets still
  apply, and hosted requests remain subject to the run’s spend ceiling.

Planet descriptions retain their names and systems. Origin, destination and configured
planet descriptions appear in the initial prompt; planets in the reference scope or named
in visible context and history are also searchable, with interpreted description changes.
Dynamic search excludes scenes already selected, and the selector states when no eligible
mission history exists.

Both policies use the same eligible scene pool. Search returns whole scenes ranked by their
strongest passage; the lower-level retrieval API also supports passage-only results. Runs
save actual prompts and retrieval traces. Prepared datasets do not save sampled prompts. Prerequisite-connected missions must share
one split. Prompts identify the current mission event and its known location without
copying the post-target state.

## Train and chat

Edit `configs/train.toml` with model, dataset and context-index paths, then run:

```sh
endless-voices train --config configs/train.toml
endless-voices chat --config configs/train.toml --adapter outputs/example-adapter
```

The trainer uses Transformers and PEFT LoRA with float32 base weights. It is not a quantized
31B training recipe. Adapter files are saved separately from base weights. Pin the model revision
and record dependencies with each run.

Training selects static context before loading the generator and reserves space for the
authored target within `data.max_length`. Selected messages and traces are saved in
`training_contexts.jsonl`. Training uses the tokenizer's chat template. `data.loss = "continuation"` trains the final
assistant passage; `"all"` trains all non-padding tokens. Inputs exceeding `data.max_length`
fail rather than being truncated. The model's tokenizer determines the required sequence limit.

`endless-voices generate --help` describes saved-response generation. `endless-voices view-samples --help`
describes the existing sample explorer.

## Benchmark

Prepare a benchmark from a rebuilt validation dataset and indexed context:

```sh
endless-voices benchmark prepare \
  --dataset outputs/prepared-scenes --root outputs/benchmark \
  --config configs/benchmark.json --context-mode static
```

Use `--context-mode dynamic` when preparing a separate dynamic run. The run configuration
fixes the mode for subsequent stages. `context.tokenizer` names the cached tokenizer used
to select the common static context; `context.max_input_tokens` sets its 8,192-token budget.
Hosted dynamic budgets are set per model in `context.dynamic_max_input_tokens`, measured
with the same reference tokenizer rather than provider billing token counts. Local
configuration sets `max_input_tokens`; input plus `max_output_tokens` must fit
`context_ceiling`. Larger hosted budgets do not enlarge the local Gemma prompt.

Judges receive the same static evidence in both modes, so a generator's retrieval choices
do not change the judge's reference context. Dynamic requests and returned context are
recorded separately for inspection.

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
