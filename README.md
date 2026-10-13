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

Generation uses one retrieval loop for local and hosted models. Models call native tools
for history, semantic search and scene reads, then continue with the results in their
conversation. The same instructions and eligible evidence apply to every model. There is
no retrieval-turn cap; context budgets and hosted spending budgets still apply. Local MLX
runs use the tokenizer's tool parser; Transformers runs support Qwen's JSON-in-tool-tags
format. Unsupported formats fail explicitly.

Training and benchmark judges receive context selected beforehand: nearby prerequisite
history first, then scenes ranked by similarity. `context.depth` controls nearby history.

Planet descriptions retain their names and systems. Origin, destination and configured
planet descriptions appear in the initial prompt; planets in the reference scope or named
in visible context and history are also searchable, with interpreted description changes.
Dynamic search excludes scenes already selected, and the prompt states when no eligible
mission history exists.

All retrieval uses the same eligible scene pool. Search returns whole scenes ranked by their
strongest passage; the lower-level retrieval API also supports passage-only results. Runs
save actual prompts and retrieval traces. Prepared datasets do not save sampled prompts. Prerequisite-connected missions must share
one split. Prompts identify the current mission event and its known location without
copying the post-target state. A compact prerequisite map and chronological list of earlier
scene phases and IDs are provided upfront; `read_context` retrieves their text.

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

`endless-voices generate --help` describes saved-response generation.

Browse prepared scenes and play continuations in the local reader:

```sh
endless-voices view-samples --serve \
  --manifest outputs/prepared-scenes/manifest.json --split all \
  --config configs/benchmark.json \
  --run outputs/benchmark/gpt-6-luna
```

`--run` is optional and repeatable. Each folder can be a generation run or a benchmark's
model folder for the same dataset. Switch to **Play a live continuation**, start before
the target or after a saved response, then choose a suggested reply or type your own.
Suggestions are generic prompts, not authored game choices. The original target stays
out of live requests. Retrieval eligibility stays fixed at the scene's starting state;
the reader does not advance game conditions or mission chains.

The selector loads text-model IDs from OpenAI and Anthropic using `.env`, plus downloaded
models in `local_configs`. Provider listings do not guarantee native tool support; rejected
requests appear as errors. Local MLX models use `data/local/screen-env/bin/python` (override
with `local_python` in the configuration), loading one model at a time. Local input budgets
come from their existing configurations; Gemma stays at 8192. Hosted budgets are editable
in the reader and default to `context.hosted_max_input_tokens`, or 65536 for unconfigured IDs.

Hosted replies are billed to your API account. `--budget` optionally caps estimated spending
for models with known prices; other models report an unavailable cost estimate. Conversation
and request logs go to a new `outputs/play-*` folder, or `--live-output`. Browser conversations
last until a restart or page reload; download the transcript to keep it. To export a reader
without live generation, use `--output samples.html` instead of `--serve`.

## Benchmark

Prepare a benchmark from a rebuilt validation dataset and indexed context:

```sh
endless-voices benchmark prepare \
  --dataset outputs/prepared-scenes --root outputs/benchmark \
  --config configs/benchmark.json
```

In the benchmark configuration, `context.tokenizer` and `context.max_input_tokens` control
selection of common judge evidence. Hosted input budgets are configured per model in
`context.hosted_max_input_tokens`, measured with that reference tokenizer rather than
provider billing token counts. Native requests include tool schemas, results and reasoning
in the budget. Local configurations set `max_input_tokens`; Gemma keeps 8,192 input tokens,
and input plus `max_output_tokens` must fit `context_ceiling`. Larger hosted budgets do not
enlarge the local prompt.

Judges receive common selected evidence, so a generator's retrieval choices
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
