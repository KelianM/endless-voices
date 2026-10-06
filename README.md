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

Run commands from the repository root. Dataset payloads, source inventory and upstream
licensing are stored in LFS. Raw source caches, model weights, secrets and run outputs
are ignored. Normal Git contains code, configuration, tests, code documentation,
dataset descriptions and architecture decisions.

## Prepare data

[Dataset preparation](data/README.md) describes the cached-source inputs and build command.
`DatasetBuilder` owns preparation; its example builder and context sampler share a stateful
interpreter. `SceneDataset` provides fixed indexed records to training and benchmarking.
See [component ownership](docs/story-context.md) and [record validation](docs/contracts.md).

The [current dataset](data/dataset/README.md) contains training, validation and test splits.
Dataset sizes, provenance and limitations belong in that README.

## Train and chat

Edit `configs/train.toml` with a model path, prepared dataset directory and split and sequence limit, then run:

```sh
train --config configs/train.toml
chat --config configs/train.toml --adapter outputs/example-adapter
```

The trainer uses Transformers and PEFT LoRA with float32 base weights. It is not a quantized
31B training recipe. Adapter files are saved separately from base weights. Pin the model revision
and record dependencies with each run.

Training uses the tokenizer's chat template. `data.loss = "continuation"` trains the final
assistant passage; `"all"` trains all non-padding tokens. Inputs exceeding `data.max_length`
fail rather than being truncated. The model's tokenizer determines the required sequence limit.

`generate --help` describes saved-response generation. `scripts/view_samples.py --help`
describes the existing sample explorer.

## Benchmark

The benchmark reads a prepared validation dataset so generators and judges share selected context:

```sh
python scripts/benchmark_generators.py prepare \
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
historical experiments are recoverable from Git history.

## Development

```sh
pytest -q
ruff check src scripts tests
```

Lasting decisions are recorded in [ADRs](docs/adr/README.md). See [NOTICE.md](NOTICE.md) for
source-data licensing; full upstream notices travel with the LFS datasets.
