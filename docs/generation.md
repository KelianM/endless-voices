# Generate comparable continuations

`endless-generate` runs an existing causal language model on curated conversation samples,
optionally with an existing LoRA adapter. Each sample produces one continuation after its final
user message. The command never trains an adapter or calls an evaluator.

The runner uses `evaluation_messages` from the [conversation contract](contracts.md). Profiles,
lore, scene assumptions and earlier authored replies remain unchanged. The original final reply,
sample metadata and private assessment fields are never passed to the tokenizer or model.
Generation uses the checkpoint's native chat template with an assistant generation prompt.

## Run locally

Install the project with `python -m pip install -e '.[dev]'`. From the repository root:

```sh
endless-generate --config configs/generate.toml \
  --manifest data/pilot-v1/samples/manifest.json \
  --sample-ids data/evaluation/smoke-sample-ids.json \
  --device mps --dtype float16 \
  --max-context-tokens 4096 --max-new-tokens 512 \
  --output outputs/qwen3-4b-validation-smoke
```

`python -m endless_voices.generate` is equivalent. The checked-in model config selects
`Qwen/Qwen3-4B-Instruct-2507` at commit `cdbee75f17c01a7cc42f958dc650907174af0554` for
infrastructure validation, not as the final experiment's chosen model. The four selected
validation samples are described in the [evaluation instructions](../data/evaluation/README.md).
The test split is reserved for a later frozen comparison.

The first run downloads approximately 8 GB of model weights into `data/local/hub/`. Add
`--offline` to require cached files. No remote model code is enabled; model and adapter weights
must be in Safetensors format. Local model directories are supported and hashed. For a different
checkpoint, provide a TOML file with `[model]`, `name_or_path`, and a full 40-character Hub commit
in `revision`. Floating Hub branches and tags are rejected. A local directory's content hashes,
rather than a claimed Hub revision, identify its contents. Model compatibility remains subject
to the installed Transformers version, causal-model architecture, tokenizer and available hardware.

The CLI defaults to greedy decoding, one sample at a time, a 4,096-token total context limit,
a 512-token output cap, float32 weights, and automatic CUDA/MPS/CPU selection. Use explicit
`--device` and `--dtype` for comparisons. The example uses float16 to fit the 4B checkpoint more
comfortably on a 24 GiB Mac. The existing training and chat commands still use float32.
Inference fitting in memory does not demonstrate that training will fit or run at a useful speed.

`--temperature 0.7 --top-p 0.9` enables sampling. The runner disables model-specific generation
default overrides and saves the full generation configuration. Greedy decoding uses one beam.
Sampling uses no top-k filter. Every sample gets a stable 32-bit seed derived from SHA-256 of
`<run seed>:<sample ID>`; changing selection order does not change a sample's seed. The default
run seed is 42. Pinning inputs, settings and software supports repetition, but the runner does
not promise bit-for-bit equality across devices, library versions or nondeterministic kernels.

## Select samples and compare an adapter

The complete manifest is validated for hashes, schema, duplicate IDs and conversation/scenario
split boundaries before generation. `--split` defaults to `validation`; `train` and `test` require
explicit selection. The validator reads all splits structurally, but only the selected split is
eligible for generation. No sample text from the test split is used for development generation.

Without `--sample-ids`, selection follows the declared shard order and physical record order.
`--sample-ids` takes a nonempty JSON list of unique IDs from the chosen split and preserves that
order. `--limit N` selects the first N entries after selection. Unknown, duplicate or wrong-split
IDs fail preflight. The saved `sample-ids.json` reproduces the exact selection.

Run each model condition separately, using the same configuration and selection:

```sh
endless-generate --config configs/generate.toml \
  --manifest data/pilot-v1/samples/manifest.json \
  --sample-ids outputs/qwen3-4b-validation-smoke/sample-ids.json \
  --adapter outputs/existing-adapter --device mps --dtype float16 \
  --max-context-tokens 4096 --max-new-tokens 512 \
  --output outputs/qwen3-4b-adapter-validation
```

A Hub adapter additionally requires `--adapter-revision` with its full commit. The runner loads
the base tokenizer for both conditions. If the adapter includes a tokenizer, its vocabulary,
special tokens, chat template and tokenizer backend must match. The adapter must declare the
same starting checkpoint; a declared revision must match too. Project `run_config.json` files
provide additional training provenance when present. Conflicting declarations are rejected.
An adapter without a recorded training revision has `revision_verified: false`; loading
successfully cannot prove which weights were used for its training. Retain the adapter's original
training evidence before interpreting an experiment.

Compare dataset hashes, selected IDs, model/tokenizer identities, precision, generation settings,
and per-sample `messages_sha256` and `input_ids_sha256` before treating two runs as paired.
The runner does not score, join or judge the conditions; those operations belong to issue #9.

## Context and failure handling

The effective context ceiling is the minimum of the requested limit, tokenizer limit and model
`max_position_embeddings` when supplied. The full tokenized prompt plus the requested output
budget must fit. The runner never truncates a prompt, removes history or reduces the output
budget. A template that omits or transforms authored message text is rejected. Unsupported roles,
tokenization errors, generation exceptions and empty replies produce explicit failure records.
The template check requires verbatim message text in order; a template that rewrites or escapes
message content is deliberately unsupported.

A reply reaching the output cap without an EOS token is recorded as `OutputLimit`, with its
partial text and token IDs retained. The partial reply is not counted as a completed response.
A normal EOS completion has `finish_reason: "eos"`; a capped reply has `finish_reason: "length"`.

An existing output directory is always rejected, including an empty directory. There is no resume
or overwrite mode. Config, selection and manifest errors fail preflight with a nonzero exit and
an error on stderr, before creating a run. After selection, setup failures produce a failed record
for every selected sample. Sample failures do not silently remove samples or stop later attempts.
A caught Ctrl-C marks remaining samples failed and the run interrupted. A process kill, power
loss or disk-write failure can leave a `running` manifest and missing response rows: use the saved
selection to identify unfinished samples, and never treat that directory as a complete run.

Exit status is 0 only when every selected sample completes, 1 for failure, and 130 for a caught
interrupt. The command prints each sample ID and status as it finishes.

## Saved format, version 1

Each run directory contains UTF-8 JSON and JSONL. JSONL files hold one JSON object per line.
Output files are organizer artifacts, not blinded judge inputs: keep condition labels out of
later judge requests.

| File | Contents |
| --- | --- |
| `run.json` | Schema version, run state/times/counts, selection, dataset manifest and hash, requested config/arguments, model/tokenizer/adapter identities and file hashes, full effective generation config, context ceiling, execution details and output hashes |
| `sample-ids.json` | Ordered JSON list of selected sample IDs, reusable with `--sample-ids` |
| `prompts.jsonl` | Attempted sample ID, model-visible messages, message/rendered-prompt hashes, and token IDs/count/hash when tokenization succeeds; no final target or evaluator fields |
| `responses.jsonl` | One row per selected ID on normal completion or a caught failure, including unsuccessful samples |
| `tokenizer/` | Saved tokenizer including the effective pad token and chat template |

`run.json.status` is `running`, `complete`, `completed_with_failures`, `failed` (run-level failure),
or `interrupted`. Counts separate `ok` and `failed`. Dataset provenance includes the full manifest,
its SHA-256, dataset version, split, exact selection and all declared split-file hashes.
Model and adapter provenance includes source, immutable Hub revision or local content hashes.
Execution provenance includes Python, OS, device, precision, library versions, attention backend,
relevant environment settings, Git commit/dirty flag and hashes of the package's Python sources.
The hashes detect source changes; they do not archive uncommitted code. Preserve code and model
artifacts alongside the saved run when reproduction matters.

Response rows always contain `sample_id`, `status` (`ok` or `failed`) and `response` (text or null).
Attempted generations also contain a seed, input count/hashes, elapsed seconds, output token IDs
and count, and finish reason when available. Failures add `error` with `stage` (`prompt`,
`generation`, or `run`), exception/type name and message. Not-yet-reached fields are absent rather
than invented. Setup failures may have no prompt rows. Caught interruption can leave a prompt
row for a sample whose response row reports interruption.

Completed manifests hash all output artifacts except `run.json` itself. Prompt and response rows
are flushed after each attempt; the manifest is atomically replaced as its state advances.
Targets remain in the versioned dataset for issue #9 to join by sample ID. Generated output text
is decoded with special tokens removed and leading/trailing whitespace stripped; raw generated
token IDs are retained. Keep source attribution and licensing with any redistributed data-derived
run artifacts; the [dataset notice](../NOTICE.md) identifies the original material.

## Validation and limits

Offline tests cover target/private-field withholding, authored history, identical base/adapter
inputs, incompatible adapters and templates, exact context boundaries, incomplete/empty replies,
setup failures, interruption, selection, provenance and overwrite protection. A tiny local model
and a zero-initialized LoRA adapter exercise the actual CLI without downloading weights. That
adapter exists only in a temporary test directory and is not a trained project adapter.

Run `ruff check .` and `pytest`. A real-model validation smoke run checks device execution and
artifact collection. Neither the tiny model nor a handful of real outputs establish model
quality, authenticity, evaluator validity or a benefit from fine-tuning. No scoring, judge calls,
training, quantization, retrieval, free-running dialogue or distributed orchestration is included.
