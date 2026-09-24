# Endless Voices

A small experimental fine-tuning project inspired by [Endless Sky](https://endless-sky.github.io/),
an open-source space exploration game whose alien species and factions have distinct histories,
cultures, attitudes, and ways of speaking.

The idea is to make small open-weight language models convincingly inhabit those identities in
conversation: what a group knows, what it values, how it behaves, and how it speaks. Eventually a
user could select a race or faction and converse with its representative. This initial draft is a
command-line foundation for learning dataset preparation, supervised fine-tuning, PyTorch,
Transformers, PEFT/LoRA, model loading, and inference through a tangible, playful experiment.

The central question is: **how much does fine-tuning improve a model's ability to maintain a
distinct fictional identity compared with prompting alone?** Later experiments can compare the
same base checkpoint with no identity context, with a carefully written identity prompt, and with
a LoRA adapter. Here, “base” means the starting checkpoint, which may already be instruction-tuned.
These are experiment directions, not demonstrated results.

## What is here

```text
src/endless_voices/         Training, chat, generation and dataset validation
scripts/                   Source fetching, preparation and dataset assembly
tests/                     Offline tests and format fixtures
configs/                   Model and training settings
docs/contracts.md          Conversation format and validator reference
docs/adr/                  Architecture decisions
data/README.md             Start here to reconstruct or annotate the dataset
data/pilot-v1/              Versioned samples, annotations and evidence (Git LFS)
data/source-review/        Source-selection policy and historical review
data/evaluation/           Evaluation protocol and calibration evidence
data/overview/             Upstream source inventory and statistics
```

There is no frontend, general game-data ingestion, synthetic data generator, or scoring runner.
Each training run writes an independent adapter directory. You can use one per faction, a shared
adapter, or another dataset organization without changing the code: faction names are not built
into the loader or model.

## Pilot data curation

The [first conversation dataset](data/README.md) covers Free Worlds representatives,
Republic Navy, mainstream Hai and Quarg. Reviewed annotations record speaker attribution,
branch routes, profiles and selected lore. Deterministic preparation builds original-speech
samples into Git LFS-versioned train, validation and test files under `data/pilot-v1/`, with frozen hashes,
source provenance, agent review evidence and attribution. The [dataset overview](data/overview/README.md)
distinguishes the available source corpus from the selected coverage. Install Git LFS, then run
`git lfs install --local` and `git lfs pull` to fetch the committed release payload.

## Setup

Use Python **3.11–3.13** (3.12 recommended) and a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run commands from the repository root. The dependency ranges deliberately use Transformers 4.x;
this draft does not claim compatibility with 5.x. For a repeatable experiment, save
`python -m pip freeze` alongside your run and pin `model.revision` to a Hub commit.

Edit `configs/train.toml` to supply a Hugging Face model ID or local model directory. Choose a
small decoder-only causal language model supported by Transformers and PEFT with a tokenizer
chat template that accepts your message roles, including `system` if used. No model is bundled or
downloaded until you run a command. Model licenses, access requirements, and memory needs differ.
For gated models, authenticate with Hugging Face first. Remote model code is not enabled.

Training and interactive chat load the full base model in float32 for a simple CPU/MPS starting point. This is
**LoRA, not quantized LoRA**: only adapter parameters train, but the base weights still occupy
memory. Start with a small model, short sequences, and batch size one. CUDA is also supported.

## Conversations

Supply UTF-8 JSONL: one JSON object per line. A readable version of the format is:

```json
{
  "messages": [
    {"role": "system", "content": "The identity and speaking style to adopt."},
    {"role": "user", "content": "A question or conversational situation."},
    {"role": "assistant", "content": "The desired in-character response."}
  ],
  "metadata": {"identity": "your-faction", "source": "provenance or author"}
}
```

The optional system message must come first, followed by one or more complete user/assistant
pairs. Text must be nonempty. `metadata` is optional and ignored by training; use it to keep track
of sources and identity labels. Put private or larger local datasets in `data/local/` (gitignored)
and change `data.path`. Paths in configuration are relative to the working directory.

For curated train/validation/test samples, use the shared
[versioned data contracts](docs/contracts.md). Validate a complete split manifest offline with
`python -m endless_voices.contracts tests/fixtures/contracts/manifest.json`. This reports
structural coverage and checks metadata, hashes, IDs, and known conversation/scenario split boundaries.

The single example is an **invented archivist**, not a claim about an Endless Sky species, and is
only suitable for checking the pipeline. Meaningful identity learning needs a larger, carefully
reviewed dataset spanning different situations. Keep provenance and permissions for future game
text or generated examples; this repository includes the curated dialogue release, but not the raw upstream corpus or game assets.

The tokenizer's native chat template formats each conversation. This first implementation uses
causal language-modelling loss on **all non-padding tokens**, including system and user text;
it does not implement assistant-only loss. Padding is masked by attention position so real EOS
tokens remain training targets, even when EOS doubles as PAD. Conversations exceeding
`data.max_length` raise an error instead of silently discarding the desired response. Set that
limit within your model's supported context window.

## Train

After selecting a model and supplying data:

```bash
train --config configs/train.toml
```

The config exposes ordinary `TrainingArguments` settings and PEFT LoRA settings.
`target_modules = "all-linear"` is a convenient starting point; select explicit layer names if
your chosen architecture needs them. Trainer selects CUDA, MPS, or CPU as available; set
`training.use_cpu = true` to force CPU. Mixed precision and memory pinning are disabled in this
minimal draft. If an MPS operation is unsupported, try starting the process with
`PYTORCH_ENABLE_MPS_FALLBACK=1`, or use CPU. Fallback can be slow and does not solve insufficient
memory.

The final output contains PEFT adapter weights/config, a saved tokenizer, and `run_config.json`.
It does not contain a copy of the base model. Use the **same base checkpoint and revision** when
loading that adapter. Choose a new `training.output_dir` for each run; nonempty directories are
rejected to avoid overwriting experiments. This draft does not implement checkpoint resumption
or validation metrics. Training loss alone is not evidence of a believable fictional identity.

## Chat

Use the same config for model identity in each comparison:

```bash
# Starting checkpoint without an identity prompt
chat --config configs/train.toml

# Prompt-only condition
chat --config configs/train.toml --system "You are a cautious alien archivist."

# Fine-tuned adapter, with an optional matching identity prompt
chat --config configs/train.toml --adapter outputs/example-adapter \
  --system "You are a cautious alien archivist."
```

Use `--system-file path/to/identity.txt` for longer prompts, `--device cpu` to force CPU, and
`--temperature 0` for greedy decoding. `/reset` clears the conversation but retains the system
prompt; `/quit` exits. History is kept in memory only. Requests exceeding the configured context
budget are rejected; reset or shorten the conversation. Adjust `--max-context-tokens` (default
2048) and `--max-new-tokens` (default 128) for your model. Adapter loading expects a tokenizer
saved alongside the adapter, as the training command does.

The first comparison uses the same identity instructions, selected lore, and authored history
for the base model and adapted model. Evaluation withholds the final assistant response and
generates one answer. Entire conversations and known scenario variants stay in one split.
The [authenticity protocol](data/evaluation/README.md) compares generated replies against original
game continuations in blinded pairs. Its small development calibration pack has a
completed initial human review, including controls; no evaluator reliability or model-quality
result is claimed. The generation command below saves continuations; judging and comparison
reports remain future work. Fixed-history evaluation does not establish persistence through a
model’s own unfolding conversation. See the
[data contract](docs/contracts.md) and [sample-format decision](docs/adr/0003-use-one-conversation-format-across-splits.md).

## Browse conversation samples

Build a standalone reader for the 101 training samples:

```bash
python scripts/view_samples.py --output outputs/training-samples.html
open outputs/training-samples.html
```

The HTML file works offline in a browser without a server or model download. Filter by identity,
search sample IDs, roles or topics, and use Previous/Next or the left/right arrow keys outside
form controls. Each sample shows authored history and the final target, with expandable model
instructions, lore and source metadata.

Use `--split validation` for validation samples. Held-out content is excluded from the default
export; `--split test` or `--split all` explicitly includes it. The complete manifest is still
validated across splits. Review of held-out content must not drive development or tuning.

Add a saved generation run to compare available model replies with their targets:

```bash
python scripts/view_samples.py --split validation \
  --run outputs/qwen3-4b-validation-cleaned-v1 \
  --output outputs/validation-responses.html
open outputs/validation-responses.html
```

Replace the run path with any compatible run directory. Missing responses, failures and samples
not selected for the run remain distinct. The reader checks the dataset manifest and response
hash when recorded. Choose a new output filename for each export; existing files are protected.
The reader displays saved data and does not generate, score, edit or approve samples. It embeds
the selected source text and target responses, so it is an organizer's reading copy,
not a blinded judge input. Keep the [dataset attribution and licensing](NOTICE.md) with shared copies.

## Generate comparable responses

Generate one final reply per selected validation sample, retaining authored history:

```bash
generate --config configs/generate.toml \
  --manifest data/pilot-v1/samples/manifest.json \
  --sample-ids data/evaluation/smoke-sample-ids.json \
  --device mps --dtype float16 --output outputs/qwen3-validation
```

The example downloads approximately 8 GB of pinned Qwen3-4B-Instruct-2507 weights into
`data/local/hub/`. Add `--offline` to require cached weights. Change `[model]` in the config to
use another checkpoint; Hub models require a full commit revision, while local directories are
identified by file hashes. Inference fitting in memory does not establish that training will fit.

Selection defaults to validation. `--sample-ids` accepts an ordered JSON list of IDs;
`--limit N` limits the selection. Reserve the test split for the final comparison.
Run `generate --help` for all options and defaults.

For an adapted condition, repeat the command with `--adapter path/to/adapter` and a new output
directory. Hub adapters also require `--adapter-revision`. Both conditions use the base tokenizer;
conflicting adapter checkpoint or tokenizer declarations are rejected. An adapter with no recorded
training revision is marked `revision_verified: false`. Compare dataset hashes, selected IDs,
prompt/token hashes and generation settings before treating runs as paired.

Generation retains authored history and withholds the final target and private metadata.
The full templated prompt plus output budget must fit the requested and model context limits.
Overlong inputs and templates that alter authored text fail without truncation. Defaults are greedy
decoding, 4,096 total tokens and at most 512 output tokens. Seeds are stable per sample;
identical outputs across hardware or library versions are not guaranteed.

Each run requires a fresh output directory and saves:

| File | Contents |
| --- | --- |
| `run.json` | Run status/counts, dataset hashes, model/tokenizer/adapter identities, effective generation settings, software/device/code provenance and artifact hashes |
| `sample-ids.json` | Ordered selection, reusable with `--sample-ids` |
| `prompts.jsonl` | Model-visible messages, prompt hashes and token IDs; no targets or evaluator metadata |
| `responses.jsonl` | One result per selected ID, including explicit failures |
| `tokenizer/` | Effective tokenizer and chat template |

Response rows contain `sample_id`, `status` (`ok` or `failed`) and `response` (text or null).
Attempted generations also record timing, seed, token IDs/counts and finish reason. Failures include
an `error` with stage, type and message. Reaching the output cap is an `OutputLimit` failure;
the partial response is retained. Targets remain in the dataset and can be joined by sample ID.

Exit codes are 0 for all samples completed, 1 for failure and 130 for a caught interrupt.
A hard process kill can leave a `running` manifest and missing rows; check the saved selection
before using a run. Keep [source attribution](NOTICE.md) with shared outputs. Smoke runs check
execution, not model quality. Generation performs no training or scoring.

## Development

```bash
ruff check .
pytest
```

Tests use a tiny randomly initialized model and local tokenizer; they do not download a model.
They validate the data boundary, padding labels, and a CPU LoRA train/save/reload/generate cycle,
not model quality or hardware performance.

Implementation references: [Transformers chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating),
[PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), and
[Transformers on Apple Silicon](https://huggingface.co/docs/transformers/v4.57.1/perf_train_special).

## Licensing

[Dataset licensing notice](NOTICE.md) identifies the upstream license and attribution bundle.
