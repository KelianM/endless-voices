# Dataset contracts, version 1

These interfaces define future curated data; they do not release a corpus or frozen benchmark.
The small [contract fixtures](../tests/fixtures/contracts/) are invented plumbing examples,
not Endless Sky canon, benchmark authoring, or approved training material. Source statistics
in `data/overview/` and evidence in `docs/curation/` retain their separate formats.

## Conversations: permissive loader and curated records

`load_conversations` still accepts the existing `data/example.jsonl`: `messages`, with an
optional first system message followed by one or more complete alternating user/assistant
pairs. Content must be nonempty text. Metadata is optional and ignored by the trainer.

The stricter curated contract adds `schema_version: 1`, requires a leading **identity-setting
system message**, and requires the metadata below. The same message-order validator serves
both contracts. A system prompt must identify the faction, species, character role, knowledge
limits, and relevant scene context; structure validation can check its presence and text,
not whether it actually establishes the right identity. That remains a review responsibility.

Readable example (serialize each entire object on one line in a UTF-8 JSONL file):

```json
{
  "schema_version": 1,
  "messages": [
    {"role": "system", "content": "You are an invented archive clerk, not a canonical game character."},
    {"role": "user", "content": "Hello."},
    {"role": "assistant", "content": "Welcome to the archive."}
  ],
  "metadata": {
    "id": "fixture-train",
    "identity": "invented-archive-guild",
    "species": "invented",
    "character_role": "Archive clerk",
    "topics": ["everyday"],
    "scenario_group": "fixture-train-family",
    "split": "train",
    "sources": [{
      "reference": "Invented fixture, not canon",
      "revision": "fixture-v1",
      "source_group": "invented-archive-setting"
    }],
    "authorship": "agent",
    "review_status": "draft"
  }
}
```

All displayed keys are required. Version 1 curated objects reject unknown fields, including
extra message fields, to catch typos and accidental mixing of record types. The permissive
training loader continues to ignore extra record metadata.

| Field | Structural constraint / meaning |
| --- | --- |
| `schema_version` | Integer `1` on every record and manifest |
| `metadata.id` | Globally unique across all files in the manifest |
| `identity`, `species` | Open labels; no pilot-faction whitelist. Keep different human factions distinct |
| `id`, `identity`, `species`, `scenario_group`, topic labels | Lowercase letters/digits separated by single `-` or `_`; e.g. `human-free-worlds` |
| `character_role` | Nonempty descriptive text; speaker role, not message role |
| `topics` | Nonempty list of topic labels; counts report each label once per record |
| `scenario_group` | Globally scoped authored scenario-family ID, assigned before variants are written |
| `split` | Exactly `train`, `development`, or `benchmark`, matching its manifest file declaration |
| `sources` | Nonempty list of objects with nonempty `reference`, `revision`, `source_group` |
| `authorship` | `human`, `agent`, or `mixed`; describes record creation, not source authorship |
| `review_status` | `draft`, `reviewed`, `approved`, or `rejected`; validity does not imply approval |

A source `reference` names a path/URL and passage or a curation evidence ID. `revision` pins
its immutable snapshot (full upstream commit for game sources, a version for original work).
The validator checks presence and type, not remote existence, revision immutability, canonical
truth, or legal permission. `source_group` identifies common provenance, such as a mission
chain, independently of the authored scenario group. Several sources may support one record.
Authorship/review labels do not substitute for later attribution and review evidence.

## Separate benchmark cases

A benchmark case has `schema_version`, the **same metadata** (with `split: "benchmark"`),
`inputs`, and `evaluation`. It has no training `messages` or gold assistant turns.
The two payload objects look like this; see the complete
[benchmark fixture](../tests/fixtures/contracts/benchmark.jsonl) for an executable record:

```json
{
  "inputs": {
    "system": "You are an invented archive clerk at a closed archive.",
    "user_turns": ["May I enter?", "When should I return?"]
  },
  "evaluation": {
    "dimensions": ["identity", "uncertainty"],
    "expected_facts": ["The archive is closed."],
    "expected_behaviours": ["Be helpful."],
    "expected_style": ["Use concise language."],
    "prohibited_contradictions": ["Claiming that the archive is open."],
    "uncertainty_expectations": ["Do not invent opening hours."],
    "sources": [{
      "reference": "Invented fixture, not canon",
      "revision": "fixture-v1",
      "source_group": "invented-archive-setting"
    }]
  }
}
```

`inputs.system` is nonempty, model-visible identity/scene context. `inputs.user_turns` is a
nonempty list of nonempty strings, including fixed follow-ups. **Everything in `metadata`
and `evaluation` is evaluator/curation-only.** Identity labels are not automatically added to
prompts; the explicit system text defines what the model may see. Do not put answer keys or
reference excerpts into this input context. Structural separation cannot detect a human
copying a secret answer into an allowed text field.

`evaluation.dimensions` is a nonempty list of applicable dimension names. It defines no scores,
weights, or rubric. The five expectation lists are required but may be empty when inapplicable;
their entries must be nonempty text. `evaluation.sources` cites the evaluator's evidence and
uses the same source format. It may differ from the record's overall provenance.

```python
from endless_voices.contracts import benchmark_messages

first_prompt = benchmark_messages(case, [])
# Run generation elsewhere; append only the actual model response.
second_prompt = benchmark_messages(case, [first_generated_response])
```

The helper returns fresh `role`/`content` dictionaries using only `inputs.system`, fixed user
turns up to the current turn, and caller-supplied generated assistant history. It never inserts
future user turns, evaluator references, or gold answers. History must contain one string per
completed turn; after all user turns have answers there is no next prompt. The caller is
responsible for keeping replies associated with this case and model run; a string's origin
cannot be verified structurally. Tests show that changing evaluator references or metadata
cannot change model inputs. This is an input-construction interface, **not an evaluation runner**.

## Versioned manifests and split boundaries

Store independent JSONL files for each split alongside a versioned JSON manifest, e.g.:

```text
data/local/curated/v1/
  manifest.json
  train.jsonl
  development.jsonl
  benchmark.jsonl
```

A manifest has exactly these keys (replace the illustrative hash placeholders with SHA-256
hashes of the exact file bytes):

```json
{
  "schema_version": 1,
  "dataset_version": "v1",
  "files": {
    "train": [{"path": "train.jsonl", "sha256": "<64 lowercase hex digits>"}],
    "development": [{"path": "development.jsonl", "sha256": "<64 lowercase hex digits>"}],
    "benchmark": [{"path": "benchmark.jsonl", "sha256": "<64 lowercase hex digits>"}]
  }
}
```

All three splits need at least one nonempty file. Multiple shards per split are supported.
Paths are relative to and must resolve inside the manifest directory. Files cannot be reused,
including symlink/hard-link aliases. IDs are unique across every shard and split. Changes to
records require reviewed checksum updates; released datasets should get a new dataset version
and retain previous manifests. `schema_version` identifies the format, `dataset_version` the
content snapshot. Source revisions remain independently pinned per source.

Complete conversations and **entire scenario families** are split units. Paraphrases, branch
alternatives, source-scene reconstructions, and shared-template variants must reuse their
family ID and stay in one split. A connected mission-chain reconstruction belongs together.
Shared canonical facts may support distinct original scenarios in different splits: equal
`source_group` alone is therefore not an automatic split violation. Reviewers must identify
when those records are actually variants and assign the same `scenario_group`.

Validation rejects cross-split scenario IDs, duplicate record IDs, mixed record types,
incorrect declared splits, malformed structures, and hash mismatches. It reports counts by
split, identity, and topic for coverage inspection; it sets no corpus-size or pilot-identity
quotas. Semantic near-duplicate detection, undisclosed shared templates, mission-chain leakage
review, approval gates, and benchmark freezing remain later corpus work. Do not feed benchmark
files or evaluator references to training or prompt development.

## Offline validation

From the repository root after installing `.[dev]`:

```sh
endless-validate tests/fixtures/contracts/manifest.json
# Equivalent module invocation (also works before refreshing installed entry points):
python -m endless_voices.contracts tests/fixtures/contracts/manifest.json
```

Structural validation uses only the standard library, reads local files, and downloads nothing.
It stops at the first error with a filename and physical JSONL line number, including blank
lines in numbering. Manifest/file-level errors identify the manifest field or affected file;
malformed manifest JSON reports its parser line/column. The intentionally invalid fixture is
not included in the valid manifest; tests assert its missing-metadata error. Inspect one file
programmatically with `read_records(Path(...), "train")`; use manifest validation for global
ID and split checks.

Optional **train/development** token checks reuse the loader's native-chat-template length
check with an already saved local tokenizer:

```sh
endless-validate data/local/curated/v1/manifest.json \
  --tokenizer /absolute/path/to/local-tokenizer --max-length 2048
```

Both flags are required together. No model weights are loaded; tokenizer loading is local-only
with remote code disabled. Overlength or fewer-than-two-token conversations fail with their
file/line; no truncation occurs. Structural-only success does not establish token fit. Benchmark
length depends on generated history and must be enforced by the later runner at generation time.
The existing trainer still computes loss on all non-padding tokens, including system/user text;
assistant-only loss and trainer integration of manifests are outside this change. Train only
on a separately validated training file selected in `data.path`.
