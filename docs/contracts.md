# Conversation sample contract, version 1

Train, validation, and test files use the same sample format. Each sample contains a
representative's identity and selected lore, authored conversation history, a final user
message, one target response, and evaluator source references. The first experiment compares
prompted and fine-tuned models given identical identity instructions, lore, and history.

[ADR 3](adr/0003-use-one-conversation-format-across-splits.md) records the design.
The [fixtures](../tests/fixtures/contracts) are invented format examples. The [pilot release](../data/README.md) reconstructs the first source-backed
corpus and freezes its test content with versioned hashes. Source statistics and curation inventories remain separate
formats because evidence passages are not conversation samples.

## Record format

Store one complete JSON object per line in UTF-8 JSONL. The example below is expanded for reading.

```json
{
  "schema_version": 1,
  "messages": [
    {
      "role": "system",
      "content": "You are an invented archive clerk. Speak plainly. Lore (fixture-v1): The archive is closed. The clerk does not know the opening hours."
    },
    {"role": "user", "content": "May I enter?"},
    {"role": "assistant", "content": "The archive is closed."},
    {"role": "user", "content": "When should I return?"},
    {"role": "assistant", "content": "I do not know the opening hours."}
  ],
  "metadata": {
    "id": "fixture-train",
    "identity": "invented-archive-guild",
    "species": "invented",
    "character_role": "Archive clerk",
    "topics": ["everyday"],
    "conversation_id": "conversation-train",
    "scenario_group": null,
    "split": "train",
    "sources": [{
      "reference": "Invented fixture, not canon",
      "revision": "fixture-v1",
      "source_group": "invented-archive-setting"
    }],
    "authorship": "agent",
    "review_status": "draft"
  },
  "evaluation": {
    "dimensions": ["authenticity"],
    "sources": [{
      "reference": "Invented fixture, not canon",
      "revision": "fixture-v1",
      "source_group": "invented-archive-setting"
    }]
  }
}
```

All displayed keys are required in every split. Unknown fields are rejected. `messages` starts
with one system message followed by one or more complete alternating user/assistant pairs.
Each message contains only `role` and nonempty `content`. The final assistant message is the
target; earlier assistant messages are authored history.

The system message supplies the representative profile, scene assumptions, knowledge limits,
and selected lore. Species, faction, and role can define a representative without a named
individual. Keep distinct human factions separate. Validation checks the system message's
presence and text, not whether the description is accurate or sufficient.

| Metadata field | Constraint and meaning |
| --- | --- |
| `id` | Sample ID, unique across the manifest |
| `identity`, `species` | Open labels; no permitted-faction list |
| `character_role` | Nonempty description of the representative's role |
| `topics` | Nonempty list of topic labels |
| `conversation_id` | Shared ID for samples from the same authored or extracted conversation |
| `scenario_group` | ID for known scenario variants, or `null` when no relationship is recorded |
| `split` | `train`, `validation`, or `test`, matching the file declaration |
| `sources` | Nonempty list of source references, revisions, and provenance groups |
| `authorship` | `human`, `agent`, or `mixed` |
| `review_status` | `draft`, `reviewed`, `approved`, or `rejected`; validity does not imply approval |

IDs and topic labels use lowercase letters/digits separated by single hyphens or underscores.
`schema_version` must be integer `1`. Each source requires nonempty `reference`, `revision`,
and `source_group` strings. References identify passages or curation evidence IDs. Revisions
should pin immutable evidence: a full upstream commit for game text or a version for original
material. Structural validation does not contact sources or verify their truth or immutability.

`evaluation` contains only `dimensions: ["authenticity"]` and `sources`. Source references use
the same format as metadata and may identify additional evidence. Per-sample checklists and
expectation lists are rejected. Authenticity uses blinded original-versus-generated
pairs and free-text reasons, as chosen in [ADR 4](adr/0004-evaluate-authenticity-against-game-continuations.md).
A target response is one acceptable answer, not an exact-match requirement.

## Training and evaluation use the same sample

The existing loader reads `messages` and ignores the other fields. The loader still accepts
`data/example.jsonl`, optional system messages, and records without curated metadata. Curated
validation is a separate, stricter check before selecting a training file in `data.path`.

For generation-based evaluation:

```python
from endless_voices.contracts import evaluation_messages

prompt = evaluation_messages(record)
# Pass prompt to generation; retain the full record separately for assessment.
```

`evaluation_messages` returns fresh message dictionaries containing `messages[:-1]`. The prompt
ends with the final user message. The final target, metadata, and evaluation source references
are never copied. The helper works identically for train, validation, and test samples.

A multi-turn history remains fixed and authored. The helper evaluates one new response; it does
not substitute generated replies into earlier turns or continue with prewritten follow-ups.
Fixed-history evaluation measures response quality in supplied context, not persistence through
a model's own unfolding conversation. Free-running dialogue evaluation is a later design.

Both model conditions receive identical system context, including selected lore. The evaluator
can use the same lore and source references. Sharing canonical facts is intentional;
exposing the withheld target or origin label is not. Authors must avoid copying private
answers into model-visible context; structure checks cannot detect that semantic mistake.

The first format stores the exact profile and selected lore text in the system message, with
provenance in source references. A shared lore store can supply that text during later dataset
preparation. This PR adds neither a retrieval service nor a second prompt renderer. Evaluators
must respect the representative's knowledge limits even when the source contains more facts.

The trainer still calculates loss on all non-padding tokens, including system and user text.
Final-response-only loss was discussed but is not implemented or selected by this contract.
The format supports that later change without changing sample structure.

## Splits and manifests

Use training samples for weight updates, validation samples for model/prompt selection, and
reserved test samples for the final comparison. A benchmark is an evaluation suite, not a
fourth split or a different record type.

Store physical split files with a versioned manifest:

```text
data/local/curated/v1/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
```

```json
{
  "schema_version": 1,
  "dataset_version": "v1",
  "files": {
    "train": [{"path": "train.jsonl", "sha256": "<64 lowercase hex digits>"}],
    "validation": [{"path": "validation.jsonl", "sha256": "<64 lowercase hex digits>"}],
    "test": [{"path": "test.jsonl", "sha256": "<64 lowercase hex digits>"}]
  }
}
```

Replace each hash placeholder with the SHA-256 of the exact file bytes. Complete manifests
require at least one nonempty file per split and support multiple shards. Paths must resolve
inside the manifest directory. Reusing a file, including through symlinks or hard links, fails
validation. New dataset releases should retain earlier manifests and use new dataset versions.
Schema versions identify the format; dataset versions identify the content snapshot.

Complete conversations stay in one split. Samples at different points in a conversation share
`conversation_id`. Explicit paraphrases, branch alternatives, or shared-template variants also
share `scenario_group`. Repeated versions of one actual encounter belong together, but a shared mission chain or
theme alone does not require a shared scenario group. Record known relationships; do not invent similarity classes for unrelated examples.

`source_group` records shared source/mission-chain provenance independently. The same canonical
fact can support different situations across splits, so source-group equality alone is not an
error. The pilot preparation adds exact dialogue overlap checks and recorded source-route review.
Those checks do not establish exhaustive semantic deduplication.

## Offline validation

From the repository root after installing `.[dev]`:

```sh
endless-validate tests/fixtures/contracts/manifest.json
# Equivalent invocation before refreshing installed entry points:
python -m endless_voices.contracts tests/fixtures/contracts/manifest.json
```

Structural validation uses only the standard library and downloads nothing. The command checks
metadata, source fields, message order, file hashes, global IDs, declared splits, and known
conversation/scenario groups crossing splits. Output contains record counts and identity/topic
coverage per split. Counts impose no faction or corpus-size quotas.

The command stops at the first error with a filename and physical JSONL line number. Blank
lines count toward line numbers and are ignored as records. Manifest-level errors identify the
manifest or affected file. The invalid fixture is excluded from the valid manifest.
For an individual file, use `read_records(Path(...), "train")`; global checks require a manifest.

Optional token checks use a saved local tokenizer and the existing loader's length check:

```sh
endless-validate data/local/curated/v1/manifest.json \
  --tokenizer /absolute/path/to/local-tokenizer --max-length 2048
```

Both flags are required together. Tokenizer loading is local-only, with remote code disabled;
no model weights are loaded. Every complete sample in every split must fit the limit and contain
at least two tokens. Failures report file/line without truncation. The future generation runner
must separately budget the prompt and generated output; an authored target's length does not
bound the model's generated response.

## Evaluation and remaining work

The [authenticity protocol](../data/evaluation/README.md) uses the same sample format. For an
authenticity trial, the final target must be attributable original game speech, not an agent-written
reference presented as original. The generator receives `messages[:-1]`; the blinded judge sees
that context plus the original and generated continuations as unlabelled alternatives. The judge
receives neither origin labels nor per-sample checklists. Training samples can still have
authored targets. This is an evaluation eligibility rule, not a schema change.

The legacy calibration pack contains only draft validation samples, without invented train/test shards.
Use `read_records(path, "validation")` for that file. The [pilot release](../data/README.md) supplies all
three physical splits and a manifest. Reserve calibration conversations and their known variants
from the eventual test set.

- Evaluation design (#6) supplies the protocol and calibration material; the initial human review, including controls, is recorded.
- Dataset construction (#5) now includes source preparation and test freezing, replacing #4 and #7.
  The pilot records selected identities, source annotations, lore, review and reconstruction hashes.
- Response generation (#8) uses existing models and adapters; it does not fine-tune.
- Assessment and reporting (#9) implement the calibrated method.
- The first training experiment (#12) selects the training objective and uses validation data for
  development before the final held-out comparison. Adversarial training is not selected.

Each issue gets its own implementation and review before work proceeds to the next issue.
