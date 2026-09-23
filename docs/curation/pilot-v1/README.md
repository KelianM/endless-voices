# First conversation dataset

This release is constructed from reviewed conversation annotations and pinned Endless Sky
speech. Agents identify speakers, coherent branch routes, profiles and relevant lore. Scripts
extract the selected speech and generate conversation prefixes, provenance, split files and
checksums. An annotation is not a handwritten copy of every resulting sample.

The release covers Free Worlds representatives, Republic Navy, mainstream Hai and Quarg. Faction labels
come from speaker evidence, not filenames or the government of the planet where a mission runs.
Human factions remain distinct. Profiles describe a particular role and story state rather than
an omniscient faction spokesperson. Topic labels organize coverage; topics do not determine splits.

The pilot contains **176 samples from 61 conversations**: 101 training, 48 validation and 27 test
samples. [Coverage and review findings](findings.md) document exclusions, corrections and limits.

## Use and reconstruct the release

The [complete release](../../../data/curated/pilot-v1/) is committed in ordinary Git, including
the three split files, provenance, review copy and license material. Agent-authored annotations,
profiles and lore are committed alongside this document. A fresh checkout contains the dataset;
no source download or reconstruction is required to read it.

Validate the committed split manifest offline:

```sh
python -m endless_voices.contracts data/curated/pilot-v1/manifest.json
```

To verify reconstruction, run these commands from the repository root with the development
dependencies installed:

```sh
python scripts/fetch_sources.py
python scripts/prepare_conversations.py catalog --output data/local/pilot-candidates-v1
```

The catalog contains 785 source conversation blocks from the human, Hai and Quarg directories.
The catalog is an unreviewed reading aid, not 785 usable conversations. Reading sheets retain
physical line numbers, branch instructions and proposed speech spans. Quoted narration, other
speakers, untranslated speech, and runtime placeholders still require review. Source conditions
outside a conversation must be checked in the original mission.

Download only the selected tokenizer and its license:

```sh
hf download Qwen/Qwen3-4B-Instruct-2507 \
  tokenizer.json tokenizer_config.json vocab.json merges.txt LICENSE \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir data/local/tokenizers/qwen3-4b-instruct-2507
python scripts/build_pilot.py \
  --output data/local/pilot-v1-rebuilt \
  --tokenizer data/local/tokenizers/qwen3-4b-instruct-2507 --max-length 8192 \
  --verify-release docs/curation/pilot-v1/release.json
```

Both preparation commands require a new output directory. Reconstruction after the source and
tokenizer downloads is offline. The frozen measurement used Transformers 4.57.6 and Tokenizers
0.22.2; those versions can be installed if another supported version changes reconstruction. The builder re-verifies the source snapshot and tokenizer file
hashes. The 8,192-token complete-sample budget includes the profile, lore, full selected history
and target; it is a dataset preparation limit, not the model's maximum context length. No sample
is truncated. The trainer's existing configuration is unchanged and must use an appropriate
length limit if this dataset is selected later. No weights, generation or training are needed.

The committed release contains separate `train.jsonl`, `validation.jsonl`, `test.jsonl`, and a
versioned `manifest.json`. `coverage.json` records counts and measured token lengths.
`provenance.json` records every extracted source span, original or agent-authored user turn,
and punctuation normalization. `review.md` is an organizer's reading copy with targets and source
labels; it is not a blinded judge input. `construction.json` records annotation and script hashes.
The committed `release.json` fixes the expected artifact hashes for reconstruction.

## Annotation format

The four faction JSON files hold reusable profiles and lore, with immutable source citations.
Every conversation selects a profile and lore entries, records its scene state, and lists reviewed
routes through its original dialogue. The builder copies that exact context into the system
message, identically for the base and adapted model conditions. There is no retrieval service or
runtime dependence on annotation IDs.

A turn annotates a source player line and the source paragraphs making up the complete next reply:

```json
{"user": 123, "assistant": [125, 126]}
```

Line numbers refer to the physical source file identified by `catalog_id`. An explicit selection
such as `{"line": 125, "quotes": [0, 2]}` selects only the attributed quoted fragments when a
paragraph contains other speakers. An implicit or narrated player action can instead use
`{"user": {"prompt": "An authored connective prompt.", "reason": "Source basis."},
"assistant": [125]}`. The reason remains in the provenance ledger, outside model input.
Agents annotate these exceptions; the script never invents a reply or fills missing dialogue.

Each route produces a sample after each complete user/assistant turn, retaining its earlier
original replies as history. Repeated targets within the same annotated conversation produce only
one sample, using the first annotated route to that target. Different paragraphs in one response
are not counted as separate turns. Branch alternatives can supply different responses, but they
remain members of the same actual conversation for split checks and reporting.

Speech extraction removes narrator insertions and enclosing quotation marks. Quoted fragments
from one source paragraph join with a space; successive paragraphs retain paragraph breaks.
A trailing comma left by a removed narrator tag becomes a period. Explicit
`sentence_breaks_after` indices can apply that same correction between selected fragments.
Every correction is recorded. Unresolved runtime placeholders are rejected. Explicit `substitutions` may render player-selected
names or source-backed mission values; `substitution_notes` identifies their basis, and the same
values appear in the system context and provenance. Rendering a declared game variable does not
author new dialogue. Other word changes are not supported.

The builder checks possible paths through source choices, labels and jumps, and rejects selected
reply paths that skip intervening quoted dialogue. Conditions are not evaluated against a save
state, so static reachability does not prove that the annotated story assumptions hold.
Annotators check labels,
choices, conditions and speaker changes and record the chosen assumptions in `review_notes`.
The catalog makes those checks auditable without implementing a general game engine.

## Lore and knowledge

The lore library supplies setting facts, faction relationships, history and customs relevant to
the selected role and story date. A young Hai and a Hai elder need not receive the same historical
knowledge. Disputed accounts remain attributed accounts; a faction's interpretation is not silently
promoted to omniscient truth. Scene context establishes what has happened before the selected route.
Earlier authored history is not replaced with generated replies.

Canonical facts may be shared across splits, including facts also discussed in a held-out
response. The withheld answer's text, distinctive phrasing, and instructions to reproduce its
particular argument do not belong in the profile or scene summary. The aim is to supply knowledge
for an in-world reply, not to test unaided lore recall or coach imitation of one answer.
The calibration pack supplied only short scene profiles; this release adds explicit selected lore.

## Splits and review

The actual source conversation is the split unit. All selected routes and earlier-response
prefixes from that conversation stay together, even when different characters speak in it.
Known repeated versions of the same encounter also stay together through `scenario_group`.
A shared mission chain, subject, faction, speaker or source file alone does not require one split.
This replaces the earlier blanket reservation of the Free Worlds reconnaissance chain.
The actual Quarg first-contact, Hai first-contact, and Free Worlds scan-request calibration
conversations and their known variants remain validation material.

The builder checks exact normalized assistant passages of at least twelve words across splits,
including paragraphs and passages appearing in earlier history. `overlap-candidates.json` also
lists shared twelve-word phrases across splits and between a sample's system context and target,
for review of partial copying and answer exposure. Shared facts can be intentional; the report
does not label every match as contamination. Short conventional replies are not automatically
classified as contamination. Review also checks source routes and known repeated encounters;
there is no broad similarity taxonomy or claim that exact matching discovers every paraphrase.
Shared source provenance is retained separately from scenario grouping.

Construction-agent review and implementing-agent review are recorded separately in `review.json`.
`reviewed` means source and construction review by agents, not human approval or a validated model
result. The project owner reviews the PR and may inspect `review.md`; approval of the PR does not
create a claim that the owner checked every sample. The original calibration judgments and their
limits remain in [calibration findings](../../evaluation/calibration-findings.md).

## Release corrections and limitations

The first release is frozen by its dataset version and artifact hashes. A correction to speech,
context, attribution, split membership or test content requires a new release directory/version,
a change record identifying affected sample IDs, and new hashes. Retain earlier releases and
report which release each experiment used. Do not silently fix a test after observing model
performance or compare results from different test versions as if they used the same questions.
A contaminated test scene must be replaced with a fresh scene in a new version and disclosed.

This dataset measures continuation of selected game dialogue with supplied context. It does not
establish free-running role-play quality. Multiple samples from one conversation are dependent;
record counts are not independent-scene counts. Coverage is uneven, source text is public, and
possible pretraining memorization remains unmeasured. The small calibration used easy authored
alternatives; its successful origin judgments are not model-quality results. Future evaluator
validation still needs fresh scenes, subtler alternatives and actual generated responses.

Materialized speech, annotations, manifests and review records stay in ordinary Git. Only raw
upstream sources, tokenizer caches and temporary reconstruction outputs stay under gitignored
`data/local/`. This source-derived dataset uses GPL-3.0-or-later. The committed bundle preserves
the upstream license, copyright manifest, credits and selected file-header notices. Source URLs
and hashes identify the original text; profiles and connective prompts are identified as agent
work. Dataset release terms do not settle licensing of any future trained adapter.
