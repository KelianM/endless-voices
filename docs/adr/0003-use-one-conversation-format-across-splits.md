# 3. All splits use one conversation sample format

- **Status:** Proposed
- **Date:** 2026-09-23
- **Sources:** [Issue #3](https://github.com/KelianM/endless-voices/issues/3), [PR #11](https://github.com/KelianM/endless-voices/pull/11), [CharacterEval, section 3](https://aclanthology.org/2024.acl-long.638.pdf)

## Context

The [training loader](../../src/endless_voices/data.py) consumes complete conversations.
Curated samples also need provenance and private assessment criteria. Split assignment must
not change the sample format, and generation inputs must exclude the target response.

The original issue proposed separate training and benchmark formats and fixed user follow-ups
with generated assistant history. A valid generated answer can make a prewritten follow-up
incoherent. Separate formats also make split assignment affect how a sample is represented.
The reviewed design instead evaluates one response against an authored history, following the
profile-plus-dialogue formulation used by CharacterEval.

## Decision

[data/contracts.md](../../data/contracts.md) defines one version 1 sample format. `SPLITS` in
[contracts.py](../../src/endless_voices/contracts.py) contains `train`, `validation`, and `test`. Every sample
contains `messages`, `metadata`, and `evaluation` with the same requirements in every split.
The system message is the model-visible context. The
[fixtures](../../tests/fixtures/contracts/) include a representative profile and selected lore
in that message; the validator requires nonempty text but does not interpret its meaning.

The final assistant message is an example target response. `evaluation_messages` returns a
copy of all preceding messages and excludes metadata and evaluation criteria. Earlier
assistant messages remain authored history. Training continues to load the complete messages
through `src/endless_voices/data.py`.

Manifests keep split files physically separate. Conversation IDs and known scenario groups
cannot cross splits. Unknown scenario relationships are represented by `null`, without a
similarity-classification requirement. Source groups retain shared provenance independently.

## Consequences

One validator and one loader cover all splits. The permissive loader and invented example remain
compatible. Curated samples require stricter metadata, a system message, and evaluation fields.
The unmerged separate benchmark format is replaced; no released dataset needs migration.

Fixed-history evaluation does not measure persistence through a model's own unfolding dialogue.
The helper constructs a prompt but performs no generation or scoring. The sample retains the
target and private criteria for the caller; neither enters the returned prompt.

Authored histories can contain answer-relevant facts by design. Authors must keep the final
answer and private criteria out of the prompt; structural validation cannot detect copied
meaning. Group checks detect declared overlap, not semantic similarity.

The trainer still computes loss on all non-padding tokens. The contract does not change that
behaviour or implement a lore store, comparison runner, or scoring rubric.

[Contract tests](../../tests/test_contracts.py) verify the same format across splits, preservation
of authored history, exclusion of targets and criteria, and rejection of cross-split groups.
