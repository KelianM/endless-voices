# 3. Use one conversation format across splits

- **Status:** Proposed
- **Date:** 2026-09-23
- **Sources:** [Issue #3](https://github.com/KelianM/endless-voices/issues/3), [PR #11](https://github.com/KelianM/endless-voices/pull/11), [CharacterEval, section 3](https://aclanthology.org/2024.acl-long.638.pdf)

## Context

The first experiment concerns conversation as a faction/species representative given identity
instructions and lore. The [adapter foundation](0001-save-lora-adapters-separately-from-base-models.md)
consumes complete conversations. The [pinned source workflow](0002-fetch-pinned-sources-outside-git.md)
provides evidence for later profiles and lore.

The original issue proposed separate training and benchmark formats and fixed user follow-ups
with generated assistant history. A valid generated answer can make a prewritten follow-up
incoherent. Separate formats also make split assignment affect how a sample is represented.
The reviewed design instead evaluates one response against an authored history, following the
profile-plus-dialogue formulation used by CharacterEval.

## Decision

`data/contracts.md` defines one version 1 sample format. `SPLITS` in
`src/endless_voices/contracts.py` contains `train`, `validation`, and `test`. Every sample
contains `messages`, `metadata`, and `evaluation` with the same requirements in every split.
The system message holds the representative profile, scene assumptions, and selected lore.

The final assistant message is an example target response. `evaluation_messages` returns a
copy of all preceding messages and excludes metadata and evaluation criteria. Earlier
assistant messages remain authored history. Training continues to load the complete messages
through `src/endless_voices/data.py`.

The base and adapted model conditions are intended to receive identical profile, lore, and
history. The sample stores that exact context; the comparison runner remains future work.
Both model and evaluator may use the supplied lore. Private criteria and the final target are
available only for assessment during generation-based evaluation.

Manifests keep split files physically separate. Conversation IDs and known scenario groups
cannot cross splits. Unknown scenario relationships are represented by `null`, without a
similarity-classification requirement. Source groups retain shared provenance independently.

## Consequences

One validator and one loader cover all splits. The permissive loader and invented example remain
compatible. Curated samples require stricter metadata, a system message, and evaluation fields.
The unmerged separate benchmark format is replaced; no released dataset needs migration.

Fixed-history evaluation does not measure persistence through a model's own unfolding dialogue.
Free-running conversations need a separate evaluation design. Targets are examples of acceptable
responses, not exact-match answer keys. Supplying lore controls information availability but
does not measure unaided lore recall. A lore store or retrieval service is not implemented.

Authored histories can contain answer-relevant facts by design. Authors must keep the final
answer and private criteria out of the prompt; structural validation cannot detect copied
meaning. Known relationships prevent declared overlap, but semantic leakage review remains
necessary when a corpus exists.

The sample format does not select a scoring rubric or change the training loss. Final-response
loss, evaluator calibration, and adaptation of published scoring methods remain later decisions.
