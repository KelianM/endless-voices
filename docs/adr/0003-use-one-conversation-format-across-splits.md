# 3. All splits contain authored conversations with one final target

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #3](https://github.com/KelianM/endless-voices/issues/3), [PR #11](https://github.com/KelianM/endless-voices/pull/11), [CharacterEval, section 3](https://aclanthology.org/2024.acl-long.638.pdf)

## Context

A character can give several reasonable answers to the same question. A prewritten follow-up
may make sense after the reference answer but not after another valid answer. Substituting a
model's replies into a fixed conversation can therefore create incoherent context and penalize
the model for a flaw in the test.

Training and evaluation both need the same underlying example: a representative's identity,
relevant lore, a conversation, and an appropriate response. Separate formats would make the
split determine how that example is represented.

## Decision

Use one sample format across train, validation, and test. Each sample contains authored
conversation history ending in a target assistant response. The system context supplies the
identity and selected lore; provenance and evaluator source references remain outside the messages.

Training reads the complete conversation. Evaluation input includes the same authored context
and withholds only the final assistant response. Earlier assistant replies are not replaced by
model generations ([data contract](../contracts.md),
[evaluation_messages](../../src/endless_voices/contracts.py)).

Keep every sample from one conversation in the same split. Known scenario variants also stay
together. Shared source provenance alone does not force unrelated conversations into one split;
unknown scenario relationships can remain unset.

## Consequences

- Split assignment changes how a sample is used, not its shape. One loader and validator serve
  all splits, and a sample's complete history remains available for inspection.
- Models can receive identical context for a response comparison. The test measures the next
  response in that context, not whether a model maintains its identity through its own unfolding
  conversation.
- An earlier authored reply may contain useful facts. Those facts are intentionally visible;
  the final target and private source references are withheld. A target is an example answer,
  not a requirement to reproduce its wording.
- Known conversation and scenario relationships prevent declared variants from crossing splits.
  The checks do not discover paraphrases, copied answers, or other semantic overlap.
- Every curated sample carries assessment fields, including training samples. The shared format
  adds authoring work but avoids a second representation for evaluation. The existing permissive
  loader still accepts the original invented example without those fields.
