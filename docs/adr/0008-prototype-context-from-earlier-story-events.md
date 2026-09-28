# 8. Prototype context from earlier story events

- **Status:** Proposed
- **Date:** 2026-09-28
- **Sources:** [Issue #9](https://github.com/KelianM/endless-voices/issues/9),
  [PR #16](https://github.com/KelianM/endless-voices/pull/16), owner discussion on chronological context

## Context

Current samples supply a profile, selected lore and the current conversation. A model can lack
prior dialogue that would establish voice and story events. Explicit style instructions substitute
editorial judgments for that evidence. Adding arbitrary campaign passages risks future information,
incompatible story outcomes and knowledge the character never acquired.

The existing conversation split policy in
[ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md) does not require chronology.
Some later training conversations follow validation conversations in the same mission chain.

## Decision

Prototype references from explicitly declared earlier encounters on a story path, with whole
conversations assigned in train–validation–test order. Earlier material in the same split is
eligible, including optional dialogue alternatives that can validly precede the current scene.
Later splits, future events and alternatives to the current encounter are excluded. Earlier
examples do not assert that every optional exchange occurred or every speaker shares its knowledge. The owner selected this
chronological design for the next dataset. The frozen pilot is unchanged.

`scripts/prepare_context_prototype.py` validates declared path split order and consistent
conversation/scenario assignments, then measures bounded whole-exchange references without opening
test dialogue. The prototype does not execute game state or establish that a declared path was
played. The [context contract](../story-context.md) separates the proposed release rules from the
small measurement implementation.

## Consequences

The prototype makes reference size and split conflicts reviewable before another benchmark.
Prerequisites alone cannot establish all causal compatibility, so path review remains necessary. Recency
selection can omit relevant facts or voice examples; the small measurements cannot establish
late-campaign context requirements.

Chronological evaluation allows a previous held-out answer to become later observed history.
This tests continuation with authored history, not accumulated model-generated history. The same
selected context is required for generators and judges, but training exposure can still differ.
A production release needs new provenance, temporal lore review and split boundaries. This proposed
record does not supersede the accepted pilot contract or authorize reopening its test content.
