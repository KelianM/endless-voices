# 13. Split datasets by source mission

- **Status:** Proposed
- **Date:** 2026-09-30
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), owner decision on mission-level holdouts

## Context

Conversation-level splits can put different passages from one mission in training and evaluation. Those passages share situations, wording and outcomes. The writing objective in [ADR 11](0011-use-a-shared-scene-continuation-instruction.md) needs held-out scenes without requiring entire campaigns or chronological character knowledge to be held out.

## Decision

The primary source mission is the split unit. Individual conversations and continuation passages remain the example unit. Known conversation and scenario variants also stay together. Supplementary lore does not merge otherwise unrelated missions.

`endless_voices.splits` assigns connected groups to one split. When migrating existing assignments, test takes precedence over validation, and validation over training; an existing held-out group is never moved into training. Newly built manifests declare `split_unit: mission`, and contract validation rejects cross-split mission ownership. Conversation-level manifests are no longer accepted. Historical code and reconstruction procedures are recoverable from Git; the current implementation has no compatibility mode.

Training context excludes held-out missions before the shared token-budget sampler runs. Sampling chooses relevant passages from the permitted pool; sampling does not establish split eligibility. For the current 101-example release, nine unusable examples are replaced with reviewed unused continuations from already eligible training missions. Existing validation and test examples are retained unchanged.

This record replaces the conversation-level split decision in [ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md). Reviewed source annotations, provenance and the distinction between agent review and human approval remain required. Historical artifacts remain unchanged; new releases carry hashes, exclusions or replacement records, and licensing.

## Consequences

A mission with many conversations can make split sizes uneven. Connected variants can group multiple missions. The policy does not require campaign-level holdouts or prevent learning later story developments elsewhere in the training set.

Mission separation is not proof against repeated text across unrelated missions. Source-coordinate and target-overlap checks remain necessary. The existing validation set remains development data; changing the split rule does not make prior evaluation unseen again.
