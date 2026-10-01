# 10. Split datasets by source mission

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Different conversations in one mission share situations, wording and outcomes. Splitting those conversations between training and evaluation can expose the held-out material’s immediate setting and related passages.

## Decision

The primary source mission is the split unit. Known conversation and scenario variants also stay together; supplementary lore does not merge unrelated missions. This replaces the conversation-level split rule in [ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md).

Mission ownership is explicit in dataset configuration. Preparation rejects conflicting assignments rather than moving missions silently. The shared loader checks all published splits before returning a requested split. Training context excludes held-out mission text before token selection.

Evaluation uncertainty resamples source missions and connected variants. Conversations within one mission are not independent groups.

## Consequences

Large missions can make split sizes uneven. Connected variants can group multiple missions, reducing the number of independent evaluation groups.

The writing objective in [ADR 8](0008-use-a-shared-scene-continuation-instruction.md) does not require campaign-level or chronological holdouts. Mission separation does not prevent repeated wording across unrelated missions, and changing the split rule does not make previously used validation scenes unseen.
