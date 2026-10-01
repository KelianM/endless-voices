# 11. Build scene examples through stateful dialogue

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Extracting speakers’ quotes invents boundaries absent from the game. Structural traversal without state can combine contradictory branches. Targets and sampled history need one interpretation of authored paragraphs, choices and game conditions.

## Decision

Replace the manual extraction approach in [ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md) with `DatasetBuilder`. Its example builder and context sampler share a `DialogueInterpreter`. Preserve complete authored paragraphs, including narration and other speakers, through the next visible player choice or endpoint. Do not extract speech fragments or rewrite punctuation. Full passages replace the direct-speech normalization described in [ADR 4](0004-evaluate-authenticity-against-game-continuations.md); the original-versus-generated authenticity question remains unchanged.

Integer state uses Z3 constraints. Unspecified initial conditions represent possible prior states; assignments constrain subsequent branches. Retain each distinct reachable target sequence and select one compatible history reproducibly. Apply [mission ownership](0010-split-datasets-by-source-mission.md) before [context budgeting](0007-save-context-selection-once-for-all-consumers.md). Required ancestors follow positive completed-mission prerequisites outside OR groups.

Publish one built dataset: split JSONL records, configuration, source/state provenance, a manifest and licensing. Record implementation revision and hashes rather than copying code. All consumers use `SceneDataset`; DataLoader batches fixed examples without resampling context. Unsupported operations, unchanged-state loops, excessive branching and context overflow fail the build without publishing partial output.

## Consequences

The interpreter checks consistency within its supported state model, not reachability from every real player save. It handles random draws, arithmetic, payments, transferable outfits and scheduled events, but does not reconstruct predecessor missions from arbitrary OR conditions or simulate combat and navigation.

Branch exploration can grow rapidly. State-equivalent histories are collapsed, and route limits fail visibly. Multiple histories for one target do not create a Cartesian product of examples. Target paragraphs already present in input are rejected.

A continuation may contain several speakers or actions. Keeping narration avoids presentation bias from speech extraction, but origin detection still does not measure storytelling quality. Expanding the mission scope requires validating the additional source operations.
