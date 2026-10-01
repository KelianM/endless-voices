# 9. Build examples from consistent game state

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Game passages depend on player choices, conditions, state changes and mission-specific substitutions. Extracting text without interpreting those dependencies can combine mutually exclusive events or give an earlier passage the current mission’s destination. Extracting only a character’s quotes also invents boundaries that the player never sees.

Targets and sampled history need a consistent interpretation of the game source. Choosing one branch for the entire dataset would discard valid authored continuations; collecting every branch into one example would imply contradictory events occurred together.

## Decision

Build examples by interpreting authored dialogue and mission state, replacing the manual extraction approach in [ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md). Preserve complete passages, including narration and other speakers, through the next visible player choice or endpoint. Full passages replace the direct-speech normalization in [ADR 4](0004-evaluate-authenticity-against-game-continuations.md); the authenticity question remains unchanged.

Retain each distinct reachable target sequence as a separate example and select a compatible history reproducibly. Unspecified initial conditions represent possible prior states; subsequent assignments constrain which branches remain valid. Target construction and history sampling share the same interpretation of conditions and effects. Apply [mission split ownership](0008-split-datasets-by-source-mission.md) before [context selection](0007-infer-scene-writing-from-sampled-game-context.md).

Resolve each passage’s variables within its owning mission. Player identity comes from shared configuration; literal mission locations come from the game source. Do not apply the current mission’s substitutions to earlier missions or invent values for unresolved dynamic substitutions. Preserve unknown markers and record source coordinates, state and resolved values in provenance.

`DatasetBuilder` owns preparation. Its example builder and context sampler share `DialogueInterpreter`; `game_variables.py` handles passage substitutions. Publish fixed examples for consumers through `SceneDataset`, rather than reinterpreting branches during training or evaluation. Unsupported operations, non-progressing loops and exceeded exploration or context limits fail preparation explicitly.

## Consequences

Valid alternative branches remain available for learning without appearing together as one history. Authored narration and presentation remain intact. Multiple compatible histories for the same target do not multiply examples into every possible combination, and target passages already present in input are rejected.

The interpreter establishes consistency within its supported state model, not reachability from every real player save. It does not simulate the entire game or reconstruct predecessor histories from arbitrary prerequisite expressions. Broader source coverage requires checking support for the additional operations; branch exploration can exceed practical limits.

Unresolved variables remain visible, so a model may reproduce symbolic markers. Runtime integration needs actual mission values before presenting those passages. Configured player identity can influence generation even when substitutions are consistent. State consistency prevents contradictory examples but does not establish storytelling quality.
