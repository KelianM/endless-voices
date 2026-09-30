# 10. Preserve authored continuation passages

- **Status:** Proposed
- **Date:** 2026-09-29
- **Sources:** [Issue 9](https://github.com/KelianM/endless-voices/issues/9), [PR 16](https://github.com/KelianM/endless-voices/pull/16)

## Context

The annotated dataset described in [ADR 5](0005-build-datasets-from-reviewed-conversation-annotations.md) extracts spoken fragments. The current source-context preparation includes narration and character actions. Comparing a generated narrative continuation with a speech-only reference makes presentation a cue for origin detection and removes part of what the player would read.

## Decision

New benchmark preparation uses complete source paragraphs from a verified continuation bundle. The selected route starts at the first annotated target paragraph and ends before the next player choice, recorded response boundary or route end. Existing anchors resolve branches; distinct unresolved outcomes cause preparation to fail. Narration from other characters can remain within the continuation.

The shared implementation in `src/endless_voices/continuations.py` preserves authored text. Only source syntax delimiters are removed and configured game variables are substituted. Dataset consumers can use the same target with the saved context. Benchmark preparation checks each target paragraph for input overlap. The initial bundle contains validation scenes. Training preparation now applies the same extraction and records ambiguous examples as exclusions.

## Consequences

Generation and reference answers can both contain narration without imposing a speech-only prompt. Formatting and length may still distinguish originals, so origin detection remains separate from storytelling quality.

Existing scene boundaries and route annotations still shape the task. This change does not establish the exact rendering of every conditional game state or automatically resolve ambiguous routes. A continuation can contain actions or additional speakers rather than a single spoken turn.

Historical speech-only artifacts remain unchanged and must not be pooled with the new targets. Restoring complete passages can increase target length and requires checking that source context has not exposed any restored paragraph. Silently stripping generated narration and rewriting originals to match generated answers were rejected because both change the evidence after generation.
