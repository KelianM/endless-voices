# 11. Use a shared scene-continuation instruction

- **Status:** Proposed
- **Date:** 2026-09-30
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), owner discussion on learning game writing

## Context

Character-specific directions about cadence, caution and institutional authority prescribe the behavior the model should learn from game text. Speech-only directions also conflict with the authored narration retained under [ADR 10](0010-preserve-authored-continuation-passages.md).

## Decision

Dataset and benchmark preparation use the shared instruction in `src/endless_voices/instructions.py`: “Continue the scene with the next passage, keeping the characters and events authentic to the supplied context.” Character references identify the character; the task instruction does not prescribe tone, length, invention, speaking authority or a dialogue-only format. Saved examples retain the rendered instruction for reproducibility, rather than defining independent instructions.

The preliminary learning objective is the game's writing and characters. Whole missions are held out; chronological character knowledge is not the release criterion proposed in [ADR 8](0008-prototype-context-from-earlier-story-events.md). Training context excludes held-out source passages. Context selection still uses prerequisite missions as a relevance rule. References preserve authored paragraphs, including narration and other speakers, and reject target paragraphs already exposed in the input.

Training loss applies to the final assistant turn, including its chat delimiters, rather than the supplied lore and encounter history. The explicit `data.loss = "all"` option retains the earlier objective for reproducibility.

## Consequences

Training and evaluation can use the same task and selected context. The model must infer presentation from source examples. Scene boundaries still determine how much text is expected; this is not unrestricted story generation or a simulation of a character's knowledge at a particular date.

Historical prompts and measurements remain unchanged. Their scores do not become measurements of the new instruction. The existing validation set has been used for development, and later training dialogue may reveal story developments. Results cannot establish performance on unseen stories or strict chronological generalization.
