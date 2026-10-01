# 8. Use a shared scene-continuation instruction

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Character-specific directions about cadence, caution and speaking authority prescribe the behavior the model should learn from game text. Speech-only directions also exclude narration that the player reads in the game.

## Decision

Use the shared instruction in `src/endless_voices/instructions.py`: “Continue the scene with the next passage, keeping the characters and events authentic to the supplied context.” Character references identify the character. The instruction does not prescribe tone, length, invention, speaking authority or a dialogue-only format.

The learning objective is the game’s writing and characters, not chronological simulation of a character’s knowledge. Examples retain the rendered instruction with their selected context. Training loss defaults to the final assistant turn, including its chat delimiters. The explicit `data.loss = "all"` option includes the saved prompt and history.

## Consequences

Training and evaluation use the same task. Models must infer presentation from authored examples. Source boundaries still determine the expected continuation, so the task is not unrestricted story generation.

All-token loss repeats shared context across examples; it is not a deduplicated corpus-training export. Validation used during development is not unseen evidence, and results do not establish strict chronological generalization.
