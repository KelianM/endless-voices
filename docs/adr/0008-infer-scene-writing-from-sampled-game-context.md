# 8. Infer scene writing from sampled game context

- **Status:** Accepted
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Early prompts used handcrafted lore summaries and character-specific directions about tone, caution and speaking authority. Those prompts prescribed behavior the model should infer from the game’s writing. Summaries also omitted background knowledge and introduced the summarizer’s interpretation, making it harder to separate model capability from prompt engineering.

The current conversation alone may not establish the characters, world or preceding events. Supplying authored lore and dialogue provides that evidence without rewriting it. Including every related passage, however, produces widely varying prompt lengths and can exceed model context limits. Context selection needs a reproducible rule instead of per-example editorial choices.

## Decision

Supply sampled authored game context with one minimal continuation instruction, rather than handcrafted summaries or directions about how a character should speak. The shared instruction in `src/endless_voices/instructions.py` is: “Continue the scene with the next passage, keeping the characters and events authentic to the supplied context.” The model infers voice, presentation and appropriate detail from the supplied text. The instruction does not prescribe tone, length, invention, speaking authority or a dialogue-only format.

Apply [mission split ownership](0007-split-datasets-by-source-mission.md) before selecting context: training context excludes held-out mission text. The context sampler uses eligible game lore and dialogue without summarizing or rewriting the passages. `endless_voices.context` separates source eligibility from selection. Its budgeted strategy preserves lore, the current encounter and nearby prerequisite missions, then fills the remaining input budget with whole older missions in a reproducible order. Preparation fails if the preserved core exceeds the budget; it does not silently truncate that core. Full-context selection is available when the eligible pool fits.

Save each selection in the dataset record. Training appends the authored continuation; generation withholds the continuation; judging receives the same selected context. Source coordinates and selection settings make the input inspectable. The sampler cannot inspect target answers.

The objective is to learn the game’s writing and characters, not simulate a character learning events chronologically. Source boundaries define the continuation task; the instruction does not direct a new plot or unrestricted story generation.

## Consequences

Authored text supplies examples of style and world knowledge instead of encoding those judgments in prose written for the model. Training, generation and judging share the task context, so differences cannot arise from independently resampling their inputs. Context selection still introduces assumptions: mission distance approximates relevance, and omitted missions can contain useful connecting history.

Longer source context costs tokens and memory. Whole-mission selection can leave unused budget, and token counts differ across model tokenizers. Split ownership and state consistency remain upstream requirements; fitting a passage within the budget does not make the passage eligible. Development results do not establish strict chronological generalization or performance on unseen scenes.
