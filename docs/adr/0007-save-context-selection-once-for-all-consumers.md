# 7. Save context selection once for all consumers

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Selecting context independently for training, generation and judging gives the models different evidence. Token budgeting also cannot determine whether a source passage is eligible for a particular scene.

## Decision

Select context once from an eligible source pool and store the resulting messages in the dataset record. Dataset preparation appends the authored target; generation withholds that target; judging uses the same context. Private provenance records selection settings and source coordinates without another copy of the dialogue.

`endless_voices.context` separates eligibility from selection. Full-context selection retains every eligible block. Mission-depth selection preserves lore, the current encounter and nearby missions, then fills the remaining complete-input budget with whole older missions in a reproducible order. The caller supplies the tokenizer. A preserved core exceeding the budget fails preparation rather than being truncated.

## Consequences

Strategies cannot inspect target answers. Compared models reuse the selected text even when their tokenizers count it differently; runtime context limits still require checks.

Mission distance approximates relevance. Whole-mission sampling can omit useful connecting history or leave unused budget. Source eligibility and split ownership remain upstream responsibilities; fitting within a token budget does not establish a valid example.
