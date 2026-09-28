# 9. Save context selection once for all consumers

- **Status:** Proposed
- **Date:** 2026-09-28
- **Sources:** [Issue #9](https://github.com/KelianM/endless-voices/issues/9),
  [PR #16](https://github.com/KelianM/endless-voices/pull/16), owner request for shared context strategies

## Context

Independent context-selection scripts can supply different evidence to training, generation and
judging. A strategy also needs to remain separate from source extraction: a token budget cannot
establish whether a passage belongs before the target or respects a split boundary. The earlier
source prototype in [ADR 8](0008-prototype-context-from-earlier-story-events.md) measures eligible
reference pools but does not make selection reusable across consumers.

## Decision

Select once from an already-eligible source pool and save the resulting messages with private
selection provenance. Dataset builders append a target after selection. Generation and judging
read the same selected messages. Loading a saved bundle verifies artifact and message hashes.

`endless_voices.context` defines the strategy interface and implements full-context and
mission-depth selection. The source-draft adapter preserves fixed lore and the current encounter.
Mission-depth selection retains nearby missions and samples whole older missions using a recorded
seed and caller-supplied tokenizer. The mission-depth budget applies to the complete rendered
input, including fixed lore, the current encounter, nearby history and chat formatting. Older
missions fill the remaining space. Preparation rejects a preserved core that exceeds the budget.
Depth and token budgets are configuration, not architectural
constants. `endless_voices.prepare_context` writes reusable bundles without model inference.

## Consequences

Strategies cannot inspect target answers because targets are not part of their input. Selection
provenance remains private; only selected messages reach model inputs. Different tokenizers may
produce different selections, so compared conditions reuse one bundle instead of selecting per
model. Runtime input limits still require explicit checks; neither strategy silently truncates.

Source eligibility, chronological splits and release validation remain upstream responsibilities.
Mission distance approximates relevance, and whole-mission sampling can omit useful connecting
history. Saved omitted blocks make those omissions inspectable. Full context remains available
for comparisons without a plugin registry or a separate implementation in each consumer.

A saved context bundle is an input-preparation artifact, not approval of a new curated dataset release.
