# 10. Bind retrieval to eligible scene pools

- **Status:** Proposed
- **Date:** 2026-10-12
- **Sources:** [Issue 18](https://github.com/KelianM/endless-voices/issues/18)

## Context

Fixed context sampling ties the available evidence to one selection policy. Comparing an
agent's retrieval choices with a fixed sampler requires both policies to share eligible
material. A semantic match alone cannot establish that a scene belongs to the selected
history, split or visible encounter.

## Decision

Preserve the eligibility boundary established by [ADR 8](0008-infer-scene-writing-from-sampled-game-context.md)
and the interpreted histories established by [ADR 9](0009-build-examples-from-consistent-game-state.md).
Save each example's eligible context independently of its fixed sampled prompt. Bind retrieval
tools to that pool, applying eligibility before similarity ranking or scene reads.

Index complete authored passages in a static database, retaining each passage's interpreted
parent scene. Authorize every passage through its parent scene and allow agents to expand
a passage into that scene. Distinct interpretations remain distinct entries. Use a retrieval
encoder's prescribed pooling and query instruction format. Reject inputs beyond the encoder's
token limit instead of silently dropping text. Mission history and independent writing
references retain their different meanings.
Keep fixed sampling as one context policy and expose tool schemas and dispatch for callers
that run an agent loop.

## Consequences

Agents can select smaller passages without receiving whole missions. Shared indexing avoids
re-embedding repeated scenes, while saved pools preserve the builder's branch and split
decisions. Exact pool membership prevents excluded scenes from influencing search ranking.
Filtering only after a global search was rejected because excluded scenes could displace
eligible results.

The initial prototype indexed one vector per scene using mean pooling and a truncated scene
prefix. That approach was replaced because passages near a scene's end could not influence
retrieval, and generic mean pooling did not follow the chosen encoder's retrieval interface.
Passage indexing retains every authored passage but increases the number of vectors. Several
results can come from the same scene. An unusually long passage or query now causes an
explicit error rather than producing a vector for incomplete text.

Prepared bundles retain more eligible text than fixed prompts require. Rebuilding the corpus
or changing the encoder requires a corresponding index. Retrieval inherits the interpreter's
state limitations and does not make prepared examples equivalent to arbitrary live game saves.
The current generation and benchmark runners still use fixed prompts; tool execution budgets
and common judging evidence remain responsibilities of an agent runner.
