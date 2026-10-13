# 10. Let agents retrieve context through shared tools

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [Issue 18](https://github.com/KelianM/endless-voices/issues/18), [PR 19](https://github.com/KelianM/endless-voices/pull/19)

## Context

Fixed mission depth and seeded sampling choose a model's context before generation. Mission
distance approximates relevance, but the model cannot request evidence for a particular gap
in continuity, lore or writing style. Comparing larger agents with smaller fine-tuned models
also requires a way to explore whether retrieval can compensate for knowledge the model has
not learned.

Changing the selection policy should not change which material is permitted. Semantic
relevance cannot establish that an event happened in the selected history or that a passage
belongs to an allowed split.

## Decision

Make context selection a policy over shared eligible material. Retain the fixed sampler from
[ADR 8](0008-infer-scene-writing-from-sampled-game-context.md) as a reproducible baseline,
and expose tools for agent-directed retrieval of mission history and writing references.
Agents can request history by prerequisite depth, search semantically and read selected
passages or scenes. The initial agent messages contain the continuation instruction and
visible encounter; an omitted search query uses that encounter.

Keep the context database static. Preparation saves each example's eligible scene pool using
the state interpretation from [ADR 9](0009-build-examples-from-consistent-game-state.md).
Fixed sampling and retrieval use that same pool. Tools enforce eligibility before ranking
or reading, preserving split ownership, interpreted branches and target exclusion. Independent
writing references remain distinct from events established in the current playthrough.

Provide tool schemas, dispatch and retrieval traces independently of a model provider.
The caller owns the agent loop and its budgets. The current generation and benchmark runners
continue to consume fixed prompts; the tools enable dynamic selection without claiming an
integrated agent evaluation. The shared machinery lives in `context.py` and `retrieval.py`.

## Consequences

Agents can choose evidence according to the current writing task, while the fixed baseline
remains available over the same permitted material. Retaining fixed sampling as the only
policy was rejected because that policy cannot test adaptive context selection.

Shared eligibility makes the available evidence comparable, but does not equalize retrieved
text, token use, latency or answer quality. Retrieval traces expose the agent's choices;
retrieval alone does not establish a fair end-to-end model comparison.

Prepared datasets retain more text than fixed prompts need. Saved pools inherit the
interpreter's limitations and do not support arbitrary live game saves. Filtering only after
a global search was rejected because excluded scenes could displace eligible results even
when the excluded text was never returned.
