# 11. Rank passages and return their parent scenes

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [PR 19](https://github.com/KelianM/endless-voices/pull/19), [retrieval comparison](../../experiments/retrieval-embeddings/README.md), [MaxP document ranking](https://aclanthology.org/2020.findings-emnlp.63.pdf), [parent-document retrieval](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever)

## Context

A scene can contain several topics, while a writing query may concern one exchange. A single
scene embedding can obscure that detail. Retrieving only a matching passage has the opposite
problem: the passage can omit the setup, speaker context or consequences needed to use the
match correctly.

Semantic retrieval therefore needs separate choices for the unit being matched, the score
assigned to a scene and the context returned to the generator.

## Decision

Within the eligibility boundary from
[ADR 10](0010-let-agents-retrieve-context-through-shared-tools.md), embed complete authored
passages with their scene headings and retain their interpreted parent scenes in a static
SQLite database. Preserve distinct scene interpretations. Follow the encoder's prescribed
query format and pooling, and reject oversized inputs rather than silently truncating text.

Compute cosine similarity against every eligible passage. Give each scene the maximum of
its passage scores, return each scene once and apply the result limit to distinct scenes.
Additional matching passages do not increase the scene's score. This is MaxP aggregation
using embedding similarity: the strongest passage determines the document ranking.

Return the complete eligible parent scene by default, together with the passage that caused
the match. Expansion neither re-scores the scene nor retrieves other source branches.
This follows parent-document retrieval: search at passage granularity and supply context at
scene granularity. Passage-only results remain available. The implementation lives in
`src/endless_voices/retrieval.py`.

## Consequences

One relevant exchange can retrieve its surrounding dialogue without diluting its score with
unrelated passages. Exact search scores every eligible vector; “exact” describes the search,
not the correctness of the semantic match. Search work grows with the eligible passage count.

Maximum scoring can favor one incidental match over several consistently relevant passages.
Longer scenes have more opportunities for a high match, and full scenes can consume substantial
prompt space. The result limit is not a token budget. Averaging all passage scores would
penalize relevant exchanges surrounded by unrelated dialogue; averaging the strongest few
would reward repeated support but introduce another selection parameter. Neither alternative
has been evaluated here, and the current use case does not yet justify that added complexity.

The initial truncated scene embeddings were replaced because later passages could not affect
retrieval and generic mean pooling did not follow the selected encoder's interface.
Passage-only output was replaced as the default after qualitative review found incomplete
context despite relevant scene matches. The recorded comparison supports the current default
on those queries, not a general retrieval advantage or better generated dialogue. Passage
indexing adds vectors, while scene expansion reuses the same index without further embedding.
