# 11. Find relevant passages and return their scenes

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [PR 19](https://github.com/KelianM/endless-voices/pull/19), [retrieval comparison](../../experiments/retrieval-embeddings/README.md)

## Context

A query may match one exchange in a scene. Embedding the whole scene can obscure that exchange,
while returning only the passage can omit the setup or what happens next. Retrieval searches
the eligible history and writing references defined in
[ADR 10](0010-let-agents-retrieve-context-through-shared-tools.md).

## Decision

- **Search complete authored passages,** using a dedicated embedding model. Passages and
  queries that exceed the model's input limit fail rather than being truncated.
- **A scene's score is its highest passage similarity.** Search compares the query with every
  eligible passage using cosine similarity. Several matching passages from one scene do not
  increase its score.
- **Return the whole scene once,** with the matching passage identified. The result limit
  counts distinct scenes. Passage-only retrieval remains available.
- **Keep passage vectors and interpreted scenes in a static SQLite database.** The retrieval
  session determines which scenes the current encounter permits.

## Consequences

- **One strong match can outweigh several weaker matches.** Averaging passage scores was not
  chosen because unrelated dialogue would lower the score of an otherwise useful scene.
- **Scene expansion supplies surrounding context at a token cost.** Passage-only output was
  replaced as the default after qualitative review found relevant passages that needed that
  context; a full scene can also bring unrelated dialogue.
- **Longer scenes have more chances to match.** A high scene score does not establish that
  the whole scene is relevant.
- **Search cost grows with the eligible passage count.** Every eligible vector is scored.
- **The initial scene-prefix embeddings omitted later passages.** Complete passage indexing
  replaced that approach so text near a scene's end can also influence retrieval.
