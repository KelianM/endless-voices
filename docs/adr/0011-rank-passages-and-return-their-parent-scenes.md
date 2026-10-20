# 11. Find relevant passages and return their scenes

- **Status:** Proposed
- **Date:** 2026-10-13
- **Sources:** [PR 19](https://github.com/KelianM/endless-voices/pull/19), [issue 21](https://github.com/KelianM/endless-voices/issues/21), [PR 22](https://github.com/KelianM/endless-voices/pull/22), [retrieval comparison](../../experiments/retrieval-embeddings/README.md)

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
- **Keep authored passages and their vectors in a static SQLite database.** Preserve every
  placeholder, including values that currently resolve to only one name or location.
- **Bind results to the current encounter's interpreted scenes.** The retrieval session
  determines eligibility and returns each scene with its owning mission's substitutions.
  Different sampled states may share vectors without sharing rendered context.
- **Keep the generated index local and ignored.** Rebuild it from the prepared dataset and
  cached encoder instead of storing successive index versions in Git LFS.

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
- **Resolved names and locations do not influence passage embeddings.** Queries mentioning
  those values may rank differently. Embedding substituted text was rejected because a
  changed player name or sampled origin required rebuilding otherwise unchanged vectors.
- **Reuse requires the same authored passages.** A different branch or world-event
  description still needs indexing if the database does not contain that authored variant.
- **Each machine must build its own index.** Local storage avoids accumulating full index
  versions in LFS, at the cost of repeating context reconstruction and embedding.
