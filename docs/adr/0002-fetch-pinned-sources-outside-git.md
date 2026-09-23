# 2. Source evidence is pinned and fetched outside Git

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #2](https://github.com/KelianM/endless-voices/issues/2), [PR #10](https://github.com/KelianM/endless-voices/pull/10)

## Context

Endless Sky's development text changes over time. A passage used to justify a character's
knowledge or behaviour must remain traceable to the version that was reviewed. Downloading
the latest source whenever data is prepared would let the evidence change without a project
review.

The available source corpus is broader than the passages reviewed for the first identities.
Restricting the download to those passages would make an initial curation choice determine
what later authors can inspect.

## Decision

Fetch the complete inventoried text corpus at one pinned upstream commit. Verify the revision
and file hashes before accepting a checkout, and reuse a verified checkout offline.

Keep the raw checkout under gitignored `data/local/`. Keep the revision, hashes, statistics,
and curation evidence in ordinary Git. Raw sources are neither committed nor stored through
Git LFS ([fetch_sources.py](../../scripts/fetch_sources.py),
[source manifest](../../data/overview/source-statistics.json)).

## Consequences

- Authors can work from the same source snapshot even after upstream changes. Updating the
  evidence requires an explicit revision and manifest change.
- A changed or incomplete local checkout is rejected, not silently repaired. Local edits cannot
  pass as the reviewed source, and recovery requires a separate checkout or deliberate cleanup.
- A fresh project checkout does not contain the raw corpus. Source access depends on an initial
  download or an existing verified copy; subsequent reuse needs no network connection.
- All inventoried text is available for inspection, but availability does not imply review or
  suitability for training. Curation remains separate from fetching.
