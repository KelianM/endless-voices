# 2. Raw sources are fetched at a pinned revision outside Git

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #2](https://github.com/KelianM/endless-voices/issues/2), [PR #10](https://github.com/KelianM/endless-voices/pull/10), [Merge commit 63a69bb](https://github.com/KelianM/endless-voices/commit/63a69bb)

## Context

Dataset preparation needs repeatable access to upstream text and attributable evidence. The
source inventory covers the complete pinned text corpus, while the reviewed passages cover
only initial identity samples. This record captures the source-storage decision merged in PR #10.

## Decision

[fetch_sources.py](../../scripts/fetch_sources.py) reads the upstream revision and file hashes from
[source-statistics.json](../../data/overview/source-statistics.json). The default checkout is under gitignored `data/local/`.
The script verifies the pinned revision and all inventoried text hashes, retains upstream
attribution files, and reuses verified checkouts offline.

Documentation, source statistics, checksums, and curation evidence remain in ordinary Git.
Raw source payloads are fetched separately; the repository does not use Git LFS. The reviewed
pilot identities do not limit which upstream text files are fetched.

## Consequences

A fresh checkout needs network access once. Changed or incomplete local sources fail verification
rather than being overwritten. Updating upstream evidence requires a deliberate revision and
manifest update. Keeping raw sources in Git or Git LFS was not selected; source availability
therefore depends on upstream access or a previously verified local checkout.

The inventory and identity briefs remain evidence metadata, not training examples. Fetching
sources does not establish permission to release derived datasets or model weights.

[Fetch tests](../../tests/test_fetch_sources.py) cover offline reuse and rejection of changed,
missing, extra, or wrong-revision sources.
