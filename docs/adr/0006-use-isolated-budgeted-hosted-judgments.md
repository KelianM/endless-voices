# 6. Use isolated budgeted hosted judgments

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Origin judgments can leak answer labels through candidate ordering, and repeated calls can share information. Hosted evaluation also needs a spending limit and a durable account of failed requests.

## Decision

Use the origin-detection task from [ADR 4](0004-evaluate-authenticity-against-game-continuations.md) with one randomized A/B assignment per generated response. Serialize A before B regardless of origin. Each hosted request contains one public trial, the supplied scene context and no previous judgments, tools or answer key. Controls remain separate from primary results.

`hosted_judge.py` runs the configured model through the shared provider adapters. Save requests before sending, reserve conservative cost against the run budget, and record returned usage when available. Unknown outcomes retain their reservation and are not resent. Preserve failures and stop on request/authentication errors or repeated failures; do not switch models automatically.

Reports retain decisions, abstentions, failures, missing trials, recognition flags and explanations. Compared conditions use shared scenes. Uncertainty groups follow source missions and connected variants. Credentials remain outside saved artifacts.

## Consequences

Hosted calls disclose public trial text to the selected provider and incur costs. Provider reasoning settings are not equivalent, and mutable model aliases limit reproducibility. Usage estimates are not invoices; the budget covers one output directory, not unrelated account activity.

Origin detection measures distinguishability, not storytelling quality. Self-judging and source recognition can bias results. Development-set agreement does not establish human approval or unseen performance. Single-order evaluation avoids duplicate calls but does not measure each pair’s sensitivity to candidate position.
