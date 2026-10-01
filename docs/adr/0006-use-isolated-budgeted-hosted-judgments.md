# 6. Use isolated budgeted hosted judgments

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Hosted evaluation incurs costs and can share information between judgments if requests reuse conversation history. Evaluation needs isolated requests, a spending limit and a durable account of failed or uncertain requests.

## Decision

Run the origin-detection task and candidate ordering defined in [ADR 4](0004-evaluate-authenticity-against-game-continuations.md). Each hosted request contains one public trial, the supplied scene context and no previous judgments, tools or answer key.

`hosted_judge.py` runs the configured model through the shared provider adapters. Save requests before sending, reserve conservative cost against the run budget, and record returned usage when available. Unknown outcomes retain their reservation and are not resent. Preserve failures and stop on request/authentication errors or repeated failures; do not switch models automatically.

Save the provider response, model identifier and request settings alongside each judgment. Credentials remain outside saved artifacts.

## Consequences

Hosted calls disclose public trial text to the selected provider and incur costs. Provider reasoning settings are not equivalent, and mutable model aliases limit reproducibility. Usage estimates are not invoices; the budget covers one output directory, not unrelated account activity.

Request isolation prevents earlier judgments from influencing later requests. A failed or uncertain request remains missing from the completed judgments; rerunning the command does not silently resend it.
