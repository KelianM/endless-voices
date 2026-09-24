# 7. Allow budgeted hosted judge experiments

- **Status:** Accepted
- **Date:** 2026-09-24
- **Sources:** [Issue #9](https://github.com/KelianM/endless-voices/issues/9),
  [PR #16](https://github.com/KelianM/endless-voices/pull/16),
  project-owner authorization for Gemini, Luna, Sol and Sonnet in the implementation task

## Context

Local judges disagreed with isolated agent reviews on most validation examples. The owner
then authorized hosted model experiments and supplied provider keys. Restricting experiments
to local inference would prevent comparing those alternatives under the same blinded task.

## Decision

Supersede the local-only execution requirement in
[ADR 6](0006-calibrate-a-local-prompted-judge-with-isolated-agents.md). Retain its authenticity
task, blinding, isolated calls, agent calibration, reporting rules and prohibition on using
test data for development. Allow explicitly authorized hosted models alongside local models.
No judge becomes approved solely because an experiment produces a high agreement percentage.

Hosted runners receive public trials only. Each request contains one trial and no conversation
history, tools or answer key. Provider schemas express the same four judgment fields; provider
reasoning and decoding settings are recorded separately rather than treated as equivalent.
Credentials stay in ignored local configuration and authentication headers. Saved evidence
contains requests, responses, hashes, returned model identifiers and usage, never credentials.

Save requests before sending and preserve results without overwriting. Interrupted requests
with unknown outcomes become failures and are not resent. Paid runs reserve conservative
cost before each request against an explicit budget, charge recorded usage when available,
and retain the reservation when usage is unknown. Error stops require investigation rather
than automatic retries or model fallback. Changed procedures produce separate evidence.

## Consequences

Hosted experiments cost money and disclose the public trial text to the selected provider.
Provider retention and training policies differ; local inference remains an available option.
Model aliases and unavailable revisions limit reproducibility even when exact requests survive.
A recorded medium effort setting does not equalize reasoning across providers.

Usage-based costs are estimates, not invoices; unknown requests can leave reserved cost above
actual billing. The current runners assume one process per output directory. The budget does
not govern unrelated requests made with the same account. Saved failures and missing results
remain visible even when a corrected procedure later succeeds.

Agent agreement is still not human validation. All validation conversations have now informed
judge selection, so success on these examples does not estimate performance on unseen work.
Neither discriminator training nor generator training is introduced by this decision.
