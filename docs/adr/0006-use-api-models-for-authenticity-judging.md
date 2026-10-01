# 6. Use API models for authenticity judging

- **Status:** Accepted
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

Local judge inference was explored, but available hardware restricts the size and capability of runnable models. Judge capability matters because a weak judge can make distinguishable generated writing appear authentic. Requiring the judge to fit the same machine as the generator would make local hardware a constraint on the evaluation itself.

The benchmark needs access to frontier-level judges even when the generator runs locally. API usage is justified for that role, with explicit spending limits and inspectable decisions. Earlier comparisons affected by candidate-order leakage do not establish the reliability of any judge.

## Decision

Use provider API models for the authenticity task defined in [ADR 4](0004-evaluate-authenticity-against-game-continuations.md), rather than requiring a locally hosted judge. Judge selection is independent of generator deployment: a locally runnable model can be evaluated by an API model. The benchmark configuration selects the judge; the decision does not require the largest or most expensive model.

`hosted_judge.py` executes judgments through shared provider adapters. Keep one isolated request per trial and enforce an explicit run budget. Save requests, responses, model identifiers, settings and usage so decisions and failures remain inspectable. Failed or uncertain requests are retained without silent retries or model substitution.

## Consequences

Judge capability is no longer bounded by local memory, and local generation remains possible. Evaluation now depends on provider availability, incurs costs and sends public trial text to the provider. Fully offline evaluation is not supported by the current judge runner.

Provider settings and mutable model aliases limit reproducibility. Spending limits constrain a run but do not account for unrelated account activity. API access does not establish judge reliability: controls, explanations, source recognition and self-preference still need review, and origin detection remains distinct from storytelling quality.
