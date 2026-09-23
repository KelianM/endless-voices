# 4. Judge authenticity against original game continuations

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #6](https://github.com/KelianM/endless-voices/issues/6), [PR #13](https://github.com/KelianM/endless-voices/pull/13), [Li et al., 2017](https://aclanthology.org/D17-1230/)

## Context

The intended experience is a conversation with someone who feels part of Endless Sky. A
persona rubric substitutes the evaluator's definition of a good character for that experience.
The game already contains examples of the authors' desired dialogue, including choices of
rhythm, restraint, and personality that a profile may fail to capture.

An original reply is evidence of the desired writing, not the only acceptable reply. A generated
alternative could fit equally well. Requiring the original to win every quality comparison
would penalize successful alternatives as well as poor ones.

## Decision

The [evaluation protocol](../../data/evaluation/README.md) makes source authenticity the primary
question: given the same speaker and scene, can a blinded judge identify the original game
continuation among an original and a generated reply? The base and adapted model are each
compared against the same source continuation, with the same supplied context.

Judges give an authenticity choice and a free-text reason. No diagnostic categories or separate
persona scores are requested; the initial review found the reasons sufficient to explain choices.
Original-identification accuracy is reported with abstentions,
failures, recognition, and control results; chance performance alone does not establish quality.

The [development calibration pack](../../data/evaluation/calibration.json) uses source continuations,
authored alternatives, wrong-context speech, and identical pairs. The
[initial human review](../../data/evaluation/calibration-findings.md), including controls, is recorded.
No automated discriminator or adversarial training loop is implemented. The fixed-context boundary
from [ADR 3](0003-use-one-conversation-format-across-splits.md) remains unchanged.

## Consequences

- The source supplies origin labels without manually ranking every answer. A judge still needs
  validation: it may recognize a passage or its formatting rather than assess contextual fit.
- This measure needs an attributable original continuation. Newly invented scenarios can support
  development or other evaluations but cannot supply original game answers by assertion.
- A valid generated alternative is not automatically a failure when distinguishable. Origin
  detection and quality are related questions, not interchangeable labels.
- Normalizing both alternatives to direct speech makes the comparison about spoken replies,
  not narrator prose or game markup. Extraction and scene summaries need review.
- A weak judge can make a poor generator look authentic. Controls and human calibration remain
  necessary even though origin labels are automatic. The small pack cannot establish reliability.
- The result concerns continuing selected game scenes. It does not establish fidelity during a
  freely evolving conversation or eliminate possible memorization of public source text.
