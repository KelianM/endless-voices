# 6. Calibrate a local prompted judge with isolated agents

- **Status:** Superseded by ADR 7
- **Date:** 2026-09-24
- **Sources:** [Issue #9](https://github.com/KelianM/endless-voices/issues/9),
  [PR #16](https://github.com/KelianM/endless-voices/pull/16),
  project-owner decisions in the implementation task on 2026-09-24

## Context

Saved model responses need an evaluator that can assess more examples than the project owner
can reasonably review. The owner initially offered 10–20 judgments, then explicitly chose
isolated sub-agents to perform the calibration reviews instead. Evaluation must run without
paid inference APIs, and the available machine has an Apple M5 Pro with 24 GiB of unified memory.

The initial calibration in [ADR 4](0004-evaluate-authenticity-against-game-continuations.md)
used three easy, agent-authored alternatives. Successful judgments on those cases do not
establish reliability on actual model responses. Existing source labels identify originals,
but origin detection can exploit length, formatting or memorization rather than voice and context.

## Decision

Retain the authenticity task from ADR 4: compare each generated continuation with the original
game continuation under the same authored profile, selected lore and history. Compare model
conditions separately against the same original; do not ask the judge to choose directly between
base and adapted responses. Record A, B or abstain, confidence, a short free-text reason, and
source recognition. Do not add persona scores, diagnostic tags or expectation checklists.

Use a prompted local language model as the operational judge. The initial implementation uses
a pinned 4-bit MLX conversion of Qwen3-14B in a separate environment from generation. The exact
checkpoint, decoding settings, prompt, tokenizer inputs, software versions and artifact hashes
are recorded by [judge.py](../../src/endless_voices/judge.py). These settings are replaceable
experimental choices, not permanent architectural requirements. Neither the judge nor the
generator is trained as part of this workflow.

Calibrate the local judge against fresh sub-agent invocations on validation examples. Each
sub-agent receives one trial without parent-task history and may read only its prepared prompt.
Sub-agents return one short reason and the same structured fields as the local judge. Preserve
the actual invocation, prompt hash, agent identity and response; unavailable serving revisions
and sampling settings remain unknown. A procedure-level summary of these calls is not a panel
of independent human reviewers. The owner has replaced the earlier requirement for human
calibration in this experiment; do not describe the resulting evidence as human validation.

Keep all judge calls isolated, including reversed candidate positions and controls. Public trial
files contain opaque identifiers, authored context and two anonymously ordered continuations.
Origin labels, source links, model conditions, run provenance and randomization keys remain in
the organizer pack. Candidate dialogue is data, not instructions. Verify dataset/run hashes,
selection and compatible authored contexts before preparing trials. Never overwrite prior runs
or submitted judgments.

Compare choices and short reasons, preserve disagreement, and inspect source recognition,
position changes, abstention and control behavior before interpreting the judge's usefulness.
There is no universal agreement threshold or automatic approval from an agreement percentage.
Report the evidence and limitations for owner review. Keep the test set outside exploratory
calibration and judge tuning, even when test responses have already been generated.

Report original-identification accuracy among decided primary trials, with explicit counts and
denominators for incorrect, abstained, failed and missing assessments. Retain generation coverage
and failures separately. Show recognized-source cases, identity coverage, controls, reversed-order
checks and free-text reasons. Report each reviewer procedure separately without majority voting.
Paired condition differences use shared decided scenes from the same reviewer procedure and
show incomplete pairs. Uncertainty calculations group related conversation/scenario samples;
additional turns, votes and reversed positions do not add independent scenes.

## Consequences

- Local inference avoids paid inference APIs for the operational judge. Sub-agent calibration
  still consumes the user's existing agent usage and depends on that service; it is not fully
  local or reproducibly pinned in the same way as the MLX model.
- Agent review removes the requirement for the owner to label the calibration examples. Shared
  model tendencies can produce agreement without matching human judgments. Human agreement and
  independent human inter-reviewer reliability remain unmeasured in this expanded round.
- The validation split has 48 samples from only 15 conversations. Reviewing every sample improves
  coverage but does not supply 48 independent conversations. Some validation conversations have
  appeared in earlier work, and fresh agent sessions do not eliminate possible pretraining memory.
- Reversed-order and control calls expose some evaluator weaknesses; easy controls do not
  establish sensitivity to subtle errors. Failed controls or dominant presentation cues limit
  interpretation rather than being hidden by aggregate accuracy.
- A local judge can share preferences with the Qwen generator. A larger judge is not assumed to
  be reliable solely because of its parameter count. Changing a judge or prompt requires new
  recorded evaluation evidence, not rewriting prior judgments.
- The report measures detection under selected contexts. Lower detection does not automatically
  mean better dialogue; chance performance does not prove indistinguishability, and failure to
  detect a difference does not establish equivalence.
- Training a discriminator or optimizing the generator to fool a judge remains a separate
  experimental choice. Available origin labels make that approach possible but do not make it
  part of this implementation.
