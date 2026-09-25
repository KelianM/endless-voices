# Validation judge calibration

The prompted local Qwen3-14B judge is not suitable for headline authenticity results in this
configuration. The local judge identified 5 of 48 originals and disagreed with the isolated agent
procedure on 43 of 48 primary trials. The isolated agents identified all 48 originals. These are
actual reviews of validation outputs, not simulated judgments or human agreement.

The round used all 48 successful Qwen3-4B validation responses from 15 conversations, covering
four identities. Each response was paired with its original game continuation under the same
profile, selected lore and authored history. Each of the 50 sub-agent calls received only one
blinded trial, without parent history: 48 primary pairs and two controls. The local judge also
reviewed 48 independently presented reversed pairs. No prompt or model setting was changed
in response to these results. No test dialogue was inspected or judged.

| Reviewer procedure | Correct / decided primary | Incorrect / decided | Abstained / scheduled primary | Failed / scheduled primary | Missing / scheduled primary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolated agents | 48/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| Local Qwen3-14B | 5/48 | 43/48 | 0/48 | 0/48 | 0/48 |

| Identity | Generated / selected | Agent correct / decided | Local correct / decided |
| --- | ---: | ---: | ---: |
| Free Worlds | 22/22 | 22/22 | 1/22 |
| Republic Navy | 9/9 | 9/9 | 1/9 |
| Mainstream Hai | 9/9 | 9/9 | 0/9 |
| Quarg | 8/8 | 8/8 | 3/8 |

Validation generation had no failures or missing responses. Both reviewer procedures reported
no recognized sources: recognized primary cases were 0/48 for each procedure, so the full and
unrecognized results coincide. Self-reported nonrecognition cannot rule out pretraining memory.
Full-dataset generation coverage, including the three retained output-limit failures, is recorded
separately in [full-generation.json](full-generation.json).

## Disagreement and controls

Local explanations repeatedly favor elaboration and explicit reuse of the supplied lore. Agent
explanations favor direct replies, distinctive dialogue and continuity with the immediate exchange,
and frequently identify unsupported additions. These patterns suggest that the local judge rewards
plausible role descriptions more than the game's actual dialogue. This interpretation is a reading
of the saved reasons, not a causal test of length bias or a new diagnostic score.

For trial `f194688bb8a05f9124a19acad70b1f91`, the agent preferred the original's concrete parole
terms; the local judge preferred the generated speech for its additional character and situational
depth. For `3d64246160a75d94d3733d54f48a9e80`, the agent identified invented escort details,
while the local judge praised those operational details. For `77c95a232e2573baae1097e0f7953bb0`,
the agent preferred a Quarg invitation to ask questions; the local judge preferred an unsolicited
lore monologue. The archive preserves every reason and all 43 disagreements for inspection.

Both procedures abstained on the identical-candidate control and selected the intended original
on the wrong-context control: 2/2 expected outcomes each. Neither procedure had a failed or
missing control assessment. Passing these two easy controls did not predict primary-trial accuracy.
Controls remain outside the primary detection denominator.

The local judge identified 5/48 originals with reversed positions, but none were the five originals
identified in the primary order. The selected continuation changed in 10/48 pairs; the other 38/48
pairs selected the generated response in both positions. Stable aggregate accuracy therefore hides
position sensitivity. Agents were not assigned reversed trials; the generic report lists those
48 records as missing, meaning unassigned checks rather than lost or failed agent reviews.

## Limits and disposition

The seeded conversation/scenario bootstrap gives descriptive 95% intervals of 100–100% for
agents and 0–21.1% for the local judge, with 15 contributing groups and no undefined resamples.
The degenerate agent interval reflects identical observed outcomes, not certainty about future
performance. Only 15 conversations are available, with one Quarg conversation and two Hai
conversations. Repeated turns and positions do not add independent conversations. The intervals
exclude evaluator uncertainty and dataset-selection uncertainty.

Agent accuracy does not establish human validity. The agents share one procedure, their exact
serving revision and sampling settings are unavailable, and a fresh session does not guarantee
unseen training material. Several conversations appeared in prior development. The local judge
and generator share the Qwen family. Neither agreement nor disagreement identifies the cause of
the local judge's preferences without further experiments.

Keep this judge configuration provisional. Do not turn its low detection rate into a claim that
the generator matches game quality, and do not silently invert its choices. A revised judge or
prompt needs a separately recorded calibration round and owner discussion. No universal agreement
threshold was applied. Only one generator condition exists, so there is no base-versus-adapter
comparison, paired condition difference or evidence of equivalence.

## Evidence and reproduction

[The evidence archive](agent-calibration-evidence.tar.gz) contains the organizer pack and licensing,
all 50 original agent answers, exact agent prompts and assignments, all 98 local reviews with
prompt/settings records, and the full report. [The inventory](agent-calibration-evidence.json)
records the archive hash and every archived file hash. Absolute paths in agent assignments record
the actual invocation; preserve those values as historical provenance. The organizer pack's older
human-gate label predates the owner decision recorded in ADR 6 and does not require a human form.

Extract into a new directory, then reproduce the report from the archived records:

```sh
mkdir outputs/calibration-evidence-copy
tar -xzf data/evaluation/prompted-judge-v1/agent-calibration-evidence.tar.gz \
  -C outputs/calibration-evidence-copy
python -m endless_voices.assessment report \
  --pack outputs/calibration-evidence-copy/pack \
  --review outputs/calibration-evidence-copy/agents/final.json \
  --review outputs/calibration-evidence-copy/local/primary/review.json \
  --review outputs/calibration-evidence-copy/local/reversed/review.json \
  --review outputs/calibration-evidence-copy/local/controls/review.json \
  --output outputs/calibration-reproduced-report
```

The report writes `report.json` for complete records and `report.md` for readable results.
Reporting uses the saved answer mapping and reviews; it does not run inference or access test text.
