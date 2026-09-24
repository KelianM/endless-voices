# Local judge comparison

Neither larger candidate provides reliable authenticity judgments in this round. Mistral Small
3.2 24B identified 11/48 originals, while Qwen3-30B-A3B identified 1/42 originals among valid
primary assessments. The existing Qwen3-14B baseline identified 5/48. These results compare
judge procedures on one generator condition; they are not generator model comparisons.

All models ran locally, sequentially, on an Apple M5 Pro with 24 GiB unified memory. The two
new 4-bit checkpoints are pinned in `configs/judge-qwen3-30b.json` and
`configs/judge-mistral-small-24b.json`. Peak MLX memory was 17.88 GB for Qwen30B and 14.02 GB
for Mistral24B. MLX peak memory excludes other applications and is not total system memory.

The same 48 validation responses, authored contexts, candidate ordering and judge instructions
were used for all models, with 48 separate reversed trials and two controls. Generation coverage
was 48/48 successful, with no missing outputs. The data covers 15 conversations. No test dialogue
was inspected or judged. The isolated agent procedure provides a reference, not human validation.

| Procedure | Correct / decided | Incorrect / decided | Failed / 48 primary | Abstained / 48 | Missing / 48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isolated agents | 48/48 | 0/48 | 0/48 | 0/48 | 0/48 |
| Qwen14B, strict JSON | 5/48 | 43/48 | 0/48 | 0/48 | 0/48 |
| Qwen30B, strict JSON | 1/42 | 41/42 | 6/48 | 0/48 | 0/48 |
| Mistral24B, JSON fence accepted | 11/48 | 37/48 | 0/48 | 0/48 | 0/48 |

The Qwen30B primary failure rate was 12.5%. Six answers used plain key-value text instead of
JSON. Four reversed answers had the same failure. Every original output remains preserved;
no choices were inferred from invalid records. Agreement with agents was 1/42 for Qwen30B and
11/48 for Mistral; disagreement was 41/42 and 37/48 respectively.

Mistral's first one-trial smoke returned valid JSON inside a Markdown fence and failed the
strict parser. After reporting that failure, a separate run enabled `--allow-json-fence`.
The option removes one enclosing Markdown fence, preserves raw output, and still validates all
four fields without coercion. The prompt, decoding and budgets did not change. The original
smoke remains failed and is excluded from full-calibration denominators. This operational change
means the model comparison uses explicitly different parsing procedures.

| Identity | Generated / selected | Qwen30B correct / decided | Qwen30B failed / scheduled | Mistral correct / decided |
| --- | ---: | ---: | ---: | ---: |
| Free Worlds | 22/22 | 1/21 | 1/22 | 2/22 |
| Republic Navy | 9/9 | 0/6 | 3/9 | 2/9 |
| Mainstream Hai | 9/9 | 0/8 | 1/9 | 5/9 |
| Quarg | 8/8 | 0/7 | 1/8 | 2/8 |

Neither new judge reported source recognition on valid primary records: 0/42 for Qwen30B and
0/48 for Mistral. The unrecognized subset therefore has the same detection accuracy as the full
valid subset. Recognition remains unknown for failed assessments; self-report cannot exclude
pretraining memory.

## Position sensitivity and controls

Qwen30B identified 2/44 originals in reversed order, with four failed assessments. Of 48
primary/reversed pairs, 40 had valid decisions in both positions: two changed the selected
continuation and 38 selected the generated response twice. Eight pairs were incomplete because
one or both judgments failed. Mistral identified 8/48 originals in reversed order and changed
its selected continuation on 19/48 complete pairs. None of Mistral's 11 correct primary choices
remained correct after reversal. Agents had no reversed assignments.

Both new judges selected the expected answer on the wrong-context control. Both attempted to
abstain on identical candidates but returned the string `"null"` for confidence instead of JSON
`null`. Each procedure therefore has one correct control and one failed control out of two
scheduled controls, with no missing controls. No malformed abstention was promoted to valid.
Controls are excluded from primary detection rates.

## Reasons and interpretation

Both larger judges often prefer additional explanation, lore and strategic detail. In trial
`f194688bb8a05f9124a19acad70b1f91`, Qwen30B correctly prefers concrete parole terms, but Mistral
prefers the generated speech's strategic depth. In `3e9acbec49024bfa80213af30071dd5d`, both
models prefer generated logistical detail over the original brief reply. Mistral sometimes
recognizes excessive exposition: on `77c95a232e2573baae1097e0f7953bb0`, Mistral prefers the
original Quarg invitation to ask questions over a lore monologue. These are interpretations of
saved reasons, not a causal test of verbosity bias. All reasons and disagreements are archived.

Descriptive 95% conversation/scenario bootstrap intervals are 0–8.7% for Qwen30B and 11.1–36.2%
for Mistral, using 2,000 seeded draws and 15 contributing groups, with no undefined draws.
The intervals exclude judge uncertainty and selection effects. Repeated turns and positions do
not add independent conversations. The initial results prompted these model choices, and all
validation conversations have now been used for development; this is not fresh confirmation.

Keep both configurations provisional. More parameters did not resolve the observed failure,
and switching families did not produce stable judgments. Mistral's higher observed detection
rate does not establish suitability. No universal agreement threshold was applied, no default
judge was replaced, and no choices were inverted. Lower detection is not automatically better
generation quality; chance performance does not establish indistinguishability or equivalence.

## Evidence and reproduction

[The archive](evidence.tar.gz) preserves the pack and attribution, agent reference, all three
local judges' complete reviews, smoke failures, exact prompts/settings, execution commands,
runner code and comparison report. [The inventory](inventory.json) records every file hash and
the archive hash. The earlier [agent findings](../prompted-judge-v1/agent-findings.md) preserve
the original interpretation and detailed full-dataset coverage.

Extract into a new directory and regenerate the report without inference:

```sh
mkdir outputs/judge-comparison-copy
tar -xzf data/evaluation/local-judge-comparison-v1/evidence.tar.gz \
  -C outputs/judge-comparison-copy
python -m endless_voices.assessment report \
  --pack outputs/judge-comparison-copy/pack \
  --review outputs/judge-comparison-copy/agents.json \
  --review outputs/judge-comparison-copy/baseline/primary/review.json \
  --review outputs/judge-comparison-copy/baseline/reversed/review.json \
  --review outputs/judge-comparison-copy/baseline/controls/review.json \
  --review outputs/judge-comparison-copy/qwen30/primary/review.json \
  --review outputs/judge-comparison-copy/qwen30/reversed/review.json \
  --review outputs/judge-comparison-copy/qwen30/controls/review.json \
  --review outputs/judge-comparison-copy/mistral/primary/review.json \
  --review outputs/judge-comparison-copy/mistral/reversed/review.json \
  --review outputs/judge-comparison-copy/mistral/controls/review.json \
  --output outputs/judge-comparison-reproduced
```
