# Authenticity evaluation

The first benchmark asks whether a generated reply could pass for a continuation written
for Endless Sky. The primary comparison presents the original continuation and one generated
alternative under identical speaker and scene context. The judge identifies the original and explains the choice in free text. No diagnostic tags
or separate persona scores are requested. [ADR 4](../../docs/adr/0004-evaluate-authenticity-against-game-continuations.md) records
this choice.

This issue supplies a protocol and a small development calibration pack. No discriminator is
trained, no generator is fine-tuned, and no model-quality result is claimed. The [initial human review](calibration-findings.md) identified all three originals with high
confidence, citing vocabulary, mannerisms, and phrasing. All four subsequent control judgments
matched their intended outcomes. A fresh subagent also identified all three originals without
seeing the human answers, with similar reasons. This small comparison does not establish evaluator
reliability. The alternatives are authored examples, not independently generated model outputs.

## What the experiment measures

For each reserved scene, generate one reply from the prompted base model and one from the
same checkpoint with its adapter. Both receive the same profile, selected lore, and authored
history. Each condition is judged separately against the **same original game continuation**;
base and adapted responses are not the two alternatives in the authenticity trial.

| Visible to the generator | Visible to the authenticity judge | Private to the organizer |
| --- | --- | --- |
| Profile, selected lore, authored history ending in a user turn | The same context and two anonymously ordered continuations | Which continuation is original, source links, model condition, private expectations, run metadata |

The generator must never receive the withheld target. The authenticity judge necessarily sees
that target as one unlabelled alternative; withholding its identity is different from hiding
its text. The judge receives no third reference answer or hints about preferred phrasing.
Reviewers explain their choices in their own words. No second assessment pass is required.
Private expectations remain outside the authenticity judgment.

The shared sample contract remains unchanged. `messages[-1]` holds the extracted game speech,
and `evaluation_messages(sample)` supplies the generator's context. Authored histories stay
fixed. This is a test of next-response authenticity in selected scenes, not free-running
conversation, universal lore accuracy, or the absence of memorized public text.

## Evidence behind the method

| Source | What we use | What we do not assume |
| --- | --- | --- |
| [Li et al., 2017](https://aclanthology.org/D17-1230/), sections 4.1–4.3 | Real/generated discrimination as an evaluation question; controls for evaluator weakness and incoherent responses | Fooling one evaluator proves dialogue quality, or adversarial training is required |
| [Bruni and Fernández, 2017](https://aclanthology.org/W17-5534/) | Compare automated discrimination with human judgments | Origin labels remove the need to check evaluator validity |
| [CharacterEval](https://aclanthology.org/2024.acl-long.638/), section 5 | Background on persona evaluation considered when choosing the method | Its dimensions, numerical scales, human-likeness, or empathy are required judge fields |
| [RAIDEN](https://aclanthology.org/2025.coling-main.735/), section 4 | Fixed-context, blinded response pairs and reasons for judgments | Preference between responses is identical to identifying their origin |
| [InCharacter](https://aclanthology.org/2024.acl-long.102/), section 3 | Behavioral probes can expose more than superficial style | Human personality inventories define an alien species or political faction |
| [RoleLLM](https://aclanthology.org/2024.findings-acl.878/), section 4 | Role knowledge and speaking style are distinct aspects of evaluation | Reference word overlap is a sufficient authenticity metric |

Our forced-choice pair with abstention is an adaptation, not a reproduction of any paper's
benchmark or a validated replacement for human judgment. A source origin is known without
manual labeling; whether a judge uses meaningful evidence still needs calibration.

## Prepare the development review

From the repository root, with the project installed:

```sh
python scripts/fetch_sources.py
python scripts/prepare_authenticity_calibration.py \
  --output data/local/authenticity-calibration-a --form a
# For an independent reviewer with every candidate position reversed:
python scripts/prepare_authenticity_calibration.py \
  --output data/local/authenticity-calibration-b --form b
```

Fetching reuses the verified local snapshot offline. Preparation itself never accesses the
network or loads a model. The recipe is [calibration.json](calibration.json). Selected lines
are checked against inventoried file hashes and passage ranges. The preparer selects explicit
quoted speech spans, omits narrator insertions, and joins successive speech spans with one space.
The Hai recipe explicitly changes the comma before the removed narrator tag to a full stop,
so extraction does not create a comma splice. Original wording is retained; normalized punctuation
is declared in the recipe. The script does not parse arbitrary game dialogue or resolve branches
automatically. Unresolved runtime
placeholders fail preparation. The selected branches and system summaries have been inspected
against the pinned source. This normalization defines **spoken-response authenticity**, not
imitation of raw game markup or full narrated prose.

Each of the three samples contains an authored scene summary, an extracted player question,
and extracted target speech. No full earlier transcript is claimed. Context omits summaries of
the withheld answer: for example, the young Hai profile states age but does not instruct the
model to defer that particular question to elders. Later authors must inspect such leakage
semantically; schema validation alone cannot detect it.

| File in the local output directory | Use |
| --- | --- |
| `review.md`, `review.jsonl` | Three blinded original-versus-authored trials; give only these and the blank review template to a reviewer |
| `review-template.json` | Record independent primary judgments before opening controls |
| `controls.md`, `controls.jsonl`, `controls-template.json` | Three wrong-context speech substitutions and one identical pair; administer after primary review |
| `samples.jsonl` | Three draft validation samples in the existing contract; organizer only |
| `answer-key.json` | Origin mapping and control expectations; organizer only |
| `ATTRIBUTION.md`, `license.txt`, `copyright`, `credits.txt` | Source provenance and licensing material; retain with the complete local pack |

The recipe contains source selectors and agent-authored alternatives rather than copied target
speech. Materialized source excerpts remain in gitignored `data/local/`. The local pack retains
upstream license material and notices; do not redistribute excerpts without those materials.
Output directories must be new to protect recorded reviews from overwrite.

These alternatives were authored by the implementing agent **with access to the source**.
They are worked calibration examples, not independently sampled model responses. Their ease
or difficulty says nothing about a base model or adapter. Three cases are enough to expose
some procedural problems, not to estimate evaluator reliability.

## Calibration and human review

Use the [judge instructions](judge-instructions.md) unchanged for the primary review. The first
review can be the project owner's independent judgments. The implementation agent has seen the
sources and written the alternatives, so its explanations cannot count as independent labels.
Save original judgments before opening the key or discussing disagreements. A reviewer who
recognizes a source should mark that fact even if the passage is otherwise convincing.

The controls ask whether the reviewer attends to **this scene**, rather than recognizing game
prose anywhere. Both alternatives in a wrong-context control are real source text, but only one
continues the presented scene. These controls are not real/generated trials and never enter
the primary detection rate. Identical pairs should produce abstention. Controls are intentionally
obvious and reuse previously seen passages; passing them does not independently demonstrate
context sensitivity or sensitivity to subtle lore or voice errors. Future evaluator validation
should use fresh scenes for human controls and isolated calls for automated judges.

For each surprising judgment, retain the choice and reason, inspect the evidence, and record
whether the disagreement concerns recognition, presentation, context, or the writing itself.
Do not adjudicate a plausible alternative into a quality failure merely because its origin is
known. Revise unclear instructions or flawed contexts, then use fresh examples for another
calibration round. Disclose reuse; repeated reviews are not independent evidence.

Compare reversed candidate positions with independent reviewers or isolated LLM calls. Count
order changes as sensitivity, not extra independent scenes. A single human review provides
initial feedback; reviewer agreement needs at least two independent judgments on the same
items. No human inter-reviewer agreement is available. The recorded human–subagent comparison
covers only three cases. Preserve disagreements rather than replacing them with a consensus label.

The original protocol required fresh human comparisons before automated headline results.
The owner replaced that requirement for the current experiment with isolated agent review on
2026-09-24, as recorded in [ADR 6](../../docs/adr/0006-calibrate-a-local-prompted-judge-with-isolated-agents.md).
Compare the local judge's choices and reasons with fresh sub-agent calls on actual validation
outputs. Check recognition, position effects, context sensitivity and abstention. Preserve and
discuss discrepancies; no universal agreement threshold is specified. Agent agreement must not
be described as human agreement. Failed controls or dominant formatting cues limit interpreting
the detection rate as authenticity.

## Reporting rule

Report original-identification accuracy among decided primary trials, together with counts of
correct, incorrect, abstained, and failed trials and their denominators. Report abstention and
failure rates over all scheduled primary trials. Never silently remove abstentions or failures.
Show recognized-source cases separately alongside the full results, rather than deleting them
post hoc. Report controls separately. Preserve free-text reasons without imposing a category scheme or weighted persona score.

With balanced A/B positions and no abstentions, random choice identifies the original 50% of
the time. With selective abstention, accuracy is conditional on the decided cases and cannot
be read alone. With three cases, a position-only strategy can appear successful; the calibration
pack cannot establish a model ranking. An evaluator far below chance may have reversed labels
or a systematic preference for generated prose; lower accuracy is not an unbounded quality reward.

For the eventual model comparison, show detection rates for each condition and their paired
change on the same scenes, alongside abstention, failures, recognition, and the reviewers' reasons.
Cluster uncertainty estimates by conversation/scenario family; repeated turns, reviewer votes,
and reversed positions are not independent examples. Report counts by representative so a
large faction does not silently dominate. The exact reporting implementation belongs to #9.

Being statistically unable to distinguish a model from source text is not proof of equivalence.
A claim of practical indistinguishability requires a justified tolerance and sufficient independent
scenes chosen before the final run. This pilot makes no such claim and sets no arbitrary pass score.

## Split reservations and remaining work

All three examples are development material (`split: validation`). Reserve their conversations
and known source-derived variants from the final test set, including the actual Free Worlds
scan-request conversation and its branch variants. Separate encounters in that chain are no
longer reserved solely because they share the chain. Quoted control donors and known alternatives
remain development material. A background citation alone does not reserve a different encounter: the
prior transport assignment is a shared story fact, while its dialogue is a separate conversation.
Shared canonical facts may still inform unrelated scenes in other splits. Do not create fake
training or test files merely to satisfy a complete-manifest check; `read_records(path,
"validation")` validates this development-only file. The [pilot release](../README.md) retains these conversation reservations
and adds selected lore to its newly prepared contexts. The legacy calibration remains reproducible
with its original short contexts; the recorded judgments are not relabelled as reviews of new prompts.

- #5 supplies independent scenes, attributable original continuations, coverage, and final split
  review. Agent-written targets can serve training or development, but cannot be labelled original
  game continuations for this authenticity measure. Do not use the public calibration scenes as
  final test material.
- #8 generates replies only, preserving the same context for base and adapter. Record selected
  checkpoint, settings, and failures; do not train a discriminator or generator there.
- #9 implements blinded judgments and reports after calibration. Use separate judge sessions for
  each pair; do not show the original repeatedly in a batch that reveals its identity by repetition.
- #12 is the separate fine-tuning experiment. Training against a discriminator would be a new
  training decision, not implied by choosing authenticity evaluation.

Primary and control judgments are recorded in [calibration findings](calibration-findings.md).
The initial human calibration is complete. The current agent-based calibration on actual validation outputs remains separate evidence
from that initial human review.

## Where review evidence lives

Keep calibration evidence in this repository, not only in conversational memory:

- `reviews/` preserves submitted judgments and reasons, reviewer identity/type, and the exact
  review sheet or prompt hashes. Missing information stays unknown. Revisions do not rewrite
  what a reviewer originally said.
- [Calibration findings](calibration-findings.md) interprets the evidence, compares reviewers,
  records limitations, and states what changes or follow-up the findings justify.
- The ADR records the lasting evaluation choice; individual trial results belong in the evidence,
  not in a growing decision log.

The next dataset and evaluator work should read these findings before authoring new calibration
material. Preserve the primary insight—voice can disappear in an otherwise sensible paraphrase—
without turning specific phrases into required catchphrases. Add new review records for fresh
scenes and model outputs, retaining the earlier results for comparison.

## Generation infrastructure smoke samples

[smoke-sample-ids.json](smoke-sample-ids.json) selects four `pilot-v1` validation samples, one
per identity. Selection takes the first multi-turn sample for each identity in file order,
so the check exercises preserved authored history. These are infrastructure checks; the
selection is neither a representative quality sample nor fresh judge-calibration evidence.
No test samples are generated or inspected for tuning. The selection does not alter split
membership or the frozen release.

Use the [generation command](../../README.md#generate-comparable-responses) to collect responses and explicit failures.
The pinned Qwen3-4B-Instruct-2507 configuration is a replaceable smoke-run checkpoint. Judge
validation still needs fresh scenes and actual model outputs under the protocol above.

## Prompted local judge implementation

The [assessment workflow](../../README.md#assess-saved-responses) verifies saved generation runs,
exports isolated blinded trials, runs a local MLX judge, and produces reproducible reports.
The [first prompted-judge round](prompted-judge-v1/README.md) records selection and execution.
Agent comparison is underway under the revised protocol; implementation does not establish judge reliability.
A trained discriminator and adversarial generator training remain outside this workflow.
