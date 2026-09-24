# Hosted judge comparison

Luna and Sol identified all 48 original continuations in both candidate orders. Sonnet identified
47 originals and abstained once in the primary order; Sonnet made one incorrect choice in the
reversed order. The three complete hosted runs cost an estimated $0.980831 from returned usage.
These results support considering Luna for further evaluation, not automatic judge approval.

## Task and coverage

Every primary trial compares one Qwen3-4B generated response with the original game continuation.
Each judge receives the same authored context and anonymously ordered candidates, in a separate
request with no tools or prior conversation. Reversed trials swap the same candidates; the 96
primary/reversed trials therefore represent 48 scenes from only 15 conversations. Two easy
procedural controls test identical-candidate abstention and wrong-context detection separately.

All 48 validation generations succeeded: 22 Free Worlds, nine Republic Navy, nine mainstream Hai,
and eight Quarg. No generation failures or missing responses were excluded from this validation
pack. The full release has 173 successful generations out of 176 and three preserved output-limit
failures; see the earlier [execution record](../prompted-judge-v1/full-generation.json).
Test dialogue was neither inspected nor judged in this comparison. No adapter condition exists.

## Primary results

Each procedure had 48 scheduled primary assessments. Correct and incorrect counts use decided
trials; abstention, failure and missing rates use all 48 scheduled trials.

| Procedure | Correct / decided | Incorrect | Abstained / 48 | Failed / 48 | Missing / 48 | Run cost estimate |
| --- | --- | --- | --- | --- | --- | --- |
| Isolated agents | 48/48 | 0 | 0/48 | 0/48 | 0/48 | Not measured |
| Local Qwen14B | 5/48 | 43 | 0/48 | 0/48 | 0/48 | Local |
| Local Qwen30B | 1/42 | 41 | 0/48 | 6/48 | 0/48 | Local |
| Local Mistral24B, fence parser | 11/48 | 37 | 0/48 | 0/48 | 0/48 | Local |
| GPT-6 Luna, medium | 48/48 | 0 | 0/48 | 0/48 | 0/48 | $0.0228667 |
| GPT-6 Sol, medium | 48/48 | 0 | 0/48 | 0/48 | 0/48 | $0.356404 |
| Claude Sonnet 5, medium, corrected schema | 47/47 | 0 | 1/48 | 0/48 | 0/48 | $0.601560 |

Run costs cover all 98 calls, including reversals and controls. Both OpenAI models agree with all
48 agent primary decisions. Sonnet agrees on 47 decided scenes and abstains on the remaining
Quarg scene. Sonnet's primary abstention rate is 2.08%; the other completed hosted procedures
have zero primary abstentions. All completed hosted procedures have zero failures and missing
assessments. The separate original Sonnet attempt retains one failed request and 97 missing
assessments; the corrected procedure does not erase that failure.

Luna and Sol identify all originals in every identity. Sonnet identifies all 22 Free Worlds,
nine Republic, nine Hai and seven decided Quarg originals, with one Quarg abstention.
Sonnet reports recognizing four primary sources (4/4 correct), leaving 43/43 correct unrecognized
decisions and one unrecognized abstention. Luna and Sol report no recognition. Self-reported
non-recognition does not rule out memorization.

## Controls, order and reasons

All three hosted procedures pass both controls: abstain on identical candidates and choose the
original against the wrong-context alternative. Controls are excluded from detection accuracy.
Luna and Sol also identify 48/48 originals in reversed positions, with no changed selections
among 48 complete decided position pairs each.

Sonnet identifies 47/48 originals in reversed positions and reports five recognized sources.
Sonnet has 47 complete decided position pairs: one changes to the incorrect candidate. One
additional pair is incomplete for decided-pair analysis because the primary judgment abstains
and the reversed judgment chooses the original. These are two changes in judgment, not two
additional scenes. The archived report preserves every position and reason.

The hosted reasons often point to specific contradictions rather than length alone. On
`diana-howl-credit-return-l3086-d2ddde92`, Luna and Sol both identify the generated response's
reward-recipient confusion and invented procedures. On
`commander-hines-harmony-talks-l1937-a6e8c8ac`, both identify a change in speaker stance or
political control. Many other reasons still use brevity as evidence, so this comparison does
not establish robustness to concise, plausible alternatives.

On `fw-northern-2c-freya-city-sensor-l950-90b1731c`, Sonnet initially favors the original's
conversational transition toward New Holland. With positions reversed, Sonnet treats that same
transition as inconsistent with the scene and chooses the generated tactical explanation.
On `quarg-first-contact-l41-66e6073a`, Sonnet abstains initially, then identifies the original
from cadence and lore after reversal. These examples show why retained reasons and order
checks matter even when primary decided accuracy reaches 100%.

## Execution and reproducibility

The runs used `gpt-6-luna`, `gpt-6-sol` and `claude-sonnet-5`; each provider returned that same
identifier on all 98 responses. No immutable revision was supplied. OpenAI used medium reasoning,
structured outputs, `store:false`, default service tier and a 4,096-token total output cap.
Sonnet used adaptive thinking with medium effort, structured outputs and the same total output
cap. No sampling override or tools were supplied. These settings do not equalize provider
reasoning, tokenization or inference. Local procedures used different decoding/output budgets.

Anthropic rejected the initial nullable confidence schema with HTTP 400. An owner-approved
non-generating token-count diagnostic isolated the error. The corrected schema uses an
`anyOf` string enum or null with the same allowed values. The original failed request, original
runner, both diagnostics and corrected run are preserved separately. No judgment was repaired.

The owner stopped the earlier Gemini round. Its partial requests, responses and stop summary
are preserved in the archive but are not treated as completed calibration. No Gemini calls
were resumed. One interrupted Gemini request has unknown outcome, not an invented response.

The archive includes the public/private pack, attribution/licensing, all comparison reviews,
complete hosted requests/responses/settings, source runners, verification script and generated
report. `inventory.json` lists each file hash and the archive hash. Verification checked all
294 completed hosted trial IDs, request hashes, public context hashes, response parsing,
aggregate records, returned model names and usage calculations. Run the archived verification
script from the extracted archive root with `PYTHONPATH=src`; no network calls are made.

Published standard token prices used were Luna $0.10/$0.50, Sol $2/$10 and Sonnet $2/$10 per
million input/output tokens, with provider cache rates when relevant. Prices are recorded in
settings and estimates are not invoices. Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing)
and [Sonnet specifications](https://platform.claude.com/docs/en/models/sonnet-5/overview),
checked 2026-09-24.

## Limits on interpretation

All validation conversations have informed development and model selection. This is an
exploratory procedure comparison on one generator condition, not a held-out estimate, human
agreement study, or base-versus-adapter comparison. Agent sessions had no parent history but
shared filesystem/tool capabilities; serving revisions were not recorded. The owner identified
the parent configuration as Astra medium, which does not recover missing historical metadata.

Conversation/scenario cluster bootstrap intervals are retained in the report. Perfect observed
decisions produce a degenerate [1, 1] interval here; that does not imply certainty about future
scenes. Fifteen groups are few, and resampling excludes model selection, memorization, judge
choice and dataset-selection uncertainty. No universal agreement threshold is applied.
Lower detection is not automatically better dialogue, chance does not prove indistinguishability,
and failure to detect a difference does not establish equivalence. Further judge selection
remains an owner decision. No additional models or training were started.
