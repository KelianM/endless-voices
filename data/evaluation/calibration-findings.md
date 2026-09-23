# Initial authenticity calibration findings

The project owner identified the original continuation in all three primary trials, with high
confidence in each judgment. The [review record](reviews/kelian-primary-v1.json) preserves the
submitted reasons and the hash of the corrected review sheet. The implementing agent checked
choices against the private provenance key after receiving the review.

| Trial | Representative | Submitted choice | Original | Reviewer confidence | Reported evidence |
| --- | --- | --- | --- | --- | --- |
| a-01 | Young Hai resident | A | A | High | In-world vocabulary and a sense of mystery; the alternative felt flatter |
| a-05 | Quarg first-contact speaker | A | A | High | Collective self-reference, unusual mannerisms, and attention to detail conveyed an alien voice |
| a-06 | Free Worlds commander | B | B | High | The conversational opening, hesitation, and stylized phrasing felt more in character than the direct explanation |

There were three correct decisions out of three submitted primary judgments, no incorrect
choices, no abstentions, and no unsubmitted primary trials. The reviewer did not remember the
Hai encounter. Recognition was not explicitly reported for the other two trials and remains
unknown; no missing answer has been inferred as a denial of recognition.

## Interpretation

The reviewer's reasons concern voice and how the speakers express themselves, rather than
formatting or a preference for more factual detail. In these examples, the authenticity question
elicited the kinds of distinctions the project intends to measure. The human example also
shows that the desired signal is not limited to visibly alien syntax.

The authored alternatives were deliberately plausible but turned out to be easy for this
reviewer to distinguish. Their failure to preserve vocabulary, cadence, and dramatic phrasing
is useful calibration evidence. It is not a measurement of any independently run model:
the implementing agent wrote the alternatives while reading the original passages.

Keep authenticity as the primary question and preserve reviewers' reasons in free text.
After completing the sheets, the reviewer found the diagnostic tags unnecessary. The instructions
and blank templates now omit those tags; the submitted reasons already explain the judgments. Do not convert the cited phrases
into mandatory catchphrases, add them to these examples' visible profiles, or require more
ornate writing everywhere. Those changes would reward imitation of these particular answers
rather than the appropriate voice of another scene or individual.

## Control review

The [control record](reviews/kelian-controls-v1.json) preserves the subsequent judgments.
All four match the intended control outcomes; they remain separate from primary results.

| Trial | Control | Submitted choice | Intended outcome |
| --- | --- | --- | --- |
| a-02 | Quarg speech in the militia scene | B | B |
| a-03 | Hai speech in the Quarg scene | A | A |
| a-04 | Militia speech in the Hai scene | B | B |
| a-07 | Identical responses | Abstain | Abstain |

The reviewer described the mismatches as obvious and noticed reused answers. Reuse was deliberate:
the controls substitute already-extracted speech instead of adding more source scenes. However,
the reviewer had seen the primary sheet and had been told which primary choices were correct.
The controls can therefore be answered through recognition as well as contextual fit. They check
that the review procedure and abstention option make sense, not independent contextual reasoning.
No numerical confidence value was supplied for the controls; none has been invented.

## Blinded subagent review

A fresh subagent received the same primary trials without the parent conversation, source labels,
or the human answers. It was instructed to read only the prepared trial prompt, with no browsing
or repository inspection. The [invocation](reviews/subagent-invocation.md) records the prompt
construction and task messages; the [primary response](reviews/subagent-primary-v1.json) preserves
its choices and reasons. Model settings inherited the parent defaults; the exact serving revision
and sampling parameters were not exposed and are not invented in the record.

| Trial | Human choice / confidence | Subagent choice / confidence | Shared reasoning |
| --- | --- | --- | --- |
| a-01 | A / high | A / medium | Culture-specific vocabulary; the alternative smooths away voice |
| a-05 | A / high | A / high | Distinctive alien syntax and mannerisms |
| a-06 | B / high | B / high | Hesitation and tactful phrasing fit the speaker better than an impersonal explanation |

The subagent identified all three originals and agreed with the human on all three choices.
There were no abstentions or reported source recognitions in its primary review. These are
self-reported recognition judgments, not proof that the model never encountered the public text.
The lower confidence on the Hai example is preserved rather than reconciled with the human rating.

The reasons converge on style without requiring diagnostic tags. That supports retaining the
free-text explanation and testing harder alternatives next. It does not establish human–model
agreement beyond these three cases. The implementing agent authored the alternatives and the
judge inherited the same task defaults, so shared model tendencies are another possible influence.

The same subagent then completed the [control sheet](reviews/subagent-controls-v1.json), without
receiving correctness feedback on its primary choices. It selected B, A, B, and abstain, matching
all four intended outcomes and the human choices. It explicitly remembered the repeated passages
in every control. Like the human control review, this is a procedural check rather than independent
evidence of contextual sensitivity.

## Outcome and follow-up

The initial human calibration round is complete. The authenticity question produced relevant
voice-based explanations, and the controls behaved as intended. Keep the primary protocol.
The control review does not justify additional routine human labeling of equally obvious cases.
Recorded review-sheet hashes identify the sheets actually seen, before diagnostic tags were
removed. The seven submitted judgments and reasons are unchanged; only empty diagnostic fields
were removed from the stored records.

For future evaluator validation, use fresh source scenes for human controls and isolate automated
judge calls from previous trials and keys. Include subtler alternatives that preserve surface
style while failing scene or speaker context, alongside obvious sanity checks. Actual model
outputs are needed before assessing an automated judge's usefulness for model comparison.

Candidate-order sensitivity and agreement between independent human reviewers remain unmeasured.
The human–subagent comparison covers only three authored examples; it does not establish evaluator
reliability or an indistinguishability threshold. No statistical or model-quality conclusion is drawn. The reviewed
scenes and their known variants remain reserved for development, not final testing.
