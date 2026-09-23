# Initial human calibration findings

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

No primary instruction change follows from this review. Keep authenticity as the primary
question and record voice-related observations as diagnostics. Do not convert the cited phrases
into mandatory catchphrases, add them to these examples' visible profiles, or require more
ornate writing everywhere. Those changes would reward imitation of these particular answers
rather than the appropriate voice of another scene or individual.

## Remaining calibration

The four control judgments have not been submitted. Candidate-order sensitivity, independent
reviewer agreement, and automated-judge agreement have not been measured. Three deliberately
small examples and one reviewer do not establish evaluator reliability or an indistinguishability
threshold. No statistical or model-quality conclusion is drawn from the three correct choices.

Complete the control review next, retaining it separately from primary detection results.
Before relying on an automated judge, use fresh development scenes and actual model outputs,
including alternatives that preserve surface style but fail contextual fit. That will test
whether the judge detects more than the obvious style differences exposed here. These reviewed
scenes and their known variants remain reserved for development, not final testing.
