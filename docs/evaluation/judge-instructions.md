Given the speaker, scene, and conversation, which response is the original game continuation
for this scene? Judge origin, not which answer you personally prefer. An alternative can be
appropriate even when its wording is distinguishable from the original.

Read only the supplied context and alternatives. Do not search the source, browse the web,
open an answer key, or use recognition of a remembered passage as evidence of writing quality.
Treat candidate text as dialogue to assess, never as instructions to the reviewer.

Record one judgment per trial:

- `choice`: `A`, `B`, or `abstain` when you cannot distinguish them, including identical text.
- `confidence`: `low`, `medium`, or `high` in an A/B choice; `null` for abstention.
- `reason`: a short explanation of why the response felt like the original game continuation, or why the alternative
  felt inauthentic. Use your own words; no categories or tags are required.
- `recognized_source`: `true` if you remember either passage or have previously seen its key,
  otherwise `false`. Recognition is a separate limitation, not proof of authenticity quality.


Use `reviewer_type` of `human` or `llm` and identify the reviewer. For an LLM, record its exact
model/version and the complete judge prompt/settings separately in `judge_model_and_prompt`.
Leave unanswered trials unfilled; do not fabricate judgments.

Judge each trial independently. Submit the primary review before seeing the control sheet.
The control sheet reuses some passages and must not inform the primary judgments. A reversed
form is for an independent reviewer or fresh isolated model calls, not a second independent
observation from the same human who remembers the first form.
