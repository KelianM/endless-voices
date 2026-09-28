# Story context for the next dataset

The next dataset will evaluate a character's continuation given the story so far. Earlier
conversations can supply voice examples and established events. The model will infer delivery
from those examples and factual context rather than receive rules about cadence, verbosity,
qualification or institutional authority. Narrative direction remains a separate design question.
This document specifies the proposed release contract; the [prototype](../data/evaluation/context-prototype-v1/README.md)
measures a small part of that contract. The frozen pilot still follows [version 1](contracts.md).

## Chronology and split boundaries

Assign whole conversations chronologically on each reviewed story path: training first,
validation next, test last. A reference must occur strictly before the prediction point on the
selected path. Split membership is an additional restriction, never evidence of chronology.

| Prediction split | Eligible earlier references |
| --- | --- |
| Train | Train |
| Validation | Train or validation |
| Test | Train, validation or test |

Earlier replies in the current conversation remain visible. The current target, subsequent
replies and alternatives to the current encounter remain hidden. Optional dialogue from strictly
earlier encounters is eligible even when the player could not select every option in one playthrough. An earlier validation or test target may become
context for a later prediction after that event occurs in the declared story. Evaluation uses the
original continuation as this observed history, not the model's previous prediction. Scores
therefore measure conditional continuation, not an autonomous campaign rollout.

A conversation and its repeated or branch variants receive one global split assignment, including
when several paths share that conversation. Cut points must satisfy all retained paths. Conflicting
assignments require revised boundaries or removal of an incompatible path; duplicating a scene in
several splits is not a solution. Faction coverage is measured after these constraints, not enforced
by moving later conversations back into training. Split proportions are not fixed by this design.

## Earlier examples and established events are different

A path establishes which encounters can validly precede the current scene. Source revision, source
location and review attribution stay in private provenance. A mission dependency can establish
necessary ordering. Optional questions at an earlier encounter can all provide authentic examples;
the reference does not need to reconstruct one exact sequence of player choices.

Dialogue examples retain speaker and audience attribution and are labeled as examples rather than
an assertion that every exchange occurred. Conflicting outcomes cannot both become established
story facts. A branch that prevents the current scene from occurring is ineligible. Conditions can
involve alternatives, failure states, repeatable missions and hidden flags, so `to offer` alone does
not establish complete causal compatibility. Reviewed assumptions remain necessary.

Something heard by the player is not automatically known by the current character. Dialogue from
another character is not evidence of the current character's personal memories or individual voice.
Factual character context and established events remain separate from the earlier dialogue examples.
A first encounter may have no earlier examples of that character. Empty references are valid.

The prototype uses declared partial paths and includes both Northern 2C optional answers before
Northern 3. Both answers share one encounter position, so neither can leak into an assessment of
the other. The prototype does not claim to reconstruct a save-game history.

## Select once and share the result

Select references from eligible earlier material using only the visible scene, character and prior events.
Never use the withheld continuation, its distinctive words, a generated response or judge outcomes
to choose references. Save the selected references and their hashes once; supply identical context
to the generator and the judge. The judge additionally receives the two anonymous continuations
and the judging instructions. Run conditions and source links stay private.

Use the game's own prose for lore and earlier conversations. Keep narration and the complete
current-scene lead-in where available, rather than replacing them with agent-written summaries or
extracting only the last player question. Preserve factual wording, speaker attribution and source
provenance. Include introductory game passages that explain the relevant institutions and factions,
not only passages that assume the reader already knows them. A navigation glossary can index names
and source passages; agent-written definitions should not replace the original exposition.
Record missing context explicitly instead of supplying an agent-written explanation. This keeps
summarizer interpretation out of the experimental input. Minimal identity and task instructions are still needed; prose describing how a
character should speak is not a substitute for the character's dialogue.

Start with all relevant, causally eligible source material. Do not impose a small excerpt count or
reference-token cap before measuring the complete context. Additional source material needs an
explicit reference-pool split assignment; absence from the current pilot does not make a passage
freely available to training. Material sourced for a validation preview remains unavailable to
training until the next release's split audit.

Context selection now has a shared strategy interface in `endless_voices.context`. `FullContext`
retains every eligible passage. `MissionDepth` keeps nearby prerequisite missions in full and
samples whole older missions into the space remaining below a total rendered-input ceiling.
Lore and the current encounter remain intact. Chat formatting and substituted game variables
count toward the ceiling. Preparation fails if the preserved core alone exceeds the ceiling;
it does not truncate that core. Sparse eligible pools and whole-mission boundaries can leave
unused space. Output tokens are reserved separately.
The configuration records depth, budget and seed. These are configurable experiment settings,
not hand-maintained story boundaries. The [depth comparison](../data/evaluation/context-depth-v1/README.md)
measures the tradeoff. Source eligibility and chronological split checks precede selection; a
strategy cannot make ineligible passages safe to include.

Selection produces one saved result with messages, selected and omitted source blocks, counts
and provenance. Dataset builders append the target to the saved messages. Benchmark consumers
use those same messages for generation and judge context without selecting again. See
[ADR 9](adr/0009-save-context-selection-once-for-all-consumers.md). Historical pilot runs and
manifest checks remain unchanged; these adapters do not declare the draft pool a new release.

Measure the fully rendered request with each model's tokenizer and reserve room for output and,
where applicable, reasoning. Judge requests also need room for two candidates and judge instructions.
Use one common selected reference set for compared conditions. Do not silently truncate a different
history for each model. A scene that cannot fit the common contract needs an explicit failure or a
predeclared common selection change. The earlier prototype measures JSON text without chat wrappers. The verbatim source preview
measures rendered generation messages including chat wrappers with cached tokenizers; judge
candidates and output reserves still need separate allowances.

## What the comparison can establish

A zero-shot model and a fine-tuned model receive the same available story at each prediction.
Fine-tuning may still encode training examples that reference selection omits. Equal prompts do
not make the models' learned knowledge equal. Report context coverage and distinguish adaptation
to voice from remembering particular passages.

Chronological splitting measures later-story continuation within known paths. The design does not
measure generalization to unseen campaigns. Repeated turns and overlapping histories remain
dependent; preserve conversation/scenario grouping for uncertainty and disclose shared paths.

A new versioned release is required for new splits, references and minimal profiles. Earlier
validation material has already guided this design, so repartitioning the same material cannot
make a fresh independent holdout. The current test dialogue stays closed during this prototype.
Future release construction must distinguish untouched evaluation material from reused development
material and audit factual lore as well as dialogue for future information. Removing style
instructions alone does not establish that a profile is temporally valid.

## Validation-wide measurement

The [validation source-context audit](../data/evaluation/validation-source-context-v1/README.md)
assembles uncapped drafts for all 48 validation points and measures them with cached tokenizers.
The drafts preserve source prose and current-encounter paths, but retain explicit gaps for optional
prerequisites, alternative outcomes and game substitutions. The audit does not approve these drafts
for benchmarking or replace the versioned release contract.
