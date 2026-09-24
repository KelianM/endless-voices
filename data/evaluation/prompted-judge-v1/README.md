# First prompted local judge round

The owner replaced the planned human review with agent-based calibration on 2026-09-24.
The original round below remains execution history; see the change record at the end.

This round checks a local Qwen3-14B judge against up to twelve human judgments collected before showing judge answers:
ten original-versus-generated pairs, followed by two procedural controls. Human judgments and
judge usefulness have not yet been established. The ten generated alternatives are actual outputs
of the pinned Qwen3-4B-Instruct-2507 generator, with no adapter.

`sample-ids.json`, `controls.json` and `selection.json` record the validation selection.
Reproduce the selection with `python scripts/select_judge_calibration.py --output NEW_DIRECTORY`.
Selection is seeded before inspecting generated outputs, uses one sample per conversation, and
prefers conversations absent from earlier calibration and smoke work. The five Free Worlds and
two Republic conversations are new to those reviews. Both Hai conversations and the only Quarg
conversation have already appeared in prior work; their selected turns are different, but that does
not make their conversations fresh. No test examples were selected for calibration.

The controls use two additional Free Worlds conversations: one identical pair and one
wrong-context substitution. The substitution donor also appears in the identical control.
Controls are reviewed after primary submission and reported separately. These checks do not
comprehensively validate sensitivity to subtle context errors or establish a universal agreement
threshold. Further examples can be selected within the agreed 10–20-example human budget if
initial discrepancies warrant a second round.

The prompted judge uses `configs/judge-model.json`: MLX Qwen3-14B at revision
`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`, 4-bit weights, greedy decoding, thinking disabled,
8,192 total context tokens and at most 512 output tokens. Each trial receives a fresh conversation
and cache. The judge sees the exact authored context and two candidates, without model condition,
source links, run metadata or the answer key. The runner saves actual prompts, package versions,
model hashes, raw outputs and failure records. Using the Qwen family for both generation and
judging may introduce shared tendencies; the human comparison must examine those limitations.

The organizer pack, judge answers and complete prompts remain in ignored local outputs until
independent human submission. Publishing those answers before review would compromise the
review. Retain the submitted human record before opening keys or discussing disagreements, then
add the completed review evidence and findings here. Do not label the judge calibrated merely
because execution succeeds.

The human review page supports brief reasons and explicit source recognition. A simulated browser
form check was discarded; it is not a human judgment. The earlier four-identity smoke run remains
integration evidence only. See the root README for commands and report denominators.

## Full-dataset generation

The requested full-dataset run produced 173 successful responses from 176 selected samples.
Three responses reached the 512-token output cap and remain failures with their partial text
preserved. No selected sample is missing. Generation used the existing pinned Qwen3-4B settings:
MPS, float16, greedy decoding, a 4,096-token context ceiling and no adapter.

| Split | Selected | Successful | Output-limit failures | Missing |
| --- | ---: | ---: | ---: | ---: |
| Train | 101 | 99 | 2 | 0 |
| Validation | 48 | 48 | 0 | 0 |
| Test | 27 | 26 | 1 | 0 |

[The execution record](full-generation.json) preserves per-identity coverage, run and artifact
hashes, timestamps and local output locations. All recorded artifact hashes and dataset hashes
were verified. Ordered selections match the complete splits; response IDs match those selections
without duplicates or omissions. Prompt IDs are unique and belong to the corresponding selection.
All three output-limit failures belong to mainstream Hai samples. No settings were changed and
no failures were retried or promoted to successful responses.

Test dialogue and generated test text were not inspected or judged. The checks used hashes,
identifiers, identity metadata and execution status only. These counts establish generation
coverage, not model quality; the expanded calibration findings are reported separately.

## Change to agent-based calibration

On 2026-09-24 the owner replaced the planned human calibration with isolated sub-agent reviews,
as recorded in [ADR 6](../../../docs/adr/0006-calibrate-a-local-prompted-judge-with-isolated-agents.md).
The earlier human pack and local judge outputs remain preserved; no human answers were submitted.
The expanded round uses all 48 successful validation outputs, one fresh sub-agent invocation per
primary trial, and two isolated control reviews. The local judge also evaluates all reversed pairs.
These calls cover 15 conversations, not 48 independent scenes. The expanded comparison is complete: agents identified 48/48 originals and the local judge 5/48.
The local judge remains unsuitable for headline results; see [findings and evidence](agent-findings.md).
