# Three-generator validation benchmark

Sol identified the original game continuation in all 48 primary comparisons for each generator.
The primary detection score therefore does not rank Gemma 4 31B, GPT-6 Sol and Claude Sonnet 5.
The benchmark completed on 2026-09-25 with 144 successful generations and 290 successful
assessments. No response or assessment is missing, failed or truncated.

| Generator | Original identified, primary | Original identified, reversed | Changed choices / complete position pairs | Generation API cost |
| --- | --- | --- | --- | --- |
| GPT-6 Sol | 48/48 (100%) | 47/48 (97.9%) | 1/48 | $0.110868 |
| Gemma 4 31B, MLX 4-bit | 48/48 (100%) | 48/48 (100%) | 0/48 | Local |
| Claude Sonnet 5 | 48/48 (100%) | 48/48 (100%) | 0/48 | $0.187052 |

Sol judged every condition, including its own responses. Each trial compared one generated
response with the original game continuation, in an isolated request with anonymous candidate
positions. Sol received the generator's profile, selected lore and authored history, without
condition labels, source links, run metadata or the answer key. The reversed trials exchanged
candidate positions; the reversed trials are sensitivity checks, not additional independent scenes.

## Coverage and uncertainty

Each generator completed the same 48 validation scenes from 15 conversation/scenario groups.
Each condition has nine Republic, 22 Free Worlds, nine mainstream Hai and eight Quarg scenes.
Primary original identification was 9/9, 22/22, 9/9 and 8/8 respectively for every condition.
Each condition has zero incorrect primary decisions, zero abstentions out of 48 scheduled primary
trials, zero failures out of 48 and zero missing assessments out of 48. No primary or reversed
assessment reported source recognition; recognition-excluded results therefore equal full results.

All three condition pairs have 48 complete shared-scene comparisons, zero incomplete pairs and
zero percentage-point difference in primary detection. The conversation/scenario bootstrap gives
[100%, 100%] for each primary rate and [0, 0] percentage points for every paired difference.
These intervals are degenerate because every primary decision was correct. The intervals do not
measure uncertainty about new scenes, another judge, memorization or model selection. Fifteen
groups provide limited independent evidence; repeated turns and positions add no independent groups.

The identical-candidate control produced an abstention and the wrong-context control selected the
original. Both controls behaved as intended and are excluded from primary detection rates.
There is one judge, so there is no between-reviewer agreement estimate.

## What the reasons show

Sol's recorded reasons often contrast an immediate conversational reply with an expanded summary
of the supplied lore. For Gemma's Alondo reply after Parliament (`l132-bfd0247c`), Sol objected to
an extended explanation that assigns motives to Parliament. For Sonnet's first Hai greeting
(`l32-2b5b5beb`), Sol described a role reversal followed by lengthy exposition. For Sol's own Alondo
reply (`l140-1302d95e`), the judge preferred the original's brisk answer over a cautious explanation.
These are attributed judge observations, not independently validated quality labels. All exact
reasons and candidate texts remain in the evidence archive.

The sole order change involved Sol's Hines response (`l1945-b31cceba`). In the primary order, the
judge accepted the original's sudden arrest order as a consequential scripted turn. In the reverse
order, the judge instead preferred the generated conciliatory response and called the arrest order
inconsistent with Hines's role. This disagreement shows that plausible contextual reasoning can
support opposite choices even when the text pair is unchanged.

Median generated lengths were 61.5 words for Sol, 139 words for Gemma and 135.5 words for Sonnet
(using whitespace-separated words). Length is descriptive, not an additional evaluation score.
The screen's favorable qualitative impression of Gemma can coexist with perfect source detection:
a response can be readable and still differ noticeably from the game's authored continuation.

## Execution and limitations

All 144 responses were generated afresh; the earlier four-scene screen remains unchanged.
Gemma used revision `696d436c404745a59f30e4939a658162b0a9e57f`, greedy decoding, thinking disabled,
a 512-token output cap and 4,096-token context ceiling without truncation. Peak MLX memory was
18.583 GB, below the 20 GB experiment ceiling. The shared MLX environment was used.
Sol used medium reasoning and Sonnet medium adaptive thinking, each with 4,096 output tokens
including reasoning. Provider settings differ from local greedy decoding and must not be treated
as a controlled architecture-only comparison. Actual returned aliases and usage are recorded;
immutable hosted model revisions are unavailable.

Sol judging cost $1.068828; combined generation and judging cost $1.366748, within the $5 cap.
Sol used medium reasoning, structured JSON, 4,096 output tokens including reasoning, no tools,
stateless requests and `store: false`. Original and reversed assessments were scheduled separately.

Sol judging its own generations creates possible self-preference; the result does not eliminate
that risk. Model selection and judge development already used these validation conversations.
The benchmark has no independent human validation and provides no held-out generalization claim.
The test set was neither inspected nor judged. No adapter was trained or evaluated. Lower detection
would not by itself establish better quality, and a tied score does not establish model equivalence.

## Reproduction and evidence

From the repository root, `scripts/benchmark_generators.py prepare` verifies and selects the saved
full validation inputs. Run `scripts/screen_generators.py local` with the recorded Gemma config,
then `scripts/benchmark_generators.py hosted` and `export`. Exact invocation arguments and stage
logs are archived in `outputs/generator-benchmark-v1/run.py` and the stage records. Preparation,
generation and export refuse existing output artifacts. A new experiment needs a new `--root`.

Use `python -m endless_voices.openai_judge --models gpt-6-sol` with the exported public pack,
a fresh output directory and the recorded $2 judge budget. The existing assessment report command
joins the private organizer pack and saved review; no text or answer mapping is assembled manually.
Normalized run files preserve raw generation evidence and map recorded normal completion to `eos`.
Hosted token IDs and prompt token counts remain null in that normalization; actual provider usage
is retained in raw responses. The benchmark is not a Transformers generation run.

`evidence.tar.gz` contains raw requests/responses, settings, source code snapshots, input selection,
public/private trials, all reasons, the full machine-readable and Markdown reports, verification
and dataset attribution. `inventory.json` records every member hash and the archive hash.
Verification checked all selected IDs, authored contexts, answer mappings, exact judge payloads,
raw completion text, artifact hashes and usage-based costs. Ruff passed and 183 offline tests passed,
including regression checks for selected-model execution, incomplete outputs and provenance mismatch.
