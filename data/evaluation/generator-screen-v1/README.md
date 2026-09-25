# Four-scene generator screen

The shortlist is **Gemma 4 31B and Gemma 4 26B-A4B for local generation**, with **GPT-6 Luna
and GPT-6 Sol as hosted references**. Luna and Sol produced the most restrained, contextually
consistent dialogue in this screen. Gemma31B was the strongest local candidate; Gemma26B offers
lower memory use and faster generation, but often turns a short exchange into a speech.
This is a qualitative shortlist from four examples, not a benchmark ranking.

## Method

Ten models answered the same four validation scenes: Danforth's defense reward, Alondo's next
step after Parliament, a Hai barkeep discussing poker, and a Quarg greeting a tourist. These
are the existing smoke scenes, one per identity, selected before new generation. Qwen4B's four
verified earlier outputs were reused; the other nine models generated four new responses each.
All 40 selected responses completed. No test dialogue was inspected and no adapters were used.

Scripts verified the dataset manifest, original run artifacts, selected IDs and exact authored
messages. Generated context withheld the original final response and private metadata. Local
models used 4-bit MLX weights, greedy decoding, thinking disabled, a 4,096-token context ceiling
and 512 output tokens. The reused Qwen4B reference used float16 Transformers inference. Hosted
models used medium reasoning and a 4,096-token output ceiling including thinking, without tools
or shared conversations. These are different procedures, not equal compute budgets.

The parent Codex assistant reviewed all four scene pages with model names replaced by randomized
letters. Each page included the authored context, original continuation and ten responses.
The assistant saved 40 concise notes and a comparative summary before opening the model key;
a hash records the notes at reveal time. The assistant had already seen the originals and some
Qwen4B outputs. This was model-label masking, not fresh blinded calibration or human review.
No automated judge was used to select the shortlist.

## Findings

| Model | Qualitative finding | Next step | Peak MLX memory |
| --- | --- | --- | --- |
| GPT-6 Luna | Strongest conversational restraint; one unwanted enclosing quotation and a generic Quarg voice | Hosted reference | Not local |
| GPT-6 Sol | Strong contextual consistency and natural uncertainty; occasional redundant exposition | Hosted reference | Not local |
| Gemma 4 31B IT | Best local balance of restraint and voice; still adds bureaucratic details and managerial phrasing | Full validation candidate | 17.94 GB |
| Gemma 4 26B-A4B IT | Some good Quarg cadence; much longer speeches and invented personal detail | Full validation candidate | 14.61 GB |
| Claude Sonnet 5 | Readable, but repeats lore, relocates Alondo prematurely and invents a numbered tenet | Reserve hosted candidate | Not local |
| Mistral Small 3.2 24B, corrected tokenizer | Invents a station/cargo at Farpoint; moves Alondo to New Wales before departure | Defer | 13.61 GB |
| Qwen3-14B | Concise in places, but invents political motives and a planet named Quarg | Defer | 8.68 GB |
| Qwen3-30B-A3B Instruct 2507 | Invented lore and anatomy, rhetorical repetition, and narration despite instructions | Defer | 17.47 GB |
| Qwen3.6-27B | Long explanations, faction confusion and a port changed into a vessel | Defer | 15.94 GB |
| Qwen3-4B Instruct 2507 | Known reference: verbose, invented reports and inconsistent scene progression | Keep as baseline | Not measured here |

Peak figures are allocator measurements, not total macOS memory use. Gemma31B fit the 20 GB
experiment ceiling on the 24 GiB machine, although MLX warned that the model approaches its
recommended working-set size. This establishes short-context inference feasibility, not
training feasibility. Generation time across four responses was approximately 67 seconds for
Gemma31B and 26 seconds for Gemma26B, excluding model setup and downloads.

Luna's Danforth response acknowledges the reward in 26 words with restrained officer humor.
Sol's Alondo response keeps temporary safe passage, distinguishes Navy from Parliament and
expresses uncertainty naturally. Gemma31B's Quarg response invites questions without a full
history lesson, although its hospitality language is less distinctive than the original.
Gemma26B begins that scene well, then volunteers a much longer account of Quarg origins.

Shorter was not automatically better: Qwen14B's relatively short Quarg reply invents a named
world, refuge policy and cosmic intervention role. Larger was not automatically better either:
Qwen30B adds narration to Danforth and invented Hai terminology; Qwen3.6 places the tourist on
a vessel rather than at the port. The full notes preserve strengths and weaknesses by scene.

All ten responses miss the original Hai explanation connecting real-money consequences with
the fantasy/reality tenet. The best responses still address skill, chance and bluffing plausibly.
Missing the withheld wording or exact continuation was not itself a failure; the shared omission
shows a useful difficult case for closer review, not a new required phrase or checklist.

## Execution and limitations

The twelve hosted generations cost an estimated **$0.0221676** from returned usage:
Luna $0.0004876, Sol $0.006962 and Sonnet $0.014718. Actual model identifiers, requests,
responses and settings are preserved. Local checkpoints use immutable Hub revisions.

Initial Gemma31B and Qwen27B setup attempts failed because metadata-only snapshot directories
were mistaken for complete downloads. Those failures and logs remain preserved. The owner
paused downloads during poor connectivity, then explicitly resumed them. Completed responses
were never overwritten. Mistral's first four responses are also preserved separately: a tokenizer
regex warning prompted an explicitly disclosed correction and four new responses, which are
the responses reviewed here. No other output was regenerated to improve its apparent quality.

The shared newer MLX environment successfully generated all six local candidate models and
completed one identical-candidate judge control. The first smoke attempt lacked sandbox GPU
access and made no judgment; the GPU-enabled attempt is preserved separately. README
now documents one active MLX setup for generation and judging; historical environment versions
remain provenance, not separately maintained requirements. The existing Transformers 4 training
path remains separate pending compatibility work.

Four previously used scenes cannot establish general quality or fine-tuning suitability. Model
selection itself uses validation data, and the review is one assistant's judgment with known
source material. Model-specific decoding and quantization also affect comparisons. Nothing here
establishes human preference, indistinguishability or equivalence.

For the next benchmark, compare the two Gemma candidates and two hosted references against the
same originals across validation, keeping Qwen4B as the existing baseline. Use independent
stateless assessments and preserve reversed positions and reasons. Luna can provide an economical
first pass, but another judge should check difficult cases and Luna-generated responses to avoid
relying solely on a model judging its own family. Full benchmarking has **not** been started.

`evidence.tar.gz` contains exact inputs, responses, revisions, settings, logs, code versions,
anonymous review pages, pre-reveal notes, model key, verification and attribution. `inventory.json`
records the archive and member hashes. No weights or credentials are included.
