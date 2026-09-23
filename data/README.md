# Endless Voices dataset: source overview and pilot status

The [first conversation dataset](../docs/curation/pilot-v1/README.md) is committed under
`data/curated/pilot-v1/`, with versioned train, validation and test files and reconstruction annotations. Original game speech is
paired with source-backed profiles, selected lore and coherent conversation history. The release
includes Free Worlds representatives, Republic Navy, mainstream Hai and Quarg; its own coverage report
records actual sample and conversation counts. `example.jsonl` remains an invented pipeline fixture.

The whole-source statistics below describe availability, not usable sample yield. The historical
three-identity source review was a starting selection, not a limit on dataset coverage.

## Curated record and benchmark contracts

[Version 1 contracts](contracts.md) define one conversation format for train,
validation, and test, with versioned split manifests and offline validation commands. The [pilot release](../docs/curation/pilot-v1/README.md) uses these contracts.
Source inventories below remain evidence metadata, not conversation samples.

## Scope and faction counts

Source: official Endless Sky development commit
[`7140eb2a29ce`](https://github.com/endless-sky/endless-sky/commit/7140eb2a29ce4d2797933075c751791a892c7d4f),
inspected on 2026-09-22. This is not a named release or a claim about today's latest version.
All 204 tracked `data/**/*.txt` files are counted, including shared, UI, and deprecated data.
Third-party plugins, artwork, audio, translations outside these files, and repository history
are outside the inventory.

There is no single authoritative “number of factions” field in these sources:

- **19 species/region content directories:** avgi, bunrodea, coalition, drak, gegno, hai, human, iije, incipias, kahet, korath, pug, quarg, remnant, rulei, sheragi, successors, vyrmeid, wanderer.
- **128 distinct government identifiers**, also 128 root government declarations in this
  snapshot. These include political groups, location/hostility variants, and technical entities
  such as `Test Dummy`, `Uninhabited`, and `Escort`. They are not 128 independently trainable voices.
- **Three identities in the initial source review:** early Free Worlds militia, Quarg, and non-Unfettered Hai. They are curation choices,
  not the full set available in the game.

Directory groups are the reproducible breakdown below, **not a completed faction taxonomy or
speaker attribution**. For example, `human` includes multiple human factions, `hai` contains
Unfettered material too, and Coalition contains several species. Quarg speech and lore also
appear outside the Quarg directory. An exact semantic faction/species census requires a separate
mapping; these statistics do not invent one. All government names are listed in
[source-statistics.json](overview/source-statistics.json).

## Whole-source scale

| Measure | Count |
| --- | ---: |
| Text files | 204 |
| Raw bytes | 10,461,684 (10.46 MB / 9.98 MiB) |
| Physical lines, including comments and blanks | 261,599 |
| Root mission declarations | 2,331 |
| Conversation blocks with children | 1,745 |
| Conversation references without children | 126 |
| Of the conversation declarations, root named conversations | 54 |
| Root phrase declarations | 868 |
| Root news declarations | 219 |
| Recognized text words across the categories below | 1,018,481 |

Missions often contain conversations; these are overlapping structural counts, not separate
pools of examples. The 54 named conversations are not additional to the 1,745 blocks. References
are not independently counted as conversations. One mission may have multiple conversations;
one conversation can contain many branches and repeated text.

| Text category | Approximate words | What is included / limitation |
| --- | ---: | --- |
| Conversation display text | 682,165 | Narrative paragraphs and player choices as well as NPC speech; not a count of spoken dialogue |
| Mission/action dialog text | 37,921 | Literal `dialog` messages; phrase references are not expanded |
| Descriptions | 156,324 | Mission summaries, ships, outfits, planets, and other `description` values |
| Spaceport text | 32,772 | Literal `spaceport` descriptions |
| Logs | 22,105 | Recorded summaries; often repeat information from conversations |
| Phrase/news fragments | 87,194 | `word` alternatives, including names and hails; not fully assembled sentences |

The categories are disjoint in this lexical count, but their **meaning is not deduplicated**.
They do not cover every possible text-bearing construct. These refined estimates replace the
initial reconnaissance's rough ~1.01-million-word estimate; the source snapshot is unchanged,
but short text and fragment handling and structural classification are now explicit.

## Data by source group

“Conv.” means a block with children. “Conv. words” includes narration and player choices.
“All text words” sums the six categories above; it is not all file tokens or exclusively lore.
KiB uses 1,024 bytes, rounded to the nearest integer. Rows are sorted by conversation-word volume.

| Source group | Files | KiB | Missions | Conv. | Conv. words | All text words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| human | 41 | 3359 | 1,124 | 688 | 249,660 | 368,083 |
| coalition | 10 | 913 | 248 | 250 | 101,539 | 120,116 |
| remnant | 10 | 677 | 151 | 186 | 68,084 | 80,541 |
| kahet | 9 | 545 | 84 | 95 | 57,408 | 63,694 |
| successors | 18 | 541 | 108 | 106 | 46,569 | 60,176 |
| wanderer | 8 | 484 | 203 | 155 | 37,973 | 53,727 |
| hai | 11 | 471 | 167 | 88 | 33,795 | 49,035 |
| avgi | 14 | 429 | 89 | 43 | 24,708 | 40,665 |
| gegno | 9 | 254 | 42 | 24 | 21,405 | 27,071 |
| sheragi | 3 | 144 | 33 | 41 | 17,199 | 19,518 |
| korath | 11 | 298 | 30 | 21 | 6,039 | 14,983 |
| quarg | 5 | 86 | 8 | 9 | 5,321 | 9,153 |
| incipias | 5 | 62 | 11 | 11 | 4,141 | 6,302 |
| (shared root) | 19 | 1574 | 5 | 6 | 2,370 | 93,440 |
| bunrodea | 5 | 57 | 1 | 2 | 2,288 | 4,443 |
| drak | 5 | 38 | 13 | 7 | 1,612 | 2,325 |
| rulei | 3 | 15 | 7 | 5 | 798 | 1,046 |
| pug | 5 | 31 | 4 | 2 | 685 | 1,981 |
| _ui | 8 | 179 | 0 | 5 | 542 | 542 |
| _deprecated | 3 | 46 | 3 | 1 | 29 | 1,393 |
| iije | 1 | 6 | 0 | 0 | 0 | 80 |
| vyrmeid | 1 | 7 | 0 | 0 | 0 | 167 |

The machine-readable report also includes per-group description, spaceport, dialog, log, and
phrase/news word counts, text-line counts, conversation references, and every source-file hash.
Human and Coalition folders together supply about 51% of conversation words. Hai has about
6.4 times the Quarg directory's conversation volume. Some groups have almost no dialogue in
their own folder; this does **not** prove they have no lore or speech elsewhere.

Shared-root files hold much of the map/planet description material. UI and deprecated rows are
included for an honest full inventory but excluded from the selected canonical authoring pool.
Numbers for text volume say nothing by themselves about topic diversity, speaker certainty,
independent scenes, or sufficient material for a benchmark.

## What the examples tell us

These are paraphrased qualitative findings from selected passages, not new training responses.
They illustrate why both volume and reading the source matter.

| Source example | Learning signal | Preparation concern |
| --- | --- | --- |
| [Free Worlds reconnaissance](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/human/free%20worlds%200%20prologue.txt#L329-L425) | Practical cooperation, defensive framing, and concern about provoking the Navy | A Republic captain speaks inside the same mission; preserve faction attribution, early-campaign timing, and actual-conversation split boundaries |
| [Quarg first contact](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/quarg/quarg%20missions.txt#L22-L70) | Patient explanation, collective identity, peaceful coexistence backed by strength | Narration and player questions interrupt speech; Quarg claims about the Drak must remain attributed |
| [Quarg at Kuwaru Efreti](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/quarg/quarg%20missions.txt#L74-L118) | More archaic register; protection of Efreti and withholding dangerous knowledge | Branches repeat answers; one speaker's register is not mandatory for every Quarg |
| [Hai first contact](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/hai/hai%20missions.txt#L14-L96) | Hospitality and curiosity about humans; a young Hai defers history to elders | Human merchant exposition is interleaved with Hai speech; characters have different knowledge |
| [Hai gambling discussion](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/hai/hai%20culture%20conversations.txt#L179-L212) | Honesty as a cultural norm, with explicit acknowledgement that Hai can bluff | Avoid converting a norm into a biological inability or inventing unspecified tenets |
| [Hai theater conversation](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/hai/hai%20culture%20conversations.txt#L14-L34) | Everyday taste, disagreement, and a human/Hai friendship | Two species speak in the same paragraph; personal taste is not a universal faction belief |
| [Coalition folklore](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/coalition/coalition%20culture%20conversations.txt#L14-L60) | Cultural performance and distinctive speech | Embedded theatrical stories are not necessarily literal history; this is reconnaissance, not pilot-approved evidence |
| [Avgi hails](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/data/avgi/avgi%20hails.txt#L17-L52) | Friendly/hostile voice material assembled from phrases | Weighted fragments and references need rendering; many combinations do not imply many independent examples |

The strongest material teaches values through decisions, explanations, disagreement, and
knowledge boundaries. The main risks are speaker mixing, lost branch context, treating opinions
as omniscient facts, and generating many near-duplicates from a small number of scenes.

## Curated pilot versus available source

The [curation package](../docs/curation/README.md) contains **21 inspected passage records**:
14 retained as evidence, two context-only, three deferred, and two excluded. Retained material
spans **11 mission groups**: three Free Worlds, two Quarg, and six Hai. The three Free Worlds
missions share one reconnaissance chain and are not independent scenario families. This is an agent source review, not
independent human adjudication or final example approval.

Quarg has strong first-contact/knowledge-restraint evidence but narrow everyday coverage.
Hai has more varied civilian contexts, but speakers' ages, tastes, political views, and
translation conditions must be preserved. Free Worlds adds practical militia dialogue about cooperation, surveillance, and avoiding
escalation; its initial sample does not represent all human-space factions or civilian life.
The old per-identity quotas have been replaced by source-supported conversation annotation.
The [pilot release documentation](../docs/curation/pilot-v1/README.md) records the current scope,
exclusions, split rules and review evidence.

## Reproduce and interpret the statistics

Fetch the pinned source using Python 3.11+ and Git (no Git LFS or Python packages required):

```sh
python scripts/fetch_sources.py
python scripts/inventory_sources.py data/local/endless-sky-7140eb2a29ce --output data/overview/source-statistics.json
```

The fetch command reads the revision and all 204 SHA-256 hashes from the committed statistics
manifest. It fetches only that commit with a sparse checkout of `data/` plus root files,
including upstream license, copyright, and credits. Game images/audio and full Git history are
not checked out. Git metadata and root files add some overhead to the 10.46 MB text corpus.
The default destination is gitignored and independent of your shell's current directory.
`--destination /another/path` selects a different location; ensure custom locations stay out of Git.

A second run verifies the existing checkout without network access. Missing, changed, extra,
or wrong-revision data fails verification rather than overwriting local work. Downloads are
staged in a temporary sibling directory and published only after verification succeeds.
To recover from a failed verification, move the old checkout aside or select a new destination.
Network access to GitHub is required only for the initial fetch.

To refresh deliberately: obtain a separate clean upstream checkout at the desired full commit,
update `REVISION` in `scripts/inventory_sources.py`, and regenerate the committed statistics
manifest from that checkout. Review the source changes, inventory counts, curation citations,
file hashes, and this README together in a PR. The fetch command then uses the new manifest
and a new revision-specific directory. Do not switch to a moving branch on each run: existing
training and benchmark artifacts must retain the revision they were built against.

The standard-library script checks the commit, a clean data tree, and completeness against the
tracked file list. It stores SHA-256 hashes and uses indentation and quoted tokens to count
structural nodes and selected text categories. Words are whitespace-separated units, **not
model tokens**. It does not execute game conditions, expand phrases, resolve speakers, deduplicate
scenes, or fully parse the game format. Tiny hand-counted fixtures test the counting boundaries.
The statistics are descriptive estimates, not an extraction pipeline or quality benchmark.

## Storage, licensing, and release status

Keep this README, inventory statistics, curation metadata, and small test fixtures in ordinary
Git so they remain reviewable. Raw upstream files are fetched into `data/local/` and ignored;
no raw corpus is committed. The complete curated release is versioned through Git LFS to keep
generated output from dominating the code diff. Agent-authored annotations, profiles, lore,
curation review records and frozen hashes remain in ordinary Git.
The release includes source notices, licensing material, provenance and a readable review copy.
Run `git lfs install --local` and `git lfs pull` after cloning to obtain the release payload.
Reconstruction checks the versioned release; it is not required to obtain the dataset.

The selected text files carry GPL-3.0-or-later notices. Preserve their provenance and assess
redistribution terms when publishing source-derived examples; do not assume “open source” means
public domain. See the [source policy](../docs/curation/source-policy.md) for header credits,
source boundaries, and the unresolved model/adapter-release question. Fetching raw sources does not constitute a reviewed dataset release.
