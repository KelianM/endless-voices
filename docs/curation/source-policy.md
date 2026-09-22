# Source and review policy

## Provenance and access

The authority for this pilot is the official Endless Sky repository at commit
`7140eb2a29ce4d2797933075c751791a892c7d4f`. The [inventory](source-inventory.json) records
three inspected files with SHA-256 checksums, author notices, immutable URLs, and inclusive
one-based passage ranges. Mission names and branch context make the evidence auditable beyond
a bare line citation. Review date: 2026-09-22. Review method: agent reading of the selected source
passages and relevant conditions; no human or independent double-review is claimed.

The upstream data uses an indentation-based format mixing prose and game instructions.
Consult the official [data format](https://github.com/endless-sky/endless-sky/wiki/DataFormat)
and [conversation documentation](https://github.com/endless-sky/endless-sky/wiki/WritingConversations)
for syntax, while the pinned source determines this pilot's content. Wiki pages can change;
they are not an additional unversioned canon source.

To inspect the exact evidence locally, run `python scripts/fetch_sources.py` from this repository
and open the listed files and ranges in `data/local/endless-sky-7140eb2a29ce/`.
See the [dataset README](../../data/README.md) for verification and explicit refresh instructions. Hash the raw file bytes (before newline conversion) with
SHA-256 and compare with the inventory. No upstream checkout, copied game dialogue, artwork,
or downloaded model is required to use the existing Endless Voices pipeline.

## Inclusion and exclusion

Retain direct attributable speech with its narrator/player boundaries and prerequisites. Use
logs and descriptions only as labeled context. Record testimony as testimony, personal tastes
as individual, and speculative explanations as uncertain. Conflicting accounts are not silently
resolved by choosing the most convenient one. Unknown speakers or incompatible story states
must be flagged, deferred, or excluded rather than guessed.

The pilot excludes third-party plugins, community lore additions, images/audio, UI/deprecated
content, unreviewed campaign outcomes, and the existing invented archivist from canonical
training. Referenced but unreviewed factions are not new target identities. The inventory also
records two deferred passages and one excluded record so later authors do not mistake all
inspected material for usable speech.

Source paragraphs are evidence, not ready-made user/assistant pairs. Post-processing must
preserve branch conditions, role attribution, and coherent placeholders. Shared source scenes,
branch repeats, and logs summarizing a scene belong to the same provenance family. Curation
summaries and identity briefs are not benchmark questions or exact-match answer targets.

## Licensing and attribution

The pinned [upstream copyright manifest](https://github.com/endless-sky/endless-sky/blob/7140eb2a29ce4d2797933075c751791a892c7d4f/copyright)
assigns its general `Files: *` entry GPL-3.0-or-later. All three selected text files have matching
GPL notices. Their file headers credit Michael Zahniser (2014) for the Quarg mission file,
Michael Zahniser (2015) for the Hai mission file, and MasterOfGrey (2021) for the Hai culture file. These are header credits, not a claim that no other
contributors changed the files; preserve the upstream contributors/credits context as well.
Artwork has separate terms and is not included in this pilot.

This PR stores source links, attribution metadata, and newly written curation summaries; it does
not vendor the game text. Before redistributing extracted/adapted dialogue, record the applicable
license and preserve required notices/attribution and license material. Do not relabel copied
source as public domain or assume generated paraphrases erase source obligations. Decide and
document the release terms for that dataset when preparing it. The implications for model or
adapter redistribution are not established by this source review and remain a release question;
this is not a claim that trained weights necessarily inherit GPL.

## Review handoff

The `retain` disposition establishes an evidence candidate, not final example approval.
Conversation authors must still review every new answer against its evidence, check that roles
and knowledge match, distinguish creative connective text from canonical claims, and record
review status. Benchmark authors must check gold expectations independently of how convincing
an authored training response sounds. Use development material for prompt/parameter choices,
never the frozen benchmark or its scoring notes.
