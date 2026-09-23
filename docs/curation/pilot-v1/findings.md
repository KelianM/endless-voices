# Pilot coverage and review findings

The release contains 176 samples from 61 source conversations. The source catalog contains 785
conversation blocks in the human, Hai and Quarg directories; this pilot does not measure their
full usable yield. The reviewed selection is below the initial working target of roughly 200
samples because uncertain speakers and incomplete constructions were removed. No paraphrases
were added to reach a quota.

| Identity | Train samples | Validation samples | Test samples | Total samples |
| --- | ---: | ---: | ---: | ---: |
| Free Worlds | 46 | 22 | 6 | 74 |
| Republic Navy | 11 | 9 | 5 | 25 |
| Mainstream Hai | 34 | 9 | 13 | 56 |
| Quarg | 10 | 8 | 3 | 21 |
| Total | 101 | 48 | 27 | 176 |
| Source conversations | 35 | 15 | 11 | 61 |

Human speakers account for 99 samples. Free Worlds coverage includes militia logistics,
reconnaissance, elected representation, diplomacy, engineering and civilian projects. Republic
Navy coverage includes inspections, a joint operation, failed peace talks and postwar service.
Hai coverage includes hospitality, history, public customs, entertainment, education, merchants
and family concerns. Quarg coverage includes contact, Korath containment, human freight and a
disputed account of the Coalition.

The 27 test samples come from 11 conversations. The Quarg test has only three continuations from
one conversation. The pilot cannot support precise per-faction quality estimates, and conversation
prefixes are dependent observations. Military and political scenes dominate human coverage;
civilian human life and additional human factions need further source review.

## Source review and corrections

Construction agents proposed annotations. The implementing agent then read the retained source
conversations with their selectors and corrected speaker, context and branch errors. The final
annotation hashes and review roles are recorded in [review.json](review.json). This is agent source
review, not independent peer adjudication or human approval of individual records.

- Lieutenant Paris belongs to Deep Security, which is distinct from the Republic Navy. Paris's
  targets were removed rather than relabelled as Navy speech. In the Luna briefing, Paris interrupts
  the admiral at line 429; the admiral's target ends at 428. The unassigned speaker at 434 is excluded.
- The independent Wolf Pack is not used as an official Free Worlds representative. Its proposed
  records are excluded from this release.
- Complete replies now follow source jumps, including Eeeya's repeated request, the Hai merchant's
  warning file and farewell, and the Greenrock captain's concluding remarks. These continuations
  are not separate samples. Other characters' interjections terminate a selected speaker's reply.
- Story assumptions distinguish the attack to retake Kornephoros from its aftermath, Bloodsea's
  prior-invasion condition, and the reconciliation outcome of Danforth's postwar visit. Static
  source reachability alone would not detect a wrong condition value.
- Punctuation corrections remove commas left by narrator tags at sentence boundaries. Commas
  inside interrupted sentences are retained. Source words and distinctive phrasing remain intact.
- Profiles and scenes establish the current role and situation. Selected lore adds shared setting
  facts and relevant relationships; later search results and construction instructions were removed
  from initial context. The Hai Academy exchange retains the game's immediate translation of the
  same speaker, identified in its scene and review notes.

The actual Quarg first-contact, Hai first-contact and Free Worlds scan-request calibration
conversations remain validation material. Kuwaru Efreti is a different Quarg encounter and supplies
training samples. Separate reconnaissance encounters can cross splits. Source filenames, a shared
speaker and a common topic are not evidence that two encounters are repeated versions.

The final overlap report contains zero cross-split twelve-word phrase matches and zero
system-to-target twelve-word phrase matches. Exact reply and paragraph checks also pass, including
history. Those checks do not rule out shared meaning, common facts or undiscovered repeated scenes.

## Validation and limits

Qwen3-4B-Instruct-2507 tokenization measures 277–1,040 tokens per complete sample, including all
system context, selected lore, history and target. Every sample fits the recorded 8,192-token
preparation limit without truncation. The frozen reconstruction records the tokenizer revision,
file hashes and the library versions used for measurement. No weights were downloaded.

Untranslated dialogue, unassigned speakers, arbitrary phrase generation, UI/deprecated material,
and unreviewed campaign outcomes are excluded. Other factions and the later Quarg satellite
mission are not covered. Named game variables are rendered only from declared player choices or
source-backed mission values. The preparation tools do not execute a save state or resolve all
possible game syntax.

Calibration remains development evidence: both the human and a fresh agent recognized all three
originals, citing vocabulary, cadence, alien mannerisms and dramatic phrasing. Meaning-preserving
paraphrases lost the game's voice, and reused controls were recognized. Those alternatives were
agent-authored and easy; the result says nothing about model quality. Future evaluator validation
needs fresh scenes and subtler alternatives. The original [calibration findings](../../evaluation/calibration-findings.md)
and review records remain unchanged.
