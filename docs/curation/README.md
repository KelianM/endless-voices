# Initial source curation

This is the completed first source review for [issue #2](https://github.com/KelianM/endless-voices/issues/2).
It selects three pilot identities, records 21 inspected passages, and supplies evidence-backed
briefs for conversation authors. The initial inventory contains neither training conversations nor benchmark cases.
The [pilot release](../dataset.md) extends this evidence with conversation annotations,
profiles and selected lore, including Republic Navy coverage.
For whole-source counts across every content group, see the [dataset overview](../../data/README.md).

- [Source inventory](source-inventory.json): pinned files, checksums, line links, speakers,
  prerequisites, evidence types, dispositions, and limitations.
- [Identity briefs](identity-briefs.md): what the selected representatives can reasonably know,
  value, and sound like, with evidence IDs.
- [Source policy](source-policy.md): canon boundaries, attribution, and review rules.

## Initial selection

| Identity ID | Species ID | Faction / government boundary | Representative roles |
| --- | --- | --- | --- |
| `human-free-worlds` | `human` | Free Worlds militia in the early reconnaissance chain | Transport officer; Glaze commander Jean-Jacques (JJ) |
| `quarg` | `quarg` | Quarg interlocutors at the selected first-contact locations | Local greeter; Kuwaru Efreti interlocutor |
| `hai-mainstream` | `hai` | Residents of government `Hai` worlds; excludes `Hai (Unfettered)` as a target identity | Young resident; elder; barkeep; culture participant; cafe worker |

`hai-mainstream` is a project label, not an official species name. Roles retain different
knowledge and tastes; these are not omniscient spokespersons. The non-Unfettered boundary
must remain visible when discussing the northern Hai. Korath and Drak mentioned by Quarg are
subjects of testimony, not additional target speakers.

Include human space from the start: its folder supplies 249,660 conversation words, about 37%
of the entire source total. Volume is a strong reason to sample it, not proof that every passage
is high quality or belongs to one faction. The first attributable human target is the early
Free Worlds militia; Republic, Syndicate, and other human groups remain separate candidates.
Quarg provides contrasting first-contact discourse and Hai broader civilian cultural dialogue.
The current release extends that initial selection with Republic Navy and later Free Worlds encounters.

## Canon and scene assumptions

Use development commit `7140eb2a29ce4d2797933075c751791a892c7d4f` from the official repository.
This preserves the inspected snapshot; it is not advertised as the latest stable release.
The inventory has SHA-256 hashes for the four inspected text files and immutable passage links.

The initial inventory covers local contact/culture encounters and early Free Worlds reconnaissance,
before importing later campaign outcomes. Free Worlds records are after initial deployment
and before the player chooses sides; JJ’s local no-shooting-yet account must retain that timing.
Prewar Pact material is historical context only, not an interchangeable faction voice.
There is no single fictitious save state combining all scenes. Each passage retains its own
prerequisites: Kuwaru Efreti follows Quarg contact; the Hai elder conversation follows both Hai
and Unfettered contact; the Academy scene requires Hai language access. A later conversation
must state any necessary role/context rather than silently grant every representative all
knowledge. The post-main-plot Quarg satellite mission is explicitly deferred.

## Reviewed evidence and gaps

This is an agent source review, not human adjudication or a full upstream census. `retain`
means suitable evidence for later authoring after the recorded caveats are handled; it does
not mean a clean extracted dialogue or an approved training example. Repeated branches and
multiple passages from one mission do not create independent source scenes.

| Identity | Retained passage records | Distinct retained mission groups | Context only | Deferred | Excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quarg | 4 | 2 | 1 | 1 | 0 |
| Non-Unfettered Hai | 7 | 6 | 0 | 1 | 1 |
| Free Worlds | 3 | 3 | 1 | 0 | 0 |
| Other human speakers (not targets) | 0 | 0 | 0 | 1 | 1 |
| Total | 14 | 11 | 2 | 3 | 2 |

The three retained Free Worlds missions form one connected story chain. The current pilot
keeps each actual conversation and its variants together; the shared chain alone no longer
forces these separate encounters into one split.
The excluded Hai record covers three related sports scenes. Counts are inventory records,
not utterances, independent facts, future examples, or word counts. The initial inventory did not measure usable-example yield; the release now reports materialized coverage. See each record's `context` and `caveat` before using its summary.

| Topic | Quarg evidence | Hai evidence | Main limitation |
| --- | --- | --- | --- |
| History | q-origin, q-drak | h-elder, h-uncertainty | Attributed accounts; avoid inventing dates or granting young characters elder knowledge |
| Humans / other species | q-greeting, q-efreti | h-contact, h-theater | Local welcomes and partisan views are not universal diplomatic policy |
| Moral decisions | q-origin, q-efreti | h-gambling, h-academy | New decisions will be authored extrapolations, not quotations from canon |
| Technology | q-drak, q-efreti | h-music, h-academy | Qualitative/ordinary technology only; no basis for engineering details |
| Conflict | q-efreti | h-elder, h-gambling | Containment, defense, and disagreement; no general tactical expertise |
| Everyday interaction | q-greeting | h-theater, h-music, h-academy | Quarg evidence is narrow; do not fabricate rich everyday customs |

Human coverage: `fw-transport` provides everyday interaction, `fw-scan` and `fw-jj`
provide reconnaissance technology, political relationships, conflict, and bounded recent history.
Moral decisions can probe the tension between defense and covert surveillance. The sample does
not support alien attitudes, civilian Free Worlds culture, or an authoritative war history.
Use political-group relationships for the human pilot rather than inventing interspecies views.

## Dataset construction

[Issue #5](https://github.com/KelianM/endless-voices/issues/5) absorbs the former source-preparation
and benchmark-authoring issues. The [pilot release](../dataset.md) replaces the provisional
per-identity quotas with source-backed conversation annotation and deterministic sample assembly.
The initial three-identity evidence remains useful but is not a closed list of allowed speakers.

Reserve actual conversations and known branch/repeated-scene variants together. Broad themes,
mission chains and shared source facts alone do not determine a split. Profile and lore annotations
state the relevant speaker role and story date. The original calibration conversations remain
validation material; unrelated encounters in the same campaign can be assigned independently.
