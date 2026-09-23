# 5. Build datasets from reviewed conversation annotations

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #5](https://github.com/KelianM/endless-voices/issues/5),
  [PR #14](https://github.com/KelianM/endless-voices/pull/14)

## Context

The authenticity comparison needs attributable game speech, but the game stores dialogue with
narration, choices, branch instructions and changing speakers. Copying individual examples by hand
would repeat mechanical work and make later corrections difficult to reproduce. Blind extraction
would mistake quotations and mission organization for speaker identity and coherent conversations.

The calibration profiles also lacked broader lore. Without relevant background, a response can
reflect missing knowledge rather than the model's ability to portray the selected representative.

Expanding the dataset requires judgment about speakers, story conditions and relevant knowledge.
Those decisions need source inspection and correction, and the pilot has no validated automated
annotation workflow. Agent delegation can share the reading and annotation work, but someone must
define each assignment, reconcile findings and decide which constructions are ready to retain.

## Decision

Use manually orchestrated agents for annotation under an owner-approved scope. A lead agent
prepares the source catalog, assigns bounded source batches to construction agents, and supplies
the annotation rules and source references. Construction agents annotate speakers, routes,
profiles, lore and scene assumptions. The lead agent inspects those annotations against the source,
requests or makes corrections, resolves attribution and split conflicts, and assembles the release
for the owner's PR review. Construction review, implementing review and human approval remain
distinct claims, as recorded in the [pilot review evidence](../curation/pilot-v1/review.json).

Source preparation, speech extraction, sample assembly and validation run in deterministic scripts.
Agent dispatch, follow-up instructions, correction cycles and acceptance of annotations are
coordinated interactively. The repository does not contain an annotation job runner that performs
that orchestration from API credentials and a command.

Keep reviewed annotations of actual conversations, speakers, branch routes, reusable profiles and
source-backed lore in Git. Deterministic scripts extract original speech, retain earlier replies,
and materialize the shared sample format from [ADR 3](0003-use-one-conversation-format-across-splits.md).
The resulting dataset supplies identical selected lore and context to both model conditions.
All target replies use game speech, consistent with
[ADR 4](0004-evaluate-authenticity-against-game-continuations.md).

Keep each actual conversation and its known repeated/branch variants in one split. Mission chains,
themes, faction labels and shared lore alone do not determine split membership. Preserve source
provenance independently from those relationships.

Version the complete curated release through Git LFS, retaining its manifest and reconstruction
hashes. Keep agent-authored annotations and curation review records in ordinary Git. Mark the
materialized release as generated so dataset output does not dominate code review. Keep the retrievable raw upstream corpus
outside Git. Agent work must remain available without rerunning generation or reconstruction.
Content corrections produce new dataset versions. Annotation and source review is explicit agent
review; human approval is recorded only when actually supplied.

## Consequences

- The committed annotations reproduce the exact dataset without repeating agent work. Producing
  annotations for additional sources requires another orchestrated curation and review pass;
  supplying an API key alone is insufficient. Repeating the agent assignments may produce different
  selections and judgments, even when the source revision is unchanged.
- Expansion consumes agent tokens and coordinator attention. Progress depends on batch assignment,
  available agent capacity and source review. The workflow accepts that operational cost for the
  pilot; an unattended annotation service is outside the implemented design.
- Review can catch speaker mixing and invalid story assumptions, but construction and implementing
  agents can share mistakes. Agent review does not establish independent adjudication or replace
  the owner's decision to accept the release.
- Agents spend their effort on interpretation and context. Changing an annotation regenerates all
  affected samples without asking an agent to recopy speech or construct record metadata.
- The preparation script can support a later annotation interface. It does not execute game state;
  branch coherence, speaker attribution and knowledge boundaries still require source review.
- Facts can recur across splits, so evaluation concerns speaking with supplied lore rather than
  recall of previously unseen facts. Public source text can still have appeared in pretraining.
- Different encounters in a campaign can cross splits. This preserves usable coverage but does
  not test generalization to entirely unseen story arcs. Conversation prefixes remain dependent
  samples and must not be counted as independent observations.
- A checkout with Git LFS contains the exact curated dataset and its review evidence. Git stores
  compact pointers while LFS stores the payload; a checkout without LFS contains only pointers
  until the objects are fetched. This adds an LFS installation and storage dependency. The pilot
  is small in bytes, but its generated text dominates the PR diff; review visibility motivates
  LFS here. Reviewers inspect materialized samples locally and review annotations in Git.
  Reconstruction hashes still detect payload drift; storage format does not establish quality.
