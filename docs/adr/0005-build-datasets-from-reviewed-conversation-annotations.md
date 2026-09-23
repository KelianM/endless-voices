# 5. Build datasets from reviewed conversation annotations

- **Status:** Accepted
- **Date:** 2026-09-23
- **Sources:** [Issue #5](https://github.com/KelianM/endless-voices/issues/5)

## Context

The authenticity comparison needs attributable game speech, but the game stores dialogue with
narration, choices, branch instructions and changing speakers. Copying individual examples by hand
would repeat mechanical work and make later corrections difficult to reproduce. Blind extraction
would mistake quotations and mission organization for speaker identity and coherent conversations.

The calibration profiles also lacked broader lore. Without relevant background, a response can
reflect missing knowledge rather than the model's ability to portray the selected representative.

## Decision

Keep reviewed annotations of actual conversations, speakers, branch routes, reusable profiles and
source-backed lore in Git. Deterministic scripts extract original speech, retain earlier replies,
and materialize the shared sample format from [ADR 3](0003-use-one-conversation-format-across-splits.md).
The resulting dataset supplies identical selected lore and context to both model conditions.
All target replies use game speech, consistent with
[ADR 4](0004-evaluate-authenticity-against-game-continuations.md).

Keep each actual conversation and its known repeated/branch variants in one split. Mission chains,
themes, faction labels and shared lore alone do not determine split membership. Preserve source
provenance independently from those relationships.

Store extracted payloads outside Git, with versioned manifests and committed reconstruction hashes.
Content corrections produce new dataset versions. Annotation and source review is explicit agent
review; human approval is recorded only when actually supplied.

## Consequences

- Agents spend their effort on interpretation and context. Changing an annotation regenerates all
  affected samples without asking an agent to recopy speech or construct record metadata.
- The preparation script can support a later annotation interface. It does not execute game state;
  branch coherence, speaker attribution and knowledge boundaries still require source review.
- Facts can recur across splits, so evaluation concerns speaking with supplied lore rather than
  recall of previously unseen facts. Public source text can still have appeared in pretraining.
- Different encounters in a campaign can cross splits. This preserves usable coverage but does
  not test generalization to entirely unseen story arcs. Conversation prefixes remain dependent
  samples and must not be counted as independent observations.
- A fresh checkout must reconstruct the payload from pinned sources. Release hashes make the
  reconstruction auditable; the recipe and a successful schema check alone do not establish quality.
