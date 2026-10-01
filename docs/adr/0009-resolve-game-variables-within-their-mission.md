# 9. Resolve game variables within their mission

- **Status:** Proposed
- **Date:** 2026-10-01
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Issue 9](https://github.com/KelianM/endless-voices/issues/9)

## Context

A global substitution can give historical passages the current mission’s destination or replace canonical geography with invented names. Player identity and mission state need different sources of values.

## Decision

`game_variables.py` separates configurable player identity from source-derived mission values. `configs/game-variables.player.json` supplies first name, last name and ship name. Resolve each passage using its owning mission; the target and current encounter share that mission’s values.

Literal destination planets and systems come from cached source. `<destination>` includes planet and system; `<marks>` and `<waypoints>` denote system lists. Dynamic filters, NPC selection and unresolved payment substitutions are not guessed. Unknown markers remain visible. Provenance records source coordinates and resolved values; resolved history is not substituted again using the current mission’s values.

## Consequences

Static geography remains grounded without a complete game simulation. Some examples retain symbolic variables, and a model may reproduce them. Runtime integration will need instantiated mission values before displaying those passages.

Synthetic player identity can still influence generation. Correct substitution alone does not establish that optional branches are mutually compatible; branch consistency requires the interpreter.
