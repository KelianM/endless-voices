# 12. Resolve game variables within their mission

- **Status:** Proposed
- **Date:** 2026-09-30
- **Sources:** [PR 16](https://github.com/KelianM/endless-voices/pull/16), [Endless Sky mission implementation](https://github.com/endless-sky/endless-sky/blob/master/source/Mission.cpp)

## Context

A global dummy mapping replaced mission destinations with fictional locations and interpreted `<marks>` as currency. Historical passages can belong to different missions, so even a correct current destination is wrong when applied to every passage. A ship named Wanderer also introduces an avoidable association with a game faction.

## Decision

`game_variables.py` separates configurable player identity from source-derived mission state. `configs/game-variables.player.json` supplies only first name, last name and ship name. Each source passage is resolved using its owning mission. The current target uses the same mission values as the current encounter.

Literal destination planets and their systems come from cached game source. `<destination>` includes both planet and system; `<marks>` and `<waypoints>` denote system lists. Dynamic location filters, NPC selection and payment calculations are not guessed. Unknown markers remain visible and are recorded for review. Source text and resolved values are retained, and resolved history is never substituted again using the current mission's values.

## Consequences

Static mission geography is preserved without implementing the game's state machine. Some examples retain symbolic variables, so the model may also return those markers. A future runtime integration needs instantiated mission values to render those examples for a player.

The synthetic player identity is controlled but can still influence generation. The historical global dummy configuration and benchmark results remain evidence of the earlier procedure; they are not a clean baseline for a corrected dataset. This rule does not establish that arbitrary optional branches occurred together or resolve conditional story outcomes.
