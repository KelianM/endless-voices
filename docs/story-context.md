# Context for preliminary scene-continuation training

The objective is to learn the game's writing and characters. Dataset and benchmark preparation use the same minimal task instruction, recorded in [ADR 11](adr/0011-use-a-shared-scene-continuation-instruction.md). The model receives a character reference, verbatim game lore, earlier mission passages and the current encounter. No directions about cadence, caution, verbosity or speaking authority are added.

## Split and target boundaries

Whole source missions form the split unit, as recorded in [ADR 13](adr/0013-split-datasets-by-source-mission.md). Existing held-out assignments take precedence when migrating conversation-level assignments. Training preparation excludes examples belonging to missions represented in validation or test, and excludes those missions from historical references. Unannotated prerequisite passages remain eligible unless they belong to a held-out mission. Ambiguous continuation boundaries are recorded as exclusions.

The target preserves complete authored paragraphs, including narration and other speakers. Preparation rejects target paragraphs already present in the current input. The current encounter ends before the target; subsequent scene text is not supplied as its lead-in.

Later training conversations may reveal story developments. That is compatible with learning game writing, but means these data do not measure chronological character knowledge or unseen-story generalization. The validation set has been used repeatedly for development. The earlier proposal to reorder every story chronologically is superseded; historical prototype measurements remain unchanged.

## Context selection

Mission prerequisites select relevant earlier material without a maintained list of important stories. Optional alternatives can supply writing examples but are labelled as alternatives, not simultaneous events. Unresolved prerequisites and exclusions are recorded instead of being filled with generated summaries.

`endless_voices.context` provides the shared strategy interface. `FullContext` retains all eligible passages. `MissionDepth` preserves the current encounter, lore and nearby prerequisite missions, then samples whole older missions into the remaining input budget. Preparation fails if the preserved core exceeds the budget. Whole-mission boundaries and sparse reference pools can leave unused space.

The saved selection records depth, token budget, seed, source coordinates, retained and omitted passages, and hashes. Generation, judging and training reuse the same selected messages. The judge additionally receives anonymous candidate continuations and its own judging task. Origin detection is not a direct measure of storytelling quality.

## Game variables

Player identity is configurable. Static mission values are resolved separately for each passage's owning mission, under [ADR 12](adr/0012-resolve-game-variables-within-their-mission.md). Unknown runtime values remain literal markers. Source text, substitutions and unresolved markers are retained for review. Historical passages are never substituted again using the current mission's destination.

The [preliminary training release](../data/scene-training-v2/README.md) records exclusions, token measurements and remaining markers. The release does not claim that every optional branch occurred in one playthrough, that the game state machine has been reconstructed, or that adapter training fits the local machine.
