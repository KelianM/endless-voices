# Published scene dataset

The dataset contains authored encounters and targets from 462 selected Human, Hai and Quarg
missions. Prerequisite-connected mission chains and known text-sharing variants stay in one
split. The selection does not cover every mission in those regions.

The conversation files contain instructions, the visible encounter and its authored target.
Provenance records game states, source passage traces and context fingerprints. No sampled
context, mission graphs or eligible-scene lists are stored per example.

Retrieval reconstructs compatible history and independent writing references from the
pinned raw source cache. The manifest locates that cache through `source_root`, relative to
this directory, and records hashes for its files. Reconstruction must match the saved
context fingerprint. The passage embedding index is a separate, rebuildable local artifact.
See the [README](../../README.md#prepare-data) for preparation and indexing commands.

Identity labels group source regions, not individual speakers. Species is unspecified
because scenes can contain multiple characters. Unresolved runtime substitutions remain
literal markers; no missing game facts are filled with generated text.

Training context excludes held-out missions. Validation can use training context, and test
can use training and validation context. References exclude the target mission and known
sibling variants. History records one compatible branch, not every possible playthrough.
The prerequisite graph follows positive completion requirements and required events
scheduled by mission offers or completions. Multiple mission producers mean at least one producer is required. Histories explore
individual producers and combinations in different orders; game conditions determine
which combinations are compatible. Offer histories stop without inventing a
completion, and only branches that schedule the required event can supply its history.
All possible producers share a split; unused predecessor routes are excluded from references.
A milestone required absent earlier and present later brings its supported mission setters
into the history. Events with other kinds of triggers and requirements inside OR conditions
remain unresolved.

The interpreter resolves supported dialogue conditions, assignments and mission effects.
Unknown starting conditions and travel durations remain constrained possibilities. Recorded
world changes do not constitute a complete game simulation; see
[ADR 9](../../docs/adr/0009-build-examples-from-consistent-game-state.md).
