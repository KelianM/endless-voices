# Dataset inputs

`data/sources/source-statistics.json` pins the game-source revision and file hashes used by
`endless-voices fetch-sources`. Raw sources and model caches stay in ignored `data/local/`;
run outputs stay in ignored `outputs/`.

`data/dataset/config.json` records mission ownership, player substitutions and target scope.
Omitting a mission's conversation list selects supported conversations and dialog passages.
An empty list excludes targets while retaining split ownership. The configuration contains
no quote selectors or per-branch routes.

[Preparation usage](../README.md#prepare-data) builds encounters, targets, eligible context
pools, provenance and licensing. Preparation needs no tokenizer or model. Unsupported
operations or exceeded exploration limits fail the build rather than dropping examples.
Context selection and token budgets apply when a run consumes the prepared examples.

The [published dataset](dataset/README.md) describes source coverage and limitations.
[ADR 7](../docs/adr/0007-split-datasets-by-source-mission.md) defines mission ownership;
[ADR 9](../docs/adr/0009-build-examples-from-consistent-game-state.md) explains interpretation.
