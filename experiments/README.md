# Experiments

Each experiment directory records one investigation: its question, runner, inputs,
procedure and experiment-specific tests. Small result summaries can live beside the
procedure. Model weights, databases and generated run outputs stay in ignored `outputs/`
or `data/local/`; the procedure records how to reproduce them.

Experiments import reusable machinery from `endless_voices`. System code does not import
experiments. An experiment runner may select inputs, collect measurements and organize
review, but encoding, retrieval, generation and evaluation behavior belong in the system.
Move reusable behavior into `src/` when the system needs it; do not copy it into experiments.

The repository separates these responsibilities:

- `src/endless_voices/`: reusable system behavior.
- `src/endless_voices/cli/`: supported commands that parse arguments and call system behavior.
- `scripts/`, if needed: development utilities or short sequences of existing CLI commands.
- `configs/`: reusable system configuration.
- `docs/` and `tests/`: system documentation and regression tests.
- `experiments/<question>/`: investigation-specific code, configuration, notes and tests.

The default `pytest` command runs system tests. Run experiment tests explicitly when changing
an experiment or the system behavior it uses: `pytest -q experiments/<question>`.

[Retrieval embeddings](retrieval-embeddings/README.md) compares embedding model sizes and
passage versus scene results using the shared retrieval system.
