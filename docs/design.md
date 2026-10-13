# System design

Endless Voices prepares game scenes, trains models to continue those scenes and compares
generated continuations with the original writing. The [README](../README.md) contains usage;
[ADRs](adr/README.md) explain lasting decisions.

## Preparation and saved examples

[DatasetBuilder](../src/endless_voices/dataset/builder.py) interprets pinned game sources
into targets and compatible context. Mission ownership separates dataset splits. The same
interpreter resolves branches and mission variables for targets and history; the interpreter
models supported source operations, not the entire game.

A prepared bundle stores split records, configuration, provenance, eligible retrieval pools
and licensing, with artifact hashes in a manifest. Each record contains saved messages ending
in the authored target, plus private identity and evaluation metadata.
[SceneDataset](../src/endless_voices/dataset/storage.py) verifies artifacts, pool consistency
and ownership across every published split before returning a requested split. Loading does
not reinterpret or resample source text. [Record validation](../src/endless_voices/contracts.py)
defines the required fields and invariants.

## Context selection

[Context selection](../src/endless_voices/context.py) separates eligible material from the
policy that chooses it. Preparation saves both a fixed sampled prompt and the broader eligible
pool for each example. Mission history represents the selected playthrough; independent
references supply writing examples without asserting that their events happened.

[Retrieval sessions](../src/endless_voices/retrieval.py) restrict search and reads to that
example's interpreted scenes, excluding its target paragraphs. A shared static SQLite index
stores passage vectors and parent scenes. Search ranks scenes by their strongest passage
match and returns complete scenes by default. [ADR 10](adr/0010-let-agents-retrieve-context-through-shared-tools.md)
explains agent-directed selection; [ADR 11](adr/0011-rank-passages-and-return-their-parent-scenes.md)
explains scoring and expansion. Tool sessions are available, but current generation and
benchmark runners still use fixed prompts.

## Training and evaluation

[Training](../src/endless_voices/train.py) learns a LoRA adapter from saved records.
[Generation](../src/endless_voices/generate.py) receives the saved messages without the final
target; private evaluation metadata does not enter the prompt. The benchmark gives the judge
the same context with original and generated continuations in blinded order.
[Evaluation documentation](evaluation.md) defines interpretation and reporting limits.

Command argument handling lives in [cli/](../src/endless_voices/cli/); reusable behavior
lives in the package. [Experiments](../experiments/README.md) consume that machinery and keep
ad-hoc procedures and findings separate from supported workflows.
