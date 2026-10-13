# System design

Endless Voices prepares game scenes, trains models to continue those scenes and compares
continuations with the original writing. The [README](../README.md) contains usage;
[ADRs](adr/README.md) explain lasting decisions.

## Preparation

[DatasetBuilder](../src/endless_voices/dataset/builder.py) interprets pinned game sources
into visible encounters, authored targets and compatible context. Prerequisite-connected mission chains stay in one split. The interpreter resolves supported branches and mission variables; the interpreter
does not reproduce arbitrary game saves.

Prepared records contain the instruction, visible encounter and target. The prompt names
the current mission event and its known location without exposing post-target state. Eligible history and
writing references are saved separately, with scene text deduplicated across examples.
[SceneDataset](../src/endless_voices/dataset/storage.py) verifies artifact hashes, pool
consistency and ownership across every published split.
[Record validation](../src/endless_voices/contracts.py) defines the record invariants.

## Retrieval

A static SQLite database holds passage vectors and interpreted scenes.
[Retrieval sessions](../src/endless_voices/retrieval.py) enforce each example's eligibility
before searching or reading. History represents preceding events; independent references
supply writing examples. Search ranks a scene by its strongest passage and returns the
whole scene ([ADR 11](adr/0011-rank-passages-and-return-their-parent-scenes.md)).

[Context selection](../src/endless_voices/context_generation.py) supplies two policies over
the same pool. Static selection fits nearby mission history, then semantic matches, within
a token budget. Dynamic generation uses the same tool loop and instructions for local and
hosted models, retaining the conversation between retrievals. Provider APIs and local chat
templates supply their native call formats. Both policies record selected context and omit
complete results that exceed the input budget
([ADR 10](adr/0010-let-agents-retrieve-context-through-shared-tools.md)).

## Training and evaluation

[Training](../src/endless_voices/train.py) selects static context before loading the model,
reserving space for the target. [Generation](../src/endless_voices/generate.py) supports
both policies and withholds targets and private evaluation metadata from the generator.
Runs save actual messages and retrieval traces rather than relying on later reconstruction.

The benchmark supplies common static evidence to judges regardless of generator mode.
Blinded evaluation compares the final writing; retrieval traces show how each generator
selected its context. [Evaluation documentation](evaluation.md) describes reporting limits.

Command parsing lives in [cli/](../src/endless_voices/cli/); reusable behavior lives in the
package. [Experiments](../experiments/README.md) contain disposable investigations.
