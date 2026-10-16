# Gameplay reconstruction

This package reconstructs condition-consistent histories from authored Endless Sky
sources. Dataset preparation consumes the histories; the gameplay package does not
import dataset, retrieval, agent or model code.

| Module | Responsibility |
| --- | --- |
| `source.py` | Parse and verify source content, preserve passage coordinates, resolve named definitions |
| `graph.py` | Infer required conditions and the mission phases that can establish them |
| `state.py` | Interpret symbolic integer conditions, actions and scheduled events |
| `dialogue.py` | Traverse satisfiable dialogue branches and preserve player choices |
| `histories.py` | Order mission phases, execute histories and select reproducible encounters |
| `variables.py` | Resolve player and mission substitutions in authored passages |

`GameCorpus.mission_graph` includes every inferred producer link. A graph edge does not
mean that every producer occurred in a particular playthrough. `HistoryReconstructor`
chooses producer combinations, executes their mission phases and rejects histories
whose interpreted conditions conflict.

`HistoryReconstructor.histories()` returns compatible predecessor histories and omitted
mission names. `encounters()` traverses the target mission and selects one compatible
history per distinct authored target. Each encounter carries the source node, dialogue
continuation, preceding history and mission phase. History blocks contain attributed
passages with game substitutions; dataset code adds prompts, retrieval IDs and records.

Callers supply a corpus, dialogue interpreter, player substitutions and sampling seed.
Callers can exclude predecessor missions by name. Dataset preparation derives those
exclusions from split eligibility; gameplay does not interpret split names or dataset
metadata. Sampling order and state projection share deterministic hashing with other
artifact consumers.

The interpreter establishes consistency within supported game semantics, not complete
reachability from a real save. Unknown initial conditions remain constrained
possibilities. Concrete solver snapshots can choose different satisfying values even
when passages and histories match. Dependency inference supports selected prerequisite
forms; mission
scheduling explores sequential and overlapping producer schedules, not every possible
phase interleaving. State comparison normalizes commutative operand order before grouping histories.
Equivalent projected states share a representative history, independently of solver
print order.
Unsupported operations and exceeded route limits fail explicitly. These limits apply
to both dataset preparation and direct gameplay use.
