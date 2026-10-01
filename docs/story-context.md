# Scene dataset preparation

`DatasetBuilder.build()` owns source preparation. The builder supplies the parsed corpus, mission ownership, interpreter and context strategy to `ExampleBuilder`. The example builder asks the interpreter for reachable targets and asks `ContextSampler` for compatible history and independent writing references. Both consumers use the same state semantics.

```text
DatasetBuilder
  ├─ GameCorpus: verified source paragraphs and mission structure
  ├─ mission split ownership
  └─ ExampleBuilder
       ├─ DialogueInterpreter: reachable targets and state constraints
       └─ ContextSampler
            ├─ DialogueInterpreter: preceding routes and independent reference scenes
            └─ ContextStrategy: preserve nearby history, then fill the input budget
  → saved SceneDataset
       ├─ training adapter → DataLoader and collator → batches
       └─ benchmark → saved context and reference
```

The interpreter preserves full authored paragraphs, including narration and embedded questions. A player choice is an explicit game node, not a sentence spoken by a character. Unknown initial conditions can produce multiple satisfiable routes. Assignments on a route determine subsequent conditions; the sampler cannot reverse those assignments to select another passage.

Every distinct reachable target paragraph sequence is retained. When several histories produce that target, preparation chooses one reproducibly. Targets and histories are resolved together before the token-budget strategy selects passages. The saved state witness includes the chosen history's conditions even when the token budget omits some history text.

Whole missions stay in one split. Training history cannot include held-out mission text. Validation may use permitted training history, and test may use permitted training or validation history. Prerequisite missions without selected targets are eligible as supporting context unless assigned to a held-out split. The current prerequisite graph follows positive completed-mission requirements; it does not reconstruct arbitrary OR dependencies or event timelines.

`MissionDepth` preserves lore, the current encounter and nearby eligible missions. Whole older prerequisite missions use the remaining budget first. Independent writing references then fill spare input space: complete conversations from the source files containing the configured identity's missions, followed by authored ship, outfit and government descriptions from those source directories. The current mission and held-out missions are excluded from the reference pool before interpretation. This rule uses source organization rather than hand-maintained related-story lists.

Reference conversations are separate scenes, labelled separately from preceding events. The interpreter samples eight seeded complete routes per conversation and retains the longest available route, avoiding a reference pool dominated by brief refusals. Each reference has its own compatible state; independent scenes need not form a shared chronology. Target extraction continues to enumerate reachable branches. Selection preserves complete reference conversations and description blocks rather than cutting text to reach an exact token count.

Overflow of the preserved core is an error. Preparation fixes the selected context for training, generation and judging; DataLoader workers do not resample or interpret game text.

The current interpreter supports integer comparisons, Boolean condition groups, assignments, increments, multiplication, minimum/maximum clamps, choices, jumps and paragraph display conditions. Fresh random draws, arithmetic expressions, payments, transferable outfit counts and scheduled events share the same state. Prerequisite history samples one compatible event timeline per route reproducibly from the dataset seed; it does not enumerate every possible travel duration. The current mission still enumerates reachable dialogue branches and event timelines. Event delays are not treated as immediate. Question menus retain each reachable answer; identical choices at the same state are not repeated indefinitely within a route. State-changing choices remain available after a state change. Unsupported operations, excessive branching and loops without player input fail explicitly. Literal dialog blocks preserve all authored paragraphs. Shop, fleet and map changes, journal entries, ship gifts and account charges remain recorded effects rather than condition assignments. Financial servicing and fleet simulation are not implemented. Outfit requirements constrain the state before a conversation is displayed. Supported mission assignments run after the initial conversation display and before the first player response, matching the game engine. Source order alone does not establish their execution order.

The current training release is built by this pipeline. [ADR 9](adr/0009-build-examples-from-consistent-game-state.md) records the implemented rule and limits.

## Training objectives

The same source ownership can support two objectives. Scene continuation trains the final passage given context. Corpus language modeling trains authored text throughout each training sequence. Both use next-token prediction, but they put loss on different tokens.

The current training adapter uses scene examples. Its `loss="all"` option includes the saved prompt and history; that is not a deduplicated full-corpus text export. A corpus export should preserve coherent branches and avoid repeatedly training on shared context or including held-out mission text. No corpus-training run is included in this change.
