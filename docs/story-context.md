# Scene dataset preparation

`DatasetBuilder.build()` owns source preparation. The builder supplies the parsed corpus, mission ownership, interpreter and context strategy to `ExampleBuilder`. The example builder asks the interpreter for reachable targets and asks `ContextSampler` for compatible history. Both consumers use the same state semantics.

```text
DatasetBuilder
  ├─ GameCorpus: verified source paragraphs and mission structure
  ├─ mission split ownership
  └─ ExampleBuilder
       ├─ DialogueInterpreter: reachable targets and state constraints
       └─ ContextSampler
            ├─ DialogueInterpreter: consistent preceding routes
            └─ ContextStrategy: select whole missions within the token budget
  → saved SceneDataset
       ├─ training adapter → DataLoader and collator → batches
       └─ benchmark → saved context and reference
```

The interpreter preserves full authored paragraphs, including narration and embedded questions. A player choice is an explicit game node, not a sentence spoken by a character. Unknown initial conditions can produce multiple satisfiable routes. Assignments on a route determine subsequent conditions; the sampler cannot reverse those assignments to select another passage.

Every distinct reachable target paragraph sequence is retained. When several histories produce that target, preparation chooses one reproducibly. Targets and histories are resolved together before the token-budget strategy selects passages. The saved state witness includes the chosen history's conditions even when the token budget omits some history text.

Whole missions stay in one split. Training history cannot include held-out mission text. Validation may use permitted training history, and test may use permitted training or validation history. Unassigned source missions are omitted until ownership is assigned. The current prerequisite graph follows positive completed-mission requirements; it does not reconstruct arbitrary OR dependencies or event timelines.

`MissionDepth` preserves lore, the current encounter and nearby eligible missions, then fills the remaining input allowance with whole older missions. Overflow of the preserved core is an error. State resolution happens before this token selection, never in DataLoader workers.

The current interpreter supports integer comparisons, Boolean condition groups, assignments, increments, multiplication, minimum/maximum clamps, choices, jumps and paragraph display conditions. Fresh random draws, arithmetic expressions, payments, transferable outfit counts and scheduled events share the same state. Unknown travel times admit compatible event timelines; event delays are not treated as immediate. State-changing question menus can repeat. Unsupported operations, excessive branching and unchanged-state loops fail explicitly. Supported mission assignments run after the initial conversation display and before the first player response, matching the game engine. Source order alone does not establish their execution order.

The current training release is built by this pipeline. [ADR 9](adr/0009-build-examples-from-consistent-game-state.md) records the implemented rule and limits.

## Training objectives

The same source ownership can support two objectives. Scene continuation trains the final passage given context. Corpus language modeling trains authored text throughout each training sequence. Both use next-token prediction, but they put loss on different tokens.

The current training adapter uses scene examples. Its `loss="all"` option includes the saved prompt and history; that is not a deduplicated full-corpus text export. A corpus export should preserve coherent branches and avoid repeatedly training on shared context or including held-out mission text. No corpus-training run is included in this change.
