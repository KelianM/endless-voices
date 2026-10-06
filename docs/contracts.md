# Prepared dataset contract

`SceneDataset.load(root, split)` is the shared entry point for training, generation and
benchmarking. The loader verifies every published split before returning the requested split.
Loading performs no sampling, interpretation or model calls.

A dataset directory contains `manifest.json`, one JSONL file per published split, and
`config.json`, `provenance.json` and licensing. Each JSONL record contains its context and target;
provenance retains source coordinates and state evidence without duplicating dialogue. The manifest declares
`format: scene-dataset-v1`, `split_unit: mission`, the published `splits`, and SHA-256 hashes
in `artifacts`. A training-only dataset is valid; requesting an unpublished split fails.
Artifact paths must remain inside the dataset directory. Modified files, reused split files,
duplicate sample IDs and mission, conversation or known-variant groups crossing splits fail
validation. There is one maintained dataset format.

Each record has `schema_version`, `metadata`, `messages` and `evaluation`. Metadata identifies
the sample, character, split, source mission and any known variants. The required `task` is
`scene_opening` or `scene_continuation`, derived from the current conversation prefix during
construction. The task selects the shared instruction and remains private metadata. Messages
contain the shared system instruction and selected context, followed by the authored conversation
and final target.
The final assistant message is the full authored continuation, including narration. Evaluation
metadata supplies source attribution and is never inserted into model input.

Generation receives `messages[:-1]`. The judge receives that same context and the original and
generated continuations in one randomized A/B assignment. Target text and private metadata cannot
change the generator prompt. Context is selected once during preparation and remains fixed for
all consumers.

Training reads a dataset directory and a split from `configs/train.toml`. Continuation loss masks
all tokens before the final target; `loss = "all"` includes the saved prompt and history.
Neither mode silently truncates overlong examples. The collator masks padding without masking
real end-of-sequence tokens when PAD and EOS share an ID.

Run `validate data/dataset/manifest.json` for structural verification and coverage. Optional
`--tokenizer /path/to/local/tokenizer --max-length 9216` checks complete sequence lengths without
downloading weights or a tokenizer. The generation runner separately budgets prompt and output.

[Dataset preparation](../data/dataset/README.md#how-this-dataset-is-produced) explains source interpretation and context selection.
[Evaluation](evaluation.md) explains blinding, denominators and uncertainty. Dataset properties
and limitations belong in [the dataset README](../data/dataset/README.md).
