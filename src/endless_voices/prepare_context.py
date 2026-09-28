"""Save reusable context selections from eligible source drafts, without model calls."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from endless_voices.context import (
    Selection,
    TokenizerCounter,
    digest,
    pool_from_draft,
    strategy_from_config,
    variable_values,
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_selections(output, selections, provenance):
    """Write an immutable organizer bundle and generation prompts with artifact hashes."""
    ids = [s.sample_id for s in selections]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must contain unique sample IDs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"selections.json": [asdict(s) for s in selections],
                 "prompts.json": [s.generation_prompt() for s in selections]}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "schema_version": 1, "sample_ids": ids, "preparation": provenance,
        "artifacts": {name: file_hash(output / name) for name in artifacts},
    }, indent=2) + "\n")


def load_selections(output):
    """Verify a saved bundle and return selections for dataset and benchmark consumers."""
    output = Path(output)
    manifest = json.loads((output / "provenance.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported context bundle version")
    for name in ("selections.json", "prompts.json"):
        if file_hash(output / name) != manifest["artifacts"][name]:
            raise ValueError(f"Context artifact hash mismatch: {name}")
    selections = [Selection(**r) for r in json.loads((output / "selections.json").read_text())]
    ids = [s.sample_id for s in selections]
    if ids != manifest["sample_ids"] or len(ids) != len(set(ids)):
        raise ValueError("Context sample IDs differ")
    prompts = json.loads((output / "prompts.json").read_text())
    if prompts != [s.generation_prompt() for s in selections]:
        raise ValueError("Generator and saved context differ")
    if any(digest(s.messages) != s.provenance["messages_sha256"] for s in selections):
        raise ValueError("Selected message hash differs")
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True, help="Eligible source-context JSONL")
    parser.add_argument("--graph", type=Path, required=True, help="Mission prerequisite JSON map")
    parser.add_argument("--strategy", type=Path, required=True, help="Context strategy JSON config")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Cached tokenizer directory")
    parser.add_argument("--output", type=Path, required=True, help="New organizer bundle directory")
    parser.add_argument("--game-vars", type=Path, help="Optional game variable JSON config")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    strategy = strategy_from_config(json.loads(args.strategy.read_text()))
    graph = json.loads(args.graph.read_text())
    variables = variable_values(json.loads(args.game_vars.read_text())) if args.game_vars else {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    counter = TokenizerCounter(tokenizer)
    selections = [strategy.select(pool_from_draft(json.loads(line), graph, variables), counter)
                  for line in args.drafts.read_text().splitlines() if line.strip()]
    import endless_voices.context as context_module

    inputs = [args.drafts, args.graph, args.strategy]
    if args.game_vars:
        inputs.append(args.game_vars)
    save_selections(args.output, selections, {
        "inputs": {str(p): file_hash(p) for p in inputs},
        "tokenizer_files": {str(p.relative_to(args.tokenizer)): file_hash(p)
                            for p in args.tokenizer.rglob("*") if p.is_file()},
        "code": {"context.py": file_hash(context_module.__file__),
                 "prepare_context.py": file_hash(__file__)},
    })
    print(f"Prepared {len(selections)} contexts in {args.output}")


if __name__ == "__main__":
    main()
