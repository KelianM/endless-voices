"""Compare shared context strategies on cached validation drafts without model calls."""

import argparse
import json
from pathlib import Path

from assemble_validation_context import load_prerequisite_graph, sha

from endless_voices.context import FullContext, MissionDepth, TokenizerCounter, pool_from_draft
from endless_voices.prepare_context import save_selections

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New measurement directory")
    parser.add_argument("--contexts", type=Path,
                        default=ROOT / "outputs/validation-source-context-v2/contexts.jsonl")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/local/endless-sky-7140eb2a29ce")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Cached tokenizer directory")
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--seed", default="context-depth-v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    if len(args.depths) != len(set(args.depths)):
        parser.error("Depths must be unique")
    strategies = [MissionDepth(d, args.max_input_tokens, args.seed) for d in args.depths]
    inventory_path = ROOT / "data/overview/source-statistics.json"
    inventory = json.loads(inventory_path.read_text())
    graph = load_prerequisite_graph(args.source, inventory)
    from transformers import AutoTokenizer

    counter = TokenizerCounter(AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True))
    drafts = [json.loads(line) for line in args.contexts.read_text().splitlines() if line.strip()]
    pools = [pool_from_draft(draft, graph) for draft in drafts]
    full = [FullContext().select(pool, counter) for pool in pools]
    args.output.mkdir(parents=True)
    (args.output / "prerequisites.json").write_text(json.dumps(graph, indent=2) + "\n")
    provenance = {
        "inputs": {str(p): sha(p) for p in (inventory_path, args.contexts, Path(__file__))},
        "source_revision": inventory["revision"],
        "tokenizer_files": {str(p.relative_to(args.tokenizer)): sha(p)
                            for p in args.tokenizer.rglob("*") if p.is_file()},
    }
    import endless_voices.context as context_module

    provenance["selection_code_sha256"] = sha(Path(context_module.__file__))
    save_selections(args.output / "full", full, provenance)
    rows = []
    for strategy in strategies:
        selections = [strategy.select(pool, counter) for pool in pools]
        save_selections(args.output / f"depth-{strategy.depth}", selections, provenance)
        for pool, baseline, selection in zip(pools, full, selections):
            p, counts = selection.provenance, selection.token_counts
            rows.append({"sample_id": pool.sample_id, "mission": pool.mission,
                         "depth": strategy.depth, "baseline_tokens": baseline.token_counts["input"],
                         "input_tokens": counts["input"],
                         "near_history_tokens": counts["full_history"],
                         "sampled_history_tokens": counts["sampled_history"],
                         "near_missions": len(p["full_missions"]),
                         "sampled_missions": len(p["sampled_missions"]),
                         "omitted_missions": len({b["mission"] for b in selection.omitted})})
    (args.output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Measured {len(rows)} contexts; no model calls")


if __name__ == "__main__":
    main()
