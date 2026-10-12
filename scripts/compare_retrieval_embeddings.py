"""Freeze real scene pools, run local retrieval models and prepare blinded review packets."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from endless_voices.artifacts import file_hash
from endless_voices.context import ContextPool, FullContext, digest
from endless_voices.retrieval import SceneDatabase, TransformersEmbedder, scene_id


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def prepare(args):
    from endless_voices.dataset.builder import DatasetBuilder
    from endless_voices.dataset.source import GameCorpus
    from endless_voices.dataset.state import GameState

    class Counter:
        def text(self, text):
            return len(text)

        def messages(self, messages):
            return sum(len(message["content"]) for message in messages)

    plan = json.loads(args.plan.read_text())
    config = json.loads(args.config.read_text())
    inventory = json.loads(args.inventory.read_text())
    corpus = GameCorpus(args.source, inventory)
    builder = DatasetBuilder(corpus, config, Counter())
    builder.sampler.strategy = FullContext()
    pools = {}
    for mission in sorted({query["mission"] for query in plan["queries"]}):
        examples = builder.examples.build(
            corpus.missions[mission], config["missions"][mission], Counter(),
            GameState.fixed(config.get("initial_state", {})))
        selection = max(examples, key=lambda row: len(row[1].pool.blocks))[1]
        pools[mission] = selection.pool.__dict__
    scenes = {scene_id(block): block for pool in pools.values()
              for block in pool["blocks"] + pool["references"]}
    payload = {
        "format": "retrieval-comparison-v1", "plan": plan, "pools": pools,
        "source_revision": corpus.revision, "source_files": corpus.files,
        "config_sha256": file_hash(args.config), "plan_sha256": file_hash(args.plan),
        "pool_selection": "Most history scenes, breaking ties by builder example order",
        "candidate_scenes": len(scenes),
        "candidate_passages": sum(len(block["passages"]) for block in scenes.values()),
        "candidate_characters": sum(len(p["text"]) for block in scenes.values()
                                    for p in block["passages"]),
    }
    save(args.output, payload)
    print(json.dumps({key: payload[key] for key in (
        "candidate_scenes", "candidate_passages", "candidate_characters")}))


def run(args):
    import torch
    import transformers

    frozen = json.loads(args.input.read_text())
    pools = {name: ContextPool(**fields) for name, fields in frozen["pools"].items()}
    scenes = {scene_id(block): block for pool in pools.values()
              for block in pool.blocks + pool.references}
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    embedder = TransformersEmbedder.from_pretrained(
        args.model, device=args.device, dtype=args.dtype,
        max_length=args.max_length, batch_size=args.batch_size)
    loaded = time.perf_counter()
    database = SceneDatabase.create(args.output / "context.sqlite", scenes.values(), embedder)
    indexed = time.perf_counter()
    results = []
    with database:
        for query in frozen["plan"]["queries"]:
            from endless_voices.retrieval import RetrievalSession

            session = RetrievalSession(database, pools[query["mission"]])
            before = time.perf_counter()
            matches = session.search_context(query["query"], limit=args.limit, expand_scenes=False)
            elapsed = time.perf_counter() - before
            results.append({
                "query_id": query["id"], "query": query["query"],
                "matches": matches, "seconds": elapsed,
                "passages": session.read_context([match["id"] for match in matches]),
            })
    ended = time.perf_counter()
    metadata = {
        "input_sha256": file_hash(args.input), "model": str(args.model),
        "model_revision": args.revision, "embedder": embedder.identity,
        "device": str(embedder.model.device), "dtype": str(embedder.model.dtype),
        "max_length": args.max_length, "batch_size": args.batch_size, "limit": args.limit,
        "result_unit": "passage",
        "load_seconds": loaded - started, "index_seconds": indexed - loaded,
        "query_seconds": sum(result["seconds"] for result in results),
        "total_seconds": ended - started, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "retrieval_source_sha256": file_hash(Path(__file__).resolve().parents[1]
                                             / "src/endless_voices/retrieval.py"),
    }
    save(args.output / "results.json", {"metadata": metadata, "queries": results})
    print(json.dumps(metadata))


def blind(args):
    frozen = json.loads(args.input.read_text())
    first, second = (json.loads(path.read_text()) for path in (args.first, args.second))
    if any(result["metadata"]["input_sha256"] != file_hash(args.input)
           for result in (first, second)):
        raise ValueError("Both runs must use the same frozen inputs")
    for setting in ("dtype", "max_length", "batch_size", "limit", "retrieval_source_sha256"):
        if first["metadata"][setting] != second["metadata"][setting]:
            raise ValueError(f"Comparison setting differs: {setting}")
    if any(result["metadata"].get("result_unit", "passage") != "passage"
           for result in (first, second)):
        raise ValueError("Blinded passage comparison requires passage results")
    packets, key = [], {}
    for query, one, two in zip(frozen["plan"]["queries"], first["queries"],
                               second["queries"], strict=True):
        if query["id"] != one["query_id"] or query["id"] != two["query_id"]:
            raise ValueError("Query order differs")
        ordered = [("first", one), ("second", two)]
        if int(digest([args.seed, query["id"]]), 16) % 2:
            ordered.reverse()
        packet = {**query, "results": {}, "rubric": (
            "Score each of the first three passages: 0 irrelevant, 1 loosely useful, "
            "2 directly useful. Choose A, B or tie for the query; reward useful variety "
            "and penalize repeated evidence. Distinguish independent references from "
            "events in the selected history. Do not infer model identity.")}
        key[query["id"]] = {}
        for label, (source, result) in zip(("A", "B"), ordered, strict=True):
            key[query["id"]][label] = source
            packet["results"][label] = result["passages"][:3]
        packets.append(packet)
    save(args.output / "blind.json", packets)
    save(args.output / "key.json", {"first": first["metadata"],
                                     "second": second["metadata"], "mapping": key})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("plan", "config", "inventory", "source", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    run_parser = commands.add_parser("run")
    for name in ("input", "model", "output"):
        run_parser.add_argument("--" + name, type=Path, required=True)
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--dtype", default="auto")
    run_parser.add_argument("--max-length", type=int, default=32768)
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--limit", type=int, default=5)
    packet = commands.add_parser("blind")
    for name in ("input", "first", "second", "output"):
        packet.add_argument("--" + name, type=Path, required=True)
    packet.add_argument("--seed", default="retrieval-comparison-v1")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "blind": blind}[args.command](args)


if __name__ == "__main__":
    main()
