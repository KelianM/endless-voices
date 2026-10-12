"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

import argparse
import json
from pathlib import Path

from endless_voices.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    SceneDatabase,
    TransformersEmbedder,
    scene_id,
)

from .storage import SceneDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/local/hub"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = SceneDataset.load_all(args.dataset)
    scenes = {}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            pool = dataset.context_pool(index)
            for block in (*pool.blocks, *pool.references):
                scenes[scene_id(block)] = block
    embedder = TransformersEmbedder.from_pretrained(
        args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size,
        max_length=args.max_length, cache_dir=args.cache_dir)
    with SceneDatabase.create(args.output, scenes.values(), embedder):
        pass
    print(json.dumps({"scenes": len(scenes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
