"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

import argparse
import json
from pathlib import Path

from endless_voices.retrieval import SceneDatabase, TransformersEmbedder, scene_id

from .storage import SceneDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = SceneDataset.load_all(args.dataset)
    scenes = {}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            pool = dataset.context_pool(index)
            for block in (*pool.blocks, *pool.references):
                scenes[scene_id(block)] = block
    embedder = TransformersEmbedder.from_pretrained(args.model)
    with SceneDatabase.create(args.output, scenes.values(), embedder):
        pass
    print(json.dumps({"scenes": len(scenes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
