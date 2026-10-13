"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

from endless_voices.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    SceneDatabase,
    TransformersEmbedder,
    scene_id,
)

from .storage import SceneDataset


def index_context(dataset, output, model=DEFAULT_EMBEDDING_MODEL, *, device="auto",
                  dtype="float16", batch_size=4, max_length=32768, cache_dir="data/local/hub"):
    """Build an index of eligible passages and return its scene count and path."""
    datasets = SceneDataset.load_all(dataset)
    scenes = {}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            pool = dataset.context_pool(index)
            for block in (*pool.blocks, *pool.references):
                scenes[scene_id(block)] = block
    embedder = TransformersEmbedder.from_pretrained(
        model, device=device, dtype=dtype, batch_size=batch_size,
        max_length=max_length, cache_dir=cache_dir)
    with SceneDatabase.create(output, scenes.values(), embedder):
        pass
    return {"scenes": len(scenes), "output": str(output)}
