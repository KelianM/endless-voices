"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

from endless_voices.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    SceneDatabase,
    TransformersEmbedder,
    authored_scene,
    scene_id,
)

from .storage import SceneDataset


def index_context(dataset, output, model=DEFAULT_EMBEDDING_MODEL, *, device="auto",
                  dtype="float16", batch_size=4, max_length=32768, cache_dir="data/local/hub",
                  progress=None):
    """Build an index of eligible passages and return its scene count and path."""
    progress = progress or (lambda message: None)
    progress("Loading dataset and source files")
    datasets = SceneDataset.load_all(dataset)
    total = sum(len(split) for split in datasets.values())
    completed = 0
    scenes = {}
    for dataset in datasets.values():
        for index in range(len(dataset)):
            progress(f"Reconstructing context [{completed}/{total} samples]")
            pool = dataset.context_pool(index)
            for block in (*pool.blocks, *pool.references):
                source = authored_scene(block)
                scenes[scene_id(source)] = source
            completed += 1
    progress(f"Loading {model} ({len(scenes)} authored scenes)")
    embedder = TransformersEmbedder.from_pretrained(
        model, device=device, dtype=dtype, batch_size=batch_size,
        max_length=max_length, cache_dir=cache_dir)
    with SceneDatabase.create(output, scenes.values(), embedder, progress=progress):
        pass
    return {"scenes": len(scenes), "output": str(output)}
