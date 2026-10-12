"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

import argparse
import json
from pathlib import Path

from endless_voices.dataset.index_context import index_context
from endless_voices.retrieval import DEFAULT_EMBEDDING_MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="endless-voices index-context", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/local/hub"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(index_context(**vars(args))))
