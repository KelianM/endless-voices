"""Prepare a scene dataset from cached sources and one configuration file."""

import argparse
import json
from pathlib import Path

from endless_voices.context import TokenizerCounter

from .builder import DatasetBuilder
from .source import GameCorpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "validation", "test"], default=["train"]
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    builder = DatasetBuilder(
        GameCorpus(args.source, json.loads(args.inventory.read_text())),
        json.loads(args.config.read_text()),
        TokenizerCounter(tokenizer),
    )
    print(json.dumps(builder.build(args.output, args.splits)))


if __name__ == "__main__":
    main()
