"""Prepare a scene dataset from cached sources and one configuration file."""

import argparse
import json
from pathlib import Path

from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.gameplay.source import GameCorpus


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="endless-voices prepare-dataset", description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "validation", "test"], default=["train"]
    )
    args = parser.parse_args(argv)
    builder = DatasetBuilder(
        GameCorpus(args.source, json.loads(args.inventory.read_text())),
        json.loads(args.config.read_text()),
    )
    print(json.dumps(builder.build(args.output, args.splits)))

