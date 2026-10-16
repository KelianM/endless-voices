"""Measure pinned Endless Sky source structure, not model-ready dialogue or speakers."""

import argparse
import json
from pathlib import Path

from endless_voices.inventory_sources import inventory


def main(argv=None):
    parser = argparse.ArgumentParser(prog="endless-voices inventory-sources", description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(json.dumps(inventory(args.upstream), indent=2) + "\n")

