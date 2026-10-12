"""Build a standalone HTML reader for curated samples and optional generated responses."""

import argparse
from pathlib import Path

from endless_voices.view_samples import build_viewer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="endless-voices view-samples", description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/dataset/manifest.json")
    )
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--run", type=Path, help="Optional generation run directory")
    parser.add_argument("--output", type=Path, required=True, help="New standalone HTML file")
    args = parser.parse_args(argv)
    try:
        count = build_viewer(args.manifest, args.output, args.split, args.run)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {count} samples to {args.output.resolve()}")
    print(f"Open in a browser: {args.output.resolve().as_uri()}")

