"""Fetch the pinned Endless Sky text sources and verify their committed SHA-256 hashes."""

import argparse
import json
import subprocess
from pathlib import Path

from endless_voices import fetch_sources as sources


def main(argv=None):
    parser = argparse.ArgumentParser(prog="endless-voices fetch-sources", description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/sources/source-statistics.json"))
    parser.add_argument(
        "--destination", type=Path,
        default=None,
        help="Checkout location; defaults to the gitignored data/local directory.",
    )
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        destination = args.destination or Path("data/local") / (
            f"endless-sky-{manifest['revision'][:12]}")
        sources.fetch(destination.resolve(), manifest)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Fetch failed: {error}\nExisting files were not overwritten.\n")
    print(f"Verified {len(manifest['files'])} source files at {destination.resolve()}")

