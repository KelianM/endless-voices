"""Browse prepared scenes, saved responses and live continuations."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from endless_voices.view_samples import build_viewer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="endless-voices view-samples", description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/dataset/manifest.json")
    )
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="Saved run or benchmark model folder; repeat for comparison")
    parser.add_argument("--output", type=Path, help="New standalone HTML file")
    parser.add_argument("--serve", action="store_true", help="Enable live hosted conversations")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--budget", type=float,
                        help="Optional USD cap; requires known model pricing")
    parser.add_argument("--live-output", type=Path, help="Directory for live conversation logs")
    args = parser.parse_args(argv)
    if args.serve:
        from endless_voices.live_viewer import serve

        output = args.live_output or Path("outputs") / datetime.now().strftime("play-%Y%m%d-%H%M%S")
        serve(args.manifest, args.split, args.run, json.loads(args.config.read_text()),
              output, args.budget, args.port)
        return
    if args.output is None:
        parser.error("provide --output or --serve")
    try:
        count = build_viewer(args.manifest, args.output, args.split, args.run)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {count} samples to {args.output.resolve()}")
    print(f"Open in a browser: {args.output.resolve().as_uri()}")

