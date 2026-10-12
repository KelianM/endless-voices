"""Generate one continuation per curated sample and save reproducible run records."""

import argparse
import math
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="endless-voices generate",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/generate.toml", help="Model TOML config")
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset split manifest")
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Split to generate",
    )
    parser.add_argument("--sample-ids", type=Path, help="JSON list; order is preserved")
    parser.add_argument("--limit", type=int, help="First N selected samples")
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--adapter", help="Existing local LoRA directory or Hub ID")
    parser.add_argument("--adapter-revision", help="Full Hub commit for the adapter")
    parser.add_argument("--cache-dir", default="data/local/hub", help="Model download cache")
    parser.add_argument("--offline", action="store_true", help="Require cached model files")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Execution device"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Weight precision",
    )
    parser.add_argument(
        "--max-context-tokens", type=int, default=4096, help="Total prompt and output budget"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Output cap; reaching it is a failure"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Zero for greedy decoding; positive for sampling",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus probability for sampling")
    parser.add_argument(
        "--seed", type=int, default=42, help="Run seed used to derive stable per-sample seeds"
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_new_tokens < args.max_context_tokens:
        parser.error("require 0 < max-new-tokens < max-context-tokens")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("require finite temperature >= 0 and 0 < top-p <= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.adapter_revision and not args.adapter:
        parser.error("adapter-revision requires adapter")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    from endless_voices.generate import run_generation
    try:
        code = run_generation(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        code = 1
    raise SystemExit(code)

