"""Validate conversation samples and split manifests offline."""

import argparse
import json
from pathlib import Path

from endless_voices.contracts import validate_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="endless-voices validate", description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="optional local tokenizer directory only")
    parser.add_argument(
        "--max-length", type=int, help="complete-sample token limit for all splits; no truncation"
    )
    args = parser.parse_args(argv)
    if (args.tokenizer is None) != (args.max_length is None):
        parser.error("--tokenizer and --max-length must be supplied together")
    try:
        tokenizer = None
        if args.tokenizer is not None:
            if not args.tokenizer.is_dir():
                raise ValueError("--tokenizer must be an existing local directory")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
            )
        counts = validate_manifest(args.manifest, tokenizer=tokenizer, max_length=args.max_length)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(counts, indent=2, sort_keys=True))

