"""Dispatch supported commands to their argument parsers."""

import argparse
from importlib import import_module

COMMANDS = {
    "prepare-dataset": "prepare_dataset",
    "index-context": "index_context",
    "fetch-sources": "fetch_sources",
    "inventory-sources": "inventory_sources",
    "view-samples": "view_samples",
    "train": "train",
    "chat": "chat",
    "generate": "generate",
    "validate": "validate",
    "benchmark": "benchmark",
}


def main(argv=None):
    """Dispatch one command and return its exit status."""
    parser = argparse.ArgumentParser(
        prog="endless-voices", description="Prepare, train and evaluate Endless Voices.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return import_module(f"endless_voices.cli.{COMMANDS[args.command]}").main(args.arguments)
