"""Prepare a scene dataset from cached sources and one configuration file."""

import argparse
import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from endless_voices.dataset.builder import DatasetBuilder
from endless_voices.gameplay.source import GameCorpus


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="endless-voices prepare-dataset", description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Defaults to data/local/endless-sky-<inventory revision>.")
    parser.add_argument("--inventory", type=Path,
                        default=Path("data/sources/source-statistics.json"))
    parser.add_argument("--config", type=Path, default=Path("data/dataset/config.json"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset-resolved"))
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "validation", "test"], default=["train"]
    )
    args = parser.parse_args(argv)
    started = monotonic()
    status = ("Loading sources and mission dependencies", started)
    finished = Event()

    def report():
        label, since = status
        now = monotonic()
        print(f"{label} | {now - since:.0f}s in stage | {now - started:.0f}s elapsed",
              file=sys.stderr, flush=True)

    def progress(completed, total, name):
        nonlocal status
        status = (f"[{completed}/{total} missions complete] {name}", monotonic())
        report()

    def heartbeat():
        while not finished.wait(10):
            report()

    report()
    timer = Thread(target=heartbeat, daemon=True)
    timer.start()
    try:
        inventory = json.loads(args.inventory.read_text())
        source = args.source or Path("data/local") / f"endless-sky-{inventory['revision'][:12]}"
        builder = DatasetBuilder(
            GameCorpus(source, inventory), json.loads(args.config.read_text()))
        result = builder.build(args.output, args.splits, progress=progress)
    finally:
        finished.set()
        timer.join()
    print(f"Finished in {monotonic() - started:.0f}s", file=sys.stderr, flush=True)
    print(json.dumps(result))

