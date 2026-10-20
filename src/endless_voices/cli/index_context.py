"""Index eligible scenes from a prepared dataset with a locally cached encoder."""

import argparse
import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from endless_voices.dataset.index_context import index_context
from endless_voices.retrieval import DEFAULT_EMBEDDING_MODEL


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="endless-voices index-context", description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float16",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/local/hub"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = monotonic()
    status = "Starting context index"
    last_report = 0
    finished = Event()

    def report():
        nonlocal last_report
        last_report = monotonic()
        print(f"{status} | {last_report - started:.0f}s elapsed", file=sys.stderr, flush=True)

    def progress(message):
        nonlocal status
        status = message
        if monotonic() - last_report >= 1:
            report()

    def heartbeat():
        while not finished.wait(10):
            report()

    report()
    timer = Thread(target=heartbeat, daemon=True)
    timer.start()
    try:
        result = index_context(**vars(args), progress=progress)
    finally:
        finished.set()
        timer.join()
    print(f"Finished in {monotonic() - started:.0f}s", file=sys.stderr, flush=True)
    print(json.dumps(result))
