"""Fetch the pinned Endless Sky text sources and verify their committed SHA-256 hashes."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data/overview/source-statistics.json"
UPSTREAM = "https://github.com/endless-sky/endless-sky.git"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify(root, manifest):
    revision = git(root, "rev-parse", "HEAD")
    if revision != manifest["revision"]:
        raise ValueError(f"Expected revision {manifest['revision']}, found {revision}")
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    actual = {str(path.relative_to(root)) for path in (root / "data").rglob("*.txt")}
    if actual != set(expected):
        raise ValueError("Source file inventory differs from the committed manifest")
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
    if git(root, "status", "--porcelain", "--", "data", "license.txt", "copyright", "credits.txt"):
        raise ValueError("Source checkout has local modifications")
    for name in ("license.txt", "copyright", "credits.txt"):
        if not (root / name).is_file():
            raise ValueError(f"Missing upstream attribution file: {name}")


def fetch(destination, manifest):
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Manifest revision must be a full Git commit SHA")
    if destination.exists():
        verify(destination, manifest)
        return  # Reuse a verified checkout without requiring network access.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete verified checkout; failed fetches leave no partial destination.
    with tempfile.TemporaryDirectory(prefix=".endless-sky-fetch-", dir=destination.parent) as temp:
        checkout = Path(temp) / "checkout"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "config", "core.autocrlf", "false")
        git(checkout, "remote", "add", "origin", UPSTREAM)
        git(checkout, "sparse-checkout", "init", "--cone")
        git(checkout, "sparse-checkout", "set", "data")
        git(checkout, "fetch", "--quiet", "--depth=1", "--filter=blob:none",
            "origin", manifest["revision"])
        git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        verify(checkout, manifest)
        checkout.rename(destination)


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path,
        default=REPO_ROOT / "data/local" / f"endless-sky-{manifest['revision'][:12]}",
        help="Checkout location; defaults to the gitignored data/local directory.",
    )
    args = parser.parse_args()
    try:
        fetch(args.destination.resolve(), manifest)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Fetch failed: {error}\nExisting files were not overwritten.\n")
    print(f"Verified {len(manifest['files'])} source files at {args.destination.resolve()}")


if __name__ == "__main__":
    main()
