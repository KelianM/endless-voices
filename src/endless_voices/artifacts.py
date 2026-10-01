"""Read, hash and write experiment artifacts with explicit overwrite behavior."""

import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Create a JSON artifact, refusing to overwrite an existing file."""
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_progress(path, value):
    """Atomically replace a mutable progress record."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded(value) + b"\n")
    temporary.replace(path)
