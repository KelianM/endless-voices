"""Measure pinned Endless Sky source structure, not model-ready dialogue or speakers."""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "7140eb2a29ce4d2797933075c751791a892c7d4f"
TOKEN = re.compile(r'`([^`]*)`|"([^"]*)"|([^\s]+)')
CATEGORIES = ("conversation", "dialog", "description", "spaceport", "log", "phrase_news")


def tokens(line):
    """Tokenize one physical line; comments outside quoted strings end the line."""
    result = []
    for match in TOKEN.finditer(line):
        if match[3] and match[3].startswith("#"):
            break
        result.append(next(value for value in match.groups() if value is not None))
    return result


def measure(text):
    counts = Counter()
    stack = []
    nodes = []
    governments = set()
    for line in text.splitlines():
        stripped = line.lstrip()
        parts = tokens(stripped)
        if not parts:
            continue
        depth = len(line) - len(stripped)
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        parent = stack[-1] if stack else None
        if parent:
            parent["children"] += 1
        key = parts[0]
        node = {"key": key, "depth": depth, "children": 0}
        nodes.append(node)
        ancestors = [item["key"] for item in stack]
        if depth == 0:
            counts[f"root_{key}"] += 1
            if key == "government" and len(parts) > 1:
                governments.add(parts[1])
        category, prose = None, []
        # Display text is directly under conversation/choice, not action/branch/label.
        if (parent and parent["key"] in {"conversation", "choice"}
                and stripped.startswith(('"', '`'))):
            category, prose = "conversation", parts[:1]
        elif key in {"description", "spaceport", "log", "dialog"} and len(parts) > 1:
            if key == "log":
                # Optional category/header precede the final prose token.
                if parts[1] != "scene":
                    category, prose = key, parts[-1:]
            elif key != "dialog" or parts[1] != "phrase":
                category, prose = key, parts[1:]
        elif parent and parent["key"] == "dialog" and stripped.startswith(('"', '`')):
            category, prose = "dialog", parts[:1]
        elif parent and parent["key"] == "word" and (
            "phrase" in ancestors or "news" in ancestors
        ):
            category, prose = "phrase_news", parts[:1]
        if category:
            counts[f"{category}_words"] += sum(len(value.split()) for value in prose)
            counts[f"{category}_text_lines"] += 1
        stack.append(node)
    for node in nodes:
        if node["key"] == "conversation":
            counts["conversation_blocks" if node["children"] else "conversation_references"] += 1
    # Ignore unrelated root types to keep the output compact and stable.
    wanted = {f"root_{key}" for key in ("mission", "conversation", "government", "news", "phrase")}
    selected = {k: v for k, v in counts.items() if not k.startswith("root_") or k in wanted}
    return selected, governments


def inventory(root):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", "data"], text=True
    )
    if dirty.strip():
        raise ValueError("Upstream data must match its clean pinned checkout")
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD", "data"], text=True
    ).splitlines()
    expected = {path for path in tracked if path.endswith(".txt")}
    actual = {str(path.relative_to(root)) for path in (root / "data").rglob("*.txt")}
    if actual != expected:
        raise ValueError("Checkout must include all tracked data/**/*.txt files")
    groups = defaultdict(Counter)
    governments = set()
    files = []
    for path in sorted((root / "data").rglob("*.txt")):
        raw = path.read_bytes()
        relative = path.relative_to(root / "data")
        group = relative.parts[0] if len(relative.parts) > 1 else "(shared root)"
        counts, names = measure(raw.decode("utf-8"))
        counts.update(files=1, bytes=len(raw), lines=len(raw.decode("utf-8").splitlines()))
        groups[group].update(counts)
        governments.update(names)
        files.append({
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    totals = Counter()
    for counts in groups.values():
        totals.update(counts)
    return {
        "revision": revision,
        "method": "Structural lexical inventory v1; words are whitespace-separated, not tokens.",
        "content_directories": sorted(g for g in groups if not g.startswith(("_", "("))),
        "government_names": sorted(governments),
        "totals": dict(sorted(totals.items())),
        "groups": {g: dict(sorted(c.items())) for g, c in sorted(groups.items())},
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inventory(args.upstream), indent=2) + "\n")


if __name__ == "__main__":
    main()
