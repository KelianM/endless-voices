"""Prepare source conversation catalogs and materialize reviewed annotations offline."""

import argparse
import json
import re
from pathlib import Path

from fetch_sources import verify
from inventory_sources import tokens

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/overview/source-statistics.json"
DEFAULT_SOURCES = ROOT / "data/local/endless-sky-7140eb2a29ce"


def tree(text):
    """Return source nodes with physical line numbers and indentation children."""
    roots, stack = [], []
    for number, raw in enumerate(text.splitlines(), 1):
        parts = tokens(raw.lstrip())
        if not parts:
            continue
        depth = len(raw) - len(raw.lstrip())
        node = {"line": number, "tokens": parts, "raw": raw, "children": []}
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1]["children"] if stack else roots).append(node)
        stack.append((depth, node))
    return roots


def walk(nodes, ancestors=()):
    for node in nodes:
        yield node, ancestors
        yield from walk(node["children"], (*ancestors, node))


def speech(raw):
    """Return proposed quoted speech spans; attribution still requires annotation."""
    content = raw.strip()
    if not content.startswith("`") or not content.endswith("`"):
        return []
    start = raw.index("`") + 1
    stop = raw.rindex("`")
    quotes = [match.start() for match in re.finditer('"', raw[start:stop])]
    spans = []
    for index in range(0, len(quotes), 2):
        left = start + quotes[index] + 1
        right = start + quotes[index + 1] if index + 1 < len(quotes) else stop
        if raw[left:right].strip():
            spans.append({"start": left, "end": right, "text": raw[left:right]})
    return spans


def flow_graph(conversation):
    """Return possible dialogue links, leaving game conditions unevaluated."""
    nodes = conversation["children"]
    labels = {n["tokens"][1]: n["line"] for n in nodes if n["tokens"][0] == "label"}
    graph = {}
    choices = []

    def destination(label):
        if label in labels:
            return labels[label]
        if label in {"accept", "decline", "defer", "launch", "flee", "die", "explode"}:
            return None
        return labels.get(label)

    def links(node, following):
        for child in node["children"]:
            command = child["tokens"]
            if command[0] == "goto":
                return [destination(command[1])]
            if command[0] in {"accept", "decline", "defer", "launch", "flee", "die", "explode"}:
                return []
        return [following]

    for index, node in enumerate(nodes):
        following = nodes[index + 1]["line"] if index + 1 < len(nodes) else None
        command = node["tokens"]
        if command[0] == "choice":
            choices.append(node["line"])
            options = [child for child in node["children"]
                       if child["raw"].lstrip().startswith(("`", '"'))]
            graph[str(node["line"])] = [child["line"] for child in options]
            for child in options:
                graph[str(child["line"])] = [n for n in links(child, following) if n is not None]
        elif command[0] == "branch":
            targets = [destination(label) for label in command[1:]]
            if len(command) < 3:
                targets.append(following)
            graph[str(node["line"])] = [n for n in targets if n is not None]
        elif command[0] == "goto":
            target = destination(command[1])
            graph[str(node["line"])] = [] if target is None else [target]
        elif command[0] in {"accept", "decline", "defer", "launch", "flee", "die", "explode"}:
            graph[str(node["line"])] = []
        else:
            graph[str(node["line"])] = [n for n in links(node, following) if n is not None]
    return {"links": graph, "choices": choices}


def reachable(flow, start, end, *, through_choices=False, blocked=()):
    """Check a possible source path; this does not prove its conditions hold in a save state."""
    pending, visited = [start], set()
    while pending:
        current = pending.pop()
        if current == end:
            return True
        if current in visited or (current in blocked and current != start):
            continue
        visited.add(current)
        if current in flow["choices"] and not through_choices:
            continue
        pending.extend(flow["links"].get(str(current), []))
    return False


def catalog(source_root, manifest):
    """Inventory actual inline/named conversation blocks without executing game state."""
    verify(source_root, manifest)
    records = []
    for entry in manifest["files"]:
        path = entry["path"]
        if not path.startswith(("data/human/", "data/hai/", "data/quarg/")):
            continue
        raw = (source_root / path).read_text(encoding="utf-8")
        lines = raw.splitlines()
        for node, ancestors in walk(tree(raw)):
            if node["tokens"][0] != "conversation" or not node["children"]:
                continue
            descendants = list(walk(node["children"]))
            last = max(child["line"] for child, _ in descendants)
            owner = ancestors[0]["tokens"] if ancestors else node["tokens"]
            slug = re.sub(r"[^a-z0-9]+", "-", path.removeprefix("data/")[:-4].lower())
            prose = []
            for child, parents in descendants:
                if child["raw"].lstrip().startswith(("`", '"')):
                    parent = parents[-1]["tokens"][0] if parents else "conversation"
                    if parent in {"conversation", "choice"}:
                        prose.append({
                            "line": child["line"], "kind": "choice" if parent == "choice"
                            else "paragraph", "raw": child["raw"].strip(),
                            "speech": speech(child["raw"]),
                        })
            records.append({
                "id": f"{slug}-l{node['line']}", "path": path,
                "lines": [node["line"], last], "owner": owner,
                "event": [item["tokens"] for item in ancestors[1:]],
                "source_sha256": entry["sha256"], "prose": prose, "flow": flow_graph(node),
                "source": "\n".join(f"{n}: {lines[n - 1]}"
                                      for n in range(node["line"], last + 1)),
            })
    return {"revision": manifest["revision"], "conversations": records}


def write_catalog(source_root, output):
    """Write an unreviewed local catalog and per-source reading sheets."""
    if output.exists():
        raise ValueError(f"{output}: already exists")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    result = catalog(source_root, manifest)
    output.mkdir(parents=True)
    (output / "catalog.json").write_text(json.dumps(result, indent=2) + "\n")
    paths = sorted({row["path"] for row in result["conversations"]})
    for path in paths:
        rows = [row for row in result["conversations"] if row["path"] == path]
        text = []
        for row in rows:
            text.extend([f"## {row['id']} | {' / '.join(row['owner'])} | {row['event']}",
                         row["source"], "\nProposed speech (unreviewed):"])
            for paragraph in row["prose"]:
                text.append(f"{paragraph['line']} {paragraph['kind']}: " +
                            json.dumps([s["text"] for s in paragraph["speech"]],
                                       ensure_ascii=False))
        name = Path(path).stem.replace(" ", "-") + ".txt"
        (output / name).write_text("\n".join(text) + "\n", encoding="utf-8")
    return len(result["conversations"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog"])
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = write_catalog(args.sources, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Prepared {count} unreviewed conversation blocks at {args.output}")


if __name__ == "__main__":
    main()
