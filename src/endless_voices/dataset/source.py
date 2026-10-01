"""Parse game source into attributed nodes without rewriting authored text."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r'`([^`]*)`|"([^"]*)"|([^\s]+)')


def tokens(line):
    """Return game tokens, excluding comments outside quoted strings."""
    result = []
    for match in TOKEN.finditer(line):
        if match[3] and match[3].startswith("#"):
            break
        result.append(next(v for v in match.groups() if v is not None))
    return result


def tree(text):
    """Return indentation nodes with original physical line numbers."""
    roots, stack = [], []
    for line, raw in enumerate(text.splitlines(), 1):
        parts = tokens(raw.lstrip())
        if not parts:
            continue
        depth = len(raw) - len(raw.lstrip())
        node = {"line": line, "tokens": parts, "raw": raw, "children": []}
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1]["children"] if stack else roots).append(node)
        stack.append((depth, node))
    return roots


def walk(nodes):
    """Yield each node with its ancestors."""

    def visit(items, ancestors):
        for node in items:
            yield node, ancestors
            yield from visit(node["children"], (*ancestors, node))

    return visit(nodes, ())


@dataclass(frozen=True)
class Mission:
    name: str
    path: str
    sha256: str
    node: dict

    @property
    def conversations(self):
        return [
            n for n, _ in walk([self.node]) if n["tokens"][:1] == ["conversation"] and n["children"]
        ]


class GameCorpus:
    """Load hash-verified source missions once for dataset preparation."""

    def __init__(self, root, inventory):
        self.root = Path(root)
        self.revision = inventory["revision"]
        self.missions = {}
        self.events = {}
        self.planet_systems = {}
        self.descriptions = {}
        self.files = {}
        for entry in inventory["files"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source path must be relative")
            raw = (self.root / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source hash differs: {relative}")
            self.files[str(relative)] = entry["sha256"]
            for node in tree(raw.decode("utf-8")):
                kind, *names = node["tokens"]
                if kind == "mission" and names:
                    name = names[0]
                    if name in self.missions:
                        raise ValueError(f"Duplicate mission: {name}")
                    self.missions[name] = Mission(name, str(relative), entry["sha256"], node)
                if kind == "event" and names:
                    self.events[names[0]] = node
                if kind == "system" and names:
                    for child in node["children"]:
                        if child["tokens"][:1] == ["object"]:
                            for obj, _ in walk([child]):
                                if obj["tokens"][:1] == ["object"] and len(obj["tokens"]) == 2:
                                    self.planet_systems[obj["tokens"][1]] = names[0]
                if kind == "planet" and names:
                    self.descriptions[names[0]] = [
                        {"path": str(relative), "line": n["line"], "text": n["tokens"][1]}
                        for n in node["children"]
                        if n["tokens"][:1] == ["description"]
                        and len(n["tokens"]) == 2
                        and not n["children"]
                    ]
