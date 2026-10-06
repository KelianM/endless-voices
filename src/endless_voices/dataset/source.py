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


def clears_on_exit(conversation, variable):
    """Return whether every structural exit clears a dialogue-local flag."""
    from .dialogue import ENDPOINTS

    nodes = conversation["children"]
    labels = {n["tokens"][1]: i for i, n in enumerate(nodes)
              if n["tokens"][:1] == ["label"] and len(n["tokens"]) == 2}

    def destination(name, endpoint=False):
        if endpoint and name in ENDPOINTS:
            return len(nodes)
        return labels[name]

    def following(node, pc):
        for child in node["children"]:
            parts = child["tokens"]
            if parts[:1] == ["goto"] and len(parts) == 2:
                return destination(parts[1])
            if len(parts) == 1 and parts[0] in ENDPOINTS:
                return len(nodes)
            if parts != ["to", "display"]:
                raise ValueError("Unknown passage control")
        return pc + 1

    pending, seen, has_exit = [(0, False)], set(), False
    try:
        while pending:
            pc, cleared = pending.pop()
            if (pc, cleared) in seen:
                continue
            seen.add((pc, cleared))
            if pc >= len(nodes):
                if not cleared:
                    return False
                has_exit = True
                continue
            node = nodes[pc]
            parts = node["tokens"]
            successors = [pc + 1]
            if parts == ["action"]:
                for operation in node["children"]:
                    if operation["tokens"] == ["clear", variable]:
                        cleared = True
                    elif operation["tokens"] == ["set", variable]:
                        cleared = False
            elif parts[:1] == ["branch"] and len(parts) in {2, 3}:
                successors = [destination(parts[1], endpoint=True),
                              destination(parts[2], endpoint=True) if len(parts) == 3 else pc + 1]
            elif parts == ["choice"]:
                successors = [following(option, pc) for option in node["children"]]
                if all(any(c["tokens"] == ["to", "display"] for c in option["children"])
                       for option in node["children"]):
                    successors.append(pc + 1)
            elif parts[:1] == ["goto"] and len(parts) == 2:
                successors = [destination(parts[1])]
            elif len(parts) == 1 and parts[0] in ENDPOINTS:
                successors = [len(nodes)]
            elif parts[:1] in (["label"], ["scene"]) and len(parts) == 2:
                pass
            elif len(parts) == 1 and node["raw"].lstrip().startswith(('"', "`")):
                successors = [following(node, pc)]
                if any(c["tokens"] == ["to", "display"] for c in node["children"]):
                    successors.append(pc + 1)
            else:
                return False
            pending.extend((successor, cleared) for successor in successors)
    except (KeyError, ValueError):
        return False
    return has_exit


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
        self.conversations = {}
        self.phrases = {}
        self.condition_writers = {}
        self.planet_systems = {}
        self.systems = {}
        self.planets = {}
        self.descriptions = {}
        self.reference_lore = []
        self.files = {}
        flag_conversations, persistent_conditions = {}, set()
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
                owner = (kind, names[0] if names else str(relative))
                for operation, ancestors in walk([node]):
                    parts = operation["tokens"]
                    variable = None
                    if len(parts) == 2 and parts[0] in {"set", "clear"}:
                        variable = parts[1]
                    elif len(parts) >= 2 and parts[1] in {
                        "=", "+=", "-=", "*=", "/=", "%=", "<?=", ">?=", "++", "--",
                    }:
                        variable = parts[0]
                    if variable is not None:
                        self.condition_writers.setdefault(variable, set()).add(owner)
                        conversation = next((n for n in reversed(ancestors)
                                             if n["tokens"][:1] == ["conversation"]), None)
                        if conversation is None or parts[0] not in {"set", "clear"}:
                            persistent_conditions.add(variable)
                        else:
                            flag_conversations.setdefault(variable, {})[
                                (str(relative), conversation["line"])] = conversation
                if kind in {"conversation", "phrase"} and names:
                    registry = self.conversations if kind == "conversation" else self.phrases
                    if kind == "conversation" and names[0] in registry:
                        raise ValueError(f"Duplicate {kind}: {names[0]}")
                    definition = {
                        **node, "source_path": str(relative),
                        "source_sha256": entry["sha256"],
                    }
                    if kind == "phrase":
                        registry.setdefault(names[0], []).append(definition)
                    else:
                        registry[names[0]] = definition
                if kind == "mission" and names:
                    name = names[0]
                    if name in self.missions:
                        raise ValueError(f"Duplicate mission: {name}")
                    self.missions[name] = Mission(name, str(relative), entry["sha256"], node)
                if kind == "event" and names:
                    self.events[names[0]] = node
                if kind == "system" and names:
                    self.systems[names[0]] = node
                    for child in node["children"]:
                        if child["tokens"][:1] == ["object"]:
                            for obj, _ in walk([child]):
                                if obj["tokens"][:1] == ["object"] and len(obj["tokens"]) == 2:
                                    self.planet_systems[obj["tokens"][1]] = names[0]
                if kind in {"ship", "outfit", "government"} and names:
                    passages = [{"line": n["line"], "text": n["tokens"][1], "role": "passage"}
                                for n in node["children"]
                                if n["tokens"][:1] == ["description"] and len(n["tokens"]) == 2
                                and not n["children"]]
                    if passages:
                        self.reference_lore.append({
                            "mission": None, "heading": names[0], "path": str(relative),
                            "conversation": node["line"], "passages": passages,
                            "reference": True, "state": {}, "source_kind": kind,
                        })
                if kind == "planet" and names:
                    self.planets[names[0]] = node
                    self.descriptions[names[0]] = [
                        {"path": str(relative), "line": n["line"], "text": n["tokens"][1]}
                        for n in node["children"]
                        if n["tokens"][:1] == ["description"]
                        and len(n["tokens"]) == 2
                        and not n["children"]
                    ]
        self.scratch_conditions = {
            name for name, conversations in flag_conversations.items()
            if name not in persistent_conditions
            and all(clears_on_exit(node, name) for node in conversations.values())
        }

    def named_conversation_users(self):
        """Return mission names grouped by referenced conversation definition."""
        users = {}
        for mission in self.missions.values():
            for node, _ in walk([mission.node]):
                parts = node["tokens"]
                if parts[:1] == ["conversation"] and len(parts) == 2 and not node["children"]:
                    users.setdefault(parts[1], set()).add(mission.name)
        return users

    def conversation_owners(self, assignments):
        """Validate shared conversation splits and return each definition's owner."""
        owners = {}
        for name, missions in self.named_conversation_users().items():
            splits = {assignments[m] for m in missions if m in assignments}
            if len(splits) > 1:
                raise ValueError(f"Named conversation {name!r} crosses mission splits")
            if splits:
                owners[name] = splits.pop()
        return owners

    def resolve_conversation(self, node, mission):
        """Return inline or named dialogue with its definition's source location."""
        from .state import UnsupportedOperation

        if node["tokens"] == ["conversation"] and node["children"]:
            return {**node, "source_path": mission.path, "source_sha256": mission.sha256}
        if len(node["tokens"]) == 2 and not node["children"]:
            name = node["tokens"][1]
            if name not in self.conversations:
                raise UnsupportedOperation(f"Undefined conversation: {name}")
            return {**self.conversations[name], "reference_line": node["line"]}
        raise UnsupportedOperation(f"Unsupported conversation form at line {node['line']}")

    def dialog_passages(self, node, mission):
        """Return literal dialog paragraphs with their authored source locations."""
        from .state import UnsupportedOperation

        parts = node["tokens"]
        source_path, source_sha256 = mission.path, mission.sha256
        paragraphs = []
        if parts[:2] == ["dialog", "phrase"] and len(parts) == 3 and not node["children"]:
            definitions = self.phrases.get(parts[2])
            if definitions is None:
                raise UnsupportedOperation(f"Undefined dialog phrase: {parts[2]}")
            if len(definitions) != 1:
                raise UnsupportedOperation(f"Nonliteral dialog phrase: {parts[2]}")
            phrase = definitions[0]
            sections = phrase["children"]
            if (len(sections) != 1 or sections[0]["tokens"] != ["word"]
                    or len(sections[0]["children"]) != 1):
                raise UnsupportedOperation(f"Nonliteral dialog phrase: {parts[2]}")
            child = sections[0]["children"][0]
            if len(child["tokens"]) != 1 or child["children"]:
                raise UnsupportedOperation(f"Nonliteral dialog phrase: {parts[2]}")
            source_path, source_sha256 = phrase["source_path"], phrase["source_sha256"]
            paragraphs.append((child["line"], child["tokens"][0]))
        elif parts[:1] == ["dialog"] and len(parts) in {1, 2}:
            if len(parts) == 2:
                paragraphs.append((node["line"], parts[1]))
            for child in node["children"]:
                if (len(child["tokens"]) != 1 or child["children"]
                        or not child["raw"].lstrip().startswith(('"', "`"))):
                    raise UnsupportedOperation(
                        f"Nonliteral dialog paragraph at line {child['line']}"
                    )
                paragraphs.append((child["line"], child["tokens"][0]))
        else:
            raise UnsupportedOperation(f"Unsupported dialog form at line {node['line']}")
        return [{"line": line, "text": text, "path": source_path, "sha256": source_sha256}
                for line, text in paragraphs]
