"""Infer mission prerequisites and the phases that can establish them."""

from .source import walk


def mission_dependencies(corpus):
    """Return required conditions and alternative mission phases that establish them."""
    events = {}
    for event, schedulers in corpus.event_schedulers.items():
        if ("event: " + event not in corpus.condition_writers
                and all(kind == "mission" and phase in {"offer", "complete"}
                        for kind, _, phase in schedulers)):
            events[event] = sorted((name, phase) for _, name, phase in schedulers)
    dependencies, present, absent = {}, {}, {}
    for name, mission in corpus.missions.items():
        required = {}
        present[name], absent[name] = set(), set()
        for child in mission.node["children"]:
            if child["tokens"] != ["to", "offer"]:
                continue
            for term, ancestors in walk(child["children"]):
                if any(p["tokens"] == ["or"] for p in ancestors):
                    continue
                parts = term["tokens"]
                if len(parts) == 2 and parts[0] == "not":
                    absent[name].add(parts[1])
                if len(parts) != 2 or parts[0] != "has":
                    continue
                flag = parts[1]
                present[name].add(flag)
                if flag.endswith(": done") and flag[:-6] in corpus.missions:
                    required[flag] = [(flag[:-6], "complete")]
                elif flag.startswith("event: ") and flag[7:] in events:
                    choices = events[flag[7:]]
                    if all(parent != name for parent, _ in choices):
                        required[flag] = choices
        dependencies[name] = required
    # A flag required absent earlier needs a witnessed transition before it can be present.
    changed = True
    while changed:
        changed = False
        for name, required in dependencies.items():
            pending = [p for choices in required.values() for p, _ in choices]
            ancestors = set()
            while pending:
                parent = pending.pop()
                if parent in ancestors or parent == name:
                    continue
                ancestors.add(parent)
                pending.extend(p for choices in dependencies[parent].values()
                               for p, _ in choices)
            earlier_absent = set().union(*(absent[p] for p in ancestors))
            for flag in sorted((present[name] & earlier_absent) - required.keys()):
                triggers = corpus.condition_triggers.get(flag, set())
                if triggers and all(kind == "mission" and parent != name
                                    and phase in {"offer", "complete"}
                                    for kind, parent, phase in triggers):
                    required[flag] = sorted((parent, phase) for _, parent, phase in triggers)
                    changed = True
    return dependencies


def mission_graph(dependencies):
    """Return every possible prerequisite link, without choosing a playthrough."""
    return {name: sorted({parent for choices in required.values() for parent, _ in choices})
            for name, required in dependencies.items()}
