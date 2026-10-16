"""Keep source missions and known conversation variants within one dataset split."""

def mission_key(metadata):
    """Identify the primary source owner without grouping supplementary lore."""
    source = metadata['sources'][0]
    return source['revision'] + ':' + source['source_group']


def check_mission_splits(records):
    """Reject primary source owners assigned to more than one split."""
    seen = {}
    for record in records:
        meta = record['metadata']
        key = mission_key(meta)
        if key in seen and seen[key] != meta['split']:
            raise ValueError(f'Mission {key!r} crosses splits')
        seen[key] = meta['split']
    return seen


def mission_chains(graph):
    """Return connected prerequisite groups, including unselected connecting missions."""
    neighbors = {name: set(parents) for name, parents in graph.items()}
    for name, parents in graph.items():
        for parent in parents:
            neighbors.setdefault(parent, set()).add(name)
    remaining = set(neighbors)
    while remaining:
        pending, chain = [min(remaining)], set()
        while pending:
            name = pending.pop()
            if name not in chain:
                chain.add(name)
                pending.extend(neighbors[name] - chain)
        remaining -= chain
        yield chain


def check_chain_splits(assignments, graph):
    """Reject prerequisite-connected missions assigned to different splits."""
    for chain in mission_chains(graph):
        owners = {assignments[name] for name in chain if name in assignments}
        if len(owners) > 1:
            raise ValueError(f"Mission chain crosses splits: {sorted(chain & assignments.keys())}")
