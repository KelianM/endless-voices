"""Keep source missions and known conversation variants within one dataset split."""

from copy import deepcopy

PRIORITY = {'train': 0, 'validation': 1, 'test': 2}


def mission_key(metadata):
    """Identify the primary source owner without grouping supplementary lore."""
    source = metadata['sources'][0]
    return source['revision'] + ':' + source['source_group']


def assign_mission_splits(records):
    """Promote connected source groups to their strongest existing held-out split."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def keys(record):
        meta = record['metadata']
        result = [('mission', mission_key(meta)), ('conversation', meta['conversation_id'])]
        if meta.get('scenario_group'):
            result.append(('scenario', meta['scenario_group']))
        return result

    for record in records:
        group = keys(record)
        for key in group[1:]:
            parent[find(key)] = find(group[0])
    splits = {}
    for record in records:
        key = find(keys(record)[0])
        split = record['metadata']['split']
        splits[key] = max(splits.get(key, 'train'), split, key=PRIORITY.get)
    result = deepcopy(records)
    for record in result:
        record['metadata']['split'] = splits[find(keys(record)[0])]
    return result


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
