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
