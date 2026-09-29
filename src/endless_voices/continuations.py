"""Preserve complete authored continuations at existing dialogue boundaries."""

import hashlib
import json
from pathlib import Path

from endless_voices.context import substitute_variables


def continuation(conversation, flow, anchors, response_boundaries=()):
    """Return source paragraphs through a selected route, stopping before a player choice."""
    anchors = list(dict.fromkeys(anchors))
    if not anchors:
        raise ValueError("Continuation requires source anchors")
    nodes = {n['line']: n for n in conversation['children']}
    if any(line not in nodes for line in anchors):
        raise ValueError("Target anchor is not a conversation paragraph")
    pending = [(anchors[0], 0, [])]
    routes = []
    while pending:
        line, index, path = pending.pop()
        if line in path:
            raise ValueError("Continuation loops before a player boundary")
        node = nodes[line]
        if index < len(anchors) and line == anchors[index]:
            index += 1
        if line in flow['choices'] or line in response_boundaries:
            if index == len(anchors):
                routes.append((path, line))
            continue
        path = [*path, line]
        destinations = flow['links'].get(str(line), [])
        if not destinations:
            if index == len(anchors):
                routes.append((path, None))
        else:
            pending.extend((n, index, path) for n in destinations)
        if len(pending) + len(routes) > 256:
            raise ValueError("Too many continuation branches")
    variants = {}
    for path, stop in routes:
        paragraphs = []
        for line in path:
            node = nodes[line]
            if node['raw'].lstrip().startswith(('`', '"')):
                if len(node['tokens']) != 1:
                    raise ValueError("Expected one authored text token")
                if node['children'] and any(
                    n['tokens'][:1] not in [['goto'], ['accept'], ['decline'], ['defer'],
                                          ['launch'], ['flee'], ['die'], ['explode']]
                    for n in node['children']
                ):
                    raise ValueError("Conditional paragraph requires an explicit route")
                paragraphs.append({'line': line, 'text': node['tokens'][0]})
        key = tuple(p['line'] for p in paragraphs)
        entry = variants.setdefault(key, {'paragraphs': paragraphs, 'stops': []})
        if stop not in entry['stops']:
            entry['stops'].append(stop)
    if len(variants) != 1:
        raise ValueError("Continuation route is missing or ambiguous before the next player choice")
    entry = next(iter(variants.values()))
    paragraphs = entry['paragraphs']
    if not paragraphs:
        raise ValueError("Continuation has no authored text")
    return {'paragraphs': paragraphs, 'stop_before_lines': entry['stops'],
            'text': '\n\n'.join(p['text'] for p in paragraphs)}


def load_targets(path):
    """Verify and return a source-continuation bundle keyed by sample ID."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    raw = (path / 'targets.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['targets_sha256']:
        raise ValueError("Source continuation hash mismatch")
    rows = json.loads(raw)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate continuation sample ID")
    if [r['sample_id'] for r in rows] != manifest['sample_ids']:
        raise ValueError("Continuation inventory differs")
    return {r['sample_id']: r for r in rows}


def target_text(row, messages, variables):
    """Render game variables and reject target paragraphs already exposed in the input."""
    paragraphs = [substitute_variables(p['text'], variables) for p in row['paragraphs']]
    if any(p.strip() and p.strip() in m['content'] for p in paragraphs for m in messages):
        raise ValueError("Source continuation paragraph appears in input context")
    return '\n\n'.join(paragraphs)
