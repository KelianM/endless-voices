"""Resolve player and mission substitutions without inventing game state."""

import re

PLAYER_MARKERS = {'<first>', '<last>', '<ship>'}


def player_values(config):
    """Validate player-only substitutions, rejecting global mission overrides."""
    values = config['values']
    if set(values) != PLAYER_MARKERS or any(
        not isinstance(v, str) or not v for v in values.values()
    ):
        raise ValueError('Provide only nonempty <first>, <last> and <ship> player values')
    return dict(values)


def format_list(values):
    """Format a game-style list of names."""
    if len(values) < 2:
        return ''.join(values)
    return ', '.join(values[:-1]) + ' and ' + values[-1]


def mission_values(node, planet_systems, player):
    """Return provable static mission substitutions and player values."""
    values = dict(player)
    children = node['children']

    def literal(key):
        entries = [c for c in children if c['tokens'][:1] == [key]]
        if len(entries) == 1 and len(entries[0]['tokens']) == 2 and not entries[0]['children']:
            return entries[0]['tokens'][1]
        return None

    origin = literal('source')
    destination = literal('destination')
    # No destination defaults to the origin, but a destination filter needs game state.
    if destination is None and not any(c['tokens'][:1] == ['destination'] for c in children):
        destination = origin
    if origin:
        values['<origin>'] = origin
    if destination:
        values['<planet>'] = destination
        system = planet_systems.get(destination)
        if system:
            values['<system>'] = system
            values['<destination>'] = f'{destination} in the {system} system'
    for key, marker in [('waypoint', '<waypoints>'), ('mark', '<marks>')]:
        entries = [c for c in children if c['tokens'][:1] == [key]]
        if entries and all(len(c['tokens']) == 2 and not c['children'] for c in entries):
            values[marker] = format_list(sorted({c['tokens'][1] for c in entries}))
    return values


def render(text, values):
    """Substitute known markers once and return unresolved markers separately."""
    result = re.sub(r'<[^>]+>', lambda m: values.get(m[0], m[0]), text)
    return result, sorted(set(re.findall(r'<[^>]+>', result)))
