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


def mission_values(node, planet_systems, player, *, corpus=None, seed="context", world=()):
    """Return mission substitutions, sampling supported locations reproducibly."""
    values = dict(player)
    children = node['children']

    def literal(key):
        entries = [c for c in children if c['tokens'][:1] == [key]]
        if len(entries) == 1 and len(entries[0]['tokens']) == 2 and not entries[0]['children']:
            return entries[0]['tokens'][1]
        return None

    origin = literal('source')
    destination = literal('destination')
    if corpus is not None:
        from .locations import Locations

        locations = Locations(corpus, world)
        for key in ('source', 'destination'):
            entries = [c for c in children if c['tokens'] == [key]]
            in_flight = any(c['tokens'][0] in {'boarding', 'assisting', 'entering', 'transition'}
                            for c in children)
            if len(entries) == 1 and not in_flight and (key == 'source' or origin):
                selected = locations.pick(entries[0]['children'], seed, node['tokens'][1], key,
                                          planet_systems.get(origin))
                if key == 'source':
                    origin = selected
                else:
                    destination = selected
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
    if corpus is not None:
        for npc in (c for c in children if c['tokens'][:1] == ['npc']):
            ships = [c for c in npc['children'] if c['tokens'][:1] == ['ship']]
            if not ships:
                if any(c['tokens'][:1] == ['fleet'] for c in npc['children']):
                    values.pop('<npc>', None)
                    values.pop('<npc model>', None)
                continue
            ship = ships[0]
            parts = ship['tokens']
            values.pop('<npc>', None)
            values.pop('<npc model>', None)
            definition = corpus.ships.get(parts[1]) if len(parts) > 1 else None
            if definition is not None and not ship['children']:
                model = definition['tokens'][1]
                base = corpus.ships.get(model, definition)
                display = next((c['tokens'][1] for c in [*definition['children'], *base['children']]
                                if c['tokens'][:1] == ['display name']), model)
                values['<npc model>'] = display
                if len(parts) == 3:
                    name, unresolved = render(parts[2], values)
                    if not unresolved and '${' not in name:
                        values['<npc>'] = name
    return values


def render(text, values):
    """Substitute known markers once and return unresolved markers separately."""
    result = re.sub(r'<[^>]+>', lambda m: values.get(m[0], m[0]), text)
    return result, sorted(set(re.findall(r'<[^>]+>', result)))
