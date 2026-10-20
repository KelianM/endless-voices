"""Select reproducible planets from supported game location filters."""

from collections import deque

from endless_voices.artifacts import fingerprint

from .state import UnsupportedOperation


def world_nodes(base, world, kind):
    """Return location definitions with interpreted field changes applied."""
    result = dict(base)
    for change in world:
        if change['tokens'][:1] != [kind] or len(change['tokens']) != 2:
            continue
        name = change['tokens'][1]
        children = list(result.get(name, {'children': []})['children'])
        for field in change['children']:
            tokens = field['tokens']
            operation = tokens[0] if tokens[0] in {'add', 'remove'} else 'set'
            tokens = tokens[1:] if operation != 'set' else tokens
            if not tokens:
                continue
            key = tokens[0]
            if operation == 'remove':
                if key == 'attributes' and len(tokens) > 1:
                    children = [{**n, 'tokens': [key, *[v for v in n['tokens'][1:]
                                                      if v not in tokens[1:]]]}
                                if n['tokens'][0] == key else n for n in children]
                else:
                    children = [n for n in children if n['tokens'][0] != key
                                or len(tokens) > 1 and n['tokens'][1:] != tokens[1:]]
            else:
                if operation == 'set':
                    children = [n for n in children if n['tokens'][0] != key]
                children.append({**field, 'tokens': tokens})
        result[name] = {**change, 'children': children}
    return result


def fields(node, key):
    return [value for child in node['children'] if child['tokens'][:1] == [key]
            for value in child['tokens'][1:]]


class Locations:
    """Match planet filters against the interpreted galaxy."""

    def __init__(self, corpus, world=()):
        self.planets = world_nodes(corpus.planets, world, 'planet')
        self.systems = world_nodes(corpus.systems, world, 'system')
        self.planet_systems = corpus.planet_systems
        self.links = {name: set(fields(node, 'link')) for name, node in self.systems.items()}
        for change in world:
            parts = change['tokens']
            if parts[0] in {'link', 'unlink'} and len(parts) == 3:
                for first, second in (parts[1:], parts[:0:-1]):
                    if parts[0] == 'link':
                        self.links.setdefault(first, set()).add(second)
                    else:
                        self.links.get(first, set()).discard(second)
        self.distances = {}

    def distance(self, center, system):
        if center not in self.distances:
            distances = {center: 0}
            pending = deque([center])
            while pending:
                current = pending.popleft()
                for neighbor in self.links.get(current, ()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        pending.append(neighbor)
            self.distances[center] = distances
        return self.distances[center].get(system, float('inf'))

    def attributes(self, node):
        attributes = set(fields(node, 'attributes'))
        for key in ('shipyard', 'outfitter'):
            if fields(node, key):
                attributes.add(key)
        if fields(node, 'spaceport') and 'uninhabited' not in attributes:
            attributes.add('spaceport')
            attributes.add('port')
        return attributes

    def matches(self, nodes, system, planet=None, origin=None):
        system_node = self.systems[system]
        node = self.planets[planet] if planet else system_node
        checks, groups = [], {}
        for child in nodes:
            key, *args = child['tokens']
            nested = child['children']
            if key in {'not', 'neighbor'}:
                filters = [{**child, 'tokens': args}] if args else nested
                if key == 'not':
                    checks.append(not self.matches(filters, system, planet, origin))
                else:
                    checks.append(any(self.matches(filters, neighbor, origin=origin)
                                      for neighbor in self.links.get(system, ())
                                      if neighbor in self.systems))
            elif key in {'planet', 'system', 'government', 'attributes', 'category'}:
                if any(n['children'] for n in nested):
                    raise UnsupportedOperation('Nested location value lists')
                args += [value for n in nested for value in n['tokens']]
                if key == 'attributes':
                    attributes = self.attributes(node)
                    if planet is None:
                        for name, p in self.planets.items():
                            if self.planet_systems.get(name) == system:
                                attributes.update(self.attributes(p))
                    checks.append(not args or bool(attributes.intersection(args)))
                elif key == 'category':
                    checks.append(not args)
                else:
                    groups.setdefault(key, set()).update(args)
            elif key in {'near', 'distance'}:
                if nested:
                    raise UnsupportedOperation('Location distance travel options')
                center = args.pop(0) if key == 'near' else origin
                if center is None:
                    continue
                low, high = (0, 1) if not args else ((0, int(args[0])) if len(args) == 1
                                                    else map(int, args))
                checks.append(low <= self.distance(center, system) <= high)
            else:
                raise UnsupportedOperation(f'Unsupported location filter: {key}')
        government = fields(node, 'government') or fields(system_node, 'government')
        for key, allowed in groups.items():
            value = {'planet': planet, 'system': system,
                     'government': government[-1] if government else None}[key]
            # Planet restrictions do not apply when matching neighboring systems.
            checks.append(not allowed or key == 'planet' and planet is None or value in allowed)
        return all(checks)

    def pick(self, nodes, seed, mission, key, origin=None):
        """Return a seeded valid spaceport, or leave unsupported filters unresolved."""
        candidates = []
        try:
            for name, planet in sorted(self.planets.items()):
                system = self.planet_systems.get(name)
                if system not in self.systems:
                    continue
                if any(n['tokens'] == ['inaccessible'] for n in self.systems[system]['children']):
                    continue
                if not any(n['tokens'][:1] == ['spaceport'] for n in planet['children']):
                    continue
                if fields(planet, 'required reputation') or fields(planet, 'wormhole'):
                    continue
                if self.matches(nodes, system, name, origin):
                    candidates.append(name)
        except UnsupportedOperation:
            return None
        if not candidates:
            return None
        return min(candidates, key=lambda name: fingerprint([seed, mission, key, name]))
