"""Enumerate authored continuations with consistent dialogue state."""

from dataclasses import dataclass
from random import Random

import z3

from endless_voices.context import digest

from .source import walk
from .state import GameState, apply, condition, projected_state, unsupported

ENDPOINTS = {"accept", "decline", "defer", "launch", "flee", "die", "explode"}


@dataclass(frozen=True)
class Passage:
    line: int
    text: str
    role: str = "passage"


@dataclass
class Continuation:
    prefix: tuple[Passage, ...]
    paragraphs: tuple[Passage, ...]
    state: GameState
    stop: int | str
    terminal: bool

    @property
    def text(self):
        return "\n\n".join(p.text for p in self.paragraphs)


class DialogueInterpreter:
    """Execute dialogue operations, forking only satisfiable branches."""

    def __init__(self, max_steps=10000, max_visits=2):
        self.max_steps = max_steps
        self.max_visits = max_visits

    def continuations(self, conversation, state=None, after_display=(), *, sample_seed=None,
                      permitted_stops=None, future_variables=None):
        """Return passages up to actual player choices or endpoints, with route history."""
        nodes = conversation["children"]
        labels = {n["tokens"][1]: i for i, n in enumerate(nodes) if n["tokens"][:1] == ["label"]}
        if len(labels) != sum(n["tokens"][:1] == ["label"] for n in nodes):
            raise ValueError("Duplicate conversation label")

        def jump(label, node, endpoint=False):
            if endpoint and label in ENDPOINTS:
                return label
            if label not in labels:
                raise ValueError(f"Line {node['line']}: unknown label {label!r}")
            return labels[label]

        def controls(node, following):
            destination = following
            for child in node["children"]:
                t = child["tokens"]
                if t[:1] == ["goto"] and len(t) == 2:
                    destination = jump(t[1], child)
                    break
                elif len(t) == 1 and t[0] in ENDPOINTS:
                    destination = t[0]
                    break
                elif t != ["to", "display"]:
                    raise unsupported(child)
            return destination

        def visible(node, current):
            conditions = [n for n in node["children"] if n["tokens"] == ["to", "display"]]
            return condition([c for n in conditions for c in n["children"]], current)

        variables = None
        if future_variables is not None:
            variables = set(future_variables)
            for node, _ in walk([conversation, *after_display]):
                t = node["tokens"]
                variables.update(t)
                if t[0] in {"require", "outfit"} and len(t) >= 2:
                    variables.add("outfit: " + t[1])
        initial = state.copy() if state is not None else GameState()
        actions = []
        for node in after_display:
            t = node["tokens"]
            if t[:1] != ["require"]:
                actions.append(node)
                continue
            if len(t) not in {2, 3} or node["children"]:
                raise unsupported(node)
            count = int(t[2]) if len(t) == 3 else 1
            if count < 0:
                raise unsupported(node)
            available = initial.value("outfit: " + t[1])
            initial = initial.assume(available == 0 if count == 0 else available >= count)
            if initial is None:
                return []
        after_display = actions
        pending = [(0, initial, (), (), {}, False)]
        results, steps = [], 0
        explored = set()
        symbol_cache = {}
        random = Random(sample_seed)
        while pending:
            pc, current, prefix, paragraphs, visits, displayed = pending.pop()
            pending_start = len(pending)
            steps += 1
            if steps > self.max_steps:
                raise ValueError("Dialogue route limit exceeded; no partial build is published")
            if isinstance(pc, str) or pc >= len(nodes):
                if not displayed:
                    current = apply(after_display, current)
                stop = pc if isinstance(pc, str) else "end"
                if permitted_stops is not None and stop not in permitted_stops:
                    continue
                results.append(
                    Continuation(
                        prefix, paragraphs, current, stop, True
                    )
                )
                if sample_seed is not None:
                    return [results[-1]]
                continue
            node = nodes[pc]
            execution_key = (
                pc,
                digest(projected_state(
                    current, variables & current.values.keys() if variables is not None else None,
                    symbol_cache)),
                tuple((p.line, p.role) for p in paragraphs),
                displayed,
            )
            # Equivalent futures need one compatible prefix, not every menu permutation.
            if execution_key in explored:
                continue
            explored.add(execution_key)
            visit_key = (
                pc,
                tuple(sorted((k, z3.simplify(v).sexpr()) for k, v in current.values.items())),
            )
            count = visits.get(visit_key, 0) + 1
            if count > self.max_visits:
                raise ValueError(f"Line {node['line']}: dialogue loop requires explicit handling")
            visits = {**visits, visit_key: count}
            t = node["tokens"]

            def enqueue(dest, candidate, pre=prefix, text=paragraphs, shown=displayed,
                        seen=visits):
                if candidate is not None:
                    pending.append((dest, candidate, pre, text, seen, shown))

            if t == ["choice"]:
                options = node["children"]
                any_visible = []
                for option in options:
                    if not option["raw"].lstrip().startswith(("`", '"')):
                        raise unsupported(option)
                    predicate = visible(option, current)
                    any_visible.append(predicate)
                    pre = (
                        *prefix,
                        *paragraphs,
                        Passage(option["line"], option["tokens"][0], "option"),
                    )
                    candidate = current.assume(predicate)
                    if candidate is not None and not displayed:
                        candidate = apply(after_display, candidate)
                    enqueue(controls(option, pc + 1), candidate, pre, (), True, seen={})
                stopping = current.assume(z3.Or(*any_visible))
                if paragraphs and stopping is not None:
                    results.append(Continuation(prefix, paragraphs, stopping, node["line"], False))
                enqueue(pc + 1, current.assume(z3.Not(z3.Or(*any_visible))))
            elif t[:1] == ["branch"] and len(t) in {2, 3}:
                predicate = condition(node["children"], current)
                enqueue(jump(t[1], node, endpoint=True), current.assume(predicate))
                enqueue(
                    jump(t[2], node, endpoint=True) if len(t) == 3 else pc + 1,
                    current.assume(z3.Not(predicate)),
                )
            elif t == ["action"]:
                enqueue(pc + 1, apply(node["children"], current))
            elif t[:1] == ["label"] and len(t) == 2:
                enqueue(pc + 1, current)
            elif t[:1] == ["goto"] and len(t) == 2:
                enqueue(jump(t[1], node), current)
            elif len(t) == 1 and t[0] in ENDPOINTS:
                enqueue(t[0], current)
            elif t[:1] == ["scene"] and len(t) == 2 and not node["children"]:
                enqueue(pc + 1, current)
            elif node["raw"].lstrip().startswith(("`", '"')) and len(t) == 1:
                predicate = visible(node, current)
                text = (*paragraphs, Passage(node["line"], t[0]))
                enqueue(controls(node, pc + 1), current.assume(predicate), text=text)
                enqueue(pc + 1, current.assume(z3.Not(predicate)))
            else:
                raise unsupported(node)
            if sample_seed is not None:
                candidates = pending[pending_start:]
                random.shuffle(candidates)
                pending[pending_start:] = candidates
        return results

    def histories(self, conversation, state, after_display=(), *, sample_seed=None,
                  permitted_stops=None, future_variables=None):
        """Return complete reachable dialogue routes for context construction."""
        return [c for c in self.continuations(
            conversation, state, after_display, sample_seed=sample_seed,
            permitted_stops=permitted_stops, future_variables=future_variables
        ) if c.terminal]
