"""Track integer game conditions and satisfiable route requirements."""

from dataclasses import dataclass, field

import z3
from z3.z3util import get_vars


class UnsupportedOperation(ValueError):
    """Report a source operation that the interpreter cannot execute faithfully."""


def unsupported(node):
    return UnsupportedOperation(f"Line {node['line']}: unsupported operation {node['tokens']!r}")


@dataclass
class GameState:
    """Carry symbolic current values and constraints on initial game conditions."""

    values: dict = field(default_factory=dict)
    constraints: tuple = ()
    initial: dict = field(default_factory=dict)

    draws: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    pending: tuple = ()
    elapsed: object = field(default_factory=lambda: z3.IntVal(0))
    world: tuple = ()
    scope: str = "dialogue"
    failures: tuple = ()
    planet_systems: dict = field(default_factory=dict)

    def copy(self):
        return GameState(
            dict(self.values),
            self.constraints,
            dict(self.initial),
            dict(self.draws),
            self.events,
            self.pending,
            self.elapsed,
            self.world,
            self.scope,
            self.failures,
            self.planet_systems,
        )

    def draw(self, kind, lower, upper=None):
        name = f"{kind}:{len(self.draws)}"
        value = z3.Int(name)
        self.draws[name] = value
        self.constraints += (value >= lower,)
        if upper is not None:
            self.constraints += (value <= upper,)
        return value

    def schedule(self, name, low, high):
        if name not in self.events:
            raise UnsupportedOperation(f"Undefined event: {name}")
        delay = z3.IntVal(low) if low == high else self.draw("event-delay", low, high)
        if low <= 0:
            if high > 0:
                raise UnsupportedOperation(
                    "Event delay straddles immediate and scheduled execution"
                )
            self.trigger(name)
        else:
            self.pending += ((name, self.elapsed + delay),)

    def trigger(self, name):
        node = self.events[name]
        assignments = []
        for child in node["children"]:
            if child["tokens"][0] in {
                "government", "system", "planet", "outfitter", "shipyard", "fleet", "link", "unlink"
            }:
                self.world += (child,)
            else:
                assignments.append(child)
        self.values["event: " + name] = z3.IntVal(1)
        visits = {"unvisit": 1, "unvisit planet": 2, "visit": 3, "visit planet": 4}
        assignments.sort(key=lambda n: visits.get(n["tokens"][0], 0))
        updated = apply(assignments, self)
        self.values, self.constraints = updated.values, updated.constraints
        self.initial, self.draws = updated.initial, updated.draws
        self.world, self.failures = updated.world, updated.failures
        self.pending, self.elapsed = updated.pending, updated.elapsed

    def advance(self, minimum_days=0):
        """Return feasible event timelines after an unknown nonnegative travel interval."""
        if not self.pending:
            return [self.copy()]
        start = self.copy()
        target = start.elapsed + start.draw("elapsed-days", minimum_days)
        pending, finished = [start], []
        while pending:
            current = pending.pop()
            waiting = current.assume(z3.And(*(due > target for _, due in current.pending)))
            if waiting is not None:
                waiting.elapsed = target
                finished.append(waiting)
            for index, (name, due) in enumerate(current.pending):
                first = z3.And(
                    due <= target,
                    *(
                        due < other if j < index else due <= other
                        for j, (_, other) in enumerate(current.pending)
                        if j != index
                    ),
                )
                ready = current.assume(first)
                if ready is not None:
                    ready.pending = current.pending[:index] + current.pending[index + 1 :]
                    ready.elapsed = due
                    ready.trigger(name)
                    pending.append(ready)
            if len(pending) + len(finished) > 1000:
                raise ValueError("Event timeline limit exceeded")
        return finished

    def value(self, token):
        if token == "random":
            return self.draw("random", 0, 99)
        try:
            return z3.IntVal(int(token))
        except ValueError:
            if token not in self.values:
                self.initial.setdefault(token, z3.Int("initial:" + token))
                self.values[token] = self.initial[token]
            return self.values[token]

    def assume(self, condition):
        condition = z3.simplify(condition)
        if z3.is_false(condition):
            return None
        result = self.copy()
        if z3.is_true(condition) or any(condition.eq(c) for c in self.constraints):
            return result
        result.constraints += (condition,)
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*result.constraints)
        status = solver.check()
        if status == z3.unknown:
            raise ValueError("State feasibility check did not finish")
        return result if status == z3.sat else None

    @classmethod
    def fixed(cls, values):
        state = cls()
        for name, value in values.items():
            if type(value) is not int:
                raise ValueError("Game state values must be integers")
            state.value(name)
            state.constraints += (state.initial[name] == value,)
        return state

    def snapshot(self):
        """Return symbolic requirements and one reproducible compatible initial state."""
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*self.constraints)
        if solver.check() != z3.sat:
            raise ValueError("Cannot save an inconsistent or unresolved state")
        model = solver.model()
        return {
            "draws": {
                k: model.eval(v, model_completion=True).as_long()
                for k, v in sorted(self.draws.items())
            },
            "elapsed_days": model.eval(self.elapsed, model_completion=True).as_long(),
            "pending_events": [
                {"name": name, "due_day": model.eval(due, model_completion=True).as_long()}
                for name, due in self.pending
            ],
            "world_changes": list(self.world),
            "failed_missions_pending": list(self.failures),
            "requirements": [z3.simplify(c).sexpr() for c in self.constraints],
            "initial_values": {
                k: model.eval(v, model_completion=True).as_long()
                for k, v in sorted(self.initial.items())
            },
            "current_values": {
                k: model.eval(v, model_completion=True).as_long()
                for k, v in sorted(self.values.items())
            },
        }


def condition(nodes, state, *, disjunction=False):
    """Translate supported game predicates without inventing values for unknown state."""
    terms = []
    for node in nodes:
        t, children = node["tokens"], node["children"]
        if t in (["and"], ["or"]):
            terms.append(condition(children, state, disjunction=t == ["or"]))
        elif t == ["never"] and not children:
            terms.append(z3.BoolVal(False))
        elif len(t) == 1 and not children:
            terms.append(state.value(t[0]) != 0)
        elif len(t) == 2 and t[0] in {"has", "not"} and not children:
            value = state.value(t[1])
            terms.append(value != 0 if t[0] == "has" else value == 0)
        elif not children and any(op in t for op in {"==", "!=", "<", ">", "<=", ">="}):
            indices = [i for i, op in enumerate(t) if op in {"==", "!=", "<", ">", "<=", ">="}]
            if len(indices) != 1:
                raise unsupported(node)
            i = indices[0]
            a, b = expression(t[:i], state), expression(t[i + 1 :], state)
            terms.append(
                {"==": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[
                    t[i]
                ]
            )
        else:
            raise unsupported(node)
    return z3.Or(*terms) if disjunction else z3.And(*terms)


def apply(nodes, state):
    """Apply supported assignments in order, preserving earlier branch constraints."""
    result = state.copy()
    for node in nodes:
        t = node["tokens"]
        if t[0] == "debt" and len(t) == 2 and int(t[1]) >= 0:
            if any(c["tokens"][0] not in {"interest", "term"} or len(c["tokens"]) != 2
                   or c["children"] for c in node["children"]):
                raise unsupported(node)
            result.world += (node,)
            continue
        if node["children"]:
            raise unsupported(node)
        if t[0] == "event" and 2 <= len(t) <= 4:
            low = int(t[2]) if len(t) >= 3 else 1
            high = int(t[3]) if len(t) == 4 else low
            result.schedule(t[1], min(low, high), max(low, high))
            continue
        if t[0] == "outfit" and len(t) in {2, 3}:
            name = "outfit: " + t[1]
            count = int(t[2]) if len(t) == 3 else 1
            available = result.value(name)
            result.constraints += (available >= max(0, -count),)
            result.values[name] = available + count
            continue
        if t[0] in {"visit", "unvisit", "visit planet", "unvisit planet"} and len(t) == 2:
            planet = t[0].endswith(" planet")
            visited = not t[0].startswith("unvisit")
            kind = "planet" if planet else "system"
            result.values[f"visited {kind}: " + t[1]] = z3.IntVal(int(visited))
            if not planet and not visited:
                for name, system in result.planet_systems.items():
                    if system == t[1]:
                        result.values["visited planet: " + name] = z3.IntVal(0)
            result.world += (node,)
            continue
        if t[0] in {"mark", "unmark"} and len(t) in {2, 3}:
            result.world += ({**node, "mission": t[2] if len(t) == 3 else result.scope},)
            continue
        if t[0] == "fail" and len(t) in {1, 2}:
            name = t[1] if len(t) == 2 else result.scope
            active = result.values.get(name + ": active", z3.IntVal(0))
            if not z3.is_true(z3.simplify(active == 0)) and name not in result.failures:
                result.failures += (name,)
            continue
        if t[:2] == ["give", "ship"] and len(t) in {3, 4}:
            result.world += (node,)
            continue
        if t[0] == "fine" and len(t) == 2 and int(t[1]) > 0:
            result.world += (node,)
            continue
        if t[0] == "log" and len(t) in {2, 4} and not node["children"]:
            result.world += (node,)
            continue
        if t[0] == "payment" and len(t) <= 3:
            base = int(t[1]) if len(t) > 1 else 0
            multiplier = int(t[2]) if len(t) == 3 else (150 if len(t) == 1 else 0)
            amount = z3.IntVal(base)
            if multiplier:
                factor = result.value("mission payment factor: " + result.scope)
                result.constraints += (factor >= 0,)
                amount += multiplier * factor
            credits = result.value("credits")
            result.constraints += (credits >= 0,)
            result.values["credits"] = z3.If(credits + amount >= 0, credits + amount, 0)
            continue
        if len(t) == 2 and t[0] in {"set", "clear"}:
            result.values[t[1]] = z3.IntVal(1 if t[0] == "set" else 0)
            continue
        if len(t) == 2 and t[1] in {"++", "--"}:
            t = [t[0], "+=" if t[1] == "++" else "-=", "1"]
        if len(t) < 3 or t[1] not in {"=", "+=", "-=", "*=", "/=", "%=", "<?=", ">?="}:
            raise unsupported(node)
        name, op = t[:2]
        b = expression(t[2:], result)
        if op == "=":
            result.values[name] = b
            continue
        a = result.value(name)
        result.values[name] = z3.simplify(
            {
                "+=": a + b,
                "-=": a - b,
                "*=": a * b,
                "/=": divide(a, b),
                "%=": z3.If(b == 0, a, a - divide(a, b) * b),
                "<?=": z3.If(a < b, a, b),
                ">?=": z3.If(a > b, a, b),
            }[op]
        )
    return result


def divide(left, right):
    """Return integer division truncated toward zero, including the game's zero sentinel."""
    quotient = z3.If(
        left * right >= 0, z3.Abs(left) / z3.Abs(right), -(z3.Abs(left) / z3.Abs(right)))
    return z3.If(right == 0, 2**63 - 1, quotient)


def expression(tokens, state):
    """Evaluate arithmetic using the game's precedence without evaluating Python code."""
    position = 0
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}

    def parse(minimum=0):
        nonlocal position
        if position >= len(tokens):
            raise UnsupportedOperation("Incomplete arithmetic expression")
        token = tokens[position]
        position += 1
        if token == "(":
            left = parse()
            if position >= len(tokens) or tokens[position] != ")":
                raise UnsupportedOperation("Unbalanced arithmetic parentheses")
            position += 1
        else:
            left = state.value(token)
        while position < len(tokens) and precedence.get(tokens[position], -1) >= minimum:
            op = tokens[position]
            position += 1
            right = parse(precedence[op] + 1)
            quotient = divide(left, right)
            left = {
                "+": left + right,
                "-": left - right,
                "*": left * right,
                "/": quotient,
                "%": z3.If(right == 0, left, left - quotient * right),
            }[op]
        return left

    result = parse()
    if position != len(tokens):
        raise UnsupportedOperation(f"Unsupported arithmetic expression: {tokens!r}")
    return z3.simplify(result)


def projected_state(state, variables=None, symbol_cache=None):
    """Return future-relevant state without changing its complete history witness."""
    symbol_cache = {} if symbol_cache is None else symbol_cache

    def symbols(expression):
        key = expression.sexpr()
        if key not in symbol_cache:
            symbol_cache[key] = {v.sexpr() for v in get_vars(expression)}
        return symbol_cache[key]

    values = state.values if variables is None else {
        name: state.values.get(name, z3.Int("initial:" + name)) for name in variables}
    constraints = [z3.simplify(c) for c in state.constraints]
    if variables is not None:
        live = set().union(*(symbols(v) for v in values.values()), symbols(state.elapsed),
                           *(symbols(due) for _, due in state.pending))
        connected = []
        while True:
            linked = [c for c in constraints if symbols(c) & live]
            if not linked:
                break
            for c in linked:
                live.update(symbols(c))
                connected.append(c)
            linked_keys = {c.sexpr() for c in linked}
            constraints = [c for c in constraints if c.sexpr() not in linked_keys]
        constraints = connected
    return {
        "values": {k: z3.simplify(v).sexpr() for k, v in sorted(values.items())},
        "constraints": sorted({c.sexpr() for c in constraints if not z3.is_true(c)}),
        "pending": [(name, z3.simplify(due).sexpr()) for name, due in state.pending],
        "elapsed": z3.simplify(state.elapsed).sexpr(),
        "world": state.world,
        "failures": state.failures,
        "scope": state.scope,
    }
