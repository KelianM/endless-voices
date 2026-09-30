"""Track integer game conditions and satisfiable route requirements."""

from dataclasses import dataclass, field

import z3


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

    def copy(self):
        return GameState(dict(self.values), self.constraints, dict(self.initial))

    def value(self, token):
        if token == "random":
            raise UnsupportedOperation("Random conditions require explicit draw semantics")
        try:
            return z3.IntVal(int(token))
        except ValueError:
            if token not in self.values:
                self.initial.setdefault(token, z3.Int("initial:" + token))
                self.values[token] = self.initial[token]
            return self.values[token]

    def assume(self, condition):
        result = self.copy()
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
        elif len(t) == 2 and t[0] in {"has", "not"} and not children:
            value = state.value(t[1])
            terms.append(value != 0 if t[0] == "has" else value == 0)
        elif len(t) == 3 and t[1] in {"==", "!=", "<", ">", "<=", ">="} and not children:
            a, b = state.value(t[0]), state.value(t[2])
            terms.append(
                {"==": a == b, "!=": a != b, "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[
                    t[1]
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
        if node["children"]:
            raise unsupported(node)
        if len(t) == 2 and t[0] in {"set", "clear"}:
            result.values[t[1]] = z3.IntVal(1 if t[0] == "set" else 0)
            continue
        if len(t) == 2 and t[1] in {"++", "--"}:
            t = [t[0], "+=" if t[1] == "++" else "-=", "1"]
        if len(t) != 3 or t[1] not in {"=", "+=", "-=", "*=", "<?=", ">?="}:
            raise unsupported(node)
        name, op, rhs = t
        b = result.value(rhs)
        if op == "=":
            result.values[name] = b
            continue
        a = result.value(name)
        result.values[name] = z3.simplify(
            {
                "+=": a + b,
                "-=": a - b,
                "*=": a * b,
                "<?=": z3.If(a < b, a, b),
                ">?=": z3.If(a > b, a, b),
            }[op]
        )
    return result
