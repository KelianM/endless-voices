"""Plan mission phases and interpret compatible histories and encounters."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from graphlib import CycleError
from itertools import permutations

import z3

from endless_voices.artifacts import fingerprint as digest

from .dialogue import Continuation, Passage
from .graph import mission_graph
from .source import walk
from .state import GameState, UnsupportedOperation, apply, condition, projected_state
from .variables import mission_values, render


@dataclass
class History:
    state: GameState
    blocks: list[dict]


@dataclass
class Encounter:
    key: str
    node: dict
    route: Continuation
    history: History
    phase: str


class HistoryReconstructor:
    """Reconstruct bounded histories and sample compatible authored encounters."""

    def __init__(self, corpus, interpreter, player, max_routes=1000, seed="context"):
        self.corpus = corpus
        self.interpreter = interpreter
        self.player = player
        self.max_routes = max_routes
        self.seed = seed
        self.history_variables = None
        self.dependencies = corpus.mission_dependencies
        self.graph = mission_graph(self.dependencies)

    def ancestors(self, mission):
        ordered, active, done = [], set(), set()

        def visit(name):
            if name in active:
                raise ValueError(f"Cyclic mission prerequisites: {name}")
            if name in done:
                return
            if name not in self.graph:
                raise ValueError(f"Prerequisite mission missing from corpus: {name}")
            active.add(name)
            for parent in self.graph[name]:
                visit(parent)
            active.remove(name)
            done.add(name)
            if name != mission:
                ordered.append(name)

        visit(mission)
        return ordered

    def bounded(self, routes):
        distinct = {}
        variables = (set().union(*(r.state.values for r in routes)) & self.history_variables
                     if self.history_variables is not None else None)
        symbol_cache = {}
        for route in routes:
            key = digest(projected_state(route.state, variables, symbol_cache))
            if key not in distinct or digest(route.blocks) < digest(distinct[key].blocks):
                distinct[key] = route
        routes = list(distinct.values())
        if len(routes) > self.max_routes:
            raise ValueError("History route limit exceeded; no branches were silently discarded")
        return routes

    def future_variables(self, missions):
        """Return conservative condition names referenced by remaining mission operations."""
        variables = {name + suffix for name in missions
                     for suffix in (": active", ": done", ": failed")}
        variables.update("mission payment factor: " + name for name in missions)
        visited = set()
        pending = [self.corpus.missions[name].node for name in missions]
        pending.extend(self.corpus.events.values())
        pending.extend(event for mission in self.corpus.missions.values()
                       for event in mission.node["children"] if event["tokens"] == ["on", "fail"])
        while pending:
            root = pending.pop()
            if id(root) in visited:
                continue
            visited.add(id(root))
            for node, _ in walk([root]):
                t = node["tokens"]
                variables.update(t)
                if t[0] == "conversation" and len(t) == 2:
                    pending.append(self.corpus.conversations[t[1]])
                elif t[0] == "event" and len(t) >= 2 and t[1] in self.corpus.events:
                    pending.append(self.corpus.events[t[1]])
                elif t[0] == "fail" and len(t) == 2 and t[1] in self.corpus.missions:
                    pending.append(self.corpus.missions[t[1]].node)
                    variables.add(t[1] + ": active")
                elif t[0] in {"outfit", "require"} and len(t) >= 2:
                    variables.add("outfit: " + t[1])

        return variables

    def event_parts(self, nodes, mission):
        conversations = [n for n in nodes if n["tokens"][:1] == ["conversation"]]
        dialogs = [n for n in nodes if n["tokens"][:1] == ["dialog"]]
        if len(conversations) > 1:
            raise UnsupportedOperation("Multiple dialogue entries in one mission event")
        conversations = [self.corpus.resolve_conversation(n, mission) for n in conversations]
        actions = [
            n for n in nodes if n["tokens"][0] not in
            {"conversation", "dialog", "log", "system", "planet"}
        ]
        return conversations, dialogs, actions

    def event(self, nodes, histories, mission, phase="offer", *, offer_only=False):
        """Execute a mission event with actions after the initial conversation display."""
        conversations, dialogs, actions = self.event_parts(nodes, mission)
        updated = []
        for history in histories:
            if conversations:
                for route in self.interpreter.histories(
                        conversations[0], history.state, actions,
                        future_variables=self.history_variables):
                    if not self.continues(route.stop, "history" if offer_only else phase):
                        continue
                    block = self.block(
                        mission, (*route.prefix, *route.paragraphs), conversations[0],
                        phase, history.state)
                    outcomes = [History(route.state, [*history.blocks, block])]
                    if offer_only:
                        outcome = route.stop if route.stop in {"decline", "defer"} else "accept"
                        outcomes = self.phase_histories(outcomes, mission, outcome)
                        for event in mission.node["children"]:
                            if event["tokens"] == ["on", outcome]:
                                outcomes = self.event(event["children"], outcomes, mission, outcome)
                        if outcome == "accept":
                            accepted = deepcopy(outcomes)
                            for next_phase in ("enter", "visit"):
                                before = deepcopy(outcomes)
                                outcomes = self.phase_histories(
                                    outcomes, mission, next_phase, sample=True)
                                for event in mission.node["children"]:
                                    if event["tokens"][:2] == ["on", next_phase]:
                                        outcomes = self.event_location(outcomes, event)
                                        outcomes = self.event(
                                            event["children"], outcomes, mission, next_phase)
                                if next_phase == "visit":
                                    outcomes = self.bounded([*before, *outcomes])
                            outcomes = self.bounded([*accepted, *outcomes])
                    updated.extend(outcomes)
            else:
                blocks = list(history.blocks)
                if dialogs:
                    passages = [p for dialog in dialogs
                                for p in self.corpus.dialog_passages(dialog, mission)]
                    grouped = {}
                    for passage in passages:
                        grouped.setdefault(passage["path"], []).append(
                            Passage(passage["line"], passage["text"]))
                    for path, paragraphs in grouped.items():
                        blocks.append(self.block(
                            mission, paragraphs, {"source_path": path}, phase, history.state))
                updated.append(History(apply(actions, history.state), blocks))
        return self.settle_failures(self.bounded(updated))

    def settle_failures(self, histories):
        pending, settled = list(histories), []
        while pending:
            history = pending.pop()
            if not history.state.failures:
                settled.append(history)
                continue
            state = history.state.copy()
            name, *remaining = state.failures
            state.failures = tuple(remaining)
            state.values[name + ": active"] = z3.IntVal(0)
            state.values[name + ": failed"] = state.values.get(name + ": failed", z3.IntVal(0)) + 1
            failed = self.corpus.missions.get(name)
            routes = [History(state, history.blocks)]
            if failed is not None:
                scope = state.scope
                state.scope = name
                for event in failed.node["children"]:
                    if event["tokens"] == ["on", "fail"]:
                        routes = self.event(event["children"], routes, failed, "fail")
                for route in routes:
                    route.state.scope = scope
            pending.extend(routes)
        return self.bounded(settled)

    @staticmethod
    def continues(stop, phase):
        return stop not in {"flee", "die", "explode"} and (
            phase != "offer" or stop not in {"decline", "defer"}
        )

    def values(self, mission, state=None):
        """Return the substitutions captured when this mission was instantiated."""
        if state is not None and mission.name in state.substitutions:
            return state.substitutions[mission.name]
        return mission_values(mission.node, self.corpus.planet_systems, self.player,
                              corpus=self.corpus, seed=self.seed,
                              world=state.world if state is not None else ())

    def block(self, mission, paragraphs, node, phase, state=None):
        values = self.values(mission, state)
        return {
            "mission": mission.name,
            "heading": mission.name,
            "phase": phase,
            "substitutions": values,
            "path": (node or {}).get("source_path", mission.path),
            "passages": [{**asdict(p), "text": render(p.text, values)[0]} for p in paragraphs],
        }

    def travel_days(self, mission, state):
        values = self.values(mission, state)
        return int(
            "<origin>" in values
            and "<planet>" in values
            and values["<origin>"] != values["<planet>"]
        )

    def enter(self, state, mission):
        state.scope = mission.name
        state.substitutions[mission.name] = self.values(mission, state)
        repeat = next((n["tokens"][1:] for n in mission.node["children"]
                       if n["tokens"][:1] == ["repeat"]), ["1"])
        for name, writers in self.corpus.condition_writers.items():
            if name not in state.values and (
                    name in self.corpus.scratch_conditions
                    or repeat == ["1"] and writers == {("mission", mission.name)}):
                state.values[name] = z3.IntVal(0)
        payload = z3.IntVal(0)
        for node in mission.node["children"]:
            t = node["tokens"]
            if t[0] not in {"cargo", "passengers"}:
                continue
            index = 2 if t[0] == "cargo" else 1
            if len(t) == index + 1 and t[index].isdigit():
                count = z3.IntVal(int(t[index]))
            else:
                count = state.value(f"mission {t[0]}: " + mission.name)
                state.constraints += (count >= 0,)
            payload += count * (10 if t[0] == "passengers" else 1)
        jumps = state.value("mission jumps: " + mission.name)
        state.constraints += (jumps >= 0,)
        state.values["mission payment factor: " + mission.name] = (jumps + 1) * payload

    def phase_histories(self, histories, mission, phase, *, sample=False):
        if phase == "offer":
            for history in histories:
                key = mission.name + ": offered"
                history.state.values[key] = history.state.values.get(key, z3.IntVal(0)) + 1
        if phase == "accept":
            for history in histories:
                history.state.values[mission.name + ": active"] = z3.IntVal(1)
        if phase == "enter":
            histories = [advanced for history in histories
                         for advanced in self.advance(
                             [history], self.travel_days(mission, history.state), sample=sample)]
        if phase in {"visit", "complete"}:
            requirements = [c for n in mission.node["children"]
                            if n["tokens"] == ["to", "complete"] for c in n["children"]]
            eligible = []
            for history in histories:
                if z3.is_true(z3.simplify(history.state.values.get(
                        mission.name + ": active", z3.IntVal(1)) == 0)):
                    continue
                ready = condition(requirements, history.state)
                if phase == "visit":
                    unfinished_objectives = any(
                        n["tokens"][0] in {"npc", "waypoint", "stopover"}
                        for n in mission.node["children"])
                    required = z3.BoolVal(True) if unfinished_objectives else z3.Not(ready)
                else:
                    required = ready
                state = history.state.assume(required)
                if state is not None:
                    if phase == "complete":
                        state.values[mission.name + ": active"] = z3.IntVal(0)
                        state.values[mission.name + ": done"] = state.values.get(
                            mission.name + ": done", z3.IntVal(0)) + 1
                    eligible.append(History(state, history.blocks))
            histories = eligible
        return histories

    def event_location(self, histories, event):
        if event["tokens"][1] != "enter":
            return histories
        named = event["tokens"][2:]
        filters = [n for n in event["children"] if n["tokens"] == ["system"]]
        candidates = named or sorted(self.corpus.systems)
        for node in filters:
            for child in node["children"]:
                if child["tokens"][0] != "government" or child["children"]:
                    raise UnsupportedOperation("Unsupported on-enter system filter")
                governments = set(child["tokens"][1:])
                candidates = [name for name in candidates if any(
                    n["tokens"][:1] == ["government"] and n["tokens"][1] in governments
                    for n in self.corpus.systems[name]["children"])]
        if filters and not candidates:
            raise ValueError("No system satisfies the on-enter location filter")
        if candidates:
            name = candidates[0]
            for history in histories:
                history.state.values["visited system: " + name] = z3.IntVal(1)
                history.state.world += ({"tokens": ["enter system", name], "children": [],
                                         "line": event["line"]},)
        return histories

    def histories(self, mission, initial, *, excluded=()):
        self.history_variables = None
        try:
            return self._histories(mission, initial, excluded)
        finally:
            self.history_variables = None

    def history_plans(self, mission):
        """Explore nonempty producer combinations and let game conditions filter histories."""
        forbidden = set()
        for node in self.corpus.missions[mission].node["children"]:
            if node["tokens"] == ["to", "offer"]:
                for term, parents in walk(node["children"]):
                    t = term["tokens"]
                    if (len(t) == 2 and t[0] == "not"
                            and not any(p["tokens"] == ["or"] for p in parents)):
                        writers = self.corpus.condition_triggers.get(t[1], set())
                        if writers and all(phase is not None for _, _, phase in writers):
                            forbidden.add(t[1])
        pending = [({mission: "offer"}, {})]
        plans = []
        while pending:
            stages, selected = pending.pop()
            unresolved = next((name for name in stages if name not in selected), None)
            if unresolved is None:
                contradicted = any(
                    action["tokens"] == ["set", flag]
                    for name, stage in stages.items() if name != mission
                    for node in self.corpus.missions[name].node["children"]
                    if node["tokens"] in ([["on", "offer"], ["on", "accept"], ["on", "complete"]]
                                          if stage == "complete" else [["on", "offer"]])
                    for action in node["children"] for flag in forbidden)
                if contradicted:
                    continue
                plans.append((stages, selected))
                if len(plans) > self.max_routes:
                    raise ValueError(
                        "History route limit exceeded; no branches were silently discarded")
                continue
            options = [[]]
            for choices in self.dependencies[unresolved].values():
                expanded_options = []
                for size in range(1, len(choices) + 1):
                    for producers in permutations(choices, size):
                        for parents in options:
                            expanded_options.append([*parents, *producers])
                            if len(expanded_options) > self.max_routes:
                                raise ValueError(
                                    "History route limit exceeded; "
                                    "no branches were silently discarded")
                options = expanded_options
            for parents in options:
                expanded = dict(stages)
                for parent, phase in parents:
                    if expanded.get(parent) != "complete":
                        expanded[parent] = phase
                pending.append((expanded, {**selected, unresolved: parents}))
        return plans

    def history_steps(self, ancestors, stages, selected, *, overlap=False):
        """Order mission starts and completions around active prerequisites."""
        peers = {name: [] for name in ancestors}
        if overlap:
            for owner, parents in selected.items():
                for choices in self.dependencies[owner].values():
                    group = list(dict.fromkeys(p for p, phase in parents if (p, phase) in choices))
                    for name in group:
                        if name in peers:
                            peers[name].extend(p for p in group if p != name and p in peers)
        active_parents = {name: set() for name in ancestors}
        for name in ancestors:
            for node in self.corpus.missions[name].node["children"]:
                if node["tokens"] != ["to", "offer"]:
                    continue
                for term, parents in walk(node["children"]):
                    parts = term["tokens"]
                    if (len(parts) == 2 and parts[0] == "has"
                            and parts[1].endswith(": active")
                            and parts[1][:-8] in active_parents
                            and not any(p["tokens"] == ["or"] for p in parents)):
                        active_parents[name].add(parts[1][:-8])
        steps, visiting, done = [], set(), set()

        def visit(name, stage):
            if stage == "start" and stages[name] == "offer":
                stage = "offer"
            if stage == "offer" and stages[name] == "complete":
                stage = "start"
            key = (name, stage)
            if key in done:
                return
            if key in visiting:
                raise CycleError(f"Cyclic mission history phases: {name}")
            visiting.add(key)
            if stage in {"start", "offer"}:
                for parent, phase in selected[name]:
                    visit(parent, phase)
                for parent in sorted(active_parents[name]):
                    visit(parent, "start")
            else:
                visit(name, "start")
                for peer in peers[name]:
                    phase = "start" if stages[peer] == "complete" else "offer"
                    if (peer, phase) not in visiting:
                        visit(peer, phase)
                for child in ancestors:
                    if name in active_parents[child]:
                        visit(child, "start")
                for node in self.corpus.missions[name].node["children"]:
                    if node["tokens"] != ["to", "complete"]:
                        continue
                    for term, parents in walk(node["children"]):
                        parts = term["tokens"]
                        if (len(parts) == 2 and parts[0] == "has"
                                and parts[1].startswith("event: ")
                                and not any(p["tokens"] == ["or"] for p in parents)):
                            producers = self.corpus.event_schedulers.get(parts[1][7:], set())
                            if len(producers) == 1:
                                kind, parent, phase = next(iter(producers))
                                if (kind == "mission" and parent in stages and parent != name
                                        and phase in {"offer", "complete"}):
                                    visit(parent, phase)
            visiting.remove(key)
            done.add(key)
            steps.append(key)

        for name in ancestors:
            visit(name, stages[name])
        return steps

    def _histories(self, mission, initial, excluded):
        histories, omitted = [], set()
        for stages, selected in self.history_plans(mission.name):
            state = initial.copy()
            state.events = self.corpus.events
            state.planet_systems = self.corpus.planet_systems
            for name in selected:
                for flag in self.dependencies[name]:
                    if flag.startswith("event: "):
                        state.values.setdefault(flag, z3.IntVal(0))
            ancestors = [name for name in stages if name != mission.name]
            schedules = [self.history_steps(ancestors, stages, selected)]
            if any(len(choices) > 1 for name in selected
                   for choices in self.dependencies[name].values()):
                try:
                    concurrent = self.history_steps(ancestors, stages, selected, overlap=True)
                except CycleError:
                    concurrent = None
                if concurrent is not None and concurrent != schedules[0]:
                    schedules.append(concurrent)
            for steps in schedules:
                routes, missing = self.execute_history(mission, state.copy(), steps, excluded)
                histories.extend(routes)
                omitted.update(missing)
        self.history_variables = None
        return self.bounded(histories), sorted(omitted)

    def execute_history(self, mission, initial, steps, excluded):
        """Interpret one choice of event producers through their required phases."""
        histories, omitted = [History(initial, [])], []
        for index, (name, stage) in enumerate(steps):
            remaining = list(dict.fromkeys(n for n, _ in steps[index:]))
            self.history_variables = self.future_variables([*remaining, mission.name])
            histories = self.bounded(histories)
            if name in excluded:
                if stage in {"start", "offer"}:
                    omitted.append(name)
                continue
            previous = self.corpus.missions[name]
            requirements = [
                c
                for n in previous.node["children"]
                if n["tokens"] == ["to", "offer"]
                for c in n["children"]
            ]
            if stage in {"start", "offer"}:
                candidates = []
                for history in histories:
                    timelines = sorted(enumerate(history.state.advance()),
                                       key=lambda item: digest([self.seed, item[0]]))
                    for _, state in timelines:
                        self.enter(state, previous)
                        state = state.assume(condition(requirements, state))
                        if state is not None:
                            candidates.append(History(state, history.blocks))
                            break
                histories = self.bounded(candidates)
            else:
                completion = [c for n in previous.node["children"]
                              if n["tokens"] == ["to", "complete"] for c in n["children"]]
                if completion:
                    candidates = []
                    for history in histories:
                        for state in history.state.advance():
                            state = state.assume(condition(completion, state))
                            if state is not None:
                                candidates.append(History(state, history.blocks))
                                break
                    histories = self.bounded(candidates)
                for history in histories:
                    history.state.scope = name
            before_visit = None
            phases = (["offer", "accept", "enter", "visit"] if stage == "start"
                      else [stage])
            for phase in phases:
                if phase == "visit":
                    before_visit = histories
                    if not any(n["tokens"] == ["on", "visit"]
                               for n in previous.node["children"]):
                        continue
                histories = self.phase_histories(histories, previous, phase, sample=True)
                events = [n for n in previous.node["children"] if n["tokens"][:2] == ["on", phase]]
                for event in events:
                    histories = self.event_location(histories, event)
                    histories = self.event(event["children"], histories, previous, phase,
                                           offer_only=stage == "offer")
                if phase == "visit":
                    histories = self.bounded([*before_visit, *histories])
        self.history_variables = None
        return histories, omitted

    def advance(self, histories, minimum_days=0, *, sample=False):
        advanced = []
        for history in histories:
            outcomes = history.state.advance(minimum_days)
            if sample:
                outcomes = [min(enumerate(outcomes),
                                key=lambda item: digest([self.seed, item[0]]))[1]]
            advanced.extend(History(state, history.blocks) for state in outcomes)
        return self.bounded(advanced)

    def encounters(self, mission, initial, *, excluded=(), selected_conversations=None):
        """Return one reproducible history per authored target, plus omitted missions."""
        histories, omitted = self.histories(mission, initial, excluded=excluded)
        histories = self.advance(histories)
        for history in histories:
            self.enter(history.state, mission)
        conditions = [
            c
            for n in mission.node["children"]
            if n["tokens"] == ["to", "offer"]
            for c in n["children"]
        ]
        eligible = []
        for h in histories:
            state = h.state.assume(condition(conditions, h.state))
            if state is not None:
                eligible.append(History(state, h.blocks))
        histories = eligible
        variants = {}
        completion_events = [n for n in mission.node["children"]
                             if n["tokens"][:2] == ["on", "complete"]]
        before_visit = None
        for phase in ["offer", "accept", "enter", "visit", "complete"]:
            if phase == "visit":
                before_visit = histories
                if not any(n["tokens"] == ["on", "visit"] for n in mission.node["children"]):
                    continue
            elif phase == "complete":
                histories = self.bounded([*before_visit, *histories])
            histories = self.phase_histories(histories, mission, phase)
            for event in mission.node["children"]:
                if event["tokens"][:2] != ["on", phase]:
                    continue
                histories = self.event_location(histories, event)
                conversations, _, actions = self.event_parts(event["children"], mission)
                if not conversations:
                    histories = self.event(event["children"], histories, mission, phase)
                    continue
                node = conversations[0]
                retain_histories = phase != "complete" or event is not completion_events[-1]
                next_histories = []
                for history in histories:
                    routes = self.interpreter.continuations(
                        node, history.state, actions,
                        future_variables=self.future_variables([mission.name]))
                    for route in routes:
                        if (retain_histories and route.terminal
                                and self.continues(route.stop, phase)):
                            block = self.block(
                                mission, (*route.prefix, *route.paragraphs), node,
                                phase, history.state)
                            next_histories.append(History(route.state, [*history.blocks, block]))
                        if not route.paragraphs or not route.text.strip():
                            continue
                        if selected_conversations and node["line"] not in selected_conversations:
                            continue
                        key = digest(
                            {
                                "revision": self.corpus.revision,
                                "mission": mission.name,
                                "conversation": node["line"],
                                "paragraphs": [p.line for p in route.paragraphs],
                            }
                        )
                        rank = (
                            len(
                                {p.line for p in route.paragraphs} & {p.line for p in route.prefix}
                            ),
                            digest(
                                [
                                    self.seed,
                                    history.blocks,
                                    [asdict(p) for p in route.prefix],
                                ]
                            ),
                        )
                        candidate = (rank, node, route, history, phase)
                        if key not in variants or rank < variants[key][0]:
                            variants[key] = candidate
                if retain_histories:
                    histories = self.settle_failures(self.bounded(next_histories))
                else:
                    histories = []
        found = {candidate[1]["line"] for candidate in variants.values()}
        if selected_conversations and set(selected_conversations) - found:
            raise ValueError(
                "Requested conversations were unreachable or use unsupported mission phases: "
                + str(sorted(set(selected_conversations) - found))
            )
        return [Encounter(key, node, route, history, phase)
                for key, (_, node, route, history, phase) in sorted(variants.items())], omitted
