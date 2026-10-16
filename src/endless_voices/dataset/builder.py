"""Build state-consistent scene examples through one preparation pipeline."""

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from graphlib import CycleError
from itertools import permutations
from pathlib import Path
from urllib.parse import quote

import z3

from endless_voices.context import ContextPool, digest
from endless_voices.context import render as render_block
from endless_voices.contracts import validate_record
from endless_voices.game_variables import mission_values, render
from endless_voices.instructions import SCENE_INSTRUCTIONS, scene_task
from endless_voices.retrieval import scene_id
from endless_voices.splits import check_chain_splits, check_mission_splits

from .dialogue import DialogueInterpreter, Passage
from .source import tokens, walk
from .state import GameState, UnsupportedOperation, apply, condition, projected_state


@dataclass
class History:
    state: GameState
    blocks: list[dict]


class ContextBuilder:
    """Construct compatible histories and eligible writing references."""

    def __init__(self, corpus, interpreter, assignments, player, max_routes=1000,
                 seed="context"):
        self.corpus = corpus
        self.interpreter = interpreter
        self.assignments = assignments
        self.conversation_owners = corpus.conversation_owners(assignments)
        self.named_users = corpus.named_conversation_users()
        self.player = player
        self.max_routes = max_routes
        self.seed = seed
        self.reference_cache = {}
        self.reference_paths = {}
        self.reference_exclusions = {}
        self.history_variables = None
        self.graph = corpus.mission_graph
        self.dependencies = corpus.mission_dependencies

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
                        mission, (*route.prefix, *route.paragraphs), conversations[0], phase)
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
                        blocks.append(self.block(mission, paragraphs, {"source_path": path}, phase))
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

    def block(self, mission, paragraphs, node, phase):
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        return {
            "mission": mission.name,
            "heading": mission.name,
            "phase": phase,
            "path": (node or {}).get("source_path", mission.path),
            "passages": [{**asdict(p), "text": render(p.text, values)[0]} for p in paragraphs],
        }

    def eligible(self, name, split):
        owner = self.assignments.get(name)
        return (
            owner is None
            or {"train": 0, "validation": 1, "test": 2}[owner]
            <= {"train": 0, "validation": 1, "test": 2}[split]
        )

    def travel_days(self, mission):
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        return int(
            "<origin>" in values
            and "<planet>" in values
            and values["<origin>"] != values["<planet>"]
        )

    def enter(self, state, mission):
        state.scope = mission.name
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
            histories = self.advance(histories, self.travel_days(mission), sample=sample)
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

    def histories(self, mission, split, initial):
        self.history_variables = None
        try:
            return self._histories(mission, split, initial)
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

    def _histories(self, mission, split, initial):
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
                routes, missing = self.execute_history(mission, split, state.copy(), steps)
                histories.extend(routes)
                omitted.update(missing)
        self.history_variables = None
        return self.bounded(histories), sorted(omitted)

    def execute_history(self, mission, split, initial, steps):
        """Interpret one choice of event producers through their required phases."""
        histories, omitted = [History(initial, [])], []
        for index, (name, stage) in enumerate(steps):
            remaining = list(dict.fromkeys(n for n, _ in steps[index:]))
            self.history_variables = self.future_variables([*remaining, mission.name])
            histories = self.bounded(histories)
            if not self.eligible(name, split):
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

    def references(self, mission, split, history):
        """Return independent reference scenes from eligible related missions."""
        excluded = {mission.name, *(b["mission"] for b in history.blocks)}
        excluded.update(self.ancestors(mission.name))
        excluded.update(self.reference_exclusions.get(mission.name, ()))
        blocks = []
        paths = self.reference_paths.get(mission.name, {mission.path})
        for path in sorted(paths):
            for other in self.corpus.missions.values():
                if (other.path != path or other.name in excluded
                        or not self.eligible(other.name, split)):
                    continue
                if other.name not in self.reference_cache:
                    scenes = []
                    for event in other.node["children"]:
                        if event["tokens"] not in (["on", "offer"], ["on", "accept"],
                                                   ["on", "complete"]):
                            continue
                        if not any(n["tokens"][:1] == ["conversation"]
                                   for n in event["children"]):
                            continue
                        conversations, _, actions = self.event_parts(event["children"], other)
                        if not conversations:
                            continue
                        state = GameState(events=self.corpus.events, scope=other.name,
                                          planet_systems=self.corpus.planet_systems)
                        self.enter(state, other)
                        node = conversations[0]
                        branches = {}
                        for route in self.interpreter.continuations(
                                node, state, actions, future_variables=set()):
                            if not route.paragraphs:
                                continue
                            key = tuple(p.line for p in route.paragraphs)
                            if key in branches:
                                continue
                            block = self.block(
                                other, (*route.prefix, *route.paragraphs), node,
                                event["tokens"][1])
                            block.update(reference=True, conversation=node["line"])
                            branches[key] = block
                        scenes.extend(branches.values())
                    self.reference_cache[other.name] = scenes
                target_definitions = {
                    self.corpus.conversations[name]["source_path"] + ":" + str(
                        self.corpus.conversations[name]["line"])
                    for name, users in self.named_users.items() if mission.name in users
                }
                for block in self.reference_cache[other.name]:
                    coordinate = block["path"] + ":" + str(block["conversation"])
                    if coordinate in target_definitions:
                        continue
                    owner = next((self.conversation_owners.get(name)
                                  for name, node in self.corpus.conversations.items()
                                  if coordinate == node["source_path"] + ":" + str(node["line"])),
                                 None)
                    if owner is not None and {"train": 0, "validation": 1, "test": 2}[owner] > {
                            "train": 0, "validation": 1, "test": 2}[split]:
                        continue
                    blocks.append(block)
        directories = {str(Path(path).parent) for path in paths}
        blocks.extend(block for block in self.corpus.reference_lore
                      if str(Path(block["path"]).parent) in directories)
        return blocks

    def restore_pool(self, record, evidence):
        """Reconstruct eligible context from the encounter's source trace and game state."""
        meta = record["metadata"]
        name = meta["sources"][0]["source_group"].removeprefix("mission / ")
        mission = self.corpus.missions[name]
        blocks = []
        for entry in evidence["history"]:
            paragraphs = []
            for line, role in entry["passages"]:
                parts = tokens(self.corpus.source_lines[entry["path"]][line - 1].lstrip())
                if len(parts) == 1:
                    text = parts[0]
                elif len(parts) == 2 and parts[0] == "dialog":
                    text = parts[1]
                else:
                    raise ValueError("History passage does not identify literal source text")
                paragraphs.append(Passage(line, text, role))
            blocks.append(self.block(self.corpus.missions[entry["mission"]], paragraphs,
                                     {"source_path": entry["path"]}, entry["phase"]))
        state = GameState(world=tuple(evidence["encounter_state"]["world_changes"]))
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        pool = ContextPool(meta["id"], meta["conversation_id"], mission.name,
                           record["messages"][0]["content"], record["messages"][1:-1],
                           blocks, self.graph, values,
                           self.references(mission, meta["split"], History(state, blocks)))
        self.add_planets(pool, mission, state.world,
                         self.specifications[mission.name].get("lore_planets", []))
        return pool

    def add_planets(self, pool, mission, world, lore_names=()):
        """Add named and region-relevant planets visible in this encounter's world."""
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        names = set(lore_names)
        names.update(values[k] for k in ("<planet>", "<origin>") if k in values)
        paths = self.reference_paths.get(mission.name, {mission.path})
        directories = {str(Path(path).parent) for path in paths}
        visible = "\n".join([pool.system_prefix, *(m["content"] for m in pool.encounter),
                             *(render_block(b) for b in pool.blocks)])
        pool.references.extend(block for block in self.corpus.planet_references(world)
                               if str(Path(block["path"]).parent) in directories
                               or block["name"] in names
                               or re.search(r"\b" + re.escape(block["name"]) + r"\b", visible))

    def pool(
        self, sample_id, conversation_id, mission, history, prefix, lore, split, task
    ):
        """Build the interpreted eligibility boundary shared by selection and retrieval."""
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        encounter = "\n\n".join(render(p.text, values)[0] for p in prefix)
        system = (
            SCENE_INSTRUCTIONS[task]
            + ("\n\n" + lore if lore else "")
        )
        return ContextPool(
            sample_id,
            conversation_id,
            mission.name,
            system,
            [{"role": "user", "content": encounter or "Write the opening passage of a scene."}],
            history.blocks,
            self.graph,
            values,
            self.references(mission, split, history),
        )



class ExampleBuilder:
    """Join every reachable target with a reproducibly selected compatible history."""

    def __init__(self, corpus, interpreter, sampler, seed):
        self.corpus, self.interpreter, self.sampler, self.seed = corpus, interpreter, sampler, seed

    def build(self, mission, specification, initial):
        split = specification["split"]
        histories, omitted = self.sampler.histories(mission, split, initial)
        histories = self.sampler.advance(histories)
        for history in histories:
            self.sampler.enter(history.state, mission)
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
        selected_conversations = specification.get("conversations")
        lore_names = set(specification.get("lore_planets", []))
        values = mission_values(mission.node, self.corpus.planet_systems, self.sampler.player)
        lore_names.update(values[k] for k in ("<planet>", "<origin>") if k in values)

        completion_events = [n for n in mission.node["children"]
                             if n["tokens"][:2] == ["on", "complete"]]
        before_visit = None
        for phase in ["offer", "accept", "enter", "visit", "complete"]:
            if phase == "visit":
                before_visit = histories
                if not any(n["tokens"] == ["on", "visit"] for n in mission.node["children"]):
                    continue
            elif phase == "complete":
                histories = self.sampler.bounded([*before_visit, *histories])
            histories = self.sampler.phase_histories(histories, mission, phase)
            for event in mission.node["children"]:
                if event["tokens"][:2] != ["on", phase]:
                    continue
                histories = self.sampler.event_location(histories, event)
                conversations, _, actions = self.sampler.event_parts(event["children"], mission)
                if not conversations:
                    histories = self.sampler.event(event["children"], histories, mission, phase)
                    continue
                node = conversations[0]
                retain_histories = phase != "complete" or event is not completion_events[-1]
                next_histories = []
                for history in histories:
                    routes = self.interpreter.continuations(
                        node, history.state, actions,
                        future_variables=self.sampler.future_variables([mission.name]))
                    for route in routes:
                        if (retain_histories and route.terminal
                                and self.sampler.continues(route.stop, phase)):
                            block = self.sampler.block(
                                mission, (*route.prefix, *route.paragraphs), node, phase)
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
                    histories = self.sampler.settle_failures(self.sampler.bounded(next_histories))
                else:
                    histories = []
        found = {candidate[1]["line"] for candidate in variants.values()}
        if selected_conversations and set(selected_conversations) - found:
            raise ValueError(
                "Requested conversations were unreachable or use unsupported mission phases: "
                + str(sorted(set(selected_conversations) - found))
            )
        examples = []
        for key, (_, node, route, history, phase) in sorted(variants.items()):
            sid = "scene-" + key[:24]
            cid = "conversation-" + digest(
                [node.get("source_path", mission.path), node["line"]])[:24]
            planets = self.corpus.planet_references(history.state.world)
            lore = "\n\n".join(render_block(block) for block in planets
                                 if block["name"] in lore_names)
            task = scene_task(has_prefix=bool(route.prefix))
            pool = self.sampler.pool(
                sid, cid, mission, history, route.prefix, lore, split, task
            )
            cue = f"Current mission: {mission.name}. Scene: on {phase}."
            location = values.get("<origin>" if phase in {"offer", "accept"} else "<planet>")
            if phase == "enter":
                entered = [n["tokens"][1] for n in history.state.world
                           if n["tokens"][:1] == ["enter system"]]
                if entered:
                    cue += f" Current system: {entered[-1]}."
            elif location:
                cue += f" Current location: {location}."
                system = self.corpus.planet_systems.get(location)
                if system:
                    cue += f" System: {system}."
            links = []
            for name in [*self.sampler.ancestors(mission.name), mission.name]:
                for choices in self.sampler.dependencies[name].values():
                    producers = " OR ".join(f"{parent} (on {phase})" for parent, phase in choices)
                    links.append(f"{producers} -> {name}")
            overview = (
                "Mission prerequisites (OR means at least one producer, possibly several):\n"
                + ("\n".join(links) or "No prerequisite missions."))
            overview += "\nEarlier scenes in this interpreted history (oldest first):\n"
            overview += "\n".join(
                f"{block['mission']} | on {block['phase']} | {scene_id(block)}"
                for block in pool.blocks) or "No earlier scenes."
            pool.system_prefix = SCENE_INSTRUCTIONS[task] + "\n\n" + cue + "\n\n" + overview
            if lore:
                pool.system_prefix += "\n\n" + lore
            self.sampler.add_planets(pool, mission, history.state.world, lore_names)
            target = "\n\n".join(render(p.text, values)[0] for p in route.paragraphs)
            target_lines = {p.line for p in route.paragraphs}
            if any(p.line in target_lines for p in route.prefix):
                raise ValueError("Target source paragraph appears in its own encounter")
            eligible_blocks = pool.blocks + pool.references
            for b in eligible_blocks:
                if b["path"] == node.get("source_path", mission.path) and any(
                    p["line"] in target_lines for p in b["passages"]
                ):
                    raise ValueError("Target source paragraph appears in eligible context")
            ref = {
                "reference": "https://github.com/endless-sky/endless-sky/blob/"
                + self.corpus.revision
                + "/"
                + quote(node.get("source_path", mission.path))
                + f"#L{node['line']}",
                "revision": self.corpus.revision,
                "source_group": "mission / " + mission.name,
            }
            meta = {
                k: deepcopy(specification[k])
                for k in ["identity", "species", "character_role", "topics", "scenario_group"]
            }
            meta.update(
                task=task,
                id=sid,
                split=split,
                conversation_id=cid,
                sources=[ref],
                authorship="human",
                review_status="draft",
            )
            record = {
                "schema_version": 1,
                "metadata": meta,
                "messages": [*pool.messages(), {"role": "assistant", "content": target}],
                "evaluation": {"dimensions": ["authenticity"], "sources": [ref]},
            }
            validate_record(record)
            provenance = {
                "state": route.state.snapshot(),
                "encounter_state": history.state.snapshot(),
                "history": [{"mission": block["mission"], "phase": block["phase"],
                             "path": block["path"],
                             "passages": [[p["line"], p["role"]] for p in block["passages"]]}
                            for block in history.blocks],
                "source_path": node.get("source_path", mission.path),
                "source_sha256": node.get("source_sha256", mission.sha256),
                "source_revision": self.corpus.revision,
                "target_lines": sorted(target_lines),
                "stop": route.stop,
                "omitted_history_missions": omitted,
                "prefix_lines": [p.line for p in route.prefix],
            }
            if route.prefix and route.prefix[-1].role == "option":
                provenance["player_choices"] = {
                    "prefix": "\n\n".join(render(p.text, values)[0] for p in route.prefix[:-1]),
                    "options": [render(p.text, values)[0] for p in route.choices],
                    "selected": render(route.prefix[-1].text, values)[0],
                }
            examples.append((record, pool, provenance))
        return examples


class DatasetBuilder:
    """Own parsing dependencies, split eligibility, example construction and publication."""

    def __init__(self, corpus, config):
        self.corpus, self.config = corpus, deepcopy(config)
        self.interpreter = DialogueInterpreter(**config.get("interpreter", {}))
        assignments = {name: spec["split"] for name, spec in config["missions"].items()}
        groups = {}
        for name, spec in config["missions"].items():
            if spec["split"] not in {"train", "validation", "test"}:
                raise ValueError(f"Invalid split for mission {name}")
            group = spec["scenario_group"]
            if group is not None and group in groups and groups[group] != spec["split"]:
                raise ValueError("Scenario variants cross mission splits")
            groups[group] = spec["split"]
        self.sampler = ContextBuilder(
            corpus,
            self.interpreter,
            assignments,
            config["player"],
            config.get("max_history_routes", 1000),
            seed=config["seed"],
        )
        self.sampler.specifications = self.config["missions"]
        check_chain_splits(assignments, self.sampler.graph)
        for name, spec in config["missions"].items():
            self.sampler.reference_paths[name] = sorted({
                corpus.missions[other].path for other, other_spec in config["missions"].items()
                if other_spec["identity"] == spec["identity"]
            })
            if spec["scenario_group"] is not None:
                self.sampler.reference_exclusions[name] = {
                    other for other, other_spec in config["missions"].items()
                    if other_spec["scenario_group"] == spec["scenario_group"]
                }
        self.examples = ExampleBuilder(corpus, self.interpreter, self.sampler, config["seed"])

    def build(self, output, splits=("train",)):
        """Publish a complete prepared bundle only after every requested mission succeeds."""
        from .storage import write_dataset

        output = Path(output)
        if output.exists():
            raise ValueError("Output exists; choose a new dataset directory")
        built, failures = [], []
        for name, spec in sorted(self.config["missions"].items()):
            if spec["split"] not in splits or spec.get("conversations") == []:
                continue
            try:
                mission = self.corpus.missions[name]
                rows = self.examples.build(
                    mission,
                    spec,
                    GameState.fixed(self.config.get("initial_state", {})),
                )
                if not rows:
                    raise ValueError("No reachable targets")
                built.extend(rows)
            except (ValueError, KeyError) as error:
                failures.append({"mission": name, "error": str(error)})
        if failures:
            raise ValueError(json.dumps({"build_failures": failures}, indent=2))
        if not built:
            raise ValueError("No missions selected")
        check_mission_splits([row[0] for row in built])
        write_dataset(output, built, self.config, self.corpus)
        return {
            "examples": len(built),
            "missions": len({r[0]["metadata"]["sources"][0]["source_group"] for r in built}),
        }
