"""Build state-consistent scene examples through one preparation pipeline."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from endless_voices.context import ContextPool, digest, strategy_from_config
from endless_voices.contracts import validate_record
from endless_voices.game_variables import mission_values, render
from endless_voices.instructions import CONTINUATION_INSTRUCTION
from endless_voices.splits import check_mission_splits

from .dialogue import DialogueInterpreter, Passage
from .source import walk
from .state import GameState, UnsupportedOperation, apply, condition


@dataclass
class History:
    state: GameState
    blocks: list[dict]


class ContextSampler:
    """Construct compatible histories before applying a token-budget strategy."""

    def __init__(self, corpus, interpreter, assignments, strategy, player, max_routes=1000):
        self.corpus = corpus
        self.interpreter = interpreter
        self.assignments = assignments
        self.strategy = strategy
        self.player = player
        self.max_routes = max_routes
        self.graph = {name: self.prerequisites(m.node) for name, m in corpus.missions.items()}

    @staticmethod
    def prerequisites(node):
        result = set()
        for child in node["children"]:
            if child["tokens"] != ["to", "offer"]:
                continue
            for term, parents in walk(child["children"]):
                if any(p["tokens"] == ["or"] for p in parents):
                    continue
                t = term["tokens"]
                if len(t) == 2 and t[0] == "has" and t[1].endswith(": done"):
                    result.add(t[1][:-6])
        return sorted(result)

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
        if len(routes) > self.max_routes:
            raise ValueError("History route limit exceeded; no branches were silently discarded")
        return routes

    @staticmethod
    def event_parts(nodes):
        conversations = [n for n in nodes if n["tokens"][:1] == ["conversation"]]
        dialogs = [n for n in nodes if n["tokens"][:1] == ["dialog"]]
        if len(conversations) > 1 or len(dialogs) > 1:
            raise UnsupportedOperation("Multiple dialogue entries in one mission event")
        if conversations and not conversations[0]["children"]:
            raise UnsupportedOperation("Named conversation references require resolution")
        actions = [
            n for n in nodes if n["tokens"][:1] not in (["conversation"], ["dialog"], ["log"])
        ]
        return conversations, dialogs, actions

    def event(self, nodes, histories, mission):
        """Execute a mission event with actions after the initial conversation display."""
        conversations, dialogs, actions = self.event_parts(nodes)
        updated = []
        for history in histories:
            if conversations:
                for route in self.interpreter.histories(conversations[0], history.state, actions):
                    if route.stop in {"decline", "defer", "flee", "die", "explode"}:
                        continue
                    block = self.block(mission, (*route.prefix, *route.paragraphs))
                    updated.append(History(route.state, [*history.blocks, block]))
            else:
                blocks = list(history.blocks)
                if dialogs:
                    node = dialogs[0]
                    if len(node["tokens"]) != 2 or node["children"]:
                        raise UnsupportedOperation("Only literal mission dialogs are supported")
                    blocks.append(self.block(mission, [Passage(node["line"], node["tokens"][1])]))
                updated.append(History(apply(actions, history.state), blocks))
        return self.bounded(updated)

    def block(self, mission, paragraphs):
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        return {
            "mission": mission.name,
            "heading": mission.name,
            "path": mission.path,
            "passages": [{**asdict(p), "text": render(p.text, values)[0]} for p in paragraphs],
        }

    def eligible(self, name, split):
        owner = self.assignments.get(name)
        return (
            owner is not None
            and {"train": 0, "validation": 1, "test": 2}[owner]
            <= {"train": 0, "validation": 1, "test": 2}[split]
        )

    def histories(self, mission, split, initial):
        histories, omitted = [History(initial, [])], []
        for name in self.ancestors(mission.name):
            if not self.eligible(name, split):
                omitted.append(name)
                continue
            previous = self.corpus.missions[name]
            requirements = [
                c
                for n in previous.node["children"]
                if n["tokens"] == ["to", "offer"]
                for c in n["children"]
            ]
            candidates = []
            for h in histories:
                state = h.state.assume(condition(requirements, h.state))
                if state is not None:
                    candidates.append(History(state, h.blocks))
            histories = candidates
            for phase in ["offer", "accept", "complete"]:
                events = [n for n in previous.node["children"] if n["tokens"] == ["on", phase]]
                for event in events:
                    histories = self.event(event["children"], histories, previous)
            for history in histories:
                history.state = apply(
                    [
                        {
                            "line": previous.node["line"],
                            "tokens": ["set", name + ": done"],
                            "children": [],
                        }
                    ],
                    history.state,
                )
        return histories, omitted

    def select(self, sample_id, conversation_id, mission, history, prefix, lore, counter):
        values = mission_values(mission.node, self.corpus.planet_systems, self.player)
        encounter = "\n\n".join(render(p.text, values)[0] for p in prefix)
        system = (
            CONTINUATION_INSTRUCTION + ("\n\n" + lore if lore else "") + "\n\nEarlier passages:\n"
        )
        pool = ContextPool(
            sample_id,
            conversation_id,
            mission.name,
            system,
            [{"role": "user", "content": encounter or "Continue the scene."}],
            history.blocks,
            self.graph,
            values,
        )
        return self.strategy.select(pool, counter)


class ExampleBuilder:
    """Join every reachable target with a reproducibly selected compatible history."""

    def __init__(self, corpus, interpreter, sampler, seed):
        self.corpus, self.interpreter, self.sampler, self.seed = corpus, interpreter, sampler, seed

    def build(self, mission, specification, counter, initial):
        split = specification["split"]
        histories, omitted = self.sampler.histories(mission, split, initial)
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
        lore = "\n\n".join(
            p["text"] for name in sorted(lore_names) for p in self.corpus.descriptions.get(name, [])
        )
        for phase in ["offer", "accept", "complete"]:
            for event in mission.node["children"]:
                if event["tokens"] != ["on", phase]:
                    continue
                conversations, _, actions = self.sampler.event_parts(event["children"])
                if not conversations:
                    histories = self.sampler.event(event["children"], histories, mission)
                    continue
                node = conversations[0]
                next_histories = []
                for history in histories:
                    routes = self.interpreter.continuations(node, history.state, actions)
                    for route in routes:
                        if route.terminal and route.stop not in {
                            "decline",
                            "defer",
                            "flee",
                            "die",
                            "explode",
                        }:
                            block = self.sampler.block(mission, (*route.prefix, *route.paragraphs))
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
                        rank = digest(
                            [
                                self.seed,
                                history.blocks,
                                [asdict(p) for p in route.prefix],
                                route.state.snapshot(),
                            ]
                        )
                        candidate = (rank, node, route, history)
                        if key not in variants or rank < variants[key][0]:
                            variants[key] = candidate
                histories = self.sampler.bounded(next_histories)
        found = {candidate[1]["line"] for candidate in variants.values()}
        if selected_conversations and set(selected_conversations) - found:
            raise ValueError(
                "Requested conversations were unreachable or use unsupported mission phases: "
                + str(sorted(set(selected_conversations) - found))
            )
        examples = []
        for key, (_, node, route, history) in sorted(variants.items()):
            sid = "scene-" + key[:24]
            cid = "conversation-" + digest([mission.path, node["line"]])[:24]
            selection = self.sampler.select(sid, cid, mission, history, route.prefix, lore, counter)
            target = "\n\n".join(render(p.text, values)[0] for p in route.paragraphs)
            target_lines = {p.line for p in route.paragraphs}
            if any(p.line in target_lines for p in route.prefix):
                raise ValueError("Target source paragraph appears in its own encounter")
            for b in selection.selected:
                if b["path"] == mission.path and any(
                    p["line"] in target_lines for p in b["passages"]
                ):
                    raise ValueError("Target source paragraph appears in sampled history")
            ref = {
                "reference": "https://github.com/endless-sky/endless-sky/blob/"
                + self.corpus.revision
                + "/"
                + quote(mission.path)
                + f"#L{node['line']}",
                "revision": self.corpus.revision,
                "source_group": "mission / " + mission.name,
            }
            meta = {
                k: deepcopy(specification[k])
                for k in ["identity", "species", "character_role", "topics", "scenario_group"]
            }
            meta.update(
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
                "messages": selection.training_messages(target),
                "evaluation": {"dimensions": ["authenticity"], "sources": [ref]},
            }
            validate_record(record)
            provenance = {
                "state": route.state.snapshot(),
                "source_path": mission.path,
                "source_sha256": mission.sha256,
                "source_revision": self.corpus.revision,
                "target_lines": sorted(target_lines),
                "stop": route.stop,
                "omitted_history_missions": omitted,
                "prefix_lines": [p.line for p in route.prefix],
            }
            examples.append((record, selection, provenance, route.paragraphs))
        return examples


class DatasetBuilder:
    """Own parsing dependencies, split eligibility, example construction and publication."""

    def __init__(self, corpus, config, counter):
        self.corpus, self.config, self.counter = corpus, deepcopy(config), counter
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
        self.sampler = ContextSampler(
            corpus,
            self.interpreter,
            assignments,
            strategy_from_config(config["context"]),
            config["player"],
            config.get("max_history_routes", 1000),
        )
        self.examples = ExampleBuilder(corpus, self.interpreter, self.sampler, config["seed"])

    def build(self, output, splits=("train",)):
        """Publish a complete prepared bundle only after every requested mission succeeds."""
        from .storage import write_dataset

        output = Path(output)
        if output.exists():
            raise ValueError("Output exists; choose a new dataset directory")
        built, failures = [], []
        for name, spec in sorted(self.config["missions"].items()):
            if spec["split"] not in splits:
                continue
            try:
                mission = self.corpus.missions[name]
                rows = self.examples.build(
                    mission,
                    spec,
                    self.counter,
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
