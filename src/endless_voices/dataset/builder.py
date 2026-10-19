"""Build state-consistent scene examples through one preparation pipeline."""

import json
import re
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

from endless_voices.artifacts import fingerprint as digest
from endless_voices.context import ContextPool
from endless_voices.context import render as render_block
from endless_voices.contracts import validate_record
from endless_voices.gameplay.dialogue import DialogueInterpreter, Passage
from endless_voices.gameplay.histories import History, HistoryReconstructor
from endless_voices.gameplay.source import tokens
from endless_voices.gameplay.state import GameState
from endless_voices.gameplay.variables import render
from endless_voices.instructions import SCENE_INSTRUCTIONS, scene_task
from endless_voices.retrieval import scene_id
from endless_voices.splits import check_chain_splits, check_mission_splits


class ContextBuilder:
    """Assemble dataset context and writing references from gameplay evidence."""

    def __init__(self, corpus, interpreter, assignments, player, max_routes=1000,
                 seed="context"):
        self.corpus = corpus
        self.assignments = assignments
        self.player = player
        self.gameplay = HistoryReconstructor(corpus, interpreter, player, max_routes, seed)
        self.graph = self.gameplay.graph
        self.conversation_owners = self._conversation_owners(assignments)
        self.named_users = corpus.named_conversation_users()
        self.reference_cache = {}
        self.reference_paths = {}
        self.reference_exclusions = {}

    def _conversation_owners(self, assignments):
        """Validate shared conversation splits and return each definition's owner."""
        owners = {}
        for name, missions in self.corpus.named_conversation_users().items():
            splits = {assignments[m] for m in missions if m in assignments}
            if len(splits) > 1:
                raise ValueError(f"Named conversation {name!r} crosses mission splits")
            if splits:
                owners[name] = splits.pop()
        return owners

    def eligible(self, name, split):
        owner = self.assignments.get(name)
        return (
            owner is None
            or {"train": 0, "validation": 1, "test": 2}[owner]
            <= {"train": 0, "validation": 1, "test": 2}[split]
        )

    def references(self, mission, split, history):
        """Return independent reference scenes from eligible related missions."""
        excluded = {mission.name, *(b["mission"] for b in history.blocks)}
        excluded.update(self.gameplay.ancestors(mission.name))
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
                        conversations, _, actions = self.gameplay.event_parts(
                            event["children"], other)
                        if not conversations:
                            continue
                        state = GameState(events=self.corpus.events, scope=other.name,
                                          planet_systems=self.corpus.planet_systems)
                        self.gameplay.enter(state, other)
                        node = conversations[0]
                        branches = {}
                        for route in self.gameplay.interpreter.continuations(
                                node, state, actions, future_variables=set()):
                            if not route.paragraphs:
                                continue
                            key = tuple(p.line for p in route.paragraphs)
                            if key in branches:
                                continue
                            block = self.gameplay.block(
                                other, (*route.prefix, *route.paragraphs), node,
                                event["tokens"][1], state)
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
            history_state = GameState(substitutions={entry["mission"]: entry["substitutions"]})
            blocks.append(self.gameplay.block(
                self.corpus.missions[entry["mission"]], paragraphs,
                {"source_path": entry["path"]}, entry["phase"], history_state))
        state = GameState(world=tuple(evidence["encounter_state"]["world_changes"]),
                          substitutions=evidence["encounter_state"].get("substitutions", {}))
        values = self.gameplay.values(mission, state)
        pool = ContextPool(meta["id"], meta["conversation_id"], mission.name,
                           record["messages"][0]["content"], record["messages"][1:-1],
                           blocks, self.graph, values,
                           self.references(mission, meta["split"], History(state, blocks)))
        self.add_planets(pool, mission, state.world,
                         self.specifications[mission.name].get("lore_planets", []))
        return pool

    def add_planets(self, pool, mission, world, lore_names=()):
        """Add named and region-relevant planets visible in this encounter's world."""
        values = pool.variables
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
        values = self.gameplay.values(mission, history.state)
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
    """Format reconstructed encounters as dataset records and provenance."""

    def __init__(self, corpus, sampler):
        self.corpus, self.sampler = corpus, sampler

    def build(self, mission, specification, initial):
        split = specification["split"]
        excluded = {name for name in self.corpus.missions
                    if not self.sampler.eligible(name, split)}
        encounters, omitted = self.sampler.gameplay.encounters(
            mission, initial, excluded=excluded,
            selected_conversations=specification.get("conversations"))
        lore_names = set(specification.get("lore_planets", []))
        examples = []
        for encounter in encounters:
            key, node, route, history, phase = (
                encounter.key, encounter.node, encounter.route, encounter.history, encounter.phase)
            values = self.sampler.gameplay.values(mission, history.state)
            encounter_lore = lore_names | {values[k] for k in ("<planet>", "<origin>")
                                           if k in values}
            sid = "scene-" + key[:24]
            cid = "conversation-" + digest(
                [node.get("source_path", mission.path), node["line"]])[:24]
            planets = self.corpus.planet_references(history.state.world)
            lore = "\n\n".join(render_block(block) for block in planets
                                 if block["name"] in encounter_lore)
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
            if values:
                cue += "\nMission values:\n" + "\n".join(
                    f"{marker[1:-1]}: {value}" for marker, value in sorted(values.items()))
            links = []
            for name in [*self.sampler.gameplay.ancestors(mission.name), mission.name]:
                for choices in self.sampler.gameplay.dependencies[name].values():
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
                             "path": block["path"], "substitutions": block["substitutions"],
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
        self.examples = ExampleBuilder(corpus, self.sampler)

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
