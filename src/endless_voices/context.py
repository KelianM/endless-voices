"""Select eligible story context once for dataset preparation and benchmarking."""

import hashlib
import json
import re
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from endless_voices.messages import validate_messages


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()



def substitute_variables(text, values):
    """Replace configured markers once while leaving unknown markers visible."""
    return re.sub(r"<[^>]+>", lambda match: values.get(match[0], match[0]), text)


def variable_values(config):
    """Validate and return configured game placeholder values."""
    values = config["values"]
    if not isinstance(values, dict) or any(
        not isinstance(k, str) or re.fullmatch(r"<[^>]+>", k) is None
        or not isinstance(v, str) for k, v in values.items()
    ):
        raise ValueError("Game variables must map <marker> strings to string values")
    return dict(values)

def distances(graph, start):
    """Return shortest prerequisite distances, counting one edge per mission."""
    result = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for parent in graph.get(node, []):
            if parent not in result:
                result[parent] = result[node] + 1
                queue.append(parent)
    return result


def render(block):
    """Render an original source block without changing its passages."""
    return block["heading"] + "\n" + "\n\n".join(
        ("Optional player response: " if p["role"] == "option" else "") + p["text"]
        for p in block["passages"]
    )


class TokenCounter(Protocol):
    """Measure text and complete generation prompts with one tokenizer."""

    def text(self, text: str) -> int: ...
    def messages(self, messages: list[dict]) -> int: ...


class TokenizerCounter:
    """Adapt a caller-supplied tokenizer without loading models or downloading files."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, messages):
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)


@dataclass
class ContextPool:
    """Hold already-eligible history, a fixed system prefix and the current encounter."""

    sample_id: str
    conversation_id: str
    mission: str
    system_prefix: str
    encounter: list[dict]
    blocks: list[dict]
    graph: dict[str, list[str]]
    variables: dict[str, str] = field(default_factory=dict)

    variables_resolved: bool = False

    def history(self, missions):
        text = "\n\n".join(render(b) for b in self.blocks if b["mission"] in missions)
        return text if self.variables_resolved else substitute_variables(text, self.variables)

    def messages(self, missions):
        values = {} if self.variables_resolved else self.variables
        return [{"role": "system", "content": substitute_variables(
            self.system_prefix, values) + self.history(missions)},
                *[{**m, "content": substitute_variables(m["content"], values)}
                  for m in self.encounter]]


@dataclass
class Selection:
    """Record the fixed context and private selection evidence."""

    sample_id: str
    messages: list[dict]
    selected: list[dict]
    omitted: list[dict]
    token_counts: dict[str, int]
    provenance: dict

    def generation_prompt(self):
        return {"sample_id": self.sample_id, "messages": deepcopy(self.messages),
                "messages_sha256": digest(self.messages)}

    def judge_context(self):
        return deepcopy(self.messages)

    def training_messages(self, target):
        messages = [*deepcopy(self.messages), {"role": "assistant", "content": target}]
        validate_messages(messages)
        return messages


class ContextStrategy(Protocol):
    """Select context from an eligible pool without access to target answers."""

    def select(self, pool: ContextPool, counter: TokenCounter) -> Selection: ...


def result(pool, counter, config, full, sampled, ds):
    chosen = full | sampled
    messages = pool.messages(chosen)
    return Selection(
        pool.sample_id, messages,
        deepcopy([b for b in pool.blocks if b["mission"] in chosen]),
        deepcopy([b for b in pool.blocks if b["mission"] not in chosen]),
        {"input": counter.messages(messages), "full_history": counter.text(pool.history(full)),
         "sampled_history": counter.text(pool.history(sampled))},
        {"strategy": config, "pool_sha256": digest(pool.__dict__),
         "messages_sha256": digest(messages), "mission_distances": ds,
         "full_missions": sorted(full), "sampled_missions": sorted(sampled),
         "game_variables": dict(pool.variables),
         "variable_policy": "mission-scoped" if pool.variables_resolved else "legacy-global"},
    )


@dataclass(frozen=True)
class FullContext:
    """Retain every eligible history block."""

    def select(self, pool, counter):
        return result(pool, counter, {"name": "full"},
                      {b["mission"] for b in pool.blocks}, set(), {})


@dataclass(frozen=True)
class MissionDepth:
    """Preserve nearby missions and fill the remaining complete-input budget."""

    depth: int
    max_input_tokens: int
    seed: str

    def __post_init__(self):
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be a positive integer")
        if not isinstance(self.seed, str) or not self.seed:
            raise ValueError("seed must be a nonempty string")

    def select(self, pool, counter):
        ds = distances(pool.graph, pool.mission)
        names = {b["mission"] for b in pool.blocks}
        if names - ds.keys():
            raise ValueError("Context mission is absent from prerequisite graph")
        near = {name for name in names if ds[name] <= self.depth}
        core_tokens = counter.messages(pool.messages(near))
        if core_tokens > self.max_input_tokens:
            raise ValueError(
                f"{pool.sample_id}: preserved context requires {core_tokens} input tokens; "
                f"ceiling is {self.max_input_tokens}; no context was truncated"
            )
        order = sorted(names - near, key=lambda m: hashlib.sha256(
            f"{self.seed}:{pool.conversation_id}:{m}".encode()).hexdigest())
        chosen = set()
        for mission in order:
            candidate = chosen | {mission}
            if counter.messages(pool.messages(near | candidate)) <= self.max_input_tokens:
                chosen = candidate
        selection = result(
            pool, counter, {"name": "mission-depth", **self.__dict__}, near, chosen, ds
        )
        selection.token_counts.update(
            preserved_input=core_tokens,
            remaining_input_budget=self.max_input_tokens - selection.token_counts["input"],
        )
        return selection


def strategy_from_config(config) -> ContextStrategy:
    """Build a supported strategy, rejecting misspelled or irrelevant settings."""
    if config == {"name": "full"}:
        return FullContext()
    if set(config) == {"name", "depth", "max_input_tokens", "seed"}:
        if config["name"] == "mission-depth":
            return MissionDepth(config["depth"], config["max_input_tokens"], config["seed"])
    raise ValueError("Invalid context strategy configuration")


def pool_from_draft(draft, graph, variables=None):
    """Adapt source-context drafts while preserving fixed lore and encounter messages."""
    variables = draft.get("game_variables", variables)
    blocks = [deepcopy(b) for b in draft["source_blocks"] if b["kind"] == "earlier-source-examples"]
    for block in blocks:
        block["mission"] = block["heading"].rsplit(" / ", 1)[0]
    history = "\n\n".join(map(render, blocks))
    marker = ("Earlier game passages. Optional alternatives are examples, "
              "not simultaneous events.\n\n")
    messages = draft["messages"]
    if messages[0]["role"] != "system" or not messages[0]["content"].endswith(marker + history):
        raise ValueError("Draft history does not match its rendered messages")
    validate_messages([*messages, {"role": "assistant", "content": "Boundary check"}])
    system = messages[0]["content"]
    return ContextPool(draft["sample_id"], draft["conversation_id"], draft["mission"],
                       system[:-len(history)] if history else system,
                       deepcopy(messages[1:]), blocks, deepcopy(graph), dict(variables or {}),
                       "game_variables" in draft)
