"""Select eligible story context once for dataset preparation and benchmarking."""

import hashlib
import json
import re
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from endless_voices.artifacts import encoded
from endless_voices.messages import validate_messages


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()



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
        ("Player choice: " if p["role"] == "option" else "") + p["text"]
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
        self._count = lru_cache(maxsize=2048)(self.text)
        self._boundaries = None
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None and not getattr(tokenizer, "split_special_tokens", False):
            added = json.loads(backend.to_str())["added_tokens"]
            if added and all(not any(t[k] for k in (
                    "single_word", "lstrip", "rstrip", "normalized")) for t in added):
                self._boundaries = re.compile("(" + "|".join(
                    re.escape(t["content"]) for t in sorted(
                        added, key=lambda t: len(t["content"]), reverse=True)) + ")")

    def text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, messages):
        if self._boundaries is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # Added tokens isolate sections before the tokenizer processes ordinary text.
            parts = self._boundaries.split(prompt)
            return sum(1 if i % 2 else self._count(part)
                       for i, part in enumerate(parts) if part)
        return self.exact_messages(messages)

    def exact_messages(self, messages):
        """Count the complete prompt without section caching."""
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
    references: list[dict] = field(default_factory=list)

    def history(self, missions):
        text = "\n\n".join(render(b) for b in self.blocks if b["mission"] in missions)
        return text

    def messages(self, missions, references=()):
        reference_text = ""
        if references:
            reference_text = (
                "\n\nOther game scenes (independent writing references):\n"
                + "\n\n".join(render(block) for block in references)
            )
        return [{"role": "system", "content": self.system_prefix + self.history(missions)
                 + reference_text},
                *deepcopy(self.encounter)]


@dataclass
class Selection:
    """Record the fixed context and private selection evidence."""

    sample_id: str
    messages: list[dict]
    selected: list[dict]
    omitted: list[dict]
    token_counts: dict[str, int]
    provenance: dict

    def evidence(self):
        """Return selection settings and source coordinates without duplicating text."""
        def coordinates(blocks, include_state=False):
            return [{"mission": b["mission"], "path": b["path"],
                     "lines": [p["line"] for p in b["passages"]],
                     **({"reference": True,
                         **({"state": b["state"]} if include_state else {}),
                         "source_kind": b.get("source_kind", "conversation")}
                        if b.get("reference") else {})} for b in blocks]

        return {**deepcopy(self.provenance), "token_counts": dict(self.token_counts),
                "selected": coordinates(self.selected, include_state=True),
                "omitted": coordinates(self.omitted)}

    def training_messages(self, target):
        messages = [*deepcopy(self.messages), {"role": "assistant", "content": target}]
        validate_messages(messages)
        return messages


class ContextStrategy(Protocol):
    """Select context from an eligible pool without access to target answers."""

    def select(self, pool: ContextPool, counter: TokenCounter) -> Selection: ...


def result(pool, counter, config, full, sampled, ds, references=()):
    chosen = full | sampled
    messages = pool.messages(chosen, references)
    return Selection(
        pool.sample_id, messages,
        deepcopy([b for b in pool.blocks if b["mission"] in chosen] + list(references)),
        deepcopy([b for b in pool.blocks if b["mission"] not in chosen]
                 + [b for b in pool.references if b not in references]),
        {"input": counter.messages(messages), "full_history": counter.text(pool.history(full)),
         "sampled_history": counter.text(pool.history(sampled))},
        {"strategy": config, "pool_sha256": digest(pool.__dict__),
         "messages_sha256": digest(messages), "mission_distances": ds,
         "full_missions": sorted(full), "sampled_missions": sorted(sampled),
         "game_variables": dict(pool.variables),
         "variable_policy": "mission-scoped"},
    )


@dataclass(frozen=True)
class FullContext:
    """Retain every eligible history block."""

    def select(self, pool, counter):
        return result(pool, counter, {"name": "full"},
                      {b["mission"] for b in pool.blocks}, set(), {}, pool.references)


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
        references = []
        for block in sorted(pool.references, key=lambda b: (
            b["mission"] is None,
            digest([self.seed, pool.conversation_id, b["path"], b["conversation"]]),
        )):
            candidate = [*references, block]
            if counter.messages(pool.messages(near | chosen, candidate)) <= self.max_input_tokens:
                references = candidate
        selection = result(
            pool, counter, {"name": "mission-depth", **self.__dict__}, near, chosen, ds,
            references,
        )
        if isinstance(counter, TokenizerCounter):
            if counter.exact_messages(selection.messages) != selection.token_counts["input"]:
                raise ValueError("Cached token count differs from the complete prompt")
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
