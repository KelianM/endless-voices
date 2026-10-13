"""Represent eligible story context and count complete generation prompts."""

import hashlib
import json
import re
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from endless_voices.artifacts import encoded


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
    heading = block["heading"] + (f" — on {block['phase']}" if "phase" in block else "")
    return heading + "\n" + "\n\n".join(
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

    def messages(self, blocks=(), references=()):
        """Render selected history and references with the unchanged visible encounter."""
        history = ""
        if blocks:
            history = "\n\nEarlier passages:\n" + "\n\n".join(render(block) for block in blocks)
        reference_text = ""
        if references:
            reference_text = (
                "\n\nOther game scenes (independent writing references):\n"
                + "\n\n".join(render(block) for block in references)
            )
        return [{"role": "system", "content": self.system_prefix + history + reference_text},
                *deepcopy(self.encounter)]
