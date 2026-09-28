"""Measure mission-depth context selection using cached validation drafts and tokenizer."""

import hashlib
import json
from collections import deque
from pathlib import Path

from assemble_validation_context import dependency_terms, sha
from prepare_conversations import tree, walk

ROOT = Path(__file__).resolve().parents[1]
SEED = "context-depth-v1"
BUDGET = 8000


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


def sample_older(groups, key, count, budget):
    """Select whole mission groups in seeded order within the rendered token budget."""
    order = sorted(groups, key=lambda m: hashlib.sha256(f"{SEED}:{key}:{m}".encode()).hexdigest())
    chosen = set()
    for mission in order:
        candidate = chosen | {mission}
        if count(candidate) <= budget:
            chosen = candidate
    return chosen


def render(block):
    return block["heading"] + "\n" + "\n\n".join(
        ("Optional player response: " if p["role"] == "option" else "") + p["text"]
        for p in block["passages"]
    )


def main():
    output = ROOT / "outputs/context-depth-v1"
    if output.exists():
        raise ValueError("Output exists; choose a new experiment version")
    source = ROOT / "data/local/endless-sky-7140eb2a29ce"
    inventory_path = ROOT / "data/overview/source-statistics.json"
    inventory = json.loads(inventory_path.read_text())
    missions, writers = {}, {}
    for entry in inventory["files"]:
        if not entry["path"].startswith(("data/human/", "data/hai/", "data/quarg/")):
            continue
        path = source / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Source hash mismatch")
        for node in tree(path.read_text()):
            if node["tokens"][:1] != ["mission"]:
                continue
            name = node["tokens"][1]
            missions[name] = node
            for item, parents in walk(node["children"]):
                if item["tokens"][:1] not in (["set"], ["event"]):
                    continue
                phase = next((p["tokens"][1] for p in parents
                              if p["tokens"][:1] == ["on"]), None)
                if phase in ("offer", "complete"):
                    term = ("event: " if item["tokens"][0] == "event" else "") + item["tokens"][1]
                    writers.setdefault(term, set()).add(name)
    graph = {}
    for name, node in missions.items():
        parents = set()
        for term in dependency_terms(node):
            if term.endswith(": done"):
                parents.add(term[:-6])
            elif term.endswith(": offered"):
                parents.add(term[:-9])
            else:
                parents.update(writers.get(term, set()))
        graph[name] = sorted(parents)
    from transformers import AutoTokenizer

    config_path = ROOT / "outputs/generator-screen-v1/gemma31b-config-v2.json"
    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["path"], local_files_only=True)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    context_path = ROOT / "outputs/validation-source-context-v2/contexts.jsonl"
    baseline_path = ROOT / "outputs/validation-source-context-v2/gemma-counts.json"
    baseline = {r["sample_id"]: r["input_tokens"]
                for r in json.loads(baseline_path.read_text())["samples"]}
    rows, selections = [], []
    for line in context_path.read_text().splitlines():
        row = json.loads(line)
        ds = distances(graph, row["mission"])
        blocks = [b for b in row["source_blocks"] if b["kind"] == "earlier-source-examples"]
        names = [b["heading"].rsplit(" / ", 1)[0] for b in blocks]
        if any(name not in ds for name in names):
            raise ValueError("Existing context mission is absent from prerequisite graph")
        rendered = [render(b) for b in blocks]
        full = "\n\n".join(rendered)
        suffix = ("Earlier game passages. Optional alternatives are examples, "
                  "not simultaneous events.\n\n")
        system = row["messages"][0]["content"]
        if not system.endswith(suffix + full):
            raise ValueError("Unexpected context serialization")
        prefix = system[:len(system) - len(full)] if full else system
        for depth in (4, 8, 12):
            near = {name for name in names if ds[name] <= depth}
            old = set(names) - near
            def old_text(selected):
                return "\n\n".join(text for name, text in zip(names, rendered) if name in selected)
            chosen = sample_older(old, row["conversation_id"],
                                  lambda selected: count(old_text(selected)), BUDGET)
            selected = near | chosen
            messages = [dict(m) for m in row["messages"]]
            messages[0]["content"] = prefix + old_text(selected)
            encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                                     add_generation_prompt=True)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            rows.append({"sample_id": row["sample_id"], "mission": row["mission"],
                         "depth": depth, "baseline_tokens": baseline[row["sample_id"]],
                         "input_tokens": len(ids), "near_history_tokens": count(old_text(near)),
                         "sampled_history_tokens": count(old_text(chosen)),
                         "near_missions": len(near), "sampled_missions": len(chosen),
                         "omitted_missions": len(old - chosen)})
            selections.append({"sample_id": row["sample_id"], "depth": depth,
                               "mission_distances": ds, "full_missions": sorted(near),
                               "sampled_missions": sorted(chosen), "messages": messages})
    output.mkdir()
    (output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "selections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selections))
    (output / "provenance.json").write_text(json.dumps({
        "seed": SEED, "older_history_budget": BUDGET, "source_revision": inventory["revision"],
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in
                   (inventory_path, config_path, context_path, baseline_path, Path(__file__))},
        "tokenizer_hashes": json.loads(baseline_path.read_text())["tokenizer_hashes"],
        "tokenizer_revision": config["revision"],
    }, indent=2) + "\n")
    print(f"Measured {len(rows)} contexts; no model calls")


if __name__ == "__main__":
    main()
