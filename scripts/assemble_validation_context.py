"""Assemble uncapped source-context drafts and coverage diagnostics for validation."""

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path
from urllib.parse import unquote

from prepare_conversations import flow_graph, tree, walk

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_terms(node):
    """Return positive offer requirements, excluding optional OR and negative branches."""
    terms = []
    for child in node["children"]:
        if child["tokens"] != ["to", "offer"]:
            continue
        for item, parents in walk(child["children"]):
            if any(p["tokens"][:1] in (["or"], ["not"]) for p in parents):
                continue
            if item["tokens"][:1] == ["has"]:
                terms.append(item["tokens"][1])
    return terms


def route_prefix(conversation, anchors, target_line):
    """Find a structural path through recorded dialogue anchors, stopping before the target."""
    graph = flow_graph(conversation)["links"]
    start = conversation["children"][0]["line"]
    anchors = list(dict.fromkeys([*anchors, target_line]))
    queue = deque([(start, 0, [])])
    seen = set()
    while queue:
        line, index, path = queue.popleft()
        if index < len(anchors) and line == anchors[index]:
            index += 1
        if (line, index) in seen:
            continue
        seen.add((line, index))
        if line == target_line:
            if index == len(anchors):
                return path
            continue
        # Future paragraphs must not be used to loop back into the selected question.
        if line > target_line:
            continue
        queue.extend((n, index, [*path, line]) for n in graph.get(str(line), []))
    raise ValueError("No source path through the recorded anchors")


def source_spans(metadata, revision):
    result = []
    for m in metadata:
        for s in m["sources"]:
            if not s["source_group"].startswith("mission /"):
                continue
            match = re.search(r"/blob/" + revision + r"/(.+)#L(\d+)(?:-L(\d+))?$", s["reference"])
            if not match:
                raise ValueError("Unrecognized held-out source range")
            result.append((unquote(match[1]), int(match[2]), int(match[3] or match[2])))
    return result


def prose(node):
    """Return original prose with its branch and condition ancestry."""
    result = []
    for n, parents in walk([node]):
        tokens = n["tokens"]
        if n["raw"].lstrip().startswith("`"):
            if any(c["tokens"][:1] in (["decline"], ["defer"]) for c in n["children"]):
                continue
            text = tokens[0]
        elif tokens[:1] == ["dialog"] and len(tokens) == 2:
            text = tokens[1]
        else:
            continue
        result.append(
            {
                "line": n["line"],
                "text": text,
                "role": "option" if any(p["tokens"] == ["choice"] for p in parents) else "passage",
                "controls": [c["tokens"] for c in n["children"]],
                "ancestry": [p["tokens"] for p in parents if not p["raw"].lstrip().startswith("`")],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    invpath = ROOT / "data/overview/source-statistics.json"
    inv = json.loads(invpath.read_text())
    revision = inv["revision"]
    index, missions, writers, files = {}, {}, {}, {}
    for entry in inv["files"]:
        path = entry["path"]
        if (
            not path.startswith(("data/human/", "data/hai/", "data/quarg/"))
            and path != "data/map planets.txt"
        ):
            continue
        file = args.source / path
        if sha(file) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        files[path] = entry["sha256"]
        nodes = tree(file.read_text())
        index[path] = nodes
        for node in nodes:
            if node["tokens"][:1] != ["mission"]:
                continue
            name = node["tokens"][1]
            if name in missions:
                raise ValueError(f"Duplicate mission: {name}")
            missions[name] = (path, node)
            for n, parents in walk(node["children"]):
                if n["tokens"][:1] not in (["set"], ["event"]):
                    continue
                action = next((p["tokens"][1] for p in parents if p["tokens"][:1] == ["on"]), None)
                if action in ("offer", "complete"):
                    term = ("event: " if n["tokens"][0] == "event" else "") + n["tokens"][1]
                    writers.setdefault(term, []).append(
                        (name, "done" if action == "complete" else "offered")
                    )
    sample_root = ROOT / "data/pilot-v1/samples"
    manifest = json.loads((sample_root / "manifest.json").read_text())
    records, test_metadata = {}, []
    for split in ("train", "validation", "test"):
        for entry in manifest["files"][split]:
            file = sample_root / entry["path"]
            if sha(file) != entry["sha256"]:
                raise ValueError("Pilot hash mismatch")
            for line in file.read_text().splitlines():
                r = json.loads(line)
                if split == "test":
                    test_metadata.append(r["metadata"])
                else:
                    records[r["metadata"]["id"]] = r
    blocked = source_spans(test_metadata, revision)
    validation = {k: v for k, v in records.items() if v["metadata"]["split"] == "validation"}
    provenance_path = ROOT / "data/pilot-v1/evidence/provenance.json"
    provenance = {
        p["id"]: p for p in json.loads(provenance_path.read_text()) if p["id"] in validation
    }

    def closure(name):
        ordered, seen, gaps, alternatives = [], set(), set(), []

        def visit(mission, phase):
            key = (mission, phase)
            if key in seen:
                return
            seen.add(key)
            if mission not in missions:
                gaps.add("Unresolved mission: " + mission)
                return
            for offer in missions[mission][1]["children"]:
                if offer["tokens"] == ["to", "offer"]:
                    optional = [
                        n["tokens"][1]
                        for n, parents in walk(offer["children"])
                        if n["tokens"][:1] == ["has"]
                        and any(a["tokens"][:1] == ["or"] for a in parents)
                    ]
                    if optional:
                        gaps.add(
                            "Optional offer requirements not expanded for "
                            + mission
                            + ": "
                            + ", ".join(optional)
                        )
            for term in dependency_terms(missions[mission][1]):
                found = []
                for suffix in (": done", ": offered"):
                    if term.endswith(suffix):
                        found = [(term[: -len(suffix)], suffix[2:])]
                        break
                if not found:
                    found = sorted(set(writers.get(term, [])))
                if not found:
                    gaps.add(term)
                if len(found) > 1:
                    alternatives.append({"condition": term, "possible_producers": found})
                for parent, parent_phase in found:
                    visit(parent, parent_phase)
            ordered.append(key)

        visit(name, "current")
        return [k for k in ordered if k[0] != name], sorted(gaps), alternatives

    def permitted(path, line):
        return not any(p == path and lo <= line <= hi for p, lo, hi in blocked)

    planet_nodes = index["data/map planets.txt"]
    source_docs = {}
    for node in planet_nodes:
        if node["tokens"][:1] != ["planet"]:
            continue
        name = node["tokens"][1]
        attrs = [
            t
            for c in node["children"]
            if c["tokens"][:1] == ["attributes"]
            for t in c["tokens"][1:]
        ]
        selected = []
        for c in node["children"]:
            if c["tokens"][:1] != ["description"]:
                continue
            # Default Earth prose contains no player-birth assertion; the alternate does.
            if c["children"] and not (name == "Earth" and c["line"] == 1405):
                continue
            selected.append({"line": c["line"], "text": c["tokens"][1], "role": "description"})
        source_docs[name] = (attrs, selected)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    outputs, summary = [], []
    for sample_id, record in validation.items():
        meta, origin = record["metadata"], provenance[sample_id]
        path = origin["source_path"]
        mission = meta["sources"][0]["source_group"].removeprefix("mission / ")
        current = next(
            n
            for n, _ in walk(index[path])
            if n["tokens"][:1] == ["conversation"]
            and n["line"] == int(meta["conversation_id"].rsplit("-l", 1)[1])
        )
        target_line = origin["messages"][-1]["spans"][0]["line"]
        anchors = [s["line"] for msg in origin["messages"][1:-1] for s in msg.get("spans", [])]
        issues, omitted, chunks = [], [], []
        ancestors, gaps, alternatives = closure(mission)
        for previous, phase in ancestors:
            previous_path, node = missions[previous]
            passages = []
            for event in node["children"]:
                action = event["tokens"]
                allowed = action[:2] == ["on", "offer"] or (
                    phase == "done"
                    and (
                        action[:2] in (["on", "accept"], ["on", "complete"])
                        or action[:1] == ["npc"]
                    )
                )
                if allowed:
                    passages.extend(prose(event))
            keep = [p for p in passages if permitted(previous_path, p["line"])]
            if len(keep) != len(passages):
                omitted.append(
                    {"mission": previous, "excluded_test_passages": len(passages) - len(keep)}
                )
            if keep:
                chunks.append(
                    {
                        "heading": previous + " / " + phase,
                        "path": previous_path,
                        "passages": keep,
                        "kind": "earlier-source-examples",
                    }
                )
        # Earlier phases can supply narration missing from the pilot conversation.
        for event in missions[mission][1]["children"]:
            if (
                event["tokens"][:2] in (["on", "offer"], ["on", "accept"])
                and event["line"] < current["line"]
            ):
                passages = [
                    p
                    for p in prose(event)
                    if p["line"] < current["line"] and permitted(path, p["line"])
                ]
                if passages:
                    chunks.append(
                        {
                            "heading": mission + " / earlier phase",
                            "path": path,
                            "passages": passages,
                            "kind": "earlier-source-examples",
                        }
                    )
        try:
            selected_lines = route_prefix(current, anchors, target_line)
            by_line = {n["line"]: n for n, _ in walk([current])}
            current_passages = [
                {"line": line, "text": by_line[line]["tokens"][0], "role": "passage"}
                for line in selected_lines
                if by_line[line]["raw"].lstrip().startswith("`")
            ]
            branch_nodes = [
                line for line in selected_lines if by_line[line]["tokens"][:1] == ["branch"]
            ]
            if branch_nodes:
                issues.append("Current route includes game-state branches requiring review")
        except ValueError as e:
            current_passages = []
            issues.append(str(e))
        if any(m["origin"] == "agent" for m in origin["messages"][1:-1]):
            issues.append(
                "Pilot connective prompt omitted; confirm source narration establishes the turn"
            )
        identity = meta["identity"]
        if identity.startswith("human"):
            planets = ["Earth", "Bourne", "Glaze"]
        elif identity == "hai" or "hai" in identity:
            planets = [name for name, (attrs, _) in source_docs.items() if "hai" in attrs]
        else:
            planets = [name for name, (attrs, _) in source_docs.items() if "human quarg" in attrs]
        lore = [
            {
                "heading": name + " / planet description",
                "path": "data/map planets.txt",
                "passages": source_docs[name][1],
                "kind": "world-reference",
            }
            for name in planets
        ]
        if identity.startswith("human"):
            ships = index["data/human/ships.txt"]
            for node in ships:
                if node["tokens"][:2] in (
                    ["ship", "Frigate"],
                    ["ship", "Cruiser"],
                    ["ship", "Carrier"],
                ):
                    lore.append(
                        {
                            "heading": node["tokens"][1] + " / ship description",
                            "path": "data/human/ships.txt",
                            "kind": "world-reference",
                            "passages": [
                                {"line": c["line"], "text": c["tokens"][1], "role": "description"}
                                for c in node["children"]
                                if c["tokens"][:1] == ["description"]
                            ],
                        }
                    )
            if mission != "FW Katya 5B":
                war_path, war_node = missions["event: war begins"]
                paragraphs = [p for p in prose(war_node) if p["line"] == 185]
                lore.append(
                    {
                        "heading": "War-begins event / historical witness passage",
                        "path": war_path,
                        "passages": paragraphs,
                        "kind": "world-reference",
                    }
                )
            else:
                issues.append("War-begins exposition omitted: date not established for Katya route")

        def render(c):
            return (
                c["heading"]
                + "\n"
                + "\n\n".join(
                    ("Optional player response: " if p["role"] == "option" else "") + p["text"]
                    for p in c["passages"]
                )
            )

        system = (
            meta["character_role"] + ".\nContinue the conversation as the character, responding "
            "as authentically as possible for the given scenario.\n\nWorld reference:\n\n"
            + "\n\n".join(render(c) for c in lore)
            + "\n\nEarlier game passages. Optional alternatives are examples, "
            "not simultaneous events.\n\n" + "\n\n".join(render(c) for c in chunks)
        )
        current_text = "\n\n".join(p["text"] for p in current_passages)
        if not current_text:
            issues.append("No pre-target source prose available on the reconstructed route")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": current_text or "Continue at this encounter."},
        ]
        text = json.dumps(messages, ensure_ascii=False)
        if record["messages"][-1]["content"] in text:
            raise ValueError("Current target copied into context")
        count = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        row = {
            "sample_id": sample_id,
            "identity": identity,
            "conversation_id": meta["conversation_id"],
            "mission": mission,
            "input_tokens_qwen": count,
            "earlier_missions": len(ancestors),
            "earlier_passages": sum(len(c["passages"]) for c in chunks),
            "current_passages": len(current_passages),
            "unresolved_conditions": gaps,
            "alternative_producers": alternatives,
            "test_exclusions": omitted,
            "review_issues": issues,
            "placeholders": sorted(set(re.findall(r"<[^>]+>", text))),
            "status": "draft_requires_path_and_variable_review",
        }
        outputs.append(
            {
                **row,
                "messages": messages,
                "source_blocks": [*lore, *chunks],
                "current_source": {"path": path, "passages": current_passages},
                "target_boundary": target_line,
            }
        )
        summary.append(row)
    args.output.mkdir(parents=True)
    (args.output / "contexts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in outputs)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "revision": revision,
        "source_hashes": files,
        "script_sha256": sha(Path(__file__)),
        "pilot_manifest_sha256": sha(sample_root / "manifest.json"),
        "pilot_provenance_sha256": sha(provenance_path),
        "reference_pool": "Validation-only drafts; unassigned passages are not training data",
        "tokenizer_hashes": {
            str(p.relative_to(args.tokenizer)): sha(p)
            for p in args.tokenizer.rglob("*")
            if p.is_file()
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    counts = sorted(r["input_tokens_qwen"] for r in summary)
    print(
        json.dumps(
            {
                "samples": len(summary),
                "min": min(counts),
                "median": (counts[23] + counts[24]) / 2,
                "max": max(counts),
                "over_4096": sum(c > 4096 for c in counts),
            }
        )
    )


if __name__ == "__main__":
    main()
