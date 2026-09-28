"""Build an offline, verbatim Recon context preview from pinned game text."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from prepare_conversations import tree, walk

SAMPLE = "fw-recon-3-jj-new-wales-base-l468-63b32cca"
MISSION_FILE = "data/human/free worlds 0 prologue.txt"
PLANET_FILE = "data/map planets.txt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prose(node):
    """Extract verbatim narrative and options, excluding terminal refusal choices."""
    result = []
    for item, ancestors in walk([node]):
        raw, parts = item["raw"].lstrip(), item["tokens"]
        if raw.startswith("`"):
            if any(c["tokens"][:1] in (["decline"], ["defer"]) for c in item["children"]):
                continue
            kind = "option" if any(a["tokens"] == ["choice"] for a in ancestors) else "passage"
            result.append({"line": item["line"], "kind": kind, "text": parts[0]})
        elif parts[:1] == ["dialog"] and len(parts) == 2:
            result.append({"line": item["line"], "kind": "passage", "text": parts[1]})
    return result


def protect_test(excerpts, test_metadata, revision):
    """Reject selected source lines overlapping held-out mission conversation spans."""
    for metadata in test_metadata:
        for source in metadata["sources"]:
            if not source["source_group"].startswith("mission /"):
                continue
            match = re.search(r"/blob/" + revision + r"/(.+)#L(\d+)(?:-L(\d+))?$",
                              source["reference"])
            if match is None:
                raise ValueError("Cannot verify a held-out mission source range")
            path, start, end = unquote(match[1]), int(match[2]), int(match[3] or match[2])
            if any(e["path"] == path and start <= p["line"] <= end
                   for e in excerpts for p in e["passages"]):
                raise ValueError("Reference overlaps a held-out conversation")


def first_opening(offer):
    """Return only the opening passage, never later choices or replies."""
    return next(n for n, _ in walk([offer]) if n["raw"].lstrip().startswith("`"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    inventory_path = Path("data/overview/source-statistics.json")
    inventory = json.loads(inventory_path.read_text())
    expected = {e["path"]: e["sha256"] for e in inventory["files"]}
    for name in (MISSION_FILE, PLANET_FILE):
        if sha(args.source / name) != expected[name]:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    samples = Path("data/pilot-v1/samples")
    manifest = json.loads((samples / "manifest.json").read_text())
    rows = {}
    test_metadata = []
    for split in ("train", "validation", "test"):
        for entry in manifest["files"][split]:
            file = samples / entry["path"]
            if sha(file) != entry["sha256"]:
                raise ValueError("Pilot hash mismatch")
            for line in file.read_text().splitlines():
                record = json.loads(line)
                if split == "test":
                    test_metadata.append(record["metadata"])
                else:
                    rows[record["metadata"]["id"]] = record
    missions = {n["tokens"][1]: n for n in tree((args.source / MISSION_FILE).read_text())
                if n["tokens"][:1] == ["mission"]}
    excerpts = []
    for number in range(3):
        mission = f"FW Recon {number}"
        for node in missions[mission]["children"]:
            if node["tokens"][:2] not in (["on", "offer"], ["on", "complete"],
                                          ["npc", "scan outfits"]):
                continue
            passages = prose(node)
            if passages:
                excerpts.append({"heading": mission + " / " + " ".join(node["tokens"]),
                                 "path": MISSION_FILE, "passages": passages})
    planet = next(n for n in tree((args.source / PLANET_FILE).read_text())
                  if n["tokens"][:2] == ["planet", "Glaze"])
    lore = {"heading": "Glaze: in-game planet description", "path": PLANET_FILE,
            "passages": [{"line": n["line"], "kind": "description", "text": n["tokens"][1]}
                         for n in planet["children"] if n["tokens"][:1] == ["description"]]}
    offer = next(n for n in missions["FW Recon 3"]["children"] if n["tokens"] == ["on", "offer"])
    # The opening is selected structurally, before any target or alternative reply.
    opening = first_opening(offer)
    current = {"heading": "Current encounter opening", "path": MISSION_FILE,
               "passages": [{"line": opening["line"], "kind": "passage",
                             "text": opening["tokens"][0]}]}
    protect_test([lore, *excerpts, current], test_metadata, inventory["revision"])

    def render(excerpt):
        return excerpt["heading"] + "\n" + "\n\n".join(
            ("Optional player response: " if p["kind"] == "option" else "") + p["text"]
            for p in excerpt["passages"]
        )

    instruction = ("You are Jean-Jacques Soleau (JJ), commander of the militia on Glaze.\n"
                   "Continue the conversation as the character, responding as authentically "
                   "as possible for the given scenario.")
    reference = ("Earlier game passages; optional alternatives are examples, "
                 "not simultaneous events.")
    system = instruction + "\n\n" + render(lore) + "\n\n" + reference + "\n\n" + "\n\n".join(
        render(e) for e in excerpts
    )
    sample = rows[SAMPLE]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": render(current) + "\n\nPlayer: "
                 + sample["messages"][-2]["content"]}]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    result = {
        "sample_id": SAMPLE, "split": "validation", "revision": inventory["revision"],
        "scope": "Source-context design preview, not a new dataset release or benchmark",
        "reference_pool": "Additional source passages provisionally reference-only validation; "
                          "no permission to add these passages to training",
        "lore": lore, "earlier_dialogue": excerpts, "current_opening": current,
        "messages": messages, "target": sample["messages"][-1]["content"],
        "measurements": {
            "baseline_input_tokens": len(tokenizer.apply_chat_template(
                sample["messages"][:-1], tokenize=True, add_generation_prompt=True)),
            "expanded_input_tokens": len(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)),
            "lore_text_tokens": count(render(lore)),
            "earlier_text_tokens": count("\n\n".join(render(e) for e in excerpts)),
            "current_opening_tokens": count(render(current)),
            "earlier_passages": sum(len(e["passages"]) for e in excerpts),
            "tokenizer": "cached Qwen3-4B effective tokenizer; includes chat template",
            "unresolved_placeholders": sorted(set(re.findall(r"<[^>]+>", system))),
        },
        "hashes": {
            "script": sha(Path(__file__)), "source_inventory": sha(inventory_path),
            "pilot_manifest": sha(samples / "manifest.json"),
            "sources": {name: expected[name] for name in (MISSION_FILE, PLANET_FILE)},
            "tokenizer": {str(p.relative_to(args.tokenizer)): sha(p)
                          for p in sorted(args.tokenizer.rglob("*")) if p.is_file()},
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["measurements"], indent=2))


if __name__ == "__main__":
    main()
