"""Build conversation samples from source-backed annotations, without generating dialogue."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from prepare_conversations import DEFAULT_SOURCES, ROOT, SOURCE_MANIFEST, catalog, reachable

from endless_voices.contracts import slug, text, validate_manifest, validate_record

ANNOTATIONS = ROOT / "data/pilot-v1/annotations"
BATCHES = ("free-worlds.json", "hai.json", "republic.json", "quarg.json")
RESERVED = {
    "data/quarg/quarg missions.txt": 46,
    "data/hai/hai missions.txt": 90,
    "data/human/free worlds 0 prologue.txt": 356,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def extract(selector, conversation, *, assistant=False, substitutions=None):
    """Extract annotated speech fragments and return their exact source coordinates."""
    selection = {"line": selector} if type(selector) is int else selector
    number = selection["line"]
    paragraphs = {row["line"]: row for row in conversation["prose"]}
    if number not in paragraphs:
        raise ValueError(f"{conversation['id']}: {number} is not conversation prose")
    row = paragraphs[number]
    if assistant and row["kind"] == "choice":
        raise ValueError(f"{conversation['id']}:{number}: player choice used as assistant")
    indices = selection.get("quotes", list(range(len(row["speech"]))))
    if not indices or any(type(i) is not int or i < 0 or i >= len(row["speech"])
                          for i in indices) or sorted(set(indices)) != indices:
        raise ValueError(f"{conversation['id']}:{number}: invalid speech selection")
    chunks, provenance = [], []
    for index in indices:
        span = row["speech"][index]
        content = span["text"].strip()
        original = content
        # Narrator tags can leave a dangling comma at the end of selected speech.
        if re.search(r",['’]?$", content) and (index == indices[-1]
                                      or index in selection.get("sentence_breaks_after", [])):
            content = re.sub(r",(['’]?)$", r".\1", content)
        chunks.append(content)
        provenance.append({"line": number, "start": span["start"], "end": span["end"],
                           "normalization": "comma-to-period" if content != original else None})
    result = " ".join(chunks)
    for key, value in (substitutions or {}).items():
        text(value, "substitution value")
        result = result.replace(f"<{key}>", value)
    if not result.strip() or re.search(r"<[^>]+>", result):
        raise ValueError(f"{conversation['id']}:{number}: empty speech or runtime placeholder")
    return result, provenance


def reference(source, source_root, revision, group):
    """Validate a citation and render an immutable upstream passage reference."""
    relative = Path(source["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "data":
        raise ValueError(f"Invalid source path: {relative}")
    lines = (source_root / relative).read_text(encoding="utf-8").splitlines()
    start, end = source["lines"]
    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
        raise ValueError(f"{relative}: invalid citation range {source['lines']}")
    return {
        "reference": f"https://github.com/endless-sky/endless-sky/blob/{revision}/"
        f"{quote(str(relative))}#L{start}-L{end}",
        "revision": revision, "source_group": group,
    }


def assemble(batch, source_catalog, source_root, *, review_status="draft"):
    """Return generated records and a separate per-message provenance ledger."""
    revision = source_catalog["revision"]
    candidates = {row["id"]: row for row in source_catalog["conversations"]}
    lore = {entry["id"]: entry for entry in batch["lore"]}
    if len(lore) != len(batch["lore"]):
        raise ValueError("Duplicate lore ID")
    records, ledger, conversation_ids = [], [], set()
    for annotation in batch["conversations"]:
        slug(annotation["id"], "annotation.id")
        if annotation["id"] in conversation_ids:
            raise ValueError(f"Duplicate annotated conversation {annotation['id']}")
        conversation_ids.add(annotation["id"])
        source = candidates[annotation["catalog_id"]]
        split = annotation["split"]
        reserved_line = RESERVED.get(source["path"])
        if reserved_line and source["lines"][0] <= reserved_line <= source["lines"][1]:
            if split != "validation":
                raise ValueError(f"{source['id']}: calibration conversation requires validation")
        for field in ("scene", "review_notes"):
            text(annotation[field], field)
        substitutions = annotation.get("substitutions", {})
        if substitutions:
            text(annotation.get("substitution_notes"), "substitution_notes")
        profile = batch["profiles"][annotation["profile"]]
        text(profile["text"], "profile.text")
        if not annotation["lore_ids"]:
            raise ValueError(f"{source['id']}: selected lore is required")
        selected_lore = [lore[key] for key in annotation["lore_ids"]]
        group = " / ".join(source["owner"])
        sources = [reference(source, source_root, revision, group)]
        for item in [profile, *selected_lore]:
            text(item["text"], "context.text")
            if not item["sources"]:
                raise ValueError("Profile and lore require evidence")
            sources.extend(reference(s, source_root, revision, "profile/lore evidence")
                           for s in item["sources"])
        sources = list({json.dumps(s, sort_keys=True): s for s in sources}.values())
        system = (
            f"{profile['text']}\n\nScene: {annotation['scene']}\n\n"
            + ("Scene variable values: " + "; ".join(f"{k} = {v}"
               for k, v in substitutions.items()) + ".\n\n" if substitutions else "")
            +
            "Relevant lore (respect the speaker's knowledge and the scene's story date):\n"
            + "\n".join(f"- {item['text']}" for item in selected_lore)
            + "\n\nContinue as this speaker in direct speech. Do not add narrator actions or "
            "quotation marks around the whole reply."
        )
        seen_targets = set()
        for route in annotation["routes"]:
            slug(route["id"], "route.id")
            messages = [{"role": "system", "content": system}]
            origin = [{"role": "system", "origin": "agent", "profile": annotation["profile"],
                       "lore_ids": annotation["lore_ids"]}]
            last_reply_line = None
            spoken_lines = {row["line"] for row in source["prose"] if row["speech"]}
            spoken_lines -= set(annotation.get("narration_lines", []))
            for turn in route["turns"]:
                user = turn["user"]
                if isinstance(user, dict) and "prompt" in user:
                    text(user["prompt"], "connective prompt")
                    text(user["reason"], "connective prompt reason")
                    user_text = user["prompt"]
                    user_origin = {"role": "user", "origin": "agent", "reason": user["reason"]}
                else:
                    user_text, spans = extract(user, source, substitutions=substitutions)
                    if last_reply_line is not None and "flow" in source:
                        if not reachable(source["flow"], last_reply_line, spans[0]["line"],
                                         through_choices=True, blocked=spoken_lines):
                            raise ValueError(f"{source['id']}: omitted intervening dialogue "
                                             f"before user line {spans[0]['line']}")
                    user_origin = {"role": "user", "origin": "upstream", "spans": spans}
                chunks, spans = [], []
                if not turn["assistant"]:
                    raise ValueError(f"{source['id']}: empty assistant response")
                for selector in turn["assistant"]:
                    content, selected = extract(selector, source, assistant=True,
                                                substitutions=substitutions)
                    chunks.append(content)
                    spans.extend(selected)
                if "flow" in source:
                    sequence = [span["line"] for span in spans]
                    if user_origin["origin"] == "upstream":
                        sequence.insert(0, user_origin["spans"][-1]["line"])
                    for previous, current in zip(sequence, sequence[1:]):
                        if not reachable(source["flow"], previous, current,
                                         blocked=spoken_lines):
                            raise ValueError(f"{source['id']}: no uninterrupted reply path "
                                             f"from line {previous} to {current}")
                last_reply_line = spans[-1]["line"]
                target = "\n\n".join(chunks)
                messages.extend([{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": target}])
                origin.extend([user_origin, {"role": "assistant", "origin": "upstream",
                                             "spans": spans}])
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()[:8]
                sample_id = f"{annotation['id']}-l{spans[0]['line']}-{target_hash}"
                record = {
                    "schema_version": 1, "messages": list(messages),
                    "metadata": {
                        "id": sample_id, "identity": batch["identity"],
                        "species": batch["species"],
                        "character_role": profile["text"].split(". ", 1)[0],
                        "topics": annotation["topics"], "conversation_id": source["id"],
                        "scenario_group": annotation["scenario_group"], "split": split,
                        "sources": sources, "authorship": "mixed", "review_status": review_status,
                    },
                    "evaluation": {
                        "dimensions": ["authenticity"], "expected_facts": [],
                        "expected_behaviours": [], "expected_style": [],
                        "prohibited_contradictions": [], "uncertainty_expectations": [],
                        "sources": sources,
                    },
                }
                validate_record(record, split)
                records.append(record)
                ledger.append({
                    "id": sample_id, "annotation_id": annotation["id"], "route": route["id"],
                    "source_path": source["path"], "source_sha256": source["source_sha256"],
                    "source_revision": revision, "messages": list(origin),
                    "review_notes": annotation["review_notes"],
                    "substitutions": substitutions,
                    "substitution_notes": annotation.get("substitution_notes"),
                })
    return records, ledger


def check_overlap(records):
    """Reject copied assistant speech across splits, including earlier history."""
    seen = {}
    for row in records:
        split, sample = row["metadata"]["split"], row["metadata"]["id"]
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            for passage in [message["content"], *message["content"].split("\n\n")]:
                normalized = " ".join(re.findall(r"\w+", passage.lower()))
                if len(normalized.split()) < 12:
                    continue
                if normalized in seen and seen[normalized][0] != split:
                    raise ValueError(f"Copied assistant speech crosses splits: {sample}, "
                                     f"{seen[normalized][1]}")
                seen[normalized] = (split, sample)


def overlap_candidates(records, width=12):
    """Report shared exact phrases for source review; shared lore can be intentional."""
    def phrases(value):
        words = re.findall(r"\w+", value.lower())
        return {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}

    owners, across, context = {}, {}, []
    for row in records:
        meta = row["metadata"]
        target = phrases(row["messages"][-1]["content"])
        shared = sorted(target & phrases(row["messages"][0]["content"]))
        if shared:
            context.append({"sample": meta["id"], "phrases": shared})
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            for phrase in sorted(phrases(message["content"])):
                for split, sample in owners.get(phrase, {}).items():
                    if split != meta["split"]:
                        key = tuple(sorted((sample, meta["id"])))
                        across.setdefault(key, set()).add(phrase)
                owners.setdefault(phrase, {}).setdefault(meta["split"], meta["id"])
    return {"cross_split_phrases": [{"samples": list(key), "phrases": sorted(values)}
                                     for key, values in sorted(across.items())],
            "context_target_phrases": context}


def build(source_root, annotations, output, *, tokenizer=None, max_length=None,
          expected_release=None):
    """Publish a complete validated release directory without replacing an earlier release."""
    if output.exists():
        raise ValueError(f"{output}: already exists; choose a new release directory")
    if (tokenizer is None) != (max_length is None):
        raise ValueError("tokenizer and max_length must be supplied together")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_catalog = catalog(source_root, source_manifest)
    release_root = annotations.parent
    specification_file = release_root / "release.json"
    specification = (json.loads(specification_file.read_text())
                     if specification_file.exists() else {})
    review_file = release_root / "evidence/review.json"
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    all_records, ledger, annotation_hashes = [], [], {}
    context_paths = set()
    for name in BATCHES:
        path = annotations / name
        annotation_hashes[name] = digest(path)
        status = "reviewed" if reviews.get("annotation_sha256", {}).get(name) == digest(path) \
            else "draft"
        batch = json.loads(path.read_text())
        context_paths.update(source["path"] for entry in [*batch["profiles"].values(),
                                                          *batch["lore"]]
                             for source in entry["sources"])
        records, evidence = assemble(batch, source_catalog, source_root,
                                     review_status=status)
        all_records.extend(records)
        ledger.extend(evidence)
    check_overlap(all_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-", dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        samples, evidence, licensing = (staged / name for name in
                                        ("samples", "evidence", "licensing"))
        for directory in (samples, evidence, licensing):
            directory.mkdir()
        shutil.copytree(annotations, staged / "annotations")
        for name in ("review.json", "findings.md"):
            source = release_root / "evidence" / name
            if source.exists():
                shutil.copyfile(source, evidence / name)
        if specification_file.exists():
            shutil.copyfile(specification_file, staged / "release.json")
        manifest = {"schema_version": 1,
                    "dataset_version": specification.get("dataset_version", release_root.name),
                    "files": {}}
        for split in ("train", "validation", "test"):
            rows = sorted((r for r in all_records if r["metadata"]["split"] == split),
                          key=lambda r: r["metadata"]["id"])
            path = samples / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                            encoding="utf-8")
            manifest["files"][split] = [{"path": path.name, "sha256": digest(path)}]
        dump(samples / "manifest.json", manifest)
        counts = validate_manifest(samples / "manifest.json", tokenizer=tokenizer,
                                   max_length=max_length)
        for split, stats in counts.items():
            rows = [r for r in all_records if r["metadata"]["split"] == split]
            stats["conversations"] = len({r["metadata"]["conversation_id"] for r in rows})
            stats["multi_turn_samples"] = sum(len(r["messages"]) > 3 for r in rows)
            if tokenizer is not None:
                lengths = sorted(len(tokenizer.apply_chat_template(r["messages"], tokenize=True,
                                                                   add_generation_prompt=False))
                                 for r in rows)
                stats["complete_sample_tokens"] = {
                    "min": min(lengths), "median": lengths[len(lengths) // 2],
                    "max": max(lengths), "limit": max_length,
                }
        dump(evidence / "coverage.json", counts)
        dump(evidence / "overlap-candidates.json", overlap_candidates(all_records))
        dump(evidence / "provenance.json", sorted(ledger, key=lambda r: r["id"]))
        dump(evidence / "construction.json", {"source_revision": source_catalog["revision"],
                                           "annotation_sha256": annotation_hashes,
                                           "review": reviews,
                                           "preparation_sha256": {
                                               name: digest(ROOT / "scripts" / name)
                                               for name in ("prepare_conversations.py",
                                                            "build_pilot.py")}})
        notices = ["# Source file notices\n"]
        paths = sorted({row["source_path"] for row in ledger} | context_paths)
        for name in paths:
            header = []
            for line in (source_root / name).read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    header.append(line)
                else:
                    break
            notices.extend([f"## {name}\n", "```text\n" + "\n".join(header) + "\n```\n"])
        (licensing / "SOURCE-NOTICES.md").write_text("\n".join(notices), encoding="utf-8")
        review_text = ["# Pilot dataset review\n",
                       "Organizer copy: includes withheld test targets and source labels.\n"]
        for row in sorted(all_records, key=lambda r: (r["metadata"]["split"],
                                                       r["metadata"]["id"])):
            meta = row["metadata"]
            review_text.append(f"## {meta['id']} ({meta['split']})\n")
            review_text.append(f"Source: {meta['sources'][0]['reference']}\n")
            for message in row["messages"]:
                review_text.append(f"**{message['role']}**\n\n{message['content']}\n")
        (evidence / "review.md").write_text("\n".join(review_text), encoding="utf-8")
        for name in ("license.txt", "copyright", "credits.txt"):
            shutil.copyfile(source_root / name, licensing / name)
        (licensing / "ATTRIBUTION.md").write_text(
            "# Attribution\n\nThis source-derived conversation dataset is distributed under "
            "GPL-3.0-or-later. Dialogue comes from Endless Sky at " + source_catalog["revision"]
            + ". Preserve license.txt, copyright, credits.txt, SOURCE-NOTICES.md, "
            "and ../evidence/provenance.json. "
            "Those files retain upstream notices and contributor attribution. Profiles, lore "
            "summaries, scene context and connective prompts were written by agents for "
            "Endless Voices; the combined dataset uses GPL-3.0-or-later. Extracted dialogue "
            "omits narration and enclosing quotation marks; narrator-tag trailing commas "
            "are normalized to periods as recorded per span. No human sample approval is "
            "implied by the review status.\n", encoding="utf-8")
        if expected_release:
            for name, expected in expected_release["artifact_sha256"].items():
                if digest(staged / name) != expected:
                    raise ValueError(f"Frozen release checksum mismatch: {name}")
        staged.rename(output)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--verify-release", type=Path,
                        help="check frozen artifact hashes after reconstruction")
    args = parser.parse_args()
    try:
        tokenizer = None
        if args.tokenizer:
            specification = json.loads(
                (args.annotations.parent / "release.json").read_text())["tokenizer"]
            for name, expected in specification["files"].items():
                if digest(args.tokenizer / name) != expected:
                    raise ValueError(f"Tokenizer checksum mismatch: {name}")
            if args.max_length != specification["max_length"]:
                raise ValueError("Use the complete-sample limit recorded in release.json")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()),
                                                      local_files_only=True,
                                                      trust_remote_code=False)
        expected = json.loads(args.verify_release.read_text()) if args.verify_release else None
        print(json.dumps(build(args.sources, args.annotations, args.output,
                               tokenizer=tokenizer, max_length=args.max_length,
                               expected_release=expected), indent=2))
        if expected:
            print("Frozen release hashes match.")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
