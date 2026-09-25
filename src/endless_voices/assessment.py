"""Prepare blinded trials and report saved authenticity judgments without loading models."""

import argparse
import hashlib
import html
import itertools
import json
import random
import secrets
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from endless_voices.contracts import evaluation_messages, read_records, validate_manifest

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return digest(path.read_bytes())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows, allowed):
    result = {}
    for row in rows:
        key = row["sample_id"]
        if key not in allowed or key in result:
            raise ValueError("Duplicate or unselected sample ID")
        result[key] = row
    return result


def load_dataset(manifest, split):
    validate_manifest(manifest)
    declaration = read_json(manifest)
    return {
        row["metadata"]["id"]: row
        for entry in declaration["files"][split]
        for _, row in read_records(manifest.parent / entry["path"], split)
    }


def load_run(directory, manifest, records, split):
    """Verify recorded artifacts, selection, and authored input before accepting responses."""
    run = read_json(directory / "run.json")
    dataset = run["dataset"]
    if run["schema_version"] != 1 or dataset["manifest_sha256"] != file_hash(manifest):
        raise ValueError("Unsupported run or incompatible dataset manifest")
    if dataset["split"] != split or dataset["manifest"] != read_json(manifest):
        raise ValueError("Incompatible dataset split or declaration")
    ids = dataset["sample_ids"]
    if not ids or len(set(ids)) != len(ids) or set(ids) - records.keys():
        raise ValueError("Invalid selected sample IDs")
    if read_json(directory / "sample-ids.json") != ids:
        raise ValueError("Selection file differs from run")
    hashes = run.get("artifacts_sha256", {})
    if (
        run["status"] != "running"
        and not {"sample-ids.json", "prompts.jsonl", "responses.jsonl"} <= hashes.keys()
    ):
        raise ValueError("Finalized run is missing artifact hashes")
    for name, expected in hashes.items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid artifact path")
        if file_hash(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    prompts = indexed(read_rows(directory / "prompts.jsonl"), ids)
    responses = indexed(read_rows(directory / "responses.jsonl"), ids)
    for sid, prompt in prompts.items():
        messages = evaluation_messages(records[sid])
        if prompt["messages"] != messages or prompt["messages_sha256"] != digest(encoded(messages)):
            raise ValueError("Prompt differs from authored context")
        if "input_ids" in prompt and (
            prompt["input_ids_sha256"] != digest(encoded(prompt["input_ids"]))
            or prompt["input_tokens"] != len(prompt["input_ids"])
        ):
            raise ValueError("Prompt token hash or count mismatch")
    for sid, response in responses.items():
        if response["status"] not in {"ok", "failed"}:
            raise ValueError("Invalid generation status")
        if response["status"] == "ok":
            if (
                sid not in prompts
                or not isinstance(response["response"], str)
                or not response["response"].strip()
                or response.get("error")
                or response.get("finish_reason") != "eos"
            ):
                raise ValueError("Successful response lacks complete generation evidence")
            for key in ("messages_sha256", "input_ids_sha256", "input_tokens"):
                if response.get(key) != prompts[sid].get(key) or key not in prompts[sid]:
                    raise ValueError("Response and prompt provenance differ")
    return run, prompts, responses


def connected_groups(records):
    """Return transitive conversation/scenario groups for uncertainty calculations."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for row in records.values():
        meta = row["metadata"]
        c = "conversation:" + meta["conversation_id"]
        find(c)
        if meta["scenario_group"]:
            s = "scenario:" + meta["scenario_group"]
            parent[find(c)] = find(s)
    return {
        sid: find("conversation:" + r["metadata"]["conversation_id"]) for sid, r in records.items()
    }


def trial_page(trial, instructions):
    sections = [
        "<!doctype html><meta charset='utf-8'><title>Authenticity review</title>",
        "<style>body{max-width:850px;margin:40px auto;padding:20px;font:18px/1.55 "
        "system-ui}pre{white-space:pre-wrap;font:inherit}h2{margin-top:2em}</style>",
        "<h1>Authenticity review</h1><pre>" + html.escape(instructions) + "</pre>",
        "<p>Trial: " + trial["trial_id"] + "</p>",
    ]
    for message in trial["context"]:
        sections.append(
            "<h2>"
            + html.escape(message["role"])
            + "</h2><pre>"
            + html.escape(message["content"])
            + "</pre>"
        )
    for label in ("A", "B"):
        sections.append("<h2>" + label + "</h2><pre>" + html.escape(trial[label]) + "</pre>")
    return "\n".join(sections)


def prepare(manifest, runs, output, split="validation", seed=None, controls=None, reverse=False):
    """Write an immutable organizer pack and isolated public trial files."""
    if output.exists():
        raise ValueError("Output already exists")
    records = load_dataset(manifest, split)
    groups = connected_groups(records)
    loaded = {name: load_run(path, manifest, records, split) for name, path in runs.items()}
    if not loaded:
        raise ValueError("At least one condition is required")
    seed = secrets.randbits(64) if seed is None else seed
    rng = random.Random(seed)
    instructions = (ROOT / "data/evaluation/judge-instructions.md").read_text()
    public, private, coverage, provenance = [], [], [], {}
    for name, (run, prompts, responses) in loaded.items():
        ids = run["dataset"]["sample_ids"]
        order = ["A", "B"] * (len(ids) // 2) + ([rng.choice(["A", "B"])] if len(ids) % 2 else [])
        rng.shuffle(order)
        provenance[name] = {
            "run": run,
            "run_sha256": file_hash(runs[name] / "run.json"),
            "verified_hashes": bool(run.get("artifacts_sha256")),
        }
        for sid, pos in zip(ids, order, strict=True):
            response = responses.get(sid)
            status = response["status"] if response else "missing"
            coverage.append(
                {
                    "condition": name,
                    "sample_id": sid,
                    "identity": records[sid]["metadata"]["identity"],
                    "status": status,
                    "error": response.get("error") if response else None,
                }
            )
            if status != "ok":
                continue
            public.append((name, sid, "primary", response["response"], pos, None))
            if reverse:
                public.append(
                    (name, sid, "reversed", response["response"], "B" if pos == "A" else "A", None)
                )
    for item in controls or []:
        sid, kind = item["sample_id"], item["kind"]
        if sid not in records or kind not in {"identical", "wrong-context"}:
            raise ValueError("Invalid control")
        if kind == "identical":
            other = records[sid]["messages"][-1]["content"]
        else:
            donor = item["donor_id"]
            if donor == sid or donor not in records:
                raise ValueError("Invalid control donor")
            other = records[donor]["messages"][-1]["content"]
        public.append((None, sid, kind, other, rng.choice(["A", "B"]), item))
    rng.shuffle(public)
    output.mkdir(parents=True)
    licensing = manifest.parent.parent / "licensing"
    if licensing.is_dir():
        shutil.copytree(licensing, output / "attribution")
    for stage in ("primary", "reversed", "controls"):
        (output / "public" / stage).mkdir(parents=True)
    for name, sid, kind, alternative, pos, control in public:
        trial_id = f"{rng.getrandbits(128):032x}"
        record = records[sid]
        original = record["messages"][-1]["content"]
        trial = {
            "trial_id": trial_id,
            "context": evaluation_messages(record),
            pos: original,
            "B" if pos == "A" else "A": alternative,
        }
        stage = kind if kind in {"primary", "reversed"} else "controls"
        relative = f"public/{stage}/{trial_id}.json"
        write_json(output / relative, trial)
        (output / relative).with_suffix(".html").write_text(trial_page(trial, instructions))
        private.append(
            {
                "trial_id": trial_id,
                "sample_id": sid,
                "condition": name,
                "kind": kind,
                "original": pos if kind != "identical" else None,
                "identity": record["metadata"]["identity"],
                "group": groups[sid],
                "control": control,
                "path": relative,
                "trial_sha256": file_hash(output / relative),
            }
        )
    (output / "public/instructions.txt").write_text(instructions)
    write_json(
        output / "private.json",
        {
            "schema_version": 1,
            "seed": seed,
            "split": split,
            "manifest_sha256": file_hash(manifest),
            "runs": provenance,
            "coverage": coverage,
            "trials": private,
            "instructions_sha256": digest(instructions.encode()),
            "calibration_status": "provisional; calibration and owner discussion required",
        },
    )
    for stage in ("primary", "reversed", "controls"):
        selected = [t for t in private if f"public/{stage}/" in t["path"]]
        write_json(
            output / f"{stage}-review-template.json",
            {
                "reviewer_id": None,
                "reviewer_type": "human",
                "judge_model_and_prompt": None,
                "reviews": [
                    {
                        "trial_id": t["trial_id"],
                        "trial_sha256": t["trial_sha256"],
                        "status": "missing",
                        "choice": None,
                        "confidence": None,
                        "reason": None,
                        "recognized_source": None,
                    }
                    for t in selected
                ],
            },
        )
    return private


def validate_judgment(row):
    """Reject malformed decisions while retaining explicit failures and unanswered trials."""
    status = row.get("status", "ok")
    if status not in {"ok", "failed", "missing"}:
        raise ValueError("Invalid judgment status")
    if status != "ok":
        if row.get("choice") is not None:
            raise ValueError("Failed or missing judgment cannot have a choice")
        if status == "failed" and not row.get("error"):
            raise ValueError("Failed judgment needs an error")
        return
    if row.get("choice") not in {"A", "B", "abstain"}:
        raise ValueError("Choice must be A, B or abstain")
    expected = {None} if row["choice"] == "abstain" else {"low", "medium", "high"}
    if row.get("confidence") not in expected:
        raise ValueError("Confidence does not match choice")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("A brief reason is required")
    if type(row.get("recognized_source")) is not bool:
        raise ValueError("Source recognition must be explicitly true or false")


def rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def counts(rows):
    tally = Counter(row["outcome"] for row in rows)
    result = {k: tally[k] for k in ("correct", "incorrect", "abstained", "failed", "missing")}
    result["scheduled"] = len(rows)
    result["accuracy"] = rate(tally["correct"], tally["correct"] + tally["incorrect"])
    for name, key in (
        ("abstention_rate", "abstained"),
        ("failure_rate", "failed"),
        ("missing_rate", "missing"),
    ):
        result[name] = rate(tally[key], len(rows))
    return result


def interval(rows, seed=42, repetitions=2000):
    """Bootstrap whole related groups, preserving every observation within each group."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["value"])
    contributing = sum(any(v is not None for v in values) for values in groups.values())
    result = {
        "method": "conversation/scenario cluster percentile bootstrap",
        "seed": seed,
        "repetitions": repetitions,
        "groups": len(groups),
        "contributing_groups": contributing,
        "interval_95": None,
        "undefined_resamples": 0,
    }
    if contributing < 2:
        return result
    rng, estimates, keys = random.Random(seed), [], sorted(groups)
    for _ in range(repetitions):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key] if v is not None]
        if values:
            estimates.append(sum(values) / len(values))
        else:
            result["undefined_resamples"] += 1
    estimates.sort()
    if estimates:
        result["interval_95"] = [estimates[int((len(estimates) - 1) * q)] for q in (0.025, 0.975)]
    return result


def report(pack, reviews, output):
    """Join immutable judgments to private keys and report explicit denominators."""
    if output.exists():
        raise ValueError("Output already exists")
    key = read_json(pack / "private.json")
    if file_hash(pack / "public/instructions.txt") != key["instructions_sha256"]:
        raise ValueError("Instructions changed after preparation")
    trials = {t["trial_id"]: t for t in key["trials"]}
    for t in trials.values():
        if file_hash(pack / t["path"]) != t["trial_sha256"]:
            raise ValueError("Trial hash mismatch")
    reviewers, all_rows = {}, []
    for path in reviews:
        review = read_json(path)
        rid = review.get("reviewer_id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("Reviewer identity is required")
        if review.get("reviewer_type") not in {"human", "llm"}:
            raise ValueError("Reviewer type is required")
        if review["reviewer_type"] == "llm" and not review.get("judge_model_and_prompt"):
            raise ValueError("LLM prompt and settings provenance is required")
        if rid not in reviewers:
            reviewers[rid] = {
                "metadata": {k: v for k, v in review.items() if k != "reviews"},
                "submitted": {},
                "files": [],
            }
        target = reviewers[rid]
        provenance = review.get("judge_model_and_prompt") or {}
        if provenance.get("prompts_sha256"):
            prompt_path = Path(provenance["prompt_records"])
            if prompt_path.is_absolute() or ".." in prompt_path.parts:
                raise ValueError("Prompt records must be beside the review")
            if file_hash(path.parent / prompt_path) != provenance["prompts_sha256"]:
                raise ValueError("Judge prompt records changed")

        def comparable(metadata):
            value = json.loads(json.dumps(metadata))
            if value.get("judge_model_and_prompt"):
                value["judge_model_and_prompt"].pop("prompts_sha256", None)
            return value

        metadata = {k: v for k, v in review.items() if k != "reviews"}
        if comparable(target["metadata"]) != comparable(metadata):
            raise ValueError("Reviewer metadata differs; use separate reviewer IDs")
        target["files"].append({"path": str(path), "sha256": file_hash(path), "metadata": metadata})
        for row in review["reviews"]:
            tid = row["trial_id"]
            if tid not in trials or tid in target["submitted"]:
                raise ValueError("Unknown or duplicate judgment")
            if row["trial_sha256"] != trials[tid]["trial_sha256"]:
                raise ValueError("Judgment belongs to a different trial")
            if set(row) & {
                "sample_id",
                "condition",
                "kind",
                "original",
                "identity",
                "group",
                "control",
                "path",
                "outcome",
                "value",
                "reviewer_id",
            }:
                raise ValueError("Judgment cannot replace organizer fields")
            validate_judgment(row)
            target["submitted"][tid] = row
    for rid, review in reviewers.items():
        for tid, trial in trials.items():
            row = review["submitted"].get(tid, {"status": "missing"})
            status = row.get("status", "ok")
            if status != "ok":
                outcome = status
            elif row["choice"] == "abstain":
                outcome = "abstained"
            else:
                outcome = "correct" if row["choice"] == trial["original"] else "incorrect"
            all_rows.append(
                {
                    **trial,
                    **row,
                    "reviewer_id": rid,
                    "outcome": outcome,
                    "value": 1 if outcome == "correct" else 0 if outcome == "incorrect" else None,
                }
            )
    summaries, paired, disagreements, reversals = [], [], [], []
    conditions = list(key["runs"])
    for rid in reviewers:
        primary = [r for r in all_rows if r["reviewer_id"] == rid and r["kind"] == "primary"]
        for condition in conditions:
            rows = [r for r in primary if r["condition"] == condition]
            summaries.append(
                {
                    "reviewer_id": rid,
                    "condition": condition,
                    **counts(rows),
                    "uncertainty": interval(rows),
                    "recognized": counts([r for r in rows if r.get("recognized_source")]),
                    "unrecognized": counts(
                        [r for r in rows if r.get("recognized_source") is False]
                    ),
                    "by_identity": {
                        ident: counts([r for r in rows if r["identity"] == ident])
                        for ident in sorted({r["identity"] for r in rows})
                    },
                }
            )
        for a, b in itertools.combinations(conditions, 2):
            left = {r["sample_id"]: r for r in primary if r["condition"] == a}
            right = {r["sample_id"]: r for r in primary if r["condition"] == b}
            selected = {c["sample_id"] for c in key["coverage"] if c["condition"] in {a, b}}
            complete, incomplete = [], []
            for sid in sorted(selected):
                x, y = left.get(sid), right.get(sid)
                if x and y and x["value"] is not None and y["value"] is not None:
                    complete.append(
                        {"sample_id": sid, "group": x["group"], "value": y["value"] - x["value"]}
                    )
                else:
                    incomplete.append(
                        {
                            "sample_id": sid,
                            "left": x["outcome"] if x else "not assessed; see generation coverage",
                            "right": y["outcome"] if y else "not assessed; see generation coverage",
                        }
                    )
            paired.append(
                {
                    "reviewer_id": rid,
                    "left": a,
                    "right": b,
                    "difference_right_minus_left": sum(r["value"] for r in complete) / len(complete)
                    if complete
                    else None,
                    "complete_pairs": complete,
                    "incomplete_pairs": incomplete,
                    "uncertainty": interval(complete),
                }
            )
        for r in primary:
            other = next(
                (
                    x
                    for x in all_rows
                    if x["reviewer_id"] == rid
                    and x["kind"] == "reversed"
                    and x["condition"] == r["condition"]
                    and x["sample_id"] == r["sample_id"]
                ),
                None,
            )
            if other:
                reversals.append(
                    {
                        "reviewer_id": rid,
                        "primary": r["trial_id"],
                        "reversed": other["trial_id"],
                        "primary_outcome": r["outcome"],
                        "reversed_outcome": other["outcome"],
                    }
                )
    for tid in trials:
        rows = [r for r in all_rows if r["trial_id"] == tid and r.get("status", "ok") == "ok"]
        if len({r["choice"] for r in rows}) > 1:
            disagreements.append({"trial_id": tid, "judgments": rows})
    controls = [r for r in all_rows if r["kind"] not in {"primary", "reversed"}]
    for row in controls:
        row["matches_control"] = (
            (
                row.get("choice") == "abstain"
                if row["kind"] == "identical"
                else row["outcome"] == "correct"
            )
            if row.get("status", "ok") == "ok"
            else None
        )
    coverage = []
    for condition in conditions:
        entries = [c for c in key["coverage"] if c["condition"] == condition]
        for identity in [None, *sorted({c["identity"] for c in entries})]:
            subset = [c for c in entries if identity is None or c["identity"] == identity]
            tally = Counter(c["status"] for c in subset)
            coverage.append(
                {
                    "condition": condition,
                    "identity": identity,
                    "selected": len(subset),
                    "ok": tally["ok"],
                    "failed": tally["failed"],
                    "missing": tally["missing"],
                }
            )
    result = {
        "schema_version": 1,
        "status": "exploratory; judge calibration not established",
        "pack_sha256": file_hash(pack / "private.json"),
        "reviewers": reviewers,
        "results": summaries,
        "generation_coverage": coverage,
        "generation_records": key["coverage"],
        "generation_provenance": key["runs"],
        "paired": paired,
        "controls": controls,
        "disagreements": disagreements,
        "position_checks": reversals,
        "trials": all_rows,
    }
    output.mkdir(parents=True)
    write_json(output / "report.json", result)
    lines = [
        "# Exploratory authenticity report",
        "",
        "Judge calibration is not established. Lower detection is not automatically better "
        "quality. Chance performance does not establish indistinguishability or equivalence.",
        "",
        "Intervals resample whole conversation/scenario groups. Few groups can produce "
        "unstable or degenerate intervals; these intervals exclude judge and dataset-selection "
        "uncertainty. Reviewer votes and reversed positions are not independent scenes.",
        "",
        "| Reviewer | Condition | Correct / decided | Incorrect | Abstained / scheduled | "
        "Failed | Missing |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['reviewer_id']} | {s['condition']} | "
            f"{s['correct']}/{s['accuracy']['denominator']} | {s['incorrect']} | "
            f"{s['abstained']}/{s['scheduled']} | {s['failed']}/{s['scheduled']} | "
            f"{s['missing']}/{s['scheduled']} |"
        )
    lines.extend(
        [
            "",
            "## Generation coverage",
            "",
            "| Condition | Identity | Successful / selected | Failed | Missing |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in coverage:
        lines.append(
            f"| {row['condition']} | {row['identity'] or 'All'} | "
            f"{row['ok']}/{row['selected']} | {row['failed']} | {row['missing']} |"
        )
    lines.extend(
        [
            "",
            "Generation failures are not sent for judging. Scheduled assessment counts "
            "refer to prepared primary trials; generation coverage retains every selected sample.",
            "",
            "## Recognition, identities and uncertainty",
            "",
        ]
    )
    for summary in summaries:
        lines.append(f"### {summary['reviewer_id']} / {summary['condition']}")
        lines.append("")
        for name in ("recognized", "unrecognized"):
            subset = summary[name]
            lines.append(
                f"{name.capitalize()} source: {subset['correct']} correct / "
                f"{subset['accuracy']['denominator']} decided; "
                f"{subset['abstained']} abstained. Recognition is unknown for "
                "unsubmitted or failed assessments."
            )
        lines.extend(
            [
                "",
                "| Identity | Correct / decided | Abstained | Failed | Missing | Scheduled |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for identity, subset in summary["by_identity"].items():
            lines.append(
                f"| {identity} | {subset['correct']}/{subset['accuracy']['denominator']} | "
                f"{subset['abstained']} | {subset['failed']} | {subset['missing']} | "
                f"{subset['scheduled']} |"
            )
        uncertainty = summary["uncertainty"]
        lines.extend(
            [
                "",
                f"Independent conversation/scenario groups: {uncertainty['groups']}; "
                f"groups with decisions: {uncertainty['contributing_groups']}. "
                f"Descriptive 95% interval: {uncertainty['interval_95']}. "
                f"Undefined bootstrap resamples: {uncertainty['undefined_resamples']}.",
                "",
            ]
        )
    lines.extend(["## Paired comparisons", ""])
    if not paired:
        lines.append("Only one model condition is present; no model comparison is available.")
    for pair in paired:
        lines.extend(
            [
                f"{pair['reviewer_id']}: {pair['right']} minus {pair['left']} = "
                f"{pair['difference_right_minus_left']}; "
                f"{len(pair['complete_pairs'])} decided pairs and "
                f"{len(pair['incomplete_pairs'])} incomplete pairs. "
                f"95% interval: {pair['uncertainty']['interval_95']}.",
                "",
            ]
        )
        for missing in pair["incomplete_pairs"]:
            lines.append(f"- {missing['sample_id']}: {missing['left']} / {missing['right']}")
    lines.extend(["", "## Controls and position checks", ""])
    for rid in reviewers:
        subset = [r for r in controls if r["reviewer_id"] == rid]
        matching = sum(r["matches_control"] is True for r in subset)
        submitted = sum(r["matches_control"] is not None for r in subset)
        lines.append(
            f"{rid}: {matching}/{submitted} submitted controls matched the intended "
            f"outcome; {len(subset)} scheduled controls. Controls do not enter detection rates."
        )
    for check in reversals:
        lines.append(
            f"- {check['reviewer_id']}, {check['primary']}: "
            f"{check['primary_outcome']} in the primary order; "
            f"{check['reversed_outcome']} in the reversed order."
        )
    lines.extend(
        [
            "",
            "## Judgments and reasons",
            "",
            f"Trials with differing submitted reviewer choices: {len(disagreements)}. "
            "The JSON report retains the complete disagreement records.",
            "",
        ]
    )
    for row in all_rows:
        lines.extend(
            [
                f"### {row['trial_id']} / {row['reviewer_id']}",
                "",
                f"{row['kind']}; {row['identity']}; {row['condition'] or 'control'}; "
                f"{row['outcome']}; confidence: {row.get('confidence')}; "
                f"recognized source: {row.get('recognized_source')}.",
                "",
                html.escape(str(row.get("reason") or row.get("error") or "No judgment submitted.")),
                "",
            ]
        )
    (output / "report.md").write_text("\n".join(lines))
    return result


def export_human(pack, output, condition, stage="primary"):
    """Export one condition with no repeated primary conversations or hidden answer mapping."""
    key = read_json(pack / "private.json")
    selected = [
        t
        for t in key["trials"]
        if (t["kind"] == "primary" and t["condition"] == condition)
        if stage == "primary"
    ]
    if stage == "controls":
        selected = [t for t in key["trials"] if t["kind"] in {"identical", "wrong-context"}]
    if not selected:
        raise ValueError("No matching trials")
    if stage == "primary" and len({t["group"] for t in selected}) != len(selected):
        raise ValueError("Human sheet would repeat related scenes; prepare separate assignments")
    instructions = (pack / "public/instructions.txt").read_text()
    if digest(instructions.encode()) != key["instructions_sha256"]:
        raise ValueError("Instructions changed after preparation")
    trials = []
    for trial in selected:
        if file_hash(pack / trial["path"]) != trial["trial_sha256"]:
            raise ValueError("Trial hash mismatch")
        trials.append({**read_json(pack / trial["path"]), "trial_sha256": trial["trial_sha256"]})
    payload = json.dumps({"instructions": instructions, "trials": trials}, ensure_ascii=False)
    template = Path(__file__).with_name("review.html").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(template.replace("__REVIEW_DATA__", payload.replace("<", "\\u003c")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Verify runs and export isolated blinded trials")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, metavar="CONDITION=DIRECTORY")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument(
        "--controls", type=Path, help="JSON control recipes; kept outside primary results"
    )
    p.add_argument("--seed", type=int, help="Private reproducibility seed; random by default")
    p.add_argument(
        "--reverse", action="store_true", help="Separate reversed trials for isolated calls"
    )
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report", help="Report saved judgments; never infer missing decisions")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--review", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("human", help="Export one-condition human review with no repeated scenes")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--stage", choices=["primary", "controls"], default="primary")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            pairs = [value.split("=", 1) for value in args.run]
            if any(len(p) != 2 or not p[0] for p in pairs) or len({p[0] for p in pairs}) != len(
                pairs
            ):
                raise ValueError("Use unique CONDITION=DIRECTORY arguments")
            prepare(
                args.manifest,
                {k: Path(v) for k, v in pairs},
                args.output,
                args.split,
                args.seed,
                read_json(args.controls) if args.controls else None,
                args.reverse,
            )
        elif args.command == "human":
            export_human(args.pack, args.output, args.condition, args.stage)
        else:
            report(args.pack, args.review, args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
