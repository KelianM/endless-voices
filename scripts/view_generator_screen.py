"""Build an offline reader for the completed four-scene generator screen."""

import json
from pathlib import Path

root = Path("outputs/generator-screen-v1")


def read(path):
    return json.loads(path.read_text())


labels = {
    "qwen4b-baseline": "Qwen3 · 4B",
    "qwen14b": "Qwen3 · 14B",
    "qwen30b": "Qwen3 · 30B A3B",
    "qwen36_27b-v2": "Qwen3.6 · 27B",
    "mistral24b-v2": "Mistral Small · 24B",
    "gemma26b": "Gemma 4 · 26B A4B",
    "gemma31b-v2": "Gemma 4 · 31B",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6-sol": "GPT-6 Sol",
    "claude-sonnet-5": "Claude Sonnet 5",
}
key = {v: k for k, v in read(root / "review-key.json").items()}
notes = read(root / "qualitative-notes-blind.json")
originals = read(root / "originals.json")
scenes = []
titles = [
    ("Republic Navy", "A reward for Farpoint"),
    ("Free Worlds", "After Parliament"),
    ("Hai", "Skill, chance and poker"),
    ("Quarg", "A visitor at the port"),
]
for i, prompt in enumerate(read(root / "prompts.json")):
    sid = prompt["sample_id"]
    responses = {}
    for model in labels:
        if model == "qwen4b-baseline":
            row = next(
                r for r in read(root / "qwen4b-baseline.json")["responses"] if r["sample_id"] == sid
            )
        else:
            row = read(root / model / (sid + ".result.json"))
        responses[model] = {
            "text": row["response"],
            "status": row["status"],
            "note": notes[f"scene_{i + 1}"][key[model]],
        }
    scenes.append(
        {
            "identity": titles[i][0],
            "title": titles[i][1],
            "id": sid,
            "context": prompt["messages"][0]["content"],
            "history": prompt["messages"][1:],
            "original": originals[sid]["messages"][-1]["content"],
            "responses": responses,
        }
    )
payload = json.dumps({"models": labels, "scenes": scenes}, ensure_ascii=False).replace(
    "<", "\\u003c"
)
template = Path("scripts/generator_screen_viewer.html").read_text()
out = Path("outputs/generator-screen-viewer/index.html")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as handle:
    handle.write(template.replace("__DATA__", payload))
print(out.resolve())
