"""Protect explicit retrieval experiments, reusable indexes and blinded review evidence."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from endless_voices.artifacts import file_hash

spec = importlib.util.spec_from_file_location(
    "retrieval_embedding_experiment", Path(__file__).with_name("run.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.fixture
def recorded(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"plan": {"queries": [{"id": "q1", "query": "How?"}]}}))
    text = "Opening. " + "Surrounding scene text. " * 100 + "Important final passage."
    for name in ("first", "second"):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "metadata": {
                "format": "retrieval-comparison-run-v2", "mode": "scene", "device": "mps",
                "model": f"private-model-{name}", "input_sha256": file_hash(frozen),
                "dtype": "float16", "max_length": 32768, "batch_size": 4, "limit": 5,
                "retrieval_source_sha256": "same-implementation", "runner_sha256": "runner",
                "query_instruction": "Find writing context", "tokenizer_identity": "common",
            },
            "queries": [{"query_id": "q1", "query": "How?", "documents": [
                {"text": text, "id": f"scene-{index}", "score": 0.95, "seconds": 1.2,
                 "model": "private-model", "matched_passage": {
                     "id": "passage", "line": 12, "passage_index": 2, "score": 0.95}}
                for index in range(5)]}],
        }))
    return Namespace(input=frozen, first=tmp_path / "first.json", second=tmp_path / "second.json",
                     output=tmp_path / "review", seed="fixed")


def test_review_retains_full_scenes_but_hides_model_scores_and_timing(recorded):
    comparison.blind(recorded)
    packet_text = (recorded.output / "review" / "blind.json").read_text()
    assert all(signal not in packet_text for signal in ("private-model", '"score"', '"seconds"'))
    packet = json.loads(packet_text)[0]
    assert set(packet["results"]) == {"A", "B"}
    original = json.loads(recorded.first.read_text())["queries"][0]["documents"]
    for documents in packet["results"].values():
        assert [document["text"] for document in documents] == [d["text"] for d in original]
        assert documents[0]["matched_passage"]["line"] == 12
    assert json.loads((recorded.output / "review" / "query-001.json").read_text()) == packet
    key = json.loads((recorded.output / "key.json").read_text())
    assert key["first"]["model"] == "private-model-first"
    assert set(key["mapping"]["q1"].values()) == {"first", "second"}


@pytest.mark.parametrize("setting,value", [
    ("mode", "passage"), ("max_length", 512), ("tokenizer_identity", "different"),
])
def test_comparison_rejects_different_settings_before_review(recorded, setting, value):
    result = json.loads(recorded.second.read_text())
    result["metadata"][setting] = value
    recorded.second.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=f"Comparison setting differs: {setting}"):
        comparison.blind(recorded)
    assert not recorded.output.exists()


def test_comparison_rejects_missing_explicit_mode(recorded):
    result = json.loads(recorded.second.read_text())
    del result["metadata"]["mode"]
    recorded.second.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="missing required settings"):
        comparison.blind(recorded)


@pytest.mark.parametrize("mode,reuse", [("passage", False), ("scene", True)])
def test_runner_uses_explicit_mode_common_tokenizer_and_existing_index(
    tmp_path, monkeypatch, mode, reuse,
):
    from transformers import AutoTokenizer

    from endless_voices import retrieval
    from endless_voices.context import ContextPool

    pool = ContextPool("sample", "conversation", "mission", "instruction", [], [], {})
    frozen = tmp_path / "input.json"
    frozen.write_text(json.dumps({
        "pools": {"mission": pool.__dict__},
        "plan": {"queries": [{"id": "query", "query": "Find context", "mission": "mission"}]},
    }))
    calls = []

    class Session:
        def __init__(self, database, pool):
            pass

        def search_context(self, query, *, limit, expand_scenes):
            calls.append(("expand", expand_scenes))
            return [{"id": "context", "score": 0.5}]

        def read_context(self, ids):
            return [{"id": "context", "text": "Complete returned context"}]

    class Database:
        def __init__(self, path, encoder):
            calls.append(("open", path))

        @classmethod
        def create(cls, path, blocks, encoder):
            calls.append(("create", path))
            path.write_text("index")
            return cls(path, encoder)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    tokenizer_path = tmp_path / "common-tokenizer"
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "tokenizer"),
                                special_tokens_map={}, encode=lambda text, **kw: text.split())

    def load_tokenizer(path, *, local_files_only):
        assert path == tokenizer_path
        assert local_files_only is True
        return tokenizer

    encoder = SimpleNamespace(identity="encoder", query_instruction="instruction",
                              model=SimpleNamespace(device="cpu", dtype="f32"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(comparison.TransformersEmbedder, "from_pretrained",
                        lambda *a, **kw: encoder)
    monkeypatch.setattr(comparison, "SceneDatabase", Database)
    monkeypatch.setattr(retrieval, "RetrievalSession", Session)
    index = tmp_path / "existing.sqlite"
    index.write_text("existing-index")
    output = tmp_path / "run"
    comparison.run(Namespace(input=frozen, model=tmp_path, revision="revision", output=output,
                             device="cpu", dtype="float32", max_length=32768, batch_size=4,
                             limit=5, mode=mode, tokenizer=tokenizer_path,
                             index=index if reuse else None))
    assert ("expand", mode == "scene") in calls
    assert any(call[0] == "create" for call in calls) is not reuse
    assert index.read_text() == "existing-index"
    result = json.loads((output / "results.json").read_text())
    assert result["metadata"]["mode"] == mode
    assert result["metadata"]["reused_index"] is reuse
    assert result["metadata"]["total_tokens"] == 3
