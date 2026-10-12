"""Protect blinded review from model identity leaks and mismatched experiment settings."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from endless_voices.artifacts import file_hash

spec = importlib.util.spec_from_file_location(
    "compare_retrieval_embeddings", "scripts/compare_retrieval_embeddings.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.fixture
def recorded(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"plan": {"queries": [{"id": "q1", "query": "How?"}]}}))
    for name in ("first", "second"):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "metadata": {
                "model": f"private-model-{name}", "input_sha256": file_hash(frozen),
                "dtype": "float16", "max_length": 32768, "batch_size": 4, "limit": 5,
                "retrieval_source_sha256": "same-implementation",
            },
            "queries": [{"query_id": "q1", "passages": [
                {"text": f"Passage {index}"} for index in range(5)]}],
        }))
    return Namespace(input=frozen, first=tmp_path / "first.json", second=tmp_path / "second.json",
                     output=tmp_path / "review", seed="fixed")


def test_review_packets_hide_model_identity_and_keep_key_separate(recorded):
    comparison.blind(recorded)
    packet_text = (recorded.output / "blind.json").read_text()
    assert "private-model" not in packet_text
    packet = json.loads(packet_text)[0]
    assert set(packet["results"]) == {"A", "B"}
    assert all(len(passages) == 3 for passages in packet["results"].values())
    key = json.loads((recorded.output / "key.json").read_text())
    assert key["first"]["model"] == "private-model-first"
    assert set(key["mapping"]["q1"].values()) == {"first", "second"}


def test_comparison_rejects_different_encoding_settings_before_review(recorded):
    result = json.loads(recorded.second.read_text())
    result["metadata"]["max_length"] = 512
    recorded.second.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Comparison setting differs: max_length"):
        comparison.blind(recorded)
    assert not Path(recorded.output).exists()


def test_runner_preserves_passage_results_when_session_default_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace

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

        def search_context(self, query, *, limit, expand_scenes=True):
            calls.append(expand_scenes)
            return []

        def read_context(self, ids):
            return []

    class Database:
        @staticmethod
        def create(*args):
            return Database()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    encoder = SimpleNamespace(identity="encoder",
                              model=SimpleNamespace(device="cpu", dtype="f32"))
    monkeypatch.setattr(comparison.TransformersEmbedder, "from_pretrained",
                        lambda *a, **kw: encoder)
    monkeypatch.setattr(comparison, "SceneDatabase", Database)
    monkeypatch.setattr(retrieval, "RetrievalSession", Session)
    output = tmp_path / "run"
    comparison.run(Namespace(input=frozen, model=tmp_path, revision="revision", output=output,
                             device="cpu", dtype="float32", max_length=32768, batch_size=4,
                             limit=5))
    assert calls == [False]
    assert json.loads((output / "results.json").read_text())["metadata"]["result_unit"] == "passage"
