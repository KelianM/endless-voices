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
