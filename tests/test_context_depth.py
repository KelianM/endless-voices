"""Protect reusable context selection and shared dataset/benchmark inputs."""

import json

import pytest

from endless_voices.context import (
    ContextPool,
    FullContext,
    MissionDepth,
    distances,
    pool_from_draft,
    strategy_from_config,
)
from endless_voices.prepare_context import load_selections, save_selections


class Counter:
    def text(self, text):
        return len(text)

    def messages(self, messages):
        return sum(len(m["content"]) for m in messages)


def pool():
    blocks = [{"mission": m, "heading": m, "path": "source.txt",
               "passages": [{"line": i, "role": "passage", "text": text}]}
              for i, (m, text) in enumerate([("recent", "hello"), ("old", "before"),
                                             ("huge", "long " * 100)])]
    return ContextPool("sample", "conversation", "now", "Lore\nHistory\n",
                       [{"role": "user", "content": "Question"}], blocks,
                       {"now": ["recent"], "recent": ["old", "huge"]})


def test_shared_ancestor_uses_shortest_distance_and_cycles_terminate():
    graph = {"now": ["long", "short"], "long": ["middle"],
             "middle": ["ancestor"], "short": ["ancestor"], "ancestor": ["now"]}
    assert distances(graph, "now")["ancestor"] == 2
    assert len(distances(graph, "now")) == 5


def test_sampling_keeps_whole_missions_within_budget_and_preserves_fixed_context():
    source = pool()
    strategy = MissionDepth(1, 15, "seed")
    selected = strategy.select(source, Counter())
    assert selected == strategy.select(source, Counter())
    assert [b["mission"] for b in selected.selected] == ["recent", "old"]
    assert [b["mission"] for b in selected.omitted] == ["huge"]
    assert selected.token_counts["sampled_history"] <= 15
    assert selected.messages[0]["content"].startswith(source.system_prefix)
    assert selected.messages[1:] == source.encounter
    source.blocks.reverse()
    reordered = strategy.select(source, Counter())
    assert reordered.provenance["sampled_missions"] == selected.provenance["sampled_missions"]


def test_full_strategy_retains_all_history_without_needing_graph_edges():
    source = pool()
    source.graph = {}
    selection = FullContext().select(source, Counter())
    assert selection.selected == source.blocks
    assert selection.omitted == []
    with pytest.raises(ValueError, match="absent"):
        MissionDepth(4, 8000, "seed").select(source, Counter())


def test_saved_selection_drives_training_generation_and_isolated_judge(tmp_path):
    selected = MissionDepth(1, 15, "seed").select(pool(), Counter())
    output = tmp_path / "bundle"
    save_selections(output, [selected], {"test_fixture": True})
    loaded = load_selections(output)[0]
    prompt = loaded.generation_prompt()
    assert prompt["messages"] == loaded.judge_context()
    assert loaded.training_messages("Private target")[:-1] == prompt["messages"]
    assert "Private target" not in json.dumps(prompt)
    assert "source.txt" not in json.dumps(loaded.judge_context())
    prompt["messages"][0]["content"] = "Changed by caller"
    assert loaded.messages == selected.messages
    with pytest.raises(FileExistsError):
        save_selections(output, [selected], {})
    (output / "prompts.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_selections(output)


@pytest.mark.parametrize("config", [
    {"name": "full", "depth": 4},
    {"name": "mission-depth", "depth": -1, "older_history_tokens": 8, "seed": "x"},
    {"name": "mission-depth", "depth": 4, "older_history_tokens": True, "seed": "x"},
])
def test_bad_strategy_configuration_cannot_silently_change_selection(config):
    with pytest.raises(ValueError):
        strategy_from_config(config)


def test_draft_adapter_rejects_mismatched_source_and_rendered_context():
    draft = {"sample_id": "s", "conversation_id": "c", "mission": "now",
             "source_blocks": [], "messages": [{"role": "system", "content": "Unrelated"},
                                                 {"role": "user", "content": "Question"}]}
    with pytest.raises(ValueError, match="does not match"):
        pool_from_draft(draft, {})
