"""Protect automatic context boundaries and whole-mission sampling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from measure_context_depth import distances, sample_older  # noqa: E402


def test_shared_ancestor_uses_shortest_distance_and_cycles_terminate():
    graph = {"now": ["long", "short"], "long": ["middle"],
             "middle": ["ancestor"], "short": ["ancestor"], "ancestor": ["now"]}
    assert distances(graph, "now")["ancestor"] == 2
    assert len(distances(graph, "now")) == 5


def test_sampling_is_reproducible_and_never_splits_an_oversized_mission():
    sizes = {"a": 3, "b": 4, "oversized": 20}
    def count(selected):
        return sum(sizes[m] for m in selected)
    first = sample_older(sizes, "scene", count, 7)
    assert first == {"a", "b"}
    assert first == sample_older(reversed(list(sizes)), "scene", count, 7)
    assert count(first) <= 7
