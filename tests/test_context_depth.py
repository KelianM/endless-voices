"""Protect shortest mission distances used to prioritize history."""

from endless_voices.context import distances


def test_shared_ancestor_uses_shortest_distance_and_cycles_terminate():
    graph = {"now": ["long", "short"], "long": ["middle"],
             "middle": ["ancestor"], "short": ["ancestor"], "ancestor": ["now"]}
    assert distances(graph, "now")["ancestor"] == 2
    assert len(distances(graph, "now")) == 5
