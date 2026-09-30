"""Prevent mission and branch variants from crossing dataset splits."""

import pytest

from endless_voices.splits import assign_mission_splits, check_mission_splits


def record(mission, conversation, split, scenario=None):
    source = {'revision': 'revision', 'source_group': 'mission / ' + mission}
    return {'metadata': {'sources': [source],
                         'conversation_id': conversation, 'scenario_group': scenario,
                         'split': split}}


def test_different_conversations_in_one_mission_keep_the_heldout_assignment():
    rows = [record('A', 'first', 'train'), record('A', 'second', 'validation')]
    with pytest.raises(ValueError, match='crosses splits'):
        check_mission_splits(rows)
    assigned = assign_mission_splits(rows)
    assert [r['metadata']['split'] for r in assigned] == ['validation', 'validation']
    assert rows[0]['metadata']['split'] == 'train'
    check_mission_splits(assigned)


def test_shared_variants_propagate_test_protection_across_missions():
    rows = [record('A', 'first', 'train', 'variant'),
            record('B', 'second', 'validation', 'variant'), record('B', 'third', 'test')]
    assert {r['metadata']['split'] for r in assign_mission_splits(rows)} == {'test'}


def test_shared_lore_does_not_merge_unrelated_missions():
    rows = [record('A', 'first', 'train'), record('B', 'second', 'test')]
    for row in rows:
        row['metadata']['sources'].append({'revision': 'revision', 'source_group': 'shared lore'})
    assert [r['metadata']['split'] for r in assign_mission_splits(rows)] == ['train', 'test']
