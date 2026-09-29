"""Protect narrative targets from speech stripping, branch mixing and input leakage."""

import pytest

from endless_voices.continuations import continuation, target_text


def node(line, text, prose=True):
    return {'line': line, 'raw': f'`{text}`' if prose else text,
            'tokens': [text], 'children': []}


def test_complete_paragraph_keeps_actions_quotes_and_punctuation():
    text = '\t"Captain," she says, "thank you." She hands you a chip.'
    conversation = {'children': [node(1, text), node(2, 'choice', False)]}
    result = continuation(conversation, {'links': {'1': [2]}, 'choices': [2]}, [1, 1])
    assert result['text'] == text
    assert result['stop_before_lines'] == [2]


def test_selected_branch_keeps_intervening_narration_but_stops_before_response():
    conversation = {'children': [node(1, 'Opening'), node(2, 'branch', False),
                                node(3, 'Wrong outcome'), node(4, 'She climbs onto a ship.'),
                                node(5, '"Swear the oath."'), node(6, '"I will," he replies.')]}
    flow = {'links': {'1': [2], '2': [3, 4], '3': [6], '4': [5], '5': [6]},
            'choices': []}
    result = continuation(conversation, flow, [1, 5], response_boundaries=[6])
    assert [p['line'] for p in result['paragraphs']] == [1, 4, 5]
    assert result['stop_before_lines'] == [6]


def test_unresolved_outcomes_are_rejected_instead_of_combined():
    conversation = {'children': [node(1, 'Opening'), node(2, 'One ending'),
                                node(3, 'Another ending')]}
    with pytest.raises(ValueError, match='ambiguous'):
        continuation(conversation, {'links': {'1': [2, 3]}, 'choices': []}, [1])


def test_anchors_cannot_cross_a_player_choice():
    conversation = {'children': [node(1, 'Opening'), node(2, 'choice', False), node(3, 'Later')]}
    with pytest.raises(ValueError, match='missing or ambiguous'):
        continuation(conversation, {'links': {'1': [2], '2': [3]}, 'choices': [2]}, [1, 3])


def test_leak_check_catches_restored_paragraph_even_when_whole_target_is_absent():
    row = {'paragraphs': [{'text': 'Captain <last> arrives.'}, {'text': '"Hello."'}]}
    with pytest.raises(ValueError, match='appears in input'):
        target_text(row, [{'content': 'Captain Morgan arrives.'}], {'<last>': 'Morgan'})
    assert target_text(row, [], {'<last>': 'Morgan'}) == 'Captain Morgan arrives.\n\n"Hello."'
