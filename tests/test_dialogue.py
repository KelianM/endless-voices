"""Protect reachable scene branches and state changes from extraction errors."""

import pytest

from endless_voices.dataset.dialogue import DialogueInterpreter
from endless_voices.dataset.source import tree
from endless_voices.dataset.state import GameState, UnsupportedOperation


def run(body, state=None):
    return DialogueInterpreter().continuations(tree('conversation\n' + body)[0], state)


def test_embedded_question_and_answer_are_one_authored_passage():
    routes = run('\t`"How much?" a captain asks. "One million," the admiral replies.`\n'
                 '\tchoice\n\t\t`Agreed.`\n\t\t\taccept\n')
    targets = [r for r in routes if r.paragraphs]
    assert len(targets) == 1
    assert targets[0].text == '"How much?" a captain asks. "One million," the admiral replies.'
    assert targets[0].prefix == ()


def test_set_flag_eliminates_impossible_merchant_warning():
    routes = run('\taction\n\t\tset explained\n\t`An explanation.`\n'
                 '\tbranch done\n\t\thas explained\n\t`Redundant warning.`\n'
                 '\tlabel done\n\t`The report.`\n\t\tdecline\n')
    assert [r.text for r in routes] == ['An explanation.\n\nThe report.']


def test_unknown_background_retains_both_authored_variants():
    body = ('\t`Freya grew up on Earth.`\n\t`You also grew up there.`\n'
            '\t\tto display\n\t\t\thas "start: syndicate"\n\t`Goodnight.`\n'
            '\t\taccept\n')
    routes = run(body)
    assert len(routes) == 2
    for route in routes:
        initial = route.state.snapshot()['initial_values']
        assert ('You also' in route.text) == bool(initial['start: syndicate'])
        assert [r.text for r in run(body, GameState.fixed(initial))] == [route.text]


def test_hidden_paragraph_skips_its_goto_as_the_game_does():
    body = ('\t`Conditional.`\n\t\tto display\n\t\t\thas flag\n\t\tgoto end\n'
            '\t`Fallthrough.`\n\tlabel end\n\t`End.`\n\t\taccept\n')
    assert [r.text for r in run(body, GameState.fixed({'flag': 0}))] == ['Fallthrough.\n\nEnd.']


def test_repeated_condition_cannot_choose_contradictory_branches():
    routes = run('\tbranch yes\n\t\thas flag\n\t`No.`\n\t\tgoto end\n'
                 '\tlabel yes\n\tbranch impossible\n\t\tnot flag\n'
                 '\t`Yes.`\n\t\tgoto end\n\tlabel impossible\n\t`Impossible.`\n'
                 '\tlabel end\n\t`End.`\n\t\taccept\n')
    assert len(routes) == 2
    assert all('Impossible.' not in r.text for r in routes)


def test_assignment_and_numeric_clamp_affect_later_conditions():
    routes = run('\taction\n\t\treputation = -2\n\t\treputation >?= 1\n'
                 '\t\tother = reputation\n\tbranch yes\n\t\tother >= 1\n'
                 '\t`Wrong.`\n\t\tdecline\n\tlabel yes\n\t`Right.`\n\t\taccept\n')
    assert [r.text for r in routes] == ['Right.']


def test_player_options_create_separate_following_examples():
    routes = run('\t`Choose.`\n\tchoice\n\t\t`Left.`\n\t\t\tgoto left\n'
                 '\t\t`Right.`\n\t\t\tgoto right\n\tlabel left\n'
                 '\t`Left result.`\n\t\taccept\n\tlabel right\n'
                 '\t`Right result.`\n\t\taccept\n')
    targets = {r.text: [p.text for p in r.prefix] for r in routes if r.paragraphs}
    assert targets['Left result.'] == ['Choose.', 'Left.']
    assert targets['Right result.'] == ['Choose.', 'Right.']


def test_goto_accept_label_does_not_end_the_conversation():
    assert [r.text for r in run('\tgoto accept\n\tlabel accept\n\t`Actual text.`\n'
                               '\t\taccept\n')] == ['Actual text.']


def test_unsupported_action_and_loops_fail_explicitly():
    with pytest.raises(UnsupportedOperation, match='outfit'):
        run('\taction\n\t\toutfit Laser\n\t`Hello.`\n')
    with pytest.raises(ValueError, match='loop'):
        run('\tlabel loop\n\t`Again.`\n\t\tgoto loop\n')


def test_all_hidden_choices_do_not_split_a_passage():
    routes = run('\t`First.`\n\tchoice\n\t\t`Unavailable.`\n'
                 '\t\t\tto display\n\t\t\t\tnever\n\t`Second.`\n\t\taccept\n')
    assert [r.text for r in routes] == ['First.\n\nSecond.']


def test_unimplemented_random_condition_is_not_a_persistent_flag():
    with pytest.raises(UnsupportedOperation, match='Random'):
        run('\tbranch end\n\t\trandom < 10\n\t`Hello.`\n\tlabel end\n')
