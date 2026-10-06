"""Protect reachable scene branches and state changes from extraction errors."""

import pytest

from endless_voices.dataset.dialogue import DialogueInterpreter
from endless_voices.dataset.source import tree
from endless_voices.dataset.state import GameState, UnsupportedOperation


def run(body, state=None):
    return DialogueInterpreter().continuations(tree("conversation\n" + body)[0], state)


def test_embedded_question_and_answer_are_one_authored_passage():
    routes = run(
        '\t`"How much?" a captain asks. "One million," the admiral replies.`\n'
        "\tchoice\n\t\t`Agreed.`\n\t\t\taccept\n"
    )
    targets = [r for r in routes if r.paragraphs]
    assert len(targets) == 1
    assert targets[0].text == '"How much?" a captain asks. "One million," the admiral replies.'
    assert targets[0].prefix == ()


def test_set_flag_eliminates_impossible_merchant_warning():
    routes = run(
        "\taction\n\t\tset explained\n\t`An explanation.`\n"
        "\tbranch done\n\t\thas explained\n\t`Redundant warning.`\n"
        "\tlabel done\n\t`The report.`\n\t\tdecline\n"
    )
    assert [r.text for r in routes] == ["An explanation.\n\nThe report."]


def test_unknown_background_retains_both_authored_variants():
    body = (
        "\t`Freya grew up on Earth.`\n\t`You also grew up there.`\n"
        '\t\tto display\n\t\t\thas "start: syndicate"\n\t`Goodnight.`\n'
        "\t\taccept\n"
    )
    routes = run(body)
    assert len(routes) == 2
    for route in routes:
        initial = route.state.snapshot()["initial_values"]
        assert ("You also" in route.text) == bool(initial["start: syndicate"])
        assert [r.text for r in run(body, GameState.fixed(initial))] == [route.text]


def test_hidden_paragraph_skips_its_goto_as_the_game_does():
    body = (
        "\t`Conditional.`\n\t\tto display\n\t\t\thas flag\n\t\tgoto end\n"
        "\t`Fallthrough.`\n\tlabel end\n\t`End.`\n\t\taccept\n"
    )
    assert [r.text for r in run(body, GameState.fixed({"flag": 0}))] == ["Fallthrough.\n\nEnd."]


def test_repeated_condition_cannot_choose_contradictory_branches():
    routes = run(
        "\tbranch yes\n\t\thas flag\n\t`No.`\n\t\tgoto end\n"
        "\tlabel yes\n\tbranch impossible\n\t\tnot flag\n"
        "\t`Yes.`\n\t\tgoto end\n\tlabel impossible\n\t`Impossible.`\n"
        "\tlabel end\n\t`End.`\n\t\taccept\n"
    )
    assert len(routes) == 2
    assert all("Impossible." not in r.text for r in routes)


def test_assignment_and_numeric_clamp_affect_later_conditions():
    routes = run(
        "\taction\n\t\treputation = -2\n\t\treputation >?= 1\n"
        "\t\tother = reputation\n\tbranch yes\n\t\tother >= 1\n"
        "\t`Wrong.`\n\t\tdecline\n\tlabel yes\n\t`Right.`\n\t\taccept\n"
    )
    assert [r.text for r in routes] == ["Right."]


def test_player_options_create_separate_following_examples():
    routes = run(
        "\t`Choose.`\n\tchoice\n\t\t`Left.`\n\t\t\tgoto left\n"
        "\t\t`Right.`\n\t\t\tgoto right\n\tlabel left\n"
        "\t`Left result.`\n\t\taccept\n\tlabel right\n"
        "\t`Right result.`\n\t\taccept\n"
    )
    targets = {r.text: [p.text for p in r.prefix] for r in routes if r.paragraphs}
    assert targets["Left result."] == ["Choose.", "Left."]
    assert targets["Right result."] == ["Choose.", "Right."]


def test_goto_accept_label_does_not_end_the_conversation():
    assert [
        r.text for r in run("\tgoto accept\n\tlabel accept\n\t`Actual text.`\n\t\taccept\n")
    ] == ["Actual text."]


def test_unsupported_action_and_loops_fail_explicitly():
    with pytest.raises(UnsupportedOperation, match="ship"):
        run("\taction\n\t\tship Sparrow\n\t`Hello.`\n")
    with pytest.raises(ValueError, match="loop"):
        run("\tlabel loop\n\t`Again.`\n\t\tgoto loop\n")


def test_all_hidden_choices_do_not_split_a_passage():
    routes = run(
        "\t`First.`\n\tchoice\n\t\t`Unavailable.`\n"
        "\t\t\tto display\n\t\t\t\tnever\n\t`Second.`\n\t\taccept\n"
    )
    assert [r.text for r in routes] == ["First.\n\nSecond."]


def test_random_reads_are_independent_and_bounded():
    routes = run(
        "\tbranch end\n\t\trandom >= 50\n"
        "\tbranch end\n\t\trandom < 50\n"
        "\t`Independent draws.\x60\n\tlabel end\n"
    )
    route = next(r for r in routes if r.text == "Independent draws.")
    draws = list(route.state.snapshot()["draws"].values())
    assert 0 <= draws[0] < 50 <= draws[1] < 100


def test_question_menu_can_repeat_when_answers_change_state():
    text = "\tlabel menu\n\tchoice\n"
    for name in ["one", "two", "three"]:
        text += f"\t\t`Ask {name}.`\n\t\t\tgoto {name}\n\t\t\tto display\n\t\t\t\tnot {name}\n"
    text += "\t\t`Leave.`\n\t\t\taccept\n"
    for name in ["one", "two", "three"]:
        text += f"\tlabel {name}\n\taction\n\t\tset {name}\n\t`Answer {name}.`\n\t\tgoto menu\n"
    routes = run(text)
    assert any(
        all(r.state.snapshot()["current_values"].get(name) == 1 for name in ["one", "two", "three"])
        for r in routes
        if r.terminal
    )


def test_repeatable_menu_retains_each_answer_and_finite_exit_histories():
    routes = run(
        "\tlabel menu\n\tchoice\n\t\t`Ask about Earth.`\n\t\t\tgoto earth\n"
        "\t\t`Ask about Mars.`\n\t\t\tgoto mars\n\t\t`Leave.`\n\t\t\taccept\n"
        "\tlabel earth\n\t`Earth answer.`\n\t\tgoto menu\n"
        "\tlabel mars\n\t`Mars answer.`\n\t\tgoto menu\n"
    )
    assert {r.text for r in routes if r.paragraphs} == {"Earth answer.", "Mars answer."}
    complete = [r for r in routes if r.terminal]
    assert complete and all(r.stop == "accept" for r in complete)
    assert any({"Earth answer.", "Mars answer."} <= {p.text for p in r.prefix}
               for r in complete)
    assert all(sum(p.text == "Ask about Earth." for p in r.prefix) <= 2 for r in complete)


def test_reference_route_sampling_finishes_without_enumerating_other_branches():
    conversation = tree('''conversation
\tchoice
\t\t`Earth.`
\t\t\taccept
\t\t`Mars.`
\t\t\tdecline
''')[0]
    interpreter = DialogueInterpreter(max_steps=2)
    sampled = interpreter.histories(conversation, GameState(), sample_seed="reference")
    assert len(sampled) == 1 and sampled[0].terminal
    assert sampled[0].prefix == interpreter.histories(
        conversation, GameState(), sample_seed="reference")[0].prefix
    with pytest.raises(ValueError, match="route limit"):
        interpreter.histories(conversation, GameState())
    assert len(DialogueInterpreter().histories(conversation, GameState())) == 2


def test_event_outfit_requirement_applies_before_initial_dialogue():
    conversation = tree('conversation\n\t`You have a brig.`')[0]
    interpreter = DialogueInterpreter()
    requirements = tree('require Brig')
    assert not interpreter.histories(conversation, GameState.fixed({"outfit: Brig": 0}),
                                     requirements)
    routes = interpreter.histories(conversation, GameState.fixed({"outfit: Brig": 1}),
                                   requirements)
    assert routes[0].text == "You have a brig."


def test_large_question_menu_keeps_answers_without_permuting_histories():
    body = "\tlabel menu\n\tchoice\n"
    for index in range(12):
        body += f"\t\t`Question {index}.`\n\t\t\tgoto answer{index}\n"
    body += "\t\t`Leave.`\n\t\t\taccept\n"
    for index in range(12):
        body += f"\tlabel answer{index}\n\t`Answer {index}.`\n\t\tgoto menu\n"
    routes = DialogueInterpreter(max_steps=1000).continuations(tree("conversation\n" + body)[0])
    assert {r.text for r in routes if r.paragraphs} == {
        f"Answer {index}." for index in range(12)
    }
    assert any(r.terminal and r.stop == "accept" for r in routes)
    for route in routes:
        if route.paragraphs:
            index = route.paragraphs[0].text.split()[1]
            assert route.prefix[-1].text == f"Question {index}"


def test_menu_state_changes_preserve_new_answers_after_shared_paths():
    routes = run(
        "\tlabel menu\n\tchoice\n\t\t`Ask.`\n\t\t\tgoto answer\n"
        "\t\t`Reveal.`\n\t\t\tgoto reveal\n\t\t`Leave.`\n\t\t\taccept\n"
        "\tlabel answer\n\tbranch known\n\t\thas secret\n"
        "\t`Before the revelation.`\n\t\tgoto menu\n"
        "\tlabel known\n\t`After the revelation.`\n\t\tgoto menu\n"
        "\tlabel reveal\n\taction\n\t\tset secret\n\tgoto menu\n",
        GameState.fixed({"secret": 0}),
    )
    assert {r.text for r in routes if r.paragraphs} == {
        "Before the revelation.", "After the revelation."
    }
    for route in routes:
        if route.text == "After the revelation.":
            assert any(p.text == "Reveal." for p in route.prefix)
            assert route.state.snapshot()["current_values"]["secret"] == 1


def test_sampled_history_searches_past_incompatible_endpoint():
    conversation = tree(
        'conversation\n\tchoice\n\t\t`Refuse.`\n\t\t\tdecline\n'
        '\t\t`Agree.`\n\t\t\taccept\n'
    )[0]
    interpreter = DialogueInterpreter()
    for seed in range(8):
        routes = interpreter.histories(
            conversation, GameState(), sample_seed=seed, permitted_stops={"accept"}
        )
        assert len(routes) == 1 and routes[0].stop == "accept"
        assert routes[0].prefix[-1].text == "Agree."


def test_repeatable_purchase_converges_when_money_cannot_change_target_dialogue():
    conversation = tree(
        'conversation\n\tlabel menu\n\tchoice\n\t\t`Buy.`\n\t\t\tgoto buy\n'
        '\t\t`Leave.`\n\t\t\taccept\n\tlabel buy\n\taction\n\t\tpayment -10\n'
        '\t`Purchased.`\n\t\tgoto menu\n'
    )[0]
    interpreter = DialogueInterpreter(max_steps=100)
    routes = interpreter.continuations(conversation, future_variables=set())
    assert {r.text for r in routes if r.paragraphs} == {"Purchased."}
    assert any(r.terminal for r in routes)
    later_observer = tree(
        'conversation\n\tchoice\n\t\t`Buy.`\n\t\t\tgoto buy\n'
        '\t\t`Save.`\n\t\t\tgoto end\n\tlabel buy\n\taction\n\t\tpayment -10\n'
        '\tlabel end\n\t`Farewell.`\n\t\taccept\n'
    )[0]
    outcomes = interpreter.histories(
        later_observer, GameState.fixed({"credits": 15}), future_variables={"credits"})
    assert {r.state.snapshot()["current_values"]["credits"] for r in outcomes} == {5, 15}
