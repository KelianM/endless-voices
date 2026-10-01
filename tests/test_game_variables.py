"""Protect source-derived mission locations from global dummy substitution."""

import pytest

from endless_voices.game_variables import mission_values, player_values, render


def node(**fields):
    return {'children': [{'tokens': [k, v], 'children': []} for k, v in fields.items()]}


def test_planet_is_destination_not_current_location_and_marks_are_systems():
    values = mission_values(node(source='Clark', destination='New Holland', mark='Wei'),
                            {'New Holland': 'Zeta Aquilae'}, {})
    assert values['<planet>'] == 'New Holland'
    assert values['<origin>'] == 'Clark'
    assert values['<destination>'] == 'New Holland in the Zeta Aquilae system'
    assert values['<marks>'] == 'Wei'


def test_dynamic_destination_does_not_fall_back_to_origin():
    mission = node(source='Clark')
    mission['children'].append({'tokens': ['destination'], 'children': [{'tokens': ['near']}]})
    assert '<planet>' not in mission_values(mission, {}, {})
    assert render('Go to <planet>.', {}) == ('Go to <planet>.', ['<planet>'])


def test_historical_missions_resolve_their_own_destination():
    first = mission_values(node(destination='Earth'), {'Earth': 'Sol'}, {})
    later = mission_values(node(destination='Bourne'), {'Bourne': 'Delta Pavonis'}, {})
    assert render('<destination>', first)[0] == 'Earth in the Sol system'
    assert render('<destination>', later)[0] == 'Bourne in the Delta Pavonis system'


def test_player_config_cannot_override_mission_state():
    with pytest.raises(ValueError, match='only'):
        player_values({'values': {'<first>': 'Alex', '<last>': 'Morgan', '<ship>': 'Finch',
                                  '<planet>': 'Example World'}})
