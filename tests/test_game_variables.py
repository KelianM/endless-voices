"""Protect source-derived mission locations from global dummy substitution."""

import pytest

from endless_voices.gameplay.variables import mission_values, player_values, render


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


def test_filtered_locations_and_npc_names_are_reproducible(tmp_path):
    from test_dataset_builder import corpus

    from endless_voices.gameplay.source import tree

    source = corpus(tmp_path, '''system Sol
	link Near
system Near
	link Sol
	link Far
system Far
	government Syndicate
	link Near
	object Alpha
	object Beta
	object Station
	object Wrong
planet Alpha
	spaceport `A port.`
planet Beta
	spaceport `A port.`
planet Station
	attributes station
	spaceport `A station.`
planet Wrong
	government Republic
	spaceport `A port.`
ship Argosy
ship Argosy "Argosy (Blaster)"
mission Hijack
	source
		government Syndicate
		attributes spaceport
		not attributes station
		not near Sol 1
	npc
		ship "Argosy (Blaster)" "Moonshake Express"
''')
    mission = source.missions['Hijack'].node

    def resolve(seed, world=()):
        return mission_values(mission, source.planet_systems, {}, corpus=source,
                              seed=seed, world=world)

    first = resolve('fixed')
    assert first == resolve('fixed')
    source.planets = dict(reversed(list(source.planets.items())))
    assert first == resolve('fixed')
    assert first['<origin>'] in {'Alpha', 'Beta'}
    assert first['<planet>'] == first['<origin>']
    assert first['<npc>'] == 'Moonshake Express'
    assert first['<npc model>'] == 'Argosy'
    assert {resolve(str(i))['<origin>'] for i in range(20)} == {'Alpha', 'Beta'}
    world = tree('planet Alpha\n\tgovernment Republic\n')
    assert resolve('fixed', world)['<origin>'] == 'Beta'
    mission['children'][0]['children'] += tree('visited planet')
    assert '<origin>' not in resolve('fixed')


def test_location_filters_keep_grouping_and_distance_bounds(tmp_path):
    from test_dataset_builder import corpus

    from endless_voices.gameplay.locations import Locations
    from endless_voices.gameplay.source import tree

    source = corpus(tmp_path, '''system A
	government Republic
	link B
	object Home
system B
	government Syndicate
	link A
	link C
	object Port
system C
	government Syndicate
	link B
	object End
planet Home
	attributes rich
	spaceport `A port.`
planet Port
	attributes rich industrial
	spaceport `A port.`
planet End
	attributes poor
	spaceport `A port.`
''')
    locations = Locations(source)
    assert locations.matches(tree('government Republic\ngovernment Syndicate'), 'B', 'Port')
    assert locations.matches(tree('attributes poor rich\nattributes industrial'), 'B', 'Port')
    assert not locations.matches(tree('attributes poor rich\nattributes industrial'), 'C', 'End')
    assert locations.matches(tree('neighbor government Republic'), 'B', 'Port')
    assert not locations.matches(tree('not\n\tgovernment Syndicate\n\tattributes rich'),
                                 'B', 'Port')
    assert locations.matches(tree('not\n\tgovernment Syndicate\n\tattributes rich'), 'C', 'End')
    assert locations.matches(tree('distance 1 1'), 'B', 'Port', 'A')
    assert not locations.matches(tree('distance 1 1'), 'C', 'End', 'A')
    changed = Locations(source, tree('unlink A B\nlink A C'))
    assert not changed.matches(tree('distance 1 1'), 'B', 'Port', 'A')
    assert changed.matches(tree('distance 1 1'), 'C', 'End', 'A')


def test_mission_substitutions_remain_fixed_after_world_changes(tmp_path):
    from test_dataset_builder import corpus

    from endless_voices.gameplay.dialogue import DialogueInterpreter
    from endless_voices.gameplay.histories import HistoryReconstructor
    from endless_voices.gameplay.source import tree
    from endless_voices.gameplay.state import GameState

    source = corpus(tmp_path, '''system Start
	government Syndicate
	object Alpha
planet Alpha
	spaceport `A port.`
mission Job
	source
		government Syndicate
''')
    gameplay = HistoryReconstructor(source, DialogueInterpreter(), {}, seed='fixed')
    state = GameState()
    mission = source.missions['Job']
    gameplay.enter(state, mission)
    state.world = tuple(tree('planet Alpha\n\tgovernment Republic'))
    copied = state.copy()
    assert gameplay.values(mission, copied)['<origin>'] == 'Alpha'
    assert copied.snapshot()['substitutions']['Job']['<origin>'] == 'Alpha'
    copied.substitutions['Job']['<origin>'] = 'Elsewhere'
    assert gameplay.values(mission, state)['<origin>'] == 'Alpha'
