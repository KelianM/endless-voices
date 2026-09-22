"""Hand-counted fixtures for descriptive source statistics, not game execution."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "inventory_sources", Path(__file__).parents[1] / "scripts" / "inventory_sources.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_dialogue_structure_and_disjoint_word_categories():
    source = '''mission "A mission"
\tdescription "A short job."
\ton offer
\t\tconversation
\t\t\t`Hello, visitor.`
\t\t\tchoice
\t\t\t\t`Hello back.`
\t\t\t\t\tgoto done
\t\t\tlabel done
\t\t\t`Goodbye.`
\t\t\t\tto display
\t\t\t\t\thas "some condition"
\t\tlog "Factions" "Hai" "A remembered visit."
\ton complete
\t\tconversation "A shared conversation"
\t\tdialog "A short message."
\t\tdialog phrase "some phrase"
'''
    counts, _ = inventory.measure(source)
    assert counts["root_mission"] == 1
    assert counts["conversation_blocks"] == 1
    assert counts["conversation_references"] == 1
    assert counts["conversation_words"] == 5
    assert counts["description_words"] == 3
    assert counts["dialog_words"] == 3
    assert counts["log_words"] == 3


def test_short_phrases_weights_comments_and_government_names():
    source = '''# no content
phrase "hail"
\tword
\t\t"Hi" 9 # weight
\t\t`Hello # captain` 2
government "Hai"
government "Hai"
government "Hai (Unfettered)"
'''
    counts, names = inventory.measure(source)
    assert counts["phrase_news_words"] == 4
    assert counts["root_government"] == 3
    assert names == {"Hai", "Hai (Unfettered)"}
    assert inventory.tokens('"some name" # comment') == ["some name"]
