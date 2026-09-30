"""Shared task wording for authored scene continuation."""

CONTINUATION_INSTRUCTION = (
    "Continue the scene with the next passage, keeping the characters and events "
    "authentic to the supplied context."
)


def character_reference(description):
    """Present character information without assigning the model a speaking role."""
    return "Character reference: " + description.removeprefix("You are ")
