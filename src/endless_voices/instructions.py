"""Shared task wording for authored scene passages."""

SCENE_INSTRUCTIONS = {
    "scene_continuation": (
        "Continue the scene with the next passage, keeping the characters and events "
        "authentic to the supplied context."
    ),
    "scene_opening": (
        "Write the opening passage of a scene, keeping the characters and events "
        "authentic to the supplied context."
    ),
}


def scene_task(*, has_prefix):
    """Derive the writing task from the presence of a current conversation prefix."""
    return "scene_continuation" if has_prefix else "scene_opening"


def character_reference(description):
    """Present character information without assigning the model a speaking role."""
    return "Character reference: " + description.removeprefix("You are ")
