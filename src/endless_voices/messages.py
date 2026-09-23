"""Dependency-free conversation structure shared by both data contracts."""


def validate_messages(messages: object) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError("roles must be system, user or assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be nonempty text")
        if role == "system" and index != 0:
            raise ValueError("system messages may only appear first")
    turns = messages[1:] if messages[0]["role"] == "system" else messages
    if len(turns) < 2 or len(turns) % 2:
        raise ValueError("provide complete user/assistant turn pairs")
    for index, message in enumerate(turns):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("turns must alternate user and assistant")
