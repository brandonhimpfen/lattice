import pytest
from pydantic import ValidationError

from lattice import Message, ToolCall


def test_message_convenience_constructors():
    assert Message.system("a").role == "system"
    assert Message.user("b").role == "user"
    assert Message.assistant("c").role == "assistant"


def test_tool_message_requires_call_id():
    with pytest.raises(ValidationError):
        Message(role="tool", content="result")


def test_empty_content_is_valid():
    assert Message.user("").content == ""


def test_message_is_immutable():
    message = Message.user("hello")
    with pytest.raises(ValidationError):
        message.content = "changed"  # type: ignore[misc]


def test_raw_is_excluded_from_serialization():
    message = Message(role="assistant", content="x", raw=object())
    assert "raw" not in message.model_dump()


def test_assistant_can_carry_tool_calls():
    call = ToolCall(id="1", name="search", arguments={})
    message = Message.assistant("", tool_calls=(call,))
    assert message.tool_calls == (call,)
