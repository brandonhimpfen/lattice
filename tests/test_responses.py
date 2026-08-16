from pydantic import BaseModel

from lattice import Response, ToolCall, Usage


class Result(BaseModel):
    value: str


def test_response_to_message_preserves_normalized_tool_calls():
    call = ToolCall(id="1", name="search", arguments={})
    response = Response[None](
        provider="fake",
        model="test",
        text="",
        output=None,
        tool_calls=[call],
        raw={"native": True},
    )
    message = response.to_message()
    assert message.role == "assistant"
    assert message.tool_calls == (call,)
    assert message.raw == {"native": True}


def test_raw_response_and_usage_are_excluded_from_dump():
    response = Response[None](
        provider="fake",
        model="test",
        text="hello",
        output=None,
        usage=Usage(input_tokens=1, raw=object()),
        raw=object(),
    )
    dump = response.model_dump()
    assert "raw" not in dump
    assert "raw" not in dump["usage"]


def test_missing_usage_values_remain_none():
    usage = Usage(input_tokens=100, output_tokens=20)
    assert usage.total_tokens is None
