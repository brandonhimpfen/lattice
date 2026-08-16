import pytest

from lattice.errors import ProviderResponseError, ToolCallParseError
from lattice.streaming import ResponseEnd, StreamAccumulator


def test_empty_text_delta_is_suppressed():
    accumulator = StreamAccumulator(provider="fake", model="test")
    assert accumulator.add_text("") is None


def test_final_response_reconstructs_text_deltas():
    accumulator = StreamAccumulator(provider="fake", model="test")
    accumulator.add_text("Hello")
    accumulator.add_text(" world")
    end = accumulator.finish(finish_reason="stop")
    assert isinstance(end, ResponseEnd)
    assert end.response.text == "Hello world"


def test_stream_accumulator_reconstructs_fragmented_tool_arguments():
    accumulator = StreamAccumulator(provider="fake", model="test")
    accumulator.add_tool_delta(index=0, id="call_1", name="weather", arguments_delta='{"ci')
    accumulator.add_tool_delta(index=0, arguments_delta='ty":"Toronto"}')
    end = accumulator.finish(finish_reason="tool_call")
    assert end.response.tool_calls[0].arguments == {"city": "Toronto"}


def test_malformed_final_tool_arguments_fail():
    accumulator = StreamAccumulator(provider="fake", model="test")
    accumulator.add_tool_delta(index=0, id="call_1", name="weather", arguments_delta='{"city":')
    with pytest.raises(ToolCallParseError):
        accumulator.finish()


def test_duplicate_tool_call_ids_for_distinct_indexes_fail():
    accumulator = StreamAccumulator(provider="fake", model="test")
    accumulator.add_tool_delta(index=0, id="same", name="a")
    with pytest.raises(ProviderResponseError):
        accumulator.add_tool_delta(index=1, id="same", name="b")
