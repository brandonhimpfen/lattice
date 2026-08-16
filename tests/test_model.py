import pytest
from pydantic import BaseModel

from lattice import Message, Model, ResponseEnd, TextDelta, tool
from lattice.errors import (
    ConfigurationError,
    InvalidRequestError,
    ProviderNotInstalledError,
    StructuredOutputValidationError,
    UnsupportedCapabilityError,
)
from lattice.providers import ProviderCapabilities


class Summary(BaseModel):
    title: str
    topics: list[str]


def test_valid_identifier_properties():
    model = Model("fake:test:model")
    assert model.provider == "fake"
    assert model.model_name == "test:model"
    assert model.identifier == "fake:test:model"


@pytest.mark.parametrize("identifier", ["test", ":x", "fake:", "Fake:test"])
def test_invalid_model_identifiers_fail(identifier):
    with pytest.raises((ConfigurationError, ProviderNotInstalledError)):
        Model(identifier)


def test_missing_provider_is_detected_during_construction(monkeypatch):
    monkeypatch.setattr("lattice.providers.discovery._entry_points_for", lambda name: [])
    with pytest.raises(ProviderNotInstalledError):
        Model("unknown:test")


def test_plain_string_becomes_user_message():
    model = Model("fake:test", provider_options={"text": "ok"})
    response = model.invoke("hello")
    provider = model._provider_instance
    assert response.text == "ok"
    assert provider.requests[0].messages == [Message.user("hello")]  # type: ignore[union-attr]


def test_input_sequence_is_not_mutated():
    messages = [Message.user("hello")]
    before = list(messages)
    Model("fake:test").invoke(messages)
    assert messages == before


def test_empty_string_is_valid():
    assert Model("fake:test").invoke("").text == "Hello from fake"


def test_empty_message_sequence_fails():
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke([])


def test_mixed_message_sequence_fails():
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke([Message.user("x"), "y"])  # type: ignore[list-item]


def test_none_input_fails():
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_invalid_max_tokens_fails(value):
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke("x", max_tokens=value)


def test_stop_string_is_rejected():
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke("x", stop="END")


def test_empty_stop_value_is_rejected():
    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke("x", stop=[""])


def test_shallow_provider_option_merge():
    model = Model("fake:test", provider_options={"a": 1, "nested": {"a": 1}})
    model.invoke("x", provider_options={"b": 2, "nested": {"b": 2}})
    provider = model._provider_instance
    assert provider.requests[0].provider_options == {  # type: ignore[union-attr]
        "a": 1,
        "b": 2,
        "nested": {"b": 2},
    }


def test_common_option_conflict_fails():
    with pytest.raises(ConfigurationError):
        Model("fake:test").invoke("x", temperature=0.2, provider_options={"temperature": 0.9})


def test_request_timeout_overrides_model_timeout():
    model = Model("fake:test", timeout=30)
    model.invoke("x", timeout=5)
    provider = model._provider_instance
    assert provider.requests[0].timeout == 5  # type: ignore[union-attr]


def test_repr_does_not_expose_api_key():
    model = Model("fake:test", api_key="secret")
    assert "secret" not in repr(model)


def test_tool_calls_are_normalized():
    @tool
    def weather(city: str) -> str:
        return city

    response = Model(
        "fake:test",
        provider_options={"tool_call": True, "tool_args": {"city": "Toronto"}},
    ).invoke(
        "weather", tools=[weather]
    )
    assert response.tool_calls[0].name == "weather"
    assert response.tool_calls[0].arguments == {"city": "Toronto"}


def test_tools_and_structured_output_are_rejected_together():
    @tool
    def search(query: str) -> str:
        return query

    with pytest.raises(InvalidRequestError):
        Model("fake:test").invoke(
            "x", tools=[search], output=Summary
        )  # type: ignore[call-overload]


def test_structured_output_is_locally_validated():
    response = Model(
        "fake:test",
        provider_options={"structured": {"title": "DNS", "topics": ["networking"]}},
    ).invoke("x", output=Summary)
    assert isinstance(response.output, Summary)
    assert response.output.title == "DNS"


def test_invalid_structured_output_raises_validation_error():
    model = Model("fake:test", provider_options={"structured": {"title": "DNS", "topics": 3}})
    with pytest.raises(StructuredOutputValidationError):
        model.invoke("x", output=Summary)


@pytest.mark.asyncio
async def test_async_invocation():
    response = await Model("fake:test").ainvoke("x")
    assert response.text == "Hello from fake"


@pytest.mark.asyncio
async def test_sync_only_provider_rejects_async():
    with pytest.raises(UnsupportedCapabilityError):
        await Model("sync:test").ainvoke("x")


def test_sync_only_provider_rejects_streaming():
    with pytest.raises(UnsupportedCapabilityError):
        next(Model("sync:test").stream("x"))


def test_tool_capability_is_checked():
    @tool
    def search(query: str) -> str:
        return query

    caps = ProviderCapabilities(async_invocation=True, streaming=True, async_streaming=True)
    model = Model("fake:test", provider_options={"capabilities": caps})
    with pytest.raises(UnsupportedCapabilityError):
        model.invoke("x", tools=[search])


def test_stream_reconstructs_final_text():
    events = list(Model("fake:test").stream("x", provider_options={"stream_text": ["a", "", "b"]}))
    deltas = [event.text for event in events if isinstance(event, TextDelta)]
    ends = [event for event in events if isinstance(event, ResponseEnd)]
    assert deltas == ["a", "b"]
    assert len(ends) == 1
    assert ends[0].response.text == "ab"


@pytest.mark.asyncio
async def test_async_streaming():
    events = [event async for event in Model("fake:test").astream("x")]
    assert any(isinstance(event, ResponseEnd) for event in events)
