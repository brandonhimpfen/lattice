from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from lattice.errors import ProviderError
from lattice.providers import ModelRequest, ProviderCapabilities, register_provider
from lattice.providers.discovery import _reset_provider_registry_for_tests
from lattice.responses import Response, Usage
from lattice.streaming import ResponseStart, StreamAccumulator, StreamEvent
from lattice.tools import ToolCall


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        provider_options: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.provider_options = dict(provider_options)
        self.requests: list[ModelRequest] = []
        self.capability_override = self.provider_options.get("capabilities")

    @property
    def capabilities(self) -> ProviderCapabilities:
        if isinstance(self.capability_override, ProviderCapabilities):
            return self.capability_override
        return ProviderCapabilities(
            async_invocation=True,
            streaming=True,
            async_streaming=True,
            tools=True,
            structured_output=True,
            system_messages=True,
        )

    def invoke(self, request: ModelRequest) -> Response[Any]:
        self.requests.append(request)
        configured_error = request.provider_options.get("raise")
        if isinstance(configured_error, ProviderError):
            raise configured_error

        if request.output_type is not None:
            value = request.provider_options.get(
                "structured", {"title": "Example", "topics": ["test"]}
            )
            return Response[Any](
                provider=self.name,
                model=self.model,
                text=json.dumps(value),
                output=value,
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason="stop",
                raw={"fake": True},
            )

        if request.tools and request.provider_options.get("tool_call", False):
            first = request.tools[0]
            args = request.provider_options.get("tool_args", {})
            return Response[Any](
                provider=self.name,
                model=self.model,
                text="",
                output=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name=first.name,
                        arguments=args,
                        raw_arguments=json.dumps(args),
                    )
                ],
                finish_reason="tool_call",
            )

        text = request.provider_options.get("text", "Hello from fake")
        return Response[Any](
            id="resp_1",
            provider=self.name,
            model=self.model,
            text=text,
            output=None,
            usage=Usage(input_tokens=1, output_tokens=2, total_tokens=None),
            finish_reason="stop",
            raw={"fake": True},
        )

    async def ainvoke(self, request: ModelRequest) -> Response[Any]:
        return self.invoke(request)

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        accumulator = StreamAccumulator(
            provider=self.name, model=self.model, response_id="stream_1"
        )
        yield ResponseStart(id="stream_1", provider=self.name, model=self.model)
        for piece in request.provider_options.get("stream_text", ["Hello", " ", "world"]):
            event = accumulator.add_text(piece)
            if event is not None:
                yield event
        yield accumulator.finish(finish_reason="stop")

    async def astream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        for event in self.stream(request):
            yield event


class SyncOnlyProvider:
    name = "sync"

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs["model"]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def invoke(self, request: ModelRequest) -> Response[Any]:
        return Response[Any](provider=self.name, model=self.model, text="sync", output=None)


def fake_factory(**kwargs: Any) -> FakeProvider:
    return FakeProvider(**kwargs)


def sync_factory(**kwargs: Any) -> SyncOnlyProvider:
    return SyncOnlyProvider(**kwargs)


@pytest.fixture(autouse=True)
def reset_registry() -> Iterator[None]:
    _reset_provider_registry_for_tests()
    register_provider("fake", fake_factory)
    register_provider("sync", sync_factory)
    yield
    _reset_provider_registry_for_tests()
