"""Supported provider-development protocols and capability metadata."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..responses import Response
from ..streaming import StreamEvent
from .request import ModelRequest


class ProviderCapabilities(BaseModel):
    """Capabilities implemented by a provider adapter."""

    async_invocation: bool = False
    streaming: bool = False
    async_streaming: bool = False
    tools: bool = False
    structured_output: bool = False
    system_messages: bool = True
    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False
    provider_features: frozenset[str] = frozenset()


@runtime_checkable
class Provider(Protocol):
    """Minimum provider interface required by Lattice."""

    name: str

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return adapter-level capabilities."""
        ...

    def invoke(self, request: ModelRequest) -> Response[Any]:
        """Perform a synchronous model invocation."""
        ...


@runtime_checkable
class AsyncProvider(Protocol):
    """Optional provider protocol for native asynchronous invocation."""

    async def ainvoke(self, request: ModelRequest) -> Response[Any]:
        """Perform an asynchronous model invocation."""
        ...


@runtime_checkable
class StreamingProvider(Protocol):
    """Optional provider protocol for synchronous streaming."""

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        """Yield normalized streaming events."""
        ...


@runtime_checkable
class AsyncStreamingProvider(Protocol):
    """Optional provider protocol for asynchronous streaming."""

    def astream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Yield normalized streaming events asynchronously."""
        ...


class ProviderFactory(Protocol):
    """Factory used to construct a provider adapter for a configured model."""

    def __call__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        provider_options: Mapping[str, Any],
    ) -> Provider:
        ...
