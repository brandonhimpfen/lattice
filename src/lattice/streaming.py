"""Normalized streaming event types and stream accumulation."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ProviderResponseError, ToolCallParseError
from .responses import FinishReason, Response, Usage
from .tools import ToolCall


class StreamEvent(BaseModel):
    """Base type for normalized stream events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    raw: Any = Field(default=None, exclude=True)


class ResponseStart(StreamEvent):
    """Marks the start of a provider response stream."""

    type: Literal["response_start"] = "response_start"
    id: str | None = None
    provider: str
    model: str


class TextDelta(StreamEvent):
    """A non-empty text fragment emitted by a provider."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallDelta(StreamEvent):
    """A partial fragment of a streamed tool call."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


class ResponseEnd(StreamEvent):
    """Marks successful stream completion and carries the reconstructed response."""

    type: Literal["response_end"] = "response_end"
    response: Response[Any]


class _PendingToolCall:
    def __init__(self) -> None:
        self.id: str | None = None
        self.name: str | None = None
        self.arguments_parts: list[str] = []
        self.raw: Any = None


class StreamAccumulator:
    """Accumulate provider deltas into one normalized final response.

    Provider adapters can use this helper while translating native streaming APIs. It
    intentionally performs only provider-neutral reconstruction.
    """

    def __init__(self, *, provider: str, model: str, response_id: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.response_id = response_id
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, _PendingToolCall] = {}
        self._seen_ids: dict[str, int] = {}

    def add_text(self, text: str) -> TextDelta | None:
        """Accumulate and return a text delta, suppressing empty fragments."""
        if text == "":
            return None
        self._text_parts.append(text)
        return TextDelta(text=text)

    def add_tool_delta(
        self,
        *,
        index: int,
        id: str | None = None,
        name: str | None = None,
        arguments_delta: str = "",
        raw: Any = None,
    ) -> ToolCallDelta:
        """Accumulate one partial tool-call fragment."""
        pending = self._tool_calls.setdefault(index, _PendingToolCall())
        if id is not None:
            other = self._seen_ids.get(id)
            if other is not None and other != index:
                raise ProviderResponseError(
                    f"Provider returned duplicate tool call id '{id}'",
                    provider=self.provider,
                    model=self.model,
                    raw=raw,
                )
            self._seen_ids[id] = index
            if pending.id is not None and pending.id != id:
                raise ProviderResponseError(
                    f"Provider changed tool call id for stream index {index}",
                    provider=self.provider,
                    model=self.model,
                    raw=raw,
                )
            pending.id = id
        if name is not None:
            if pending.name is not None and pending.name != name:
                raise ProviderResponseError(
                    f"Provider changed tool name for stream index {index}",
                    provider=self.provider,
                    model=self.model,
                    raw=raw,
                )
            pending.name = name
        pending.arguments_parts.append(arguments_delta)
        if raw is not None:
            pending.raw = raw
        return ToolCallDelta(
            index=index,
            id=id,
            name=name,
            arguments_delta=arguments_delta,
            raw=raw,
        )

    def finish(
        self,
        *,
        usage: Usage | None = None,
        finish_reason: FinishReason | None = None,
        raw: Any = None,
    ) -> ResponseEnd:
        """Validate accumulated tool calls and construct the final response event."""
        calls: list[ToolCall] = []
        for index in sorted(self._tool_calls):
            pending = self._tool_calls[index]
            if not pending.name:
                raise ProviderResponseError(
                    f"Provider did not supply a tool name for stream index {index}",
                    provider=self.provider,
                    model=self.model,
                    raw=pending.raw,
                )
            raw_arguments = "".join(pending.arguments_parts) or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ToolCallParseError(
                    f"Unable to parse arguments for tool '{pending.name}'"
                ) from exc
            if not isinstance(arguments, dict):
                raise ToolCallParseError(
                    f"Tool arguments for '{pending.name}' must decode to an object"
                )
            if pending.id is None:
                call = ToolCall.with_generated_id(
                    name=pending.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    raw=pending.raw,
                )
            else:
                call = ToolCall(
                    id=pending.id,
                    name=pending.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    raw=pending.raw,
                )
            calls.append(call)

        response = Response[Any](
            id=self.response_id,
            provider=self.provider,
            model=self.model,
            text="".join(self._text_parts),
            output=None,
            tool_calls=calls,
            usage=usage,
            finish_reason=finish_reason,
            raw=raw,
        )
        return ResponseEnd(response=response)
