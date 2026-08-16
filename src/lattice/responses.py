"""Normalized model response objects."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .messages import Message
from .tools import ToolCall

T = TypeVar("T")
FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error", "other"]


class Usage(BaseModel):
    """Normalized token-accounting data returned by a provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: Any = Field(default=None, exclude=True)


class Response(BaseModel, Generic[T]):
    """A normalized provider response.

    ``output`` is ``None`` for ordinary text calls and contains the validated Pydantic
    object requested by a structured-output call.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | None = None
    provider: str
    model: str
    text: str = ""
    output: T
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    raw: Any = Field(default=None, exclude=True)

    def to_message(self) -> Message:
        """Convert the response into an assistant message for conversation continuation."""
        return Message.assistant(
            self.text,
            tool_calls=tuple(self.tool_calls),
            raw=self.raw,
        )
