"""Provider-neutral message objects."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tools import ToolCall

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """An immutable provider-neutral conversational message."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_tool_message(self) -> "Message":
        """Require a call identifier for messages carrying tool results."""
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool messages require tool_call_id")
        return self

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
        raw: Any = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls, raw=raw)

    @classmethod
    def tool(
        cls,
        content: str,
        *,
        tool_call_id: str,
        name: str | None = None,
    ) -> "Message":
        """Create a tool-result message."""
        return cls(
            role="tool",
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )
