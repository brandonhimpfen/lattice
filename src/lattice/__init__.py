"""Lattice: a lightweight provider-neutral interface for language models."""

from .messages import Message
from .model import Model
from .prompts import Prompt
from .responses import FinishReason, Response, Usage
from .streaming import ResponseEnd, ResponseStart, StreamEvent, TextDelta, ToolCallDelta
from .tools import Tool, ToolCall, ToolResult, tool

__version__ = "0.1.0a1"

__all__ = [
    "FinishReason",
    "Message",
    "Model",
    "Prompt",
    "Response",
    "ResponseEnd",
    "ResponseStart",
    "StreamEvent",
    "TextDelta",
    "Tool",
    "ToolCall",
    "ToolCallDelta",
    "ToolResult",
    "Usage",
    "tool",
]
