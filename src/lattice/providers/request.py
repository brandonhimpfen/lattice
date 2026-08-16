"""Normalized request object passed from Lattice core to provider adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..messages import Message
from ..tools import Tool


class ModelRequest(BaseModel):
    """Provider-neutral representation of one model request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    output_type: type[BaseModel] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    timeout: float | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
