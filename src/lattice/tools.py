"""Tool definitions, schema generation, tool calls, and tool results."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping
from enum import Enum
from functools import wraps
from types import NoneType, UnionType
from typing import Any, Literal, TypeAlias, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .errors import ToolResultSerializationError, ToolSchemaError

_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_TOOL_METADATA_ATTR = "__lattice_tool_metadata__"


class ToolCall(BaseModel):
    """A normalized tool invocation requested by a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    raw: Any = Field(default=None, exclude=True)

    @classmethod
    def with_generated_id(
        cls,
        *,
        name: str,
        arguments: dict[str, Any],
        raw_arguments: str | None = None,
        raw: Any = None,
    ) -> "ToolCall":
        """Create a tool call with an invocation-local identifier."""
        return cls(
            id=f"call_{uuid4().hex}",
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            raw=raw,
        )


class ToolResult(BaseModel):
    """The application-produced result of a tool call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    name: str
    output: Any
    is_error: bool = False

    def to_message(self) -> "Message":
        """Convert this result into a normalized tool message."""
        from .messages import Message

        content = _serialize_tool_result(self.output)
        return Message.tool(
            content,
            tool_call_id=self.tool_call_id,
            name=self.name,
        )


class Tool(BaseModel):
    """Provider-neutral metadata describing a Python callable exposed to a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any]
    function: Callable[..., Any]

    @classmethod
    def from_callable(cls, function: Callable[..., Any]) -> "Tool":
        """Build a tool from a callable and any metadata attached by :func:`tool`."""
        metadata = getattr(function, _TOOL_METADATA_ATTR, None) or {}
        name = metadata.get("name") or function.__name__
        description = metadata.get("description")
        if description is None:
            description = inspect.getdoc(function) or ""
        _validate_tool_name(name)
        return cls(
            name=name,
            description=description,
            parameters=build_tool_schema(function),
            function=function,
        )


ToolLike: TypeAlias = Tool | Callable[..., Any]


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Attach Lattice tool metadata while preserving ordinary callable behavior.

    The decorator can be used as ``@tool`` or ``@tool(name=..., description=...)``.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if name is not None:
            _validate_tool_name(name)
        metadata = {"name": name, "description": description}

        # Preserve the original calling semantics and metadata while returning a normal
        # function object. The wrapper also gives us a safe place for Lattice metadata.
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await function(*args, **kwargs)

            setattr(async_wrapper, _TOOL_METADATA_ATTR, metadata)
            return async_wrapper

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        setattr(wrapper, _TOOL_METADATA_ATTR, metadata)
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate


def build_tool_schema(function: Callable[..., Any]) -> dict[str, Any]:
    """Generate a JSON Schema object for a callable's input parameters."""
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function, include_extras=True)
    except Exception as exc:  # pragma: no cover - defensive around arbitrary annotations
        raise ToolSchemaError(
            f"Unable to resolve type annotations for '{function.__name__}'"
        ) from exc

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            raise ToolSchemaError(
                f"Tool '{function.__name__}' cannot use variadic parameter '{parameter.name}'"
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ToolSchemaError(
                f"Tool '{function.__name__}' cannot use positional-only parameter "
                f"'{parameter.name}'"
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise ToolSchemaError(
                f"Tool parameter '{parameter.name}' on '{function.__name__}' "
                "requires a type annotation"
            )
        schema = _schema_for_annotation(annotation)
        if parameter.default is not inspect.Signature.empty:
            schema = dict(schema)
            schema["default"] = parameter.default
        else:
            required.append(parameter.name)
        properties[parameter.name] = schema

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def normalize_tools(tools: list[ToolLike] | tuple[ToolLike, ...] | None) -> list[Tool]:
    """Normalize callables and explicit Tool objects and reject duplicate names."""
    if not tools:
        return []
    normalized = [item if isinstance(item, Tool) else Tool.from_callable(item) for item in tools]
    seen: set[str] = set()
    for item in normalized:
        _validate_tool_name(item.name)
        if item.name in seen:
            raise ToolSchemaError(f"Duplicate tool name '{item.name}'")
        seen.add(item.name)
    return normalized


def _validate_tool_name(name: str) -> None:
    if not _TOOL_NAME_RE.fullmatch(name):
        raise ToolSchemaError(
            "Tool names must start with a letter or underscore, contain only letters, "
            "numbers, underscores, or hyphens, and be at most 64 characters"
        )


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """Return JSON Schema for the supported 0.1 annotation subset."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is Any:
        raise ToolSchemaError("Tool parameters annotated as Any are not supported in Lattice 0.1")
    if annotation is None or annotation is NoneType:
        return {"type": "null"}
    if annotation in {str, int, float, bool}:
        return TypeAdapter(annotation).json_schema()
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return TypeAdapter(annotation).json_schema()
    if origin is Literal:
        return TypeAdapter(annotation).json_schema()
    if origin in {list, dict, Union, UnionType}:
        if origin is dict:
            if len(args) != 2 or args[0] is not str:
                raise ToolSchemaError("Only dict[str, T] tool parameters are supported")
        try:
            schema = TypeAdapter(annotation).json_schema()
        except Exception as exc:
            raise ToolSchemaError(f"Unsupported tool annotation: {annotation!r}") from exc
        _reject_unconstrained_schema(schema, annotation)
        return schema

    raise ToolSchemaError(f"Unsupported tool annotation: {annotation!r}")


def _reject_unconstrained_schema(schema: Mapping[str, Any], annotation: Any) -> None:
    if schema == {}:
        raise ToolSchemaError(f"Unsupported or unconstrained tool annotation: {annotation!r}")


def _serialize_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (int, float, bool, list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ToolResultSerializationError("Tool result is not JSON serializable") from exc
    if isinstance(value, BaseModel):
        try:
            return json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ToolResultSerializationError(
                "Tool result model is not JSON serializable"
            ) from exc
    raise ToolResultSerializationError(
        f"Tool result of type '{type(value).__name__}' cannot be serialized to a "
        "model-facing message"
    )
