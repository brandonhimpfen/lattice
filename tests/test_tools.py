from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel

from lattice import Tool, ToolResult, tool
from lattice.errors import ToolResultSerializationError, ToolSchemaError
from lattice.tools import normalize_tools


class Mode(str, Enum):
    FAST = "fast"
    SAFE = "safe"


class Options(BaseModel):
    query: str
    limit: int = 10


def test_decorated_tool_remains_callable():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_missing_description_is_allowed():
    def lookup(query: str) -> str:
        return query

    value = Tool.from_callable(lookup)
    assert value.description == ""


def test_missing_parameter_annotation_fails():
    def lookup(query):
        return query

    with pytest.raises(ToolSchemaError):
        Tool.from_callable(lookup)


def test_schema_includes_defaults_and_required_fields():
    def search(query: str, limit: int = 10, language: str | None = None) -> str:
        return query

    schema = Tool.from_callable(search).parameters
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"]["default"] == 10
    assert schema["properties"]["language"]["default"] is None


def test_schema_supports_literal_enum_and_model():
    def search(mode: Literal["a", "b"], enum_mode: Mode, options: Options) -> str:
        return "ok"

    schema = Tool.from_callable(search).parameters
    assert "mode" in schema["properties"]
    assert "enum_mode" in schema["properties"]
    assert "options" in schema["properties"]


def test_duplicate_names_fail():
    @tool(name="search")
    def first(query: str) -> str:
        return query

    @tool(name="search")
    def second(query: str) -> str:
        return query

    with pytest.raises(ToolSchemaError, match="Duplicate"):
        normalize_tools([first, second])


def test_variadic_tools_fail():
    def invalid(*args: str) -> str:
        return ""

    with pytest.raises(ToolSchemaError):
        Tool.from_callable(invalid)


def test_tool_result_serializes_none_as_json_null():
    message = ToolResult(tool_call_id="1", name="delete", output=None).to_message()
    assert message.content == "null"


def test_tool_result_serializes_pydantic_model():
    message = ToolResult(tool_call_id="1", name="search", output=Options(query="q")).to_message()
    assert message.content == '{"query":"q","limit":10}'


def test_tool_result_rejects_arbitrary_objects():
    with pytest.raises(ToolResultSerializationError):
        ToolResult(tool_call_id="1", name="x", output=object()).to_message()
