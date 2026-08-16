import pytest
from pydantic import BaseModel

from lattice import Prompt
from lattice.errors import PromptError, PromptVariableError


class Record(BaseModel):
    name: str


def test_prompt_renders_named_variables():
    prompt = Prompt("Hello {name}")
    assert prompt.render(name="Ada").content == "Hello Ada"


def test_prompt_missing_variable_fails():
    prompt = Prompt("{a} {b}")
    with pytest.raises(PromptVariableError, match="missing"):
        prompt.render(a="x")


def test_prompt_extra_variable_fails():
    prompt = Prompt("{a}")
    with pytest.raises(PromptVariableError, match="unexpected"):
        prompt.render(a="x", b="y")


def test_prompt_serializes_dict_as_compact_json():
    prompt = Prompt("{value}")
    assert prompt.render(value={"name": "Ada"}).content == '{"name":"Ada"}'


def test_prompt_serializes_pydantic_model_as_json():
    prompt = Prompt("{value}")
    assert prompt.render(value=Record(name="Ada")).content == '{"name":"Ada"}'


def test_prompt_supports_escaped_braces():
    prompt = Prompt("{{example}} {name}")
    assert prompt.render(name="Ada").content == "{example} Ada"


def test_prompt_rejects_attribute_expression():
    with pytest.raises(PromptError):
        Prompt("{user.name}")
