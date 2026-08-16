"""Small named-template prompts for Lattice."""

from __future__ import annotations

import json
import string
from typing import Any, Literal

from pydantic import BaseModel

from .errors import PromptError, PromptVariableError
from .messages import Message


class Prompt:
    """A reusable text template that renders to a user or system message."""

    def __init__(
        self,
        template: str,
        *,
        role: Literal["system", "user"] = "user",
    ) -> None:
        self.template = template
        self.role = role
        self._variables = self._extract_variables(template)

    @classmethod
    def system(cls, template: str) -> "Prompt":
        """Create a system-role prompt."""
        return cls(template, role="system")

    @classmethod
    def user(cls, template: str) -> "Prompt":
        """Create a user-role prompt."""
        return cls(template, role="user")

    def render(self, **values: Any) -> Message:
        """Render the template with exactly the variables it declares."""
        supplied = set(values)
        missing = self._variables - supplied
        extra = supplied - self._variables
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing variables: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"unexpected variables: {', '.join(sorted(extra))}")
            raise PromptVariableError("; ".join(parts))

        serialized = {key: _serialize_prompt_value(value) for key, value in values.items()}
        try:
            rendered = self.template.format(**serialized)
        except (KeyError, ValueError, IndexError) as exc:
            raise PromptError("Unable to render prompt template") from exc
        return Message(role=self.role, content=rendered)

    @staticmethod
    def _extract_variables(template: str) -> set[str]:
        variables: set[str] = set()
        formatter = string.Formatter()
        try:
            for _, field_name, format_spec, conversion in formatter.parse(template):
                if field_name is None:
                    continue
                # Lattice supports only direct named substitution. Attribute/index lookup,
                # conversions, and format specs would make templates executable-ish or
                # create surprising non-string behavior.
                if not field_name.isidentifier() or format_spec or conversion:
                    raise PromptError(
                        "Prompt fields must be simple named variables without expressions, "
                        "format specs, or conversions"
                    )
                variables.add(field_name)
        except ValueError as exc:
            raise PromptError("Invalid prompt template") from exc
        return variables


def _serialize_prompt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise PromptError("Prompt value is not JSON serializable") from exc
    return str(value)
