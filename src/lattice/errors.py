"""Normalized exception hierarchy for Lattice."""

from __future__ import annotations

from typing import Any


class LatticeError(Exception):
    """Base class for expected Lattice failures."""


class ConfigurationError(LatticeError):
    """Raised when Lattice configuration is invalid or ambiguous."""


class UnsupportedCapabilityError(LatticeError):
    """Raised when a model/provider cannot perform a requested capability."""

    def __init__(
        self,
        capability: str,
        *,
        provider: str,
        model: str | None = None,
        message: str | None = None,
    ) -> None:
        self.capability = capability
        self.provider = provider
        self.model = model
        detail = message or f"Provider '{provider}' does not support capability '{capability}'"
        if model:
            detail += f" for model '{model}'"
        super().__init__(detail)


class ProviderError(LatticeError):
    """Base class for normalized failures originating at a provider boundary."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str | None = None,
        original: BaseException | None = None,
        raw: Any = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.original = original
        self.raw = raw
        super().__init__(message)


class ProviderNotInstalledError(ProviderError):
    """Raised when the requested provider integration cannot be discovered."""


class AuthenticationError(ProviderError):
    """Raised when provider authentication fails."""


class AuthorizationError(ProviderError):
    """Raised when credentials do not authorize a provider operation."""


class RateLimitError(ProviderError):
    """Raised when a provider rate limit prevents an operation."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when a provider cannot be reached."""


class ModelNotFoundError(ProviderError):
    """Raised when a requested model identifier is unknown to the provider."""


class ModelUnavailableError(ProviderError):
    """Raised when a known model is temporarily unavailable."""


class InvalidRequestError(ProviderError):
    """Raised when an invocation is invalid or cannot be represented."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response cannot be normalized safely."""


class PromptError(LatticeError):
    """Base class for prompt construction failures."""


class PromptVariableError(PromptError):
    """Raised when prompt variables are missing or unexpected."""


class ToolError(LatticeError):
    """Base class for tool definition and tool-message failures."""


class ToolSchemaError(ToolError):
    """Raised when a Python callable cannot be represented as a tool schema."""


class ToolCallError(ToolError):
    """Base class for invalid model-generated tool calls."""


class ToolCallParseError(ToolCallError):
    """Raised when finalized tool-call arguments cannot be parsed."""


class ToolResultSerializationError(ToolError):
    """Raised when a tool result cannot be converted into a model-facing message."""


class StructuredOutputError(LatticeError):
    """Base class for structured-output failures."""


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when provider output does not validate against the requested schema."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        validation_error: BaseException,
        raw_response: Any = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.validation_error = validation_error
        self.raw_response = raw_response
        super().__init__(message)


class StreamingError(LatticeError):
    """Raised for stream-state errors that do not map to a provider exception."""
