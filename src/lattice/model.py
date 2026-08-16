"""Primary user-facing model façade."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, TypeAlias, TypeVar, cast, overload

from pydantic import BaseModel, ValidationError

from .errors import (
    ConfigurationError,
    InvalidRequestError,
    StructuredOutputValidationError,
    UnsupportedCapabilityError,
)
from .messages import Message
from .providers import (
    AsyncProvider,
    AsyncStreamingProvider,
    ModelRequest,
    Provider,
    ProviderCapabilities,
    StreamingProvider,
    get_provider,
)
from .responses import Response
from .streaming import StreamEvent
from .tools import ToolLike, normalize_tools

T = TypeVar("T", bound=BaseModel)
PromptInput: TypeAlias = str | Message | Sequence[Message]


class Model:
    """A configured provider-neutral model endpoint.

    Construction validates the model identifier and provider availability without making
    a network request. The provider instance itself is created lazily on first use.
    """

    def __init__(
        self,
        identifier: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> None:
        provider, model_name = _parse_identifier(identifier)
        if timeout is not None:
            _validate_timeout(timeout)
        self._identifier = identifier
        self._provider_name = provider
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._provider_options = dict(provider_options or {})
        self._factory = get_provider(provider)
        self._provider_instance: Provider | None = None

    @property
    def identifier(self) -> str:
        """Return the canonical ``provider:model`` identifier."""
        return self._identifier

    @property
    def provider(self) -> str:
        """Return the provider identifier."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Return the provider-native model name."""
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return capabilities declared by the configured provider adapter."""
        return self._get_provider().capabilities

    def __repr__(self) -> str:
        return f"Model(identifier={self.identifier!r})"

    @overload
    def invoke(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        output: None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[None]: ...

    @overload
    def invoke(
        self,
        input: PromptInput,
        *,
        tools: None = None,
        output: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[T]: ...

    def invoke(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        output: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[T] | Response[None]:
        """Invoke the model synchronously and return a normalized response."""
        request = self._build_request(
            input,
            tools=tools,
            output=output,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
            provider_options=provider_options,
        )
        provider = self._get_provider()
        self._ensure_request_capabilities(provider.capabilities, request)
        response = provider.invoke(request)
        return self._validate_response_output(response, output)

    @overload
    async def ainvoke(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        output: None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[None]: ...

    @overload
    async def ainvoke(
        self,
        input: PromptInput,
        *,
        tools: None = None,
        output: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[T]: ...

    async def ainvoke(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        output: type[T] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Response[T] | Response[None]:
        """Invoke the model asynchronously when the provider supports native async."""
        request = self._build_request(
            input,
            tools=tools,
            output=output,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
            provider_options=provider_options,
        )
        provider = self._get_provider()
        self._require_capability(provider.capabilities.async_invocation, "async_invocation")
        self._ensure_request_capabilities(provider.capabilities, request)
        if not isinstance(provider, AsyncProvider):
            raise UnsupportedCapabilityError(
                "async_invocation", provider=self.provider, model=self.model_name
            )
        response = await provider.ainvoke(request)
        return self._validate_response_output(response, output)

    def stream(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield normalized synchronous stream events."""
        request = self._build_request(
            input,
            tools=tools,
            output=None,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
            provider_options=provider_options,
        )
        provider = self._get_provider()
        self._require_capability(provider.capabilities.streaming, "streaming")
        self._ensure_request_capabilities(provider.capabilities, request)
        if not isinstance(provider, StreamingProvider):
            raise UnsupportedCapabilityError(
                "streaming", provider=self.provider, model=self.model_name
            )
        yield from provider.stream(request)

    async def astream(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
        provider_options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield normalized asynchronous stream events."""
        request = self._build_request(
            input,
            tools=tools,
            output=None,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
            provider_options=provider_options,
        )
        provider = self._get_provider()
        self._require_capability(provider.capabilities.async_streaming, "async_streaming")
        self._ensure_request_capabilities(provider.capabilities, request)
        if not isinstance(provider, AsyncStreamingProvider):
            raise UnsupportedCapabilityError(
                "async_streaming", provider=self.provider, model=self.model_name
            )
        async for event in provider.astream(request):
            yield event

    def _get_provider(self) -> Provider:
        if self._provider_instance is None:
            self._provider_instance = self._factory(
                model=self.model_name,
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                provider_options=dict(self._provider_options),
            )
        return self._provider_instance

    def _build_request(
        self,
        input: PromptInput,
        *,
        tools: Sequence[ToolLike] | None,
        output: type[T] | None,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        timeout: float | None,
        provider_options: Mapping[str, Any] | None,
    ) -> ModelRequest:
        messages = _normalize_input(input, provider=self.provider, model=self.model_name)
        normalized_tools = normalize_tools(list(tools) if tools is not None else None)
        if normalized_tools and output is not None:
            raise _invalid_request(
                "Tools and structured output cannot be requested together in Lattice 0.1",
                self.provider,
                self.model_name,
            )
        if output is not None and not issubclass(output, BaseModel):
            raise _invalid_request(
                "Structured output must be a Pydantic BaseModel subclass",
                self.provider,
                self.model_name,
            )
        _validate_temperature(temperature, self.provider, self.model_name)
        _validate_max_tokens(max_tokens, self.provider, self.model_name)
        normalized_stop = _validate_stop(stop, self.provider, self.model_name)
        effective_timeout = self._timeout if timeout is None else timeout
        if effective_timeout is not None:
            _validate_timeout(effective_timeout)
        options = _merge_options(self._provider_options, provider_options)
        _reject_common_option_conflicts(
            options,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=normalized_stop,
            timeout=timeout,
        )
        return ModelRequest(
            model=self.model_name,
            messages=messages,
            tools=normalized_tools,
            output_type=output,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=normalized_stop,
            timeout=effective_timeout,
            provider_options=options,
        )

    def _ensure_request_capabilities(
        self,
        capabilities: ProviderCapabilities,
        request: ModelRequest,
    ) -> None:
        if request.tools:
            self._require_capability(capabilities.tools, "tools")
        if request.output_type is not None:
            self._require_capability(capabilities.structured_output, "structured_output")
        if any(message.role == "system" for message in request.messages):
            self._require_capability(capabilities.system_messages, "system_messages")

    def _require_capability(self, available: bool, capability: str) -> None:
        if not available:
            raise UnsupportedCapabilityError(
                capability,
                provider=self.provider,
                model=self.model_name,
            )

    def _validate_response_output(
        self,
        response: Response[Any],
        output: type[T] | None,
    ) -> Response[T] | Response[None]:
        if output is None:
            # Provider adapters may use Response[Any]; the normalized contract for a plain
            # invocation has no structured output.
            if response.output is not None:
                response = response.model_copy(update={"output": None})
            return cast(Response[None], response)
        try:
            if isinstance(response.output, output):
                validated = response.output
            else:
                validated = output.model_validate(response.output)
        except (ValidationError, TypeError, ValueError) as exc:
            raise StructuredOutputValidationError(
                f"Structured output did not validate as {output.__name__}",
                provider=self.provider,
                model=self.model_name,
                validation_error=exc,
                raw_response=response.raw,
            ) from exc
        return cast(Response[T], response.model_copy(update={"output": validated}))


def _parse_identifier(identifier: str) -> tuple[str, str]:
    if ":" not in identifier:
        raise ConfigurationError("Model identifiers must use the form 'provider:model'")
    provider, model = identifier.split(":", 1)
    from .providers.discovery import validate_provider_name

    validate_provider_name(provider)
    if not model:
        raise ConfigurationError("Model identifiers must contain a non-empty model name")
    return provider, model


def _normalize_input(input: PromptInput, *, provider: str, model: str) -> list[Message]:
    if isinstance(input, str):
        return [Message.user(input)]
    if isinstance(input, Message):
        return [input]
    if isinstance(input, Sequence) and not isinstance(input, (str, bytes, bytearray)):
        if not input:
            raise _invalid_request("Message sequences cannot be empty", provider, model)
        if not all(isinstance(item, Message) for item in input):
            raise _invalid_request(
                "Message sequences may contain only Message objects", provider, model
            )
        return list(input)
    raise _invalid_request(
        "Model input must be a string, Message, or sequence of Message objects",
        provider,
        model,
    )


def _validate_temperature(value: float | None, provider: str, model: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _invalid_request("temperature must be a finite number", provider, model)


def _validate_max_tokens(value: int | None, provider: str, model: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _invalid_request("max_tokens must be an integer greater than zero", provider, model)


def _validate_stop(
    value: Sequence[str] | None,
    provider: str,
    model: str,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise _invalid_request(
            "stop must be a sequence of non-empty strings, not a string", provider, model
        )
    result = list(value)
    if any(not isinstance(item, str) or item == "" for item in result):
        raise _invalid_request("stop must contain only non-empty strings", provider, model)
    return result


def _validate_timeout(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigurationError("timeout must be a finite number greater than zero")


def _merge_options(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(base)
    if override:
        result.update(override)
    return result


def _reject_common_option_conflicts(
    options: Mapping[str, Any],
    *,
    temperature: float | None,
    max_tokens: int | None,
    stop: list[str] | None,
    timeout: float | None,
) -> None:
    supplied = {
        "temperature": temperature is not None,
        "max_tokens": max_tokens is not None,
        "stop": stop is not None,
        "timeout": timeout is not None,
    }
    conflicts = [key for key, is_supplied in supplied.items() if is_supplied and key in options]
    if conflicts:
        raise ConfigurationError(
            "Common Lattice arguments conflict with provider_options: " + ", ".join(conflicts)
        )


def _invalid_request(message: str, provider: str, model: str) -> InvalidRequestError:
    return InvalidRequestError(message, provider=provider, model=model)
