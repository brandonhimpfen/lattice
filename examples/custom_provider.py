"""Register a small local provider without any external SDK."""

from collections.abc import Mapping
from typing import Any

from lattice import Model, Response
from lattice.providers import ModelRequest, ProviderCapabilities, register_provider


class EchoProvider:
    name = "echo"

    def __init__(self, *, model: str, **_: Any) -> None:
        self.model = model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def invoke(self, request: ModelRequest) -> Response[Any]:
        text = request.messages[-1].content
        return Response[Any](
            provider=self.name,
            model=self.model,
            text=f"Echo: {text}",
            output=None,
        )


def create_provider(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
    provider_options: Mapping[str, Any],
) -> EchoProvider:
    del api_key, base_url, timeout, provider_options
    return EchoProvider(model=model)


register_provider("echo", create_provider)
print(Model("echo:local").invoke("Hello").text)
