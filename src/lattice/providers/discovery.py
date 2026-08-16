"""Manual and entry-point discovery for provider integrations."""

from __future__ import annotations

import re
from importlib import metadata

from ..errors import ConfigurationError, ProviderNotInstalledError
from .base import ProviderFactory

_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_MANUAL: dict[str, ProviderFactory] = {}
_DISCOVERED: dict[str, ProviderFactory] = {}
_SUGGESTED_PACKAGES = {
    "openai": "lattice-openai",
    "anthropic": "lattice-anthropic",
}


def validate_provider_name(name: str) -> None:
    """Validate a provider identifier used by model names and registrations."""
    if not _PROVIDER_NAME_RE.fullmatch(name):
        raise ConfigurationError(
            "Provider identifiers must start with a lowercase letter and contain only "
            "lowercase letters, numbers, underscores, or hyphens"
        )


def register_provider(name: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Register a provider factory for the current Python process."""
    validate_provider_name(name)
    if name in _MANUAL and not replace:
        raise ConfigurationError(f"Provider '{name}' is already registered")
    _MANUAL[name] = factory
    if replace:
        _DISCOVERED.pop(name, None)


def get_provider(name: str) -> ProviderFactory:
    """Return the registered or lazily discovered provider factory."""
    validate_provider_name(name)
    if name in _MANUAL:
        return _MANUAL[name]
    if name in _DISCOVERED:
        return _DISCOVERED[name]

    matches = _entry_points_for(name)
    if len(matches) > 1:
        owners = ", ".join(sorted(_entry_point_owner(ep) for ep in matches))
        raise ConfigurationError(
            f"Multiple installed provider integrations register '{name}': {owners}"
        )
    if not matches:
        package = _SUGGESTED_PACKAGES.get(name)
        message = f"Provider '{name}' is not installed."
        if package:
            message += f" Install provider support with: pip install {package}"
        raise ProviderNotInstalledError(message, provider=name)

    entry_point = matches[0]
    try:
        factory = entry_point.load()
    except Exception as exc:
        raise ConfigurationError(f"Unable to load provider integration '{name}'") from exc
    if not callable(factory):
        raise ConfigurationError(f"Provider entry point '{name}' did not load a callable factory")
    _DISCOVERED[name] = factory
    return factory


def provider_is_available(name: str) -> bool:
    """Return whether a provider is registered/discoverable, raising on conflicts."""
    get_provider(name)
    return True


def _entry_points_for(name: str) -> list[metadata.EntryPoint]:
    points = metadata.entry_points()
    if hasattr(points, "select"):
        return list(points.select(group="lattice.providers", name=name))
    # Compatibility with older importlib.metadata APIs, retained at negligible cost.
    legacy = points.get("lattice.providers", [])  # type: ignore[attr-defined]
    return [ep for ep in legacy if ep.name == name]


def _entry_point_owner(entry_point: metadata.EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    return getattr(dist, "name", None) or entry_point.value


def _reset_provider_registry_for_tests() -> None:
    """Clear process-local provider state. Intended only for Lattice's own tests."""
    _MANUAL.clear()
    _DISCOVERED.clear()
