from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lattice.errors import ConfigurationError, ProviderNotInstalledError
from lattice.providers import get_provider, register_provider


def factory(**kwargs):
    raise AssertionError("not called")


def test_manual_registration_and_lookup():
    register_provider("acme", factory)
    assert get_provider("acme") is factory


def test_duplicate_manual_registration_requires_replace():
    register_provider("acme", factory)
    with pytest.raises(ConfigurationError):
        register_provider("acme", factory)


def test_replace_allows_explicit_override():
    register_provider("acme", factory)

    def replacement(**kwargs):
        raise AssertionError("not called")

    register_provider("acme", replacement, replace=True)
    assert get_provider("acme") is replacement


def test_missing_provider_has_actionable_error():
    with patch("lattice.providers.discovery._entry_points_for", return_value=[]):
        with pytest.raises(ProviderNotInstalledError) as caught:
            get_provider("openai")
    assert "lattice-openai" in str(caught.value)


def test_entry_point_is_loaded_lazily():
    ep = SimpleNamespace(load=lambda: factory, value="pkg:create", dist=SimpleNamespace(name="pkg"))
    with patch("lattice.providers.discovery._entry_points_for", return_value=[ep]):
        assert get_provider("acme") is factory


def test_entry_point_conflict_fails_deterministically():
    ep1 = SimpleNamespace(value="a:create", dist=SimpleNamespace(name="a"))
    ep2 = SimpleNamespace(value="b:create", dist=SimpleNamespace(name="b"))
    with patch("lattice.providers.discovery._entry_points_for", return_value=[ep1, ep2]):
        with pytest.raises(ConfigurationError, match="Multiple"):
            get_provider("acme")
