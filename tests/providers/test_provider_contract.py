from lattice.providers import ProviderCapabilities


def test_capabilities_default_to_small_sync_provider():
    capabilities = ProviderCapabilities()
    assert capabilities.async_invocation is False
    assert capabilities.streaming is False
    assert capabilities.tools is False
    assert capabilities.system_messages is True
