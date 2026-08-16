from lattice.errors import LatticeError, ProviderError, RateLimitError, UnsupportedCapabilityError


def test_provider_errors_inherit_from_lattice_error():
    assert issubclass(ProviderError, LatticeError)
    assert issubclass(RateLimitError, ProviderError)


def test_provider_error_preserves_context():
    cause = RuntimeError("native")
    error = ProviderError("failed", provider="fake", model="x", original=cause, raw={"a": 1})
    assert error.provider == "fake"
    assert error.model == "x"
    assert error.original is cause
    assert error.raw == {"a": 1}


def test_unsupported_capability_contains_capability_context():
    error = UnsupportedCapabilityError("streaming", provider="fake", model="x")
    assert error.capability == "streaming"
    assert "streaming" in str(error)
