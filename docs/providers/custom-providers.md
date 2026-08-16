# Writing a Provider Integration

A provider integration supplies a factory and at least a synchronous `Provider` implementation. Native async and streaming support use separate protocols so adapters are not required to emulate unsupported behavior.

Provider packages should translate `ModelRequest` into their SDK's request format, normalize SDK responses into `Response`, map known failures into Lattice exceptions, and preserve useful native objects in `.raw` fields.

Provider packages register through the `lattice.providers` entry-point group:

```toml
[project.entry-points."lattice.providers"]
acme = "lattice_acme:create_provider"
```

Adapters must not silently ignore portable Lattice arguments. If a requested capability or argument cannot be supported faithfully, fail explicitly.

See `examples/custom_provider.py` for a minimal provider that uses no external SDK.
