# Architecture

Lattice separates a small normalized core from provider-specific adapters.

Application code creates a `Model`, supplies normalized messages, tools, or a structured-output type, and receives a normalized `Response`. The model façade converts the request into a `ModelRequest`, then delegates provider-specific translation and network behavior to the selected integration.

Core Lattice does not contain branches for individual providers. Provider packages register factories through the `lattice.providers` Python entry-point group and implement only the capabilities they support.

This boundary allows core applications to stay portable while preserving provider-specific access through `provider_options` and `.raw` fields.
