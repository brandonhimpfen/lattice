# Core API Notes

The public convenience API is exported from `lattice`. Provider-development contracts are exported from `lattice.providers` and normalized exceptions are exported from `lattice.errors`.

`Model` is the primary application-facing entry point. `Message`, `Response`, `Prompt`, `Tool`, `ToolCall`, `ToolResult` and streaming events are provider-neutral value objects.

The 0.1 API deliberately avoids automatic tool execution and higher-level orchestration. Applications can build explicit loops from these primitives, while later releases can add higher-level behavior without changing the fundamental provider boundary.
