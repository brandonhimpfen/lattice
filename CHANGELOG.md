# Changelog

All notable changes to Lattice will be documented in this file.

## 0.1.0a1

Initial core implementation.

- Added provider-neutral `Model`, `Message`, `Response`, `Prompt`, and tool primitives.
- Added Pydantic structured-output validation.
- Added typed streaming events and stream accumulation.
- Added provider registration and lazy entry-point discovery.
- Added capability-specific async and streaming provider protocols.
- Added normalized framework and provider error hierarchy.
- Added a network-free fake provider test suite.
