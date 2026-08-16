# Contributing to Lattice

Contributions are welcome. Lattice is intentionally small, so changes should preserve a narrow core and avoid adding abstractions before they are needed by real applications.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate the virtual environment with the appropriate command for the shell in use.

## Quality checks

Before submitting a change, run:

```bash
pytest
ruff check .
mypy src/lattice
```

Core tests must not require network access, provider SDKs, or API credentials.

## Testing style

Prefer tests that describe observable behavior. For example:

```python
def test_plain_string_becomes_user_message():
    ...
```

is preferable to a test named after a private normalization helper.

Provider integrations should test translation and error mapping independently from live API calls. Live integration tests should be explicit and opt-in.

## Scope

The 0.1 core is limited to model invocation, messages, prompts, structured output, tools, streaming, provider discovery, and normalized errors. Agents, pipelines, retrieval frameworks, vector stores, automatic tool execution, retries, and hosted services are outside the initial core scope.

Proposals that expand the public API should explain why the capability belongs in a provider-neutral composition layer rather than application code or a separate integration.
