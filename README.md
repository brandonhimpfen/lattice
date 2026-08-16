# Lattice

Lattice is a lightweight, provider-neutral Python framework for working with large language models. It provides a small set of typed primitives for model invocation, messages, prompts, structured output, tools, streaming and provider integrations without requiring applications to adopt a large orchestration runtime.

Lattice 0.1 focuses on the model boundary. Agents, pipelines, retrieval systems, vector stores, automatic retries and hosted infrastructure are intentionally outside the initial scope.

## Design principles

Lattice keeps the common path small while leaving provider-specific capabilities accessible. Core Lattice does not depend on provider SDKs, model calls are explicit, tools remain ordinary Python functions and unsupported features fail rather than being silently ignored.

Applications can use normalized Lattice objects for portable behavior and access `.raw` fields or `provider_options` when provider-specific functionality is required.

## Installation

Install the core package:

```bash
pip install lattice
```

Provider integrations are installed separately. For example, when the corresponding packages are available:

```bash
pip install lattice-openai
pip install lattice-anthropic
```

Installing Lattice core does not install or import provider SDKs.

## Basic model invocation

```python
from lattice import Model

model = Model("openai:gpt-5")
response = model.invoke("Explain DNS in two sentences.")

print(response.text)
```

Switching providers changes the identifier rather than the application interface:

```python
model = Model("anthropic:claude-sonnet")
```

A provider integration must be installed and discoverable before its identifier can be used.

## Messages

```python
from lattice import Message, Model

model = Model("openai:gpt-5")

response = model.invoke([
    Message.system("You are a concise technical writer."),
    Message.user("Explain DNS caching."),
])

messages = [
    Message.user("Explain DNS."),
    response.to_message(),
    Message.user("Now compare recursive and iterative resolution."),
]

follow_up = model.invoke(messages)
```

Messages are immutable. Lattice does not mutate a caller-supplied conversation sequence.

## Prompts

`Prompt` provides intentionally small named-template support:

```python
from lattice import Prompt

prompt = Prompt("Summarize {text} in {length} sentences.")
message = prompt.render(text=article, length=3)
```

Missing or unexpected variables raise `PromptVariableError`. Dictionaries, lists and Pydantic models are rendered as compact JSON.

## Structured output

Structured output uses Pydantic models:

```python
from pydantic import BaseModel
from lattice import Model

class Summary(BaseModel):
    title: str
    topics: list[str]

model = Model("openai:gpt-5")
response = model.invoke(article, output=Summary)

print(response.output.title)
```

Lattice validates structured output locally even when a provider offers native schema enforcement. Providers that cannot support structured output raise `UnsupportedCapabilityError`.

## Tools

Tools are ordinary Python functions with type annotations:

```python
from lattice import tool

@tool
def weather(city: str) -> str:
    """Return the current weather for a city."""
    return lookup_weather(city)
```

The decorated function remains directly callable:

```python
weather("Toronto")
```

Expose it to a model with:

```python
response = model.invoke(
    "What's the weather in Toronto?",
    tools=[weather],
)

for call in response.tool_calls:
    print(call.name, call.arguments)
```

Lattice 0.1 does not execute model-requested tools automatically. Application code decides whether and when a tool runs.

### Manual tool round trip

```python
from lattice import Message, ToolResult

messages = [Message.user("What's the weather in Toronto?")]

response = model.invoke(messages, tools=[weather])
messages.append(response.to_message())

for call in response.tool_calls:
    if call.name == "weather":
        value = weather(**call.arguments)
        messages.append(
            ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=value,
            ).to_message()
        )

final = model.invoke(messages, tools=[weather])
print(final.text)
```

## Streaming

Streaming yields typed events rather than raw strings:

```python
from lattice import ResponseEnd, TextDelta

final = None

for event in model.stream("Write a short story."):
    if isinstance(event, TextDelta):
        print(event.text, end="")
    elif isinstance(event, ResponseEnd):
        final = event.response
```

A successful stream emits exactly one `ResponseEnd`. If streaming fails, the underlying normalized exception is raised and no synthetic end event is produced.

## Async

Providers that support asynchronous invocation expose the same request shape:

```python
response = await model.ainvoke("Explain DNS.")
```

Async streaming is available through:

```python
async for event in model.astream("Write a short story."):
    ...
```

Lattice does not emulate missing async capabilities by silently moving synchronous calls into worker threads. Unsupported behavior raises `UnsupportedCapabilityError`.

## Provider-specific options

Portable options such as `temperature`, `max_tokens`, `stop`, and `timeout` are first-class arguments. Provider-specific settings use `provider_options`:

```python
model = Model(
    "openai:gpt-5",
    provider_options={"reasoning_effort": "medium"},
)
```

Request-level provider options shallowly override model-level provider options.

Provider integrations must not silently ignore a supplied common Lattice argument.

## Custom providers

Third-party providers register factories under the `lattice.providers` entry-point group or explicitly at runtime:

```python
from lattice.providers import register_provider

register_provider("acme", create_acme_provider)
```

A provider implements synchronous `invoke()` as its minimum execution contract. Async and streaming capabilities are represented by separate protocols so small providers are not required to expose meaningless methods.

See `examples/custom_provider.py` for a complete local example.

## Errors

Expected framework and provider failures derive from `LatticeError`. Provider-specific SDK exceptions are normalized while preserving the original exception when available.

```python
from lattice.errors import RateLimitError

try:
    response = model.invoke("Hello")
except RateLimitError as exc:
    print(exc.provider)
```

Provider connection and timeout exceptions are named `ProviderConnectionError` and `ProviderTimeoutError` to avoid collisions with Python built-ins.

## What Lattice 0.1 does not do

Lattice 0.1 intentionally does not include agents, automatic tool execution, pipelines, graphs, sessions, memory systems, RAG abstractions, vector databases, model routing, automatic retries, tracing platforms or hosted services.

These boundaries keep the initial package small and allow higher-level features to be designed from demonstrated needs rather than added speculatively.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the quality checks:

```bash
pytest
ruff check .
mypy src/lattice
```

Core tests do not require network access, API credentials, or provider SDKs.

## License

Lattice is available under the MIT License.
