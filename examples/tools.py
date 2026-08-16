"""Expose a typed Python function as a model tool."""

from lattice import Model, tool


@tool
def weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Weather lookup for {city}"


model = Model("openai:gpt-5")
response = model.invoke("What's the weather in Toronto?", tools=[weather])

for call in response.tool_calls:
    print(call.name, call.arguments)
