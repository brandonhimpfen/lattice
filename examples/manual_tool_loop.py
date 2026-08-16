"""Perform an explicit model-tool-model round trip."""

from lattice import Message, Model, ToolResult, tool


@tool
def weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny in {city}"


model = Model("openai:gpt-5")
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

print(model.invoke(messages, tools=[weather]).text)
