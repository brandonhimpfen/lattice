"""Consume typed streaming events."""

from lattice import Model, ResponseEnd, TextDelta

model = Model("openai:gpt-5")

for event in model.stream("Write a short story."):
    if isinstance(event, TextDelta):
        print(event.text, end="")
    elif isinstance(event, ResponseEnd):
        print("\n\nCompleted with:", event.response.finish_reason)
