"""Render a small reusable prompt template."""

from lattice import Model, Prompt

model = Model("openai:gpt-5")
prompt = Prompt("Explain {topic} in {sentences} sentences.")
message = prompt.render(topic="DNS", sentences=2)

print(model.invoke(message).text)
