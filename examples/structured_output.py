"""Request and validate structured output with Pydantic."""

from pydantic import BaseModel

from lattice import Model


class Summary(BaseModel):
    title: str
    topics: list[str]


model = Model("openai:gpt-5")
response = model.invoke("Summarize DNS.", output=Summary)
print(response.output)
