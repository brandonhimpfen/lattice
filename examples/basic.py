"""Basic model invocation.

Install a provider integration and replace the model identifier with one it supports.
"""

from lattice import Model

model = Model("openai:gpt-5")
response = model.invoke("Explain DNS in two sentences.")
print(response.text)
