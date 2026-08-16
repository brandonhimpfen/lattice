"""Use normalized messages and continue a conversation."""

from lattice import Message, Model

model = Model("openai:gpt-5")
messages = [
    Message.system("You are a concise technical writer."),
    Message.user("Explain DNS."),
]

response = model.invoke(messages)
messages.append(response.to_message())
messages.append(Message.user("Now explain DNS caching."))

print(model.invoke(messages).text)
