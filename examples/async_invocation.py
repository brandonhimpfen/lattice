"""Use native asynchronous model invocation."""

import asyncio

from lattice import Model


async def main() -> None:
    model = Model("openai:gpt-5")
    response = await model.ainvoke("Explain DNS.")
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
