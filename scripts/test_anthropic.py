"""Smoke test: Anthropic API, cost tracking, and retries (no JSON forcing)."""
from __future__ import annotations

import asyncio

from src.llm_client import LLMClient


async def main() -> None:
    client = LLMClient(model="claude-sonnet-4-6", max_cost_usd=1.0)
    result = await client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with exactly: hello world",
    )
    print(f"Response: {result!r}")
    print(f"Cost: {client.cost_tracker.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
