"""Smoke test: JSON prefill path for Anthropic (response_format json_object)."""
from __future__ import annotations

import asyncio
import json

from src.llm_client import LLMClient


async def main() -> None:
    client = LLMClient(model="claude-sonnet-4-6", max_cost_usd=1.0)
    result = await client.complete(
        system_prompt="You are a propaganda detector. Output JSON only.",
        user_prompt=(
            'Find loaded language in: "These corrupt politicians are destroying '
            'our country." Reply with {"annotations": [{"text": "...", '
            '"type": "Loaded_Language", "reason": "..."}]}'
        ),
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    print(f"Raw response: {result}")
    parsed = json.loads(result)
    print(f"Parsed: {parsed}")
    print(f"Cost: {client.cost_tracker.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
