"""
LLM client with caching, cost tracking, and retry logic.

Wraps OpenAI API. All calls are cached to disk so re-running
experiments is free after the first pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)
load_dotenv()

# ── Pricing (USD per 1K tokens) ──────────────────────────────────────────────

PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-2024-11-20": (0.0025, 0.01),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
}


@dataclass
class CostTracker:
    """Tracks cumulative API costs across an experiment."""
    calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = PRICING.get(model, (0.0, 0.0))
        cost = (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)

        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": time.time(),
        })
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        return cost

    def summary(self) -> str:
        return (
            f"${self.total_cost_usd:.4f} "
            f"({self.total_input_tokens} in / {self.total_output_tokens} out, "
            f"{len(self.calls)} calls)"
        )

    def to_dict(self) -> dict:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_calls": len(self.calls),
            "calls": self.calls,
        }


class LLMClient:
    """Async OpenAI client with disk caching and cost tracking.

    Usage:
        client = LLMClient(model="gpt-4o", cache_dir="outputs/cache")
        response = await client.complete(system_prompt, user_prompt)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        cache_dir: str | Path = "outputs/cache",
        max_cost_usd: float = 50.0,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cost_usd = max_cost_usd
        self.cost_tracker = CostTracker()
        self._client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        """Send a completion request, returning the response text.

        Checks cache first. If not cached, calls the API and caches the result.
        Raises RuntimeError if budget exceeded.
        """
        cache_key = self._cache_key(
            system_prompt, user_prompt, temperature, max_tokens, response_format
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {cache_key[:12]}")
            return cached

        # Budget check
        if self.cost_tracker.total_cost_usd >= self.max_cost_usd:
            raise RuntimeError(
                f"Budget exceeded: ${self.cost_tracker.total_cost_usd:.4f} "
                f">= ${self.max_cost_usd:.2f}"
            )

        # API call with retry on rate-limit / transient errors
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        max_retries = 10
        base_delay = 2.0
        max_delay = 120.0

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries:
                    logger.error(f"API call failed after {max_retries} retries: {e}")
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Retryable error (attempt {attempt}/{max_retries}): {e}. "
                    f"Waiting {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"API call failed (non-retryable): {e}")
                raise

        text = response.choices[0].message.content or ""
        usage = response.usage

        # Track cost
        if usage:
            cost = self.cost_tracker.record(
                self.model, usage.prompt_tokens, usage.completion_tokens
            )
            logger.debug(
                f"API call: {usage.prompt_tokens}in/{usage.completion_tokens}out "
                f"(${cost:.4f}, total: {self.cost_tracker.summary()})"
            )

        # Cache
        self._cache_set(cache_key, text)
        return text

    def _cache_key(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        raw = json.dumps({
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "system": system,
            "user": user,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("response")
        return None

    def _cache_set(self, key: str, response: str) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "response": response,
            "model": self.model,
            "timestamp": time.time(),
        }))
