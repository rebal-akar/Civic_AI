"""
LLM client with async support, caching, retries, and cost tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.schemas import LLMResponse, TokenUsage, CostSummary

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI API client with:
    - Disk-based response caching (saves money during iteration)
    - Automatic retries with exponential backoff
    - Token/cost tracking
    - Both sync and async interfaces
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_dir: str | Path = "data/cache",
        use_cache: bool = True,
        api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self.cost = CostSummary()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._async_client = AsyncOpenAI(**kwargs)
        self._sync_client = OpenAI(**kwargs)

    # ── Cache ────────────────────────────────────────────────────────────

    def _cache_key(self, messages: list[dict]) -> str:
        blob = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, messages: list[dict]) -> LLMResponse | None:
        if not self.use_cache:
            return None
        path = self._cache_path(self._cache_key(messages))
        if path.exists():
            data = json.loads(path.read_text())
            resp = LLMResponse(**data)
            resp.cached = True
            return resp
        return None

    def _cache_set(self, messages: list[dict], response: LLMResponse) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(self._cache_key(messages))
        path.write_text(response.model_dump_json(indent=2))

    # ── Async API ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        ),
    )
    async def _call_api(self, messages: list[dict]) -> LLMResponse:
        t0 = time.monotonic()
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"} if self.temperature == 0 else None,
        )
        latency = (time.monotonic() - t0) * 1000

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        self.cost.add(usage, self.model)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            model=response.model,
            latency_ms=latency,
        )

    async def complete(self, messages: list[dict]) -> LLMResponse:
        """Send messages to the API, with caching."""
        cached = self._cache_get(messages)
        if cached:
            logger.debug("Cache hit")
            return cached

        response = await self._call_api(messages)
        self._cache_set(messages, response)
        return response

    def complete_sync(self, messages: list[dict]) -> LLMResponse:
        """Synchronous wrapper."""
        cached = self._cache_get(messages)
        if cached:
            return cached
        return asyncio.run(self.complete(messages))

    # ── Batch API ────────────────────────────────────────────────────────

    async def complete_batch(
        self,
        message_sets: list[list[dict]],
        concurrency: int = 5,
        progress_callback=None,
    ) -> list[LLMResponse]:
        """Run multiple completions concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[LLMResponse | None] = [None] * len(message_sets)

        async def _run_one(idx: int, messages: list[dict]):
            async with semaphore:
                results[idx] = await self.complete(messages)
                if progress_callback:
                    progress_callback(1)

        tasks = [_run_one(i, msgs) for i, msgs in enumerate(message_sets)]
        await asyncio.gather(*tasks)
        return results  # type: ignore

    def reset_cost(self) -> CostSummary:
        """Reset cost tracker and return the accumulated cost."""
        old = self.cost
        self.cost = CostSummary()
        return old
