"""Multi-provider LLM client with disk caching, cost tracking, and retries.

Routes to OpenAI or Anthropic based on model name prefix (gpt-* / claude-*).
The complete() interface is identical regardless of provider.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

try:
    from anthropic import (
        AsyncAnthropic,
        APIConnectionError as AnthropicAPIConnectionError,
        APITimeoutError as AnthropicAPITimeoutError,
        RateLimitError as AnthropicRateLimitError,
    )
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None  # type: ignore[misc, assignment]
    AnthropicAPIConnectionError = Exception  # type: ignore[misc, assignment]
    AnthropicAPITimeoutError = Exception  # type: ignore[misc, assignment]
    AnthropicRateLimitError = Exception  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)
load_dotenv()

# Anthropic SDK reads ANTHROPIC_API_KEY; allow CLAUDE_API_KEY as alias.
if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]

# USD per 1K tokens (input, output). Verify against provider pricing pages.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-sonnet-4-6": (0.003, 0.015),
}


def _provider_for(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


@dataclass
class CostTracker:
    """Tracks API cost, tokens, and calls. Supports per-stage attribution via stage()."""

    calls: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)
    by_stage: dict[str, dict] = field(default_factory=dict)
    _stage_stack: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str):
        self._stage_stack.append(name)
        try:
            yield
        finally:
            self._stage_stack.pop()

    def _current_stage(self) -> str:
        return self._stage_stack[-1] if self._stage_stack else "unknown"

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = PRICING.get(model, (0.0, 0.0))
        cost = (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
        stage = self._current_stage()

        self.calls.append({
            "model": model,
            "stage": stage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": time.time(),
        })
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        for bucket, key in ((self.by_model, model), (self.by_stage, stage)):
            b = bucket.setdefault(key, {"input": 0, "output": 0, "cost": 0.0, "calls": 0})
            b["input"] += input_tokens
            b["output"] += output_tokens
            b["cost"] += cost
            b["calls"] += 1

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
            "by_model": self.by_model,
            "by_stage": self.by_stage,
            "calls": self.calls,
        }


class LLMClient:
    """Async LLM client with disk caching, cost tracking, and exponential-backoff retries."""

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

        self._openai = AsyncOpenAI()
        self._anthropic: Any = None

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if _ANTHROPIC_AVAILABLE and anthropic_key:
            self._anthropic = AsyncAnthropic(api_key=anthropic_key)
        elif _provider_for(model) == "anthropic":
            raise RuntimeError(
                "Anthropic model requested but anthropic package is missing "
                "or ANTHROPIC_API_KEY / CLAUDE_API_KEY is not set."
            )

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
        model_override: str | None = None,
        seed: int | None = None,
    ) -> str:
        active_model = model_override or self.model
        provider = _provider_for(active_model)

        cache_key = self._cache_key(
            system_prompt, user_prompt, temperature, max_tokens,
            response_format, active_model, seed,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self.cost_tracker.total_cost_usd >= self.max_cost_usd:
            raise RuntimeError(
                f"Budget exceeded: ${self.cost_tracker.total_cost_usd:.4f} "
                f">= ${self.max_cost_usd:.2f}"
            )

        if provider == "openai":
            text, in_toks, out_toks = await self._call_openai(
                active_model, system_prompt, user_prompt,
                temperature, max_tokens, response_format, seed,
            )
        elif provider == "anthropic":
            text, in_toks, out_toks = await self._call_anthropic(
                active_model, system_prompt, user_prompt,
                temperature, max_tokens, response_format,
            )
        else:
            raise RuntimeError(f"Unknown provider for model {active_model!r}")

        self.cost_tracker.record(active_model, in_toks, out_toks)
        self._cache_set(cache_key, text, active_model)
        return text

    async def _call_openai(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        seed: int | None,
    ) -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed

        response = await self._retry(
            lambda: self._openai.chat.completions.create(**kwargs),
            provider="OpenAI",
            model=model,
            retryable=(RateLimitError, APITimeoutError, APIConnectionError),
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0

    async def _call_anthropic(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> tuple[str, int, int]:
        """Claude 4 dropped prefill support for JSON, so we use a forced tool call.

        When response_format is {"type": "json_object"}, we declare a pseudo-tool
        `emit_json` with an open-ended schema and force the model to call it.
        The returned tool_use.input is serialised back to a JSON string so the
        parser (which expects raw JSON text) works unchanged.
        """
        if self._anthropic is None:
            raise RuntimeError("Anthropic client not initialised.")

        force_json = (
            response_format is not None
            and response_format.get("type") == "json_object"
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if force_json:
            kwargs["tools"] = [{
                "name": "emit_json",
                "description": "Emit the structured JSON response for this task.",
                "input_schema": {"type": "object", "additionalProperties": True},
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit_json"}

        response = await self._retry(
            lambda: self._anthropic.messages.create(**kwargs),
            provider="Anthropic",
            model=model,
            retryable=(
                AnthropicRateLimitError,
                AnthropicAPITimeoutError,
                AnthropicAPIConnectionError,
            ),
        )

        text_parts: list[str] = []
        tool_use_dict: dict | None = None
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", "") == "emit_json":
                tool_use_dict = block.input
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        if force_json:
            if tool_use_dict is not None:
                text = json.dumps(tool_use_dict, ensure_ascii=False)
            elif text_parts:
                logger.warning(f"Anthropic ({model}): no tool_use block, falling back to text.")
                text = "".join(text_parts)
            else:
                raise RuntimeError(f"Anthropic ({model}) returned no content.")
        else:
            text = "".join(text_parts)

        return text, response.usage.input_tokens, response.usage.output_tokens

    async def _retry(self, call, provider: str, model: str, retryable: tuple):
        """Exponential-backoff retry wrapper; 10 attempts, 2s–120s delay."""
        base_delay, max_delay, max_retries = 2.0, 120.0, 10
        for attempt in range(1, max_retries + 1):
            try:
                return await call()
            except retryable as e:
                if attempt == max_retries:
                    logger.error(f"{provider} call failed after {max_retries} retries: {e}")
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"{provider} retryable error (attempt {attempt}/{max_retries}, "
                    f"{model}): {e}. Waiting {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"{provider} call failed (non-retryable, {model}): {e}")
                raise

    def _cache_key(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        model: str,
        seed: int | None,
    ) -> str:
        raw = json.dumps({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "seed": seed,
            "system": system,
            "user": user,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text()).get("response")
        return None

    def _cache_set(self, key: str, response: str, model: str) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "response": response,
            "model": model,
            "timestamp": time.time(),
        }))