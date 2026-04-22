"""
LLM client with caching, cost tracking, retry logic, and cross-provider support.

Wraps OpenAI and Anthropic APIs behind a single interface. All calls are
cached to disk so re-running experiments is free after the first pass.

Provider routing is by model name:
- gpt-* / o1-*       -> OpenAI
- claude-*           -> Anthropic

Both providers share the same cache, cost tracker, stage tracker, and
retry policy. The complete() interface is identical regardless of provider.
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

# ── Pricing (USD per 1K tokens) ──────────────────────────────────────────────
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-2024-11-20": (0.0025, 0.01),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
    # Anthropic — verify against https://www.anthropic.com/pricing
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-3-5-sonnet-latest": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-opus-4-6": (0.015, 0.075),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    # DeepSeek (requires base_url override; not wired)
    "deepseek-chat": (0.00028, 0.00042),
    "deepseek-reasoner": (0.00055, 0.00219),
    # Groq (requires base_url override)
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "qwen-qwq-32b": (0.00029, 0.00039),
}


def _provider_for(model: str) -> str:
    """Return 'openai' or 'anthropic' based on the model name prefix."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1"):
        return "openai"
    return "openai"


@dataclass
class CostTracker:
    """Tracks cumulative API costs across an experiment.

    Provides three breakdowns:
    - Total: total_cost_usd, total_input_tokens, total_output_tokens
    - Per-model: by_model[model] = {input, output, cost, calls}
    - Per-stage: by_stage[stage_name] = {input, output, cost, calls}

    Per-stage tracking uses the `stage()` context manager. Wrap each pipeline
    stage in a `with tracker.stage("stage_name"):` block, and any API calls
    made inside that block are attributed to that stage.
    """

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

        mb = self.by_model.setdefault(
            model, {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
        )
        mb["input"] += input_tokens
        mb["output"] += output_tokens
        mb["cost"] += cost
        mb["calls"] += 1

        sb = self.by_stage.setdefault(
            stage, {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
        )
        sb["input"] += input_tokens
        sb["output"] += output_tokens
        sb["cost"] += cost
        sb["calls"] += 1

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
    """Multi-provider LLM client with disk caching, cost tracking, and retries.

    Routes to OpenAI or Anthropic based on model name. The complete() method
    has identical semantics across providers — caller code does not need to
    know which provider is being used.

    Usage:
        client = LLMClient(model="gpt-4o")
        text = await client.complete(system_prompt, user_prompt)

        # Cross-provider override on a single call:
        text = await client.complete(..., model_override="claude-sonnet-4-6")
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

        self._openai = AsyncOpenAI()

        self._anthropic: Any = None
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if _ANTHROPIC_AVAILABLE and anthropic_key:
            self._anthropic = AsyncAnthropic(api_key=anthropic_key)
        elif _provider_for(model) == "anthropic":
            raise RuntimeError(
                "Anthropic model requested but the anthropic package is missing "
                "or ANTHROPIC_API_KEY / CLAUDE_API_KEY is not set. "
                "Run: uv add anthropic, and add a key to your .env"
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
        """Send a completion request, returning the response text.

        Behaviour is identical across providers from the caller's perspective.
        Cache hits short-circuit before any API call.

        Args:
            response_format: For OpenAI, passed through (e.g. {"type": "json_object"}).
                For Anthropic, {"type": "json_object"} triggers prefill-based JSON
                forcing; the returned text is normalised to start with "{".
            model_override: If set, use this model instead of self.model.
            seed: For OpenAI only; ignored by Anthropic.
        """
        active_model = model_override or self.model
        provider = _provider_for(active_model)

        cache_key = self._cache_key(
            system_prompt, user_prompt, temperature, max_tokens,
            response_format, active_model, seed,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit ({provider}): {cache_key[:12]}")
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

        cost = self.cost_tracker.record(active_model, in_toks, out_toks)
        logger.debug(
            f"API call ({active_model}, {provider}): "
            f"{in_toks}in/{out_toks}out (${cost:.4f}, "
            f"total: {self.cost_tracker.summary()})"
        )

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
        """OpenAI chat completion with retry. Returns (text, input_tokens, output_tokens)."""
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

        max_retries = 10
        base_delay = 2.0
        max_delay = 120.0

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._openai.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries:
                    logger.error(f"OpenAI call failed after {max_retries} retries: {e}")
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"OpenAI retryable error (attempt {attempt}/{max_retries}, "
                    f"{model}): {e}. Waiting {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"OpenAI call failed (non-retryable, {model}): {e}")
                raise

        text = response.choices[0].message.content or ""
        usage = response.usage
        in_toks = usage.prompt_tokens if usage else 0
        out_toks = usage.completion_tokens if usage else 0
        return text, in_toks, out_toks

    async def _call_anthropic(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> tuple[str, int, int]:
        """Anthropic Messages API with retry and tool-use JSON forcing.

        JSON forcing on Claude 4 (which dropped prefill support):
        when response_format is {"type": "json_object"}, we define a
        pseudo-tool ``emit_json`` whose input schema is open-ended (any
        object), and force the model to call it via tool_choice.  Anthropic
        guarantees the tool_use.input field is a valid object matching the
        schema.

        We re-serialise that dict back to a JSON string so the existing
        parser (which expects raw JSON text) does not need to change.
        """
        if self._anthropic is None:
            raise RuntimeError(
                "Anthropic client not initialised. Install `anthropic` and "
                "set ANTHROPIC_API_KEY or CLAUDE_API_KEY in your environment."
            )

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
            kwargs["tools"] = [
                {
                    "name": "emit_json",
                    "description": (
                        "Emit the structured JSON response for this task. "
                        "The input object should match the schema described "
                        "in the system prompt."
                    ),
                    "input_schema": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit_json"}

        max_retries = 10
        base_delay = 2.0
        max_delay = 120.0

        for attempt in range(1, max_retries + 1):
            try:
                response = await self._anthropic.messages.create(**kwargs)
                break
            except (
                AnthropicRateLimitError,
                AnthropicAPITimeoutError,
                AnthropicAPIConnectionError,
            ) as e:
                if attempt == max_retries:
                    logger.error(
                        f"Anthropic call failed after {max_retries} retries: {e}"
                    )
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Anthropic retryable error (attempt {attempt}/{max_retries}, "
                    f"{model}): {e}. Waiting {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Anthropic call failed (non-retryable, {model}): {e}")
                raise

        text_parts: list[str] = []
        tool_use_dict: dict | None = None

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", "") == "emit_json":
                tool_use_dict = block.input
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        if force_json:
            if tool_use_dict is None:
                if text_parts:
                    logger.warning(
                        f"Anthropic ({model}): tool_choice=emit_json was set but "
                        f"no tool_use block returned. Falling back to text content."
                    )
                    text = "".join(text_parts)
                else:
                    raise RuntimeError(
                        f"Anthropic ({model}) returned neither tool_use nor text "
                        f"despite force_json=True. Response: {response}"
                    )
            else:
                text = json.dumps(tool_use_dict, ensure_ascii=False)
        else:
            text = "".join(text_parts)

        in_toks = response.usage.input_tokens
        out_toks = response.usage.output_tokens
        return text, in_toks, out_toks

    def _cache_key(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        model: str,
        seed: int | None = None,
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
            data = json.loads(path.read_text())
            return data.get("response")
        return None

    def _cache_set(self, key: str, response: str, model: str) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "response": response,
            "model": model,
            "timestamp": time.time(),
        }))
