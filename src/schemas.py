"""
Core data models for the manipulation detection pipeline.
Single source of truth for all data structures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


# ── Taxonomy ─────────────────────────────────────────────────────────────────

class Tactic(str, Enum):
    """SemEval-2020 Task 11 — 14-technique propaganda taxonomy."""
    LOADED_LANGUAGE = "loaded_language"
    NAME_CALLING = "name_calling"
    REPETITION = "repetition"
    EXAGGERATION = "exaggeration"
    DOUBT = "doubt"
    APPEAL_TO_FEAR = "appeal_to_fear"
    FLAG_WAVING = "flag_waving"
    CAUSAL_OVERSIMPLIFICATION = "causal_oversimplification"
    SLOGANS = "slogans"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    BLACK_AND_WHITE = "black_and_white"
    THOUGHT_TERMINATING = "thought_terminating"
    WHATABOUTISM = "whataboutism"       # includes straw man, red herring
    BANDWAGON = "bandwagon"             # includes reductio ad hitlerum

ALL_TACTICS: list[str] = [t.value for t in Tactic]
NUM_TACTICS: int = len(ALL_TACTICS)
TACTIC_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(ALL_TACTICS)}
IDX_TO_TACTIC: dict[int, str] = {i: t for t, i in TACTIC_TO_IDX.items()}

# ── Data Models ──────────────────────────────────────────────────────────────

class Sample(BaseModel):
    """A single annotated text sample (political ad or SemEval paragraph)."""
    id: str
    text: str
    labels: list[str] = Field(default_factory=list)   # list of Tactic values
    source: str = "unknown"                            # "semeval_train", "semeval_test", "creative"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("labels", mode="before")
    @classmethod
    def normalise_labels(cls, v: list) -> list[str]:
        """Ensure labels are valid tactic values, lowercased."""
        out = []
        for label in v:
            label = str(label).lower().strip()
            if label in ("none", "no_manipulation", ""):
                continue
            if label not in ALL_TACTICS:
                # Try fuzzy matching
                label = _fuzzy_match_tactic(label)
            if label:
                out.append(label)
        return sorted(set(out))

    def label_vector(self) -> np.ndarray:
        """Convert labels to a binary vector of length NUM_TACTICS."""
        vec = np.zeros(NUM_TACTICS, dtype=int)
        for label in self.labels:
            if label in TACTIC_TO_IDX:
                vec[TACTIC_TO_IDX[label]] = 1
        return vec

    def cache_key(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class Prediction(BaseModel):
    """Model prediction for a single sample."""
    sample_id: str
    tactics: list[str] = Field(default_factory=list)
    raw_response: str = ""
    confidence: float = 0.0
    stage_outputs: dict[str, Any] = Field(default_factory=dict)  # for ASV stages
    parse_success: bool = True
    error_message: str = ""

    def label_vector(self) -> np.ndarray:
        vec = np.zeros(NUM_TACTICS, dtype=int)
        for t in self.tactics:
            if t in TACTIC_TO_IDX:
                vec[TACTIC_TO_IDX[t]] = 1
        return vec


class TokenUsage(BaseModel):
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    """Raw response from an LLM API call."""
    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: float = 0.0
    cached: bool = False


# ── Experiment Config ────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Full experiment specification — serialisable, reproducible."""
    name: str
    strategy: Literal[
        "zero_shot", 
        "few_shot", 
        "cot",  
        "asv"
    ]                        # zero_shot, few_shot, cot, cot_sc, hierarchical, asv
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    dataset: str = "semeval_train"         # semeval_train, semeval_test, creative
    sample_size: int | None = None
    num_articles: int | None = None  # when set, run on this many articles (all paragraphs from those articles)
    seed: int = 42
    batch_size: int = 10
    use_cache: bool = True
    notes: str = ""
    debug_asv: bool = False  # when True, print prosecutor/defense/verdict trace tables after ASV
    prompts: str = "asv"  # "asv" or "asvS" — which prompt module to use for ASV

    def run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.strategy}_{self.model.replace('-', '')}_{ts}"


ExperimentConfig.model_rebuild()


class ExperimentResult(BaseModel):
    """Results from a completed experiment run."""
    config: ExperimentConfig
    run_id: str
    predictions: list[Prediction]
    metrics: dict[str, Any] = Field(default_factory=dict)
    cost: CostSummary = Field(default_factory=lambda: CostSummary())
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    errors: list[str] = Field(default_factory=list)

    def duration_seconds(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


class CostSummary(BaseModel):
    """API cost tracking."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    num_api_calls: int = 0

    def add(self, usage: TokenUsage, model: str) -> None:
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.num_api_calls += 1
        self.total_cost_usd += _estimate_cost(usage, model)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


# ── Helpers ──────────────────────────────────────────────────────────────────

# Pricing per 1K tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":    (0.00015, 0.0006),
    "gpt-4o":         (0.0025,  0.01),
    "gpt-4-turbo":    (0.01,    0.03),
    "gpt-4":          (0.03,    0.06),
    "gpt-3.5-turbo":  (0.0005,  0.0015),
}


def _estimate_cost(usage: TokenUsage, model: str) -> float:
    inp, out = MODEL_PRICING.get(model, (0.001, 0.002))
    return (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1000


_TACTIC_ALIASES: dict[str, str] = {
    "loaded language": "loaded_language",
    "name calling": "name_calling",
    "name_calling_labeling": "name_calling",
    "name calling / labeling": "name_calling",
    "name_calling,labeling": "name_calling",
    "exaggeration,minimisation": "exaggeration",
    "exaggeration,minimization": "exaggeration",
    "whataboutism,straw_men,red_herring": "whataboutism",
    "bandwagon,reductio_ad_hitlerum": "bandwagon",
    "appeal to fear": "appeal_to_fear",
    "appeal to fear / prejudice": "appeal_to_fear",
    "appeal_to_fear_prejudice": "appeal_to_fear",
    "flag waving": "flag_waving",
    "flag-waving": "flag_waving",
    "causal oversimplification": "causal_oversimplification",
    "appeal to authority": "appeal_to_authority",
    "black-and-white fallacy": "black_and_white",
    "black and white fallacy": "black_and_white",
    "false dilemma": "black_and_white",
    "black_and_white_fallacy": "black_and_white",
    "thought-terminating cliche": "thought_terminating",
    "thought terminating cliche": "thought_terminating",
    "thought-terminating cliché": "thought_terminating",
    "whataboutism / straw man / red herring": "whataboutism",
    "straw man": "whataboutism",
    "straw_man": "whataboutism",
    "red herring": "whataboutism",
    "red_herring": "whataboutism",
    "bandwagon / reductio ad hitlerum": "bandwagon",
    "reductio ad hitlerum": "bandwagon",
    "exaggeration / minimization": "exaggeration",
    "exaggeration_minimization": "exaggeration",
    "minimization": "exaggeration",
    "slogan": "slogans",
}


def _fuzzy_match_tactic(raw: str) -> str | None:
    raw_clean = raw.lower().strip()
    # Direct alias lookup (exact)
    if raw_clean in _TACTIC_ALIASES:
        return _TACTIC_ALIASES[raw_clean]
    # Try with underscores as spaces
    raw_spaced = raw_clean.replace("_", " ")
    if raw_spaced in _TACTIC_ALIASES:
        return _TACTIC_ALIASES[raw_spaced]
    # Try as-is in ALL_TACTICS
    normalised = raw_clean.replace(" ", "_").replace(",", "_").replace("-", "_")
    if normalised in ALL_TACTICS:
        return normalised
    # Brute force: check all aliases
    for alias_key, alias_val in _TACTIC_ALIASES.items():
        if raw_clean.replace("_", " ").replace(",", " ").replace("-", " ") == alias_key.replace("_", " ").replace(",", " ").replace("-", " "):
            return alias_val
    return None
