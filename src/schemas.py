"""
Core data models for span-level propaganda detection.

Defines the SemEval-2020 Task 11 taxonomy (14 techniques),
span annotations, predictions, and experiment configuration.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── SemEval-2020 Task 11 Taxonomy ────────────────────────────────────────────

class Technique(str, Enum):
    """14 propaganda techniques from SemEval-2020 Task 11."""
    LOADED_LANGUAGE = "Loaded_Language"
    NAME_CALLING = "Name_Calling,Labeling"
    REPETITION = "Repetition"
    EXAGGERATION = "Exaggeration,Minimisation"
    DOUBT = "Doubt"
    APPEAL_TO_FEAR = "Appeal_to_Fear-Prejudice"
    FLAG_WAVING = "Flag-Waving"
    CAUSAL_OVERSIMPLIFICATION = "Causal_Oversimplification"
    SLOGANS = "Slogans"
    APPEAL_TO_AUTHORITY = "Appeal_to_Authority"
    BLACK_AND_WHITE = "Black-and-White_Fallacy"
    THOUGHT_TERMINATING = "Thought-terminating_Cliches"
    WHATABOUTISM = "Whataboutism"
    BANDWAGON = "Bandwagon,Reductio_ad_hitlerum"


# Canonical name -> Technique for reverse lookup
_TECHNIQUE_BY_VALUE = {t.value: t for t in Technique}

# Common LLM output aliases -> canonical Technique
_ALIASES: dict[str, Technique] = {}
for t in Technique:
    canonical = t.value
    _ALIASES[canonical] = t
    _ALIASES[canonical.lower()] = t
    # Underscore-only variant
    simple = canonical.replace("-", "_").replace(",", "_")
    _ALIASES[simple] = t
    _ALIASES[simple.lower()] = t

# Manual aliases for common LLM outputs
_MANUAL_ALIASES = {
    "loaded language": Technique.LOADED_LANGUAGE,
    "name calling": Technique.NAME_CALLING,
    "name_calling": Technique.NAME_CALLING,
    "labeling": Technique.NAME_CALLING,
    "appeal to fear": Technique.APPEAL_TO_FEAR,
    "appeal_to_fear": Technique.APPEAL_TO_FEAR,
    "appeal to fear-prejudice": Technique.APPEAL_TO_FEAR,
    "flag waving": Technique.FLAG_WAVING,
    "flag_waving": Technique.FLAG_WAVING,
    "causal oversimplification": Technique.CAUSAL_OVERSIMPLIFICATION,
    "causal_oversimplification": Technique.CAUSAL_OVERSIMPLIFICATION,
    "appeal to authority": Technique.APPEAL_TO_AUTHORITY,
    "appeal_to_authority": Technique.APPEAL_TO_AUTHORITY,
    "black and white": Technique.BLACK_AND_WHITE,
    "black_and_white": Technique.BLACK_AND_WHITE,
    "black-and-white fallacy": Technique.BLACK_AND_WHITE,
    "false dilemma": Technique.BLACK_AND_WHITE,
    "false_dilemma": Technique.BLACK_AND_WHITE,
    "thought terminating cliche": Technique.THOUGHT_TERMINATING,
    "thought_terminating_cliche": Technique.THOUGHT_TERMINATING,
    "thought-terminating cliches": Technique.THOUGHT_TERMINATING,
    "thought terminating cliches": Technique.THOUGHT_TERMINATING,
    "bandwagon": Technique.BANDWAGON,
    "reductio ad hitlerum": Technique.BANDWAGON,
    "exaggeration": Technique.EXAGGERATION,
    "minimisation": Technique.EXAGGERATION,
    "exaggeration minimisation": Technique.EXAGGERATION,
    "slogans": Technique.SLOGANS,
    "repetition": Technique.REPETITION,
    "doubt": Technique.DOUBT,
    "whataboutism": Technique.WHATABOUTISM,
}
_ALIASES.update({k.lower(): v for k, v in _MANUAL_ALIASES.items()})


def normalise_technique(raw: str) -> Technique | None:
    """Resolve a raw technique string from LLM output to canonical Technique.

    Uses exact match, then alias lookup, then fuzzy lowercase matching.
    Returns None if unresolvable.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Exact enum value match
    if raw in _TECHNIQUE_BY_VALUE:
        return _TECHNIQUE_BY_VALUE[raw]

    # Alias lookup (case-insensitive)
    key = raw.lower().strip()
    if key in _ALIASES:
        return _ALIASES[key]

    # Normalise punctuation and retry
    key = key.replace("-", "_").replace(",", "_").replace(" ", "_")
    if key in _ALIASES:
        return _ALIASES[key]

    return None


# ── Span-Level Data Models ────────────────────────────────────────────────────

class GoldSpan(BaseModel):
    """A single ground-truth span annotation from SemEval data."""
    technique: Technique
    start: int  # Character offset (inclusive)
    end: int    # Character offset (exclusive)

    @property
    def text_length(self) -> int:
        return self.end - self.start


class PredictedSpan(BaseModel):
    """A single predicted span from an LLM."""
    technique: Technique
    span_text: str           # Literal text copied by the LLM
    reasoning: str = ""      # LLM's justification
    start: int = -1          # Resolved character offset (-1 = unresolved)
    end: int = -1            # Resolved character offset (-1 = unresolved)

    # ASV metadata
    pass_id: int = 0         # Which detection pass (Stage 1 multi-pass)
    agreement_count: int = 1 # How many passes detected this
    confidence: float = -1.0 # Stage 3 adjudicated confidence (-1 = unset)
    verdict: str = ""        # Stage 3 verdict: CONFIRMED / POSSIBLE / REJECTED

    @property
    def resolved(self) -> bool:
        return self.start >= 0 and self.end >= 0


class Article(BaseModel):
    """A single article/text with ground-truth span annotations."""
    id: str
    text: str
    gold_spans: list[GoldSpan] = Field(default_factory=list)

    @property
    def techniques_present(self) -> set[Technique]:
        return {s.technique for s in self.gold_spans}


class Prediction(BaseModel):
    """Full prediction for a single article across all ASV stages."""
    article_id: str
    spans: list[PredictedSpan] = Field(default_factory=list)

    # Raw LLM outputs per stage (for debugging / error analysis)
    stage_outputs: dict[str, Any] = Field(default_factory=dict)

    # Cost tracking
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    model: str = ""
    strategy: str = ""

    @property
    def confirmed_spans(self) -> list[PredictedSpan]:
        """Spans that survived ASV (or all spans for non-ASV).

        For ASV: returns CONFIRMED and POSSIBLE spans. Only REJECTED spans
        are filtered out. This ensures ASV can still reduce false positives
        (by rejecting weak detections) without blanket-zeroing when the
        adjudicator is overly conservative.

        For non-ASV strategies (no verdicts set), all spans are returned.
        """
        if any(s.verdict for s in self.spans):
            return [s for s in self.spans if s.verdict in ("CONFIRMED", "POSSIBLE")]
        return self.spans


# ── Experiment Configuration ──────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""
    name: str
    strategy: str  # zero_shot, few_shot, cot, asv, [5th method TBD]
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42

    # ASV-specific
    asv_num_passes: int = 3
    asv_temperatures: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7])
    asv_stages: int = 2  # 2 = detect+critique, 3 = detect+critique+adjudicate

    # Data paths
    articles_dir: str = "data/train/articles"
    labels_path: str = "data/train/train-task2-TC.labels"

    # Experiment size (smoke test)
    max_articles: int | None = None

    # Budget
    max_cost_usd: float = 50.0

    def run_id(self) -> str:
        """Unique identifier for this experiment run."""
        key = f"{self.name}_{self.strategy}_{self.model}_{self.seed}"
        return hashlib.md5(key.encode()).hexdigest()[:8]



