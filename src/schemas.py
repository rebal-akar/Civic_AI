"""
Core data models for span-level propaganda detection.
SemEval-2020 Task 11 taxonomy (14 techniques), spans, predictions, config.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Technique(str, Enum):
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
    WHATABOUTISM = "Whataboutism,Straw_Men,Red_Herring"
    BANDWAGON = "Bandwagon,Reductio_ad_hitlerum"


_TECHNIQUE_BY_VALUE = {t.value: t for t in Technique}
_ALIASES: dict[str, Technique] = {}
for t in Technique:
    canonical = t.value
    _ALIASES[canonical] = t
    _ALIASES[canonical.lower()] = t
    simple = canonical.replace("-", "_").replace(",", "_")
    _ALIASES[simple] = t
    _ALIASES[simple.lower()] = t

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
    "straw_men": Technique.WHATABOUTISM,
    "straw men": Technique.WHATABOUTISM,
    "straw man": Technique.WHATABOUTISM,
    "red_herring": Technique.WHATABOUTISM,
    "red herring": Technique.WHATABOUTISM,
    "whataboutism,straw_men,red_herring": Technique.WHATABOUTISM,
    "whataboutism, straw_men, red_herring": Technique.WHATABOUTISM,
    "whataboutism straw_men red_herring": Technique.WHATABOUTISM,
}
_ALIASES.update({k.lower(): v for k, v in _MANUAL_ALIASES.items()})


def normalise_technique(raw: str) -> Technique | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw in _TECHNIQUE_BY_VALUE:
        return _TECHNIQUE_BY_VALUE[raw]
    key = raw.lower().strip()
    if key in _ALIASES:
        return _ALIASES[key]
    key = key.replace("-", "_").replace(",", "_").replace(" ", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    return None


class GoldSpan(BaseModel):
    technique: Technique
    start: int
    end: int

    @property
    def text_length(self) -> int:
        return self.end - self.start


class PredictedSpan(BaseModel):
    technique: Technique
    span_text: str
    reasoning: str = ""
    start: int = -1
    end: int = -1

    pass_id: int = 0
    agreement_count: int = 1
    confidence: float = -1.0
    verdict: str = ""

    # Diagnostics
    original_technique: Technique | None = None
    original_span_text: str = ""
    original_start: int = -1
    original_end: int = -1
    was_relabeled: bool = False
    was_trimmed: bool = False
    consol_action: str = ""
    stage_history: list[str] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.start >= 0 and self.end >= 0


class Article(BaseModel):
    id: str
    text: str
    gold_spans: list[GoldSpan] = Field(default_factory=list)

    @property
    def techniques_present(self) -> set[Technique]:
        return {s.technique for s in self.gold_spans}


class Prediction(BaseModel):
    article_id: str
    spans: list[PredictedSpan] = Field(default_factory=list)
    stage_outputs: dict[str, Any] = Field(default_factory=dict)
    stage_snapshots: dict[str, list[dict]] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    model: str = ""
    strategy: str = ""

    # "permissive" = CONFIRMED + POSSIBLE; "strict" = CONFIRMED only.
    # Default permissive matches OLD behaviour (avoids zero-recall on
    # over-conservative adjudicators). Set "strict" for ablation.
    eval_mode: str = "permissive"

    @property
    def confirmed_spans(self) -> list[PredictedSpan]:
        if not any(s.verdict for s in self.spans):
            return self.spans  # Non-ASV strategies have no verdicts
        if self.eval_mode == "strict":
            return [s for s in self.spans if s.verdict == "CONFIRMED"]
        return [s for s in self.spans if s.verdict in ("CONFIRMED", "POSSIBLE")]


class ExperimentConfig(BaseModel):
    name: str
    strategy: str  # zero_shot, few_shot, cot, asv, consol, hybrid
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42

    asv_num_passes: int = 3
    asv_temperatures: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7])
    asv_stages: int = 2

    verify_model: str | None = None
    eval_mode: str = "permissive"

    articles_dir: str = "data/train/articles"
    labels_path: str = "data/train/train-task2-TC.labels"
    max_articles: int | None = None

    max_cost_usd: float = 50.0

    def run_id(self) -> str:
        key = (
            f"{self.name}_{self.strategy}_{self.model}_"
            f"{self.verify_model}_{self.eval_mode}_{self.seed}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:8]