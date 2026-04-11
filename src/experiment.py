"""
Experiment runner — orchestrates the full pipeline.

Usage:
    python -m scripts.run_experiment --strategy zero_shot --model gpt-4o-mini
    python -m scripts.run_experiment --strategy asv --model gpt-4o
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.data.loader import load_semeval
from src.evaluation.metrics import SpanMetrics, evaluate, print_results
from src.llm_client import LLMClient
from src.parser import merge_multipass, parse_llm_response
from src.prompts.detection import (
    COT_SYSTEM,
    COT_USER,
    STAGE1_SYSTEM,
    STAGE1_USER,
)
from src.prompts.baseline import (
    FEW_SHOT_SYSTEM,
    FEW_SHOT_USER,
    ZERO_SHOT_SYSTEM,
    ZERO_SHOT_USER,
)
from src.prompts.critique import (
    STAGE2_SYSTEM,
    STAGE2_USER,
    format_detections_for_stage2,
)
from src.prompts.adjudication import (
    STAGE3_SYSTEM,
    STAGE3_USER,
    format_detections_for_stage3,
    format_critiques_for_stage3,
)
from src.prompts.consolidation import (
    CONSOL_SYSTEM,
    CONSOL_USER,
    format_candidates_for_consol,
)
from src.schemas import (
    Article,
    ExperimentConfig,
    Prediction,
    PredictedSpan,
    normalise_technique,
)

logger = logging.getLogger(__name__)


async def run_single_article(
    article: Article,
    client: LLMClient,
    config: ExperimentConfig,
) -> Prediction:
    """Run a prompting strategy on a single article and return the prediction."""

    if config.strategy == "asv":
        return await _run_asv(article, client, config)
    elif config.strategy == "consol":
        return await _run_consol(article, client, config)
    else:
        return await _run_baseline(article, client, config)


async def _run_baseline(
    article: Article,
    client: LLMClient,
    config: ExperimentConfig,
) -> Prediction:
    """Run a single-pass baseline strategy (zero_shot, few_shot, cot)."""

    use_json_mode = True

    if config.strategy == "cot":
        system_prompt = COT_SYSTEM
        user_template = COT_USER
        # CoT must NOT force JSON mode — it suppresses the reasoning chain.
        # The model reasons freely, then outputs JSON at the end.
        use_json_mode = False
    elif config.strategy == "few_shot":
        system_prompt = FEW_SHOT_SYSTEM
        user_template = FEW_SHOT_USER
    else:  # zero_shot (default)
        system_prompt = ZERO_SHOT_SYSTEM
        user_template = ZERO_SHOT_USER

    user_prompt = user_template.format(text=article.text)

    raw_output = await client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"} if use_json_mode else None,
    )

    spans, errors = parse_llm_response(raw_output, article.text, pass_id=0)
    if errors:
        logger.warning(f"Parse errors for {article.id}: {errors}")

    return Prediction(
        article_id=article.id,
        spans=spans,
        stage_outputs={"detection": raw_output},
        model=config.model,
        strategy=config.strategy,
    )


async def _run_asv(
    article: Article,
    client: LLMClient,
    config: ExperimentConfig,
) -> Prediction:
    """Run full ASV pipeline: Stage 1 (detect) → Stage 2 (critique) → Stage 3 (adjudicate).

    Stage 1: 3 independent detection passes, merged via union (high recall).
    Stage 2: Adversarial critique challenges each detection (independence constraint).
    Stage 3: Adjudicator weighs both sides, assigns confidence + verdict.
    """
    stage_outputs: dict[str, Any] = {}

    # ── Stage 1: Multi-pass detection ─────────────────────────────────────
    user_prompt = STAGE1_USER.format(text=article.text)

    pass_results: list[list[PredictedSpan]] = []
    raw_s1_outputs: list[str] = []

    for i, temp in enumerate(config.asv_temperatures[:config.asv_num_passes]):
        raw = await client.complete(
            system_prompt=STAGE1_SYSTEM,
            user_prompt=user_prompt,
            temperature=temp,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
        raw_s1_outputs.append(raw)
        spans, errors = parse_llm_response(raw, article.text, pass_id=i)
        pass_results.append(spans)
        if errors:
            logger.warning(f"ASV S1 parse errors [{article.id}] pass {i}: {errors}")

    merged = merge_multipass(pass_results, article.text)

    stage_outputs["stage1_raw"] = raw_s1_outputs
    stage_outputs["stage1_per_pass_counts"] = [len(p) for p in pass_results]
    stage_outputs["stage1_merged_count"] = len(merged)

    logger.info(
        f"ASV S1 [{article.id}]: "
        f"{sum(len(p) for p in pass_results)} total -> {len(merged)} merged"
    )

    if not merged:
        return Prediction(
            article_id=article.id, spans=[], stage_outputs=stage_outputs,
            model=config.model, strategy="asv",
        )

    # ── Stage 2: Critique (independence constraint) ───────────────────────
    # Per CoVe: Stage 2 sees ONLY span text + technique + agreement count.
    # It does NOT see Stage 1's reasoning chains.
    detections_for_s2 = [
        {
            "span_text": s.span_text,
            "technique": s.technique.value,
            "agreement_count": s.agreement_count,
        }
        for s in merged
    ]

    s2_user = STAGE2_USER.format(
        text=article.text,
        detections_json=format_detections_for_stage2(detections_for_s2),
    )

    raw_s2 = await client.complete(
        system_prompt=STAGE2_SYSTEM,
        user_prompt=s2_user,
        temperature=0.0,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
    )
    stage_outputs["stage2_raw"] = raw_s2

    critiques = _parse_critiques(raw_s2)
    stage_outputs["stage2_critiques"] = critiques
    stage_outputs["stage2_critiques_count"] = len(critiques)

    logger.info(
        f"ASV S2 [{article.id}]: {len(critiques)} critiques "
        f"({sum(1 for c in critiques if c.get('verdict') == 'CHALLENGE')} CHALLENGE, "
        f"{sum(1 for c in critiques if c.get('verdict') == 'CONCEDE')} CONCEDE)"
    )

    if config.asv_stages >= 3:
        # ── Stage 3: Adjudication ─────────────────────────────────────────
        s3_user = STAGE3_USER.format(
            text=article.text,
            detections_json=format_detections_for_stage3(detections_for_s2),
            critiques_json=format_critiques_for_stage3(critiques),
        )

        raw_s3 = await client.complete(
            system_prompt=STAGE3_SYSTEM,
            user_prompt=s3_user,
            temperature=0.0,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
        stage_outputs["stage3_raw"] = raw_s3

        verdicts = _parse_verdicts(raw_s3)
        stage_outputs["stage3_verdicts_count"] = len(verdicts)

        # Apply verdicts to merged spans
        _apply_verdicts(merged, verdicts)
    else:
        # ── 2-stage mode: use Stage 2 critique verdicts directly ──────────
        # CONCEDE (critique couldn't find innocent explanation) -> CONFIRMED
        # CHALLENGE (critique found plausible innocent explanation) -> POSSIBLE/REJECTED
        _apply_critique_verdicts(merged, critiques)

    confirmed = [s for s in merged if s.verdict == "CONFIRMED"]
    possible = [s for s in merged if s.verdict == "POSSIBLE"]
    rejected = [s for s in merged if s.verdict == "REJECTED"]
    unset = [s for s in merged if not s.verdict]

    stage_label = "S3" if config.asv_stages >= 3 else "S2-direct"
    logger.info(
        f"ASV {stage_label} [{article.id}]: {len(confirmed)} CONFIRMED, "
        f"{len(possible)} POSSIBLE, {len(rejected)} REJECTED, "
        f"{len(unset)} unmatched"
    )

    # For evaluation: use CONFIRMED spans only
    # (POSSIBLE spans are logged but not counted as predictions)
    return Prediction(
        article_id=article.id,
        spans=merged,
        stage_outputs=stage_outputs,
        model=config.model,
        strategy="asv",
    )


async def _run_consol(
    article: Article,
    client: LLMClient,
    config: ExperimentConfig,
) -> Prediction:
    """Run the Consol pipeline: multi-pass detection -> consolidation.

    Stage 1: Same as ASV — 3 independent detection passes, union merged.
             Designed for HIGH RECALL (find everything).
    Stage 2: Consolidation — given the article + all candidates, the LLM
             acts as an expert consolidator that SELECTS genuine propaganda,
             CORRECTS technique labels, and REFINES span boundaries.

    Research grounding:
    - Hasanain et al. (2024): Consolidator role achieved F1=0.671 vs
      Annotator F1=0.050 — 13x improvement from giving candidates to select.
    - Kasner et al. (2025): Main error is technique misclassification
      (soft-hard delta=0.218) — consolidator targets this directly.
    - Wang et al. (2023): Self-consistency via multi-pass improves accuracy.
    """
    stage_outputs: dict[str, Any] = {}

    # ── Stage 1: Multi-pass detection (same as ASV Stage 1) ──────────────
    user_prompt = STAGE1_USER.format(text=article.text)

    pass_results: list[list[PredictedSpan]] = []
    raw_s1_outputs: list[str] = []

    for i, temp in enumerate(config.asv_temperatures[:config.asv_num_passes]):
        raw = await client.complete(
            system_prompt=STAGE1_SYSTEM,
            user_prompt=user_prompt,
            temperature=temp,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
        raw_s1_outputs.append(raw)
        spans, errors = parse_llm_response(raw, article.text, pass_id=i)
        pass_results.append(spans)
        if errors:
            logger.warning(f"Consol S1 parse errors [{article.id}] pass {i}: {errors}")

    merged = merge_multipass(pass_results, article.text)

    stage_outputs["stage1_raw"] = raw_s1_outputs
    stage_outputs["stage1_per_pass_counts"] = [len(p) for p in pass_results]
    stage_outputs["stage1_merged_count"] = len(merged)

    logger.info(
        f"Consol S1 [{article.id}]: "
        f"{sum(len(p) for p in pass_results)} total -> {len(merged)} merged"
    )

    if not merged:
        return Prediction(
            article_id=article.id, spans=[], stage_outputs=stage_outputs,
            model=config.model, strategy="consol",
        )

    # ── Stage 2: Consolidation ────────────────────────────────────────────
    candidates_for_consol = [
        {
            "span_text": s.span_text,
            "technique": s.technique.value,
            "agreement_count": s.agreement_count,
            "reasoning": s.reasoning or "",
        }
        for s in merged
    ]

    consol_user = CONSOL_USER.format(
        text=article.text,
        num_passes=config.asv_num_passes,
        candidates_json=format_candidates_for_consol(candidates_for_consol),
    )

    raw_s2 = await client.complete(
        system_prompt=CONSOL_SYSTEM,
        user_prompt=consol_user,
        temperature=0.0,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
    )
    stage_outputs["stage2_consol_raw"] = raw_s2

    consol_annotations = _parse_consol_output(raw_s2)
    stage_outputs["stage2_consol_count"] = len(consol_annotations)

    _apply_consol_decisions(merged, consol_annotations, article.text)

    confirmed = [s for s in merged if s.verdict == "CONFIRMED"]
    rejected = [s for s in merged if s.verdict == "REJECTED"]

    logger.info(
        f"Consol S2 [{article.id}]: {len(confirmed)} CONFIRMED, "
        f"{len(rejected)} REJECTED (of {len(merged)} candidates)"
    )

    return Prediction(
        article_id=article.id,
        spans=merged,
        stage_outputs=stage_outputs,
        model=config.model,
        strategy="consol",
    )


def _parse_consol_output(raw: str) -> list[dict]:
    """Parse the consolidation stage JSON output."""
    import re
    from src.parser import _extract_json

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    obj = _extract_json(cleaned)
    if obj is None:
        logger.warning("Failed to parse consolidation output")
        return []

    annotations = obj.get("annotations", [])
    if not isinstance(annotations, list):
        logger.warning("Consolidation output 'annotations' is not a list")
        return []

    return annotations


def _apply_consol_decisions(
    spans: list[PredictedSpan],
    consol_annotations: list[dict],
    article_text: str,
) -> None:
    """Apply consolidation decisions to the merged spans.

    The consolidator outputs ONLY the kept annotations (possibly with
    corrected labels or trimmed spans). Spans NOT in the consolidator's
    output were deliberately dropped and get REJECTED.

    Matching strategy: exact on original_text or text, then substring fallback.
    """
    import re

    kept_lookup: dict[str, dict] = {}
    for ann in consol_annotations:
        for key_field in ["original_text", "text"]:
            raw_text = ann.get(key_field, "")
            if raw_text:
                norm_key = re.sub(r"\s+", " ", raw_text.lower().strip())
                kept_lookup[norm_key] = ann

    for span in spans:
        norm_span = re.sub(r"\s+", " ", span.span_text.lower().strip())

        # Exact match first
        match = kept_lookup.get(norm_span)

        # Substring fallback: consolidator may have trimmed the span text
        if match is None:
            for key, ann in kept_lookup.items():
                if key in norm_span or norm_span in key:
                    match = ann
                    break

        if match is None:
            # Consolidator did NOT include this span -> it was dropped
            span.verdict = "REJECTED"
            span.confidence = 0.0
            continue

        action = match.get("action", "kept").lower()
        if action == "dropped":
            span.verdict = "REJECTED"
            span.confidence = 0.0
            continue

        span.verdict = "CONFIRMED"
        span.confidence = 80.0

        new_type = match.get("type", "")
        if new_type:
            corrected = normalise_technique(new_type)
            if corrected and corrected != span.technique:
                logger.info(
                    f"Consol relabeled: '{span.span_text[:40]}' "
                    f"{span.technique.value} -> {corrected.value}"
                )
                span.technique = corrected

        new_text = match.get("text", "")
        if new_text and new_text != span.span_text:
            idx = article_text.find(new_text)
            if idx >= 0:
                span.span_text = new_text
                span.start = idx
                span.end = idx + len(new_text)
                logger.info(
                    f"Consol trimmed: '{span.span_text[:40]}' "
                    f"to [{span.start}:{span.end}]"
                )


def _parse_critiques(raw: str) -> list[dict]:
    """Parse Stage 2 critique JSON output."""
    import re
    from src.parser import _extract_json

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    obj = _extract_json(cleaned)
    if obj is None:
        logger.warning("Failed to parse Stage 2 critiques")
        return []
    return obj.get("critiques", [])


def _parse_verdicts(raw: str) -> list[dict]:
    """Parse Stage 3 adjudication JSON output."""
    import re
    from src.parser import _extract_json

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    obj = _extract_json(cleaned)
    if obj is None:
        logger.warning("Failed to parse Stage 3 verdicts")
        return []
    return obj.get("verdicts", [])


def _apply_verdicts(spans: list[PredictedSpan], verdicts: list[dict]) -> None:
    """Match Stage 3 verdicts back to merged spans by span_text + technique."""
    import re

    verdict_lookup: dict[tuple[str, str], dict] = {}
    for v in verdicts:
        key = (
            re.sub(r"\s+", " ", v.get("span_text", "").lower().strip()),
            v.get("technique", "").lower().strip(),
        )
        verdict_lookup[key] = v

    for span in spans:
        key = (
            re.sub(r"\s+", " ", span.span_text.lower().strip()),
            span.technique.value.lower().strip(),
        )
        v = verdict_lookup.get(key)
        if v:
            span.verdict = v.get("verdict", "").upper()
            try:
                span.confidence = float(v.get("confidence", -1))
            except (ValueError, TypeError):
                span.confidence = -1.0

            # Normalise verdict values
            if span.verdict not in ("CONFIRMED", "POSSIBLE", "REJECTED"):
                if "confirm" in span.verdict.lower():
                    span.verdict = "CONFIRMED"
                elif "reject" in span.verdict.lower():
                    span.verdict = "REJECTED"
                elif "possible" in span.verdict.lower():
                    span.verdict = "POSSIBLE"
                else:
                    span.verdict = "POSSIBLE"  # Default uncertain
        else:
            # No matching verdict — explicitly mark as POSSIBLE so it is not
            # silently dropped by confirmed_spans (fix for logic bug).
            span.verdict = "POSSIBLE"
            span.confidence = -1.0
            logger.warning(
                f"No verdict match for span: '{span.span_text[:40]}' ({span.technique.value})"
            )


def _apply_critique_verdicts(spans: list[PredictedSpan], critiques: list[dict]) -> None:
    """Apply Stage 2 critique verdicts directly (2-stage ASV mode).

    Mapping:
    - CONCEDE -> CONFIRMED
    - CHALLENGE -> POSSIBLE if agreement_count >= 2, else REJECTED
    - Unmatched/unknown -> POSSIBLE
    """
    import re

    critique_lookup: dict[tuple[str, str], dict] = {}
    for c in critiques:
        key = (
            re.sub(r"\s+", " ", c.get("span_text", "").lower().strip()),
            c.get("technique", "").lower().strip(),
        )
        critique_lookup[key] = c

    for span in spans:
        key = (
            re.sub(r"\s+", " ", span.span_text.lower().strip()),
            span.technique.value.lower().strip(),
        )
        c = critique_lookup.get(key)
        if c:
            critique_verdict = c.get("verdict", "").upper()
            if critique_verdict == "CONCEDE":
                span.verdict = "CONFIRMED"
                span.confidence = 80.0
            elif critique_verdict == "CHALLENGE":
                if span.agreement_count >= 2:
                    span.verdict = "POSSIBLE"
                    span.confidence = 50.0
                else:
                    span.verdict = "REJECTED"
                    span.confidence = 30.0
            else:
                span.verdict = "POSSIBLE"
                span.confidence = 50.0
        else:
            span.verdict = "POSSIBLE"
            span.confidence = 50.0


def _norm_key(span_text: str, technique: str) -> tuple[str, str]:
    import re

    return (
        re.sub(r"\s+", " ", span_text.lower().strip()),
        technique.lower().strip(),
    )


def _build_critique_lookup(stage_outputs: dict[str, Any]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for c in stage_outputs.get("stage2_critiques", []):
        key = _norm_key(str(c.get("span_text", "")), str(c.get("technique", "")))
        lookup[key] = c
    return lookup


def _build_consol_lookup(stage_outputs: dict[str, Any]) -> dict[tuple[str, str], dict]:
    """Build lookup for consolidation annotations keyed by (text, type)."""
    import re
    from src.parser import _extract_json

    raw = stage_outputs.get("stage2_consol_raw", "")
    if not raw:
        return {}

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    obj = _extract_json(cleaned)
    if obj is None:
        return {}

    lookup: dict[tuple[str, str], dict] = {}
    for ann in obj.get("annotations", []):
        for text_field in ["original_text", "text"]:
            text = ann.get(text_field, "")
            technique = ann.get("original_type", "") or ann.get("type", "")
            if text and technique:
                lookup[_norm_key(text, technique)] = ann
        # Also key by corrected type so relabeled spans can be found
        text = ann.get("text", "") or ann.get("original_text", "")
        corrected_type = ann.get("type", "")
        if text and corrected_type:
            lookup[_norm_key(text, corrected_type)] = ann
    return lookup


async def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a full experiment: load data, predict all articles, evaluate.

    Returns a dict with config, metrics, cost info, and per-article results.
    """
    logger.info(f"Starting experiment: {config.name} ({config.strategy}, {config.model})")

    # Load data
    articles = load_semeval(
        articles_dir=config.articles_dir,
        labels_dir=config.labels_path,
    )
    logger.info(f"Loaded {len(articles)} articles")

    if config.max_articles is not None:
        if config.max_articles <= 0:
            raise ValueError("max_articles must be a positive integer (or None)")
        original_len = len(articles)
        articles = articles[: config.max_articles]
        logger.info(
            f"Limiting articles: {original_len} -> {len(articles)} "
            f"(max_articles={config.max_articles})"
        )

    # Create client
    client = LLMClient(
        model=config.model,
        cache_dir=f"outputs/cache/{config.run_id()}",
        max_cost_usd=config.max_cost_usd,
    )

    # Run predictions
    predictions: dict[str, Prediction] = {}
    start_time = time.time()

    for i, article in enumerate(articles):
        try:
            pred = await run_single_article(article, client, config)
            predictions[article.id] = pred
        except RuntimeError as e:
            if "Budget exceeded" in str(e):
                logger.error(f"Budget exceeded after {i} articles: {e}")
                break
            raise
        except Exception as e:
            logger.error(f"Error on article {article.id}: {e}")
            continue

        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{len(articles)} articles, "
                f"cost: {client.cost_tracker.summary()}"
            )

    elapsed = time.time() - start_time

    # Evaluate
    metrics = evaluate(articles, predictions)
    print_results(metrics, strategy=f"{config.strategy} ({config.model})")

    # Package results — save ALL spans for error analysis, mark verdict status
    def serialize_prediction(pred: Prediction) -> dict[str, Any]:
        is_asv = pred.strategy == "asv"
        is_consol = pred.strategy == "consol"

        critique_lookup = _build_critique_lookup(pred.stage_outputs) if is_asv else {}
        consol_lookup = _build_consol_lookup(pred.stage_outputs) if is_consol else {}

        all_spans = []
        for s in pred.spans:
            entry: dict[str, Any] = {
                "technique": s.technique.value,
                "span_text": s.span_text,
                "start": s.start,
                "end": s.end,
                "reasoning": s.reasoning,
                "agreement_count": s.agreement_count,
                "confidence": s.confidence,
                "verdict": s.verdict,
            }

            if is_asv:
                critique = critique_lookup.get(
                    _norm_key(s.span_text, s.technique.value), {}
                )
                entry["critique_substitution_test"] = critique.get("substitution_test", "")
                entry["critique_innocent_explanation"] = critique.get("innocent_explanation", "")
                entry["critique_weakness"] = critique.get("weakness", "")
                entry["critique_verdict"] = str(critique.get("verdict", "")).upper()

            if is_consol:
                consol = consol_lookup.get(
                    _norm_key(s.span_text, s.technique.value), {}
                )
                entry["consol_action"] = consol.get("action", "")
                entry["consol_reason"] = consol.get("reason", "")
                entry["consol_original_type"] = consol.get("original_type", "")
                entry["consol_original_text"] = consol.get("original_text", "")

            all_spans.append(entry)

        return {
            "all_spans": all_spans,
            "confirmed_count": len(pred.confirmed_spans),
            "total_count": len(pred.spans),
            "stage_outputs": pred.stage_outputs,
        }

    results = {
        "config": config.model_dump(),
        "metrics": metrics.to_dict(),
        "cost": client.cost_tracker.to_dict(),
        "elapsed_seconds": round(elapsed, 1),
        "articles_processed": len(predictions),
        "predictions": {
            aid: serialize_prediction(pred)
            for aid, pred in predictions.items()
        },
    }

    # Save results
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.name}_{config.run_id()}.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved to {output_path}")

    return results