"""Experiment runner for the five prompting strategies.

Strategies: zero_shot, few_shot, asv (detect + critique), consol (detect +
consolidate), hybrid (detect + critique + refinement).
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from src.data.loader import load_semeval
from src.evaluation.charts import generate_diagnostic_charts
from src.evaluation.diagnostics import compute_all_diagnostics
from src.evaluation.metrics import evaluate, print_results
from src.llm_client import LLMClient
from src.parser import merge_multipass, parse_llm_response, _extract_json
from src.prompts.consolidation import (
    CONSOL_SYSTEM, CONSOL_USER, format_candidates_for_consol,
)
from src.prompts.refinement import (
    REFINE_SYSTEM, REFINE_USER, format_candidates_for_refine,
)
from src.prompts.critique import (
    STAGE2_SYSTEM, STAGE2_USER, format_detections_for_stage2,
)
from src.prompts.detection import (
    FEW_SHOT_SYSTEM, FEW_SHOT_USER,
    STAGE1_SYSTEM, STAGE1_USER,
    ZERO_SHOT_SYSTEM, ZERO_SHOT_USER,
)
from src.schemas import (
    Article, ExperimentConfig, PredictedSpan, Prediction, normalise_technique,
)

logger = logging.getLogger(__name__)


async def run_single_article(
    article: Article, client: LLMClient, config: ExperimentConfig,
) -> Prediction:
    if config.strategy == "asv":
        return await _run_asv(article, client, config)
    if config.strategy == "consol":
        return await _run_consol(article, client, config)
    if config.strategy == "hybrid":
        return await _run_hybrid(article, client, config)
    return await _run_baseline(article, client, config)


def _snapshot_span(s: PredictedSpan) -> dict:
    return {
        "technique": s.technique.value,
        "span_text": s.span_text,
        "start": s.start,
        "end": s.end,
        "verdict": s.verdict,
        "agreement_count": s.agreement_count,
        "was_relabeled": s.was_relabeled,
        "was_trimmed": s.was_trimmed,
        "consol_action": s.consol_action,
        "original_technique": (
            s.original_technique.value if s.original_technique else None
        ),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _serialise_pass_spans(pass_results: list[list[PredictedSpan]]) -> list[list[dict]]:
    return [
        [
            {
                "technique": s.technique.value,
                "span_text": s.span_text,
                "start": s.start,
                "end": s.end,
                "pass_id": s.pass_id,
                "reasoning": s.reasoning,
                "resolved": s.resolved,
            }
            for s in pass_spans
        ]
        for pass_spans in pass_results
    ]


# Baseline strategies: zero-shot and few-shot.

async def _run_baseline(
    article: Article, client: LLMClient, config: ExperimentConfig,
) -> Prediction:
    if config.strategy == "few_shot":
        system_prompt = FEW_SHOT_SYSTEM
        user_template = FEW_SHOT_USER
    else:
        system_prompt = ZERO_SHOT_SYSTEM
        user_template = ZERO_SHOT_USER

    user_prompt = user_template.format(text=article.text)

    with client.cost_tracker.stage(f"{config.strategy}_detection"):
        raw_output = await client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            seed=config.seed,
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
        eval_mode=config.eval_mode,
    )


# Stage 1: multi-pass detection (shared by asv, consol, hybrid).

async def _run_stage1(
    article: Article, client: LLMClient, config: ExperimentConfig, label: str,
) -> tuple[list[PredictedSpan], dict[str, Any], list[dict]]:
    user_prompt = STAGE1_USER.format(text=article.text)
    pass_results: list[list[PredictedSpan]] = []
    raw_outputs: list[str] = []

    with client.cost_tracker.stage("stage1_detection"):
        for i, temp in enumerate(config.asv_temperatures[:config.asv_num_passes]):
            raw = await client.complete(
                system_prompt=STAGE1_SYSTEM,
                user_prompt=user_prompt,
                temperature=temp,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"},
                seed=config.seed + i,
            )
            raw_outputs.append(raw)
            spans, errors = parse_llm_response(raw, article.text, pass_id=i)
            for s in spans:
                s.stage_history.append(f"S1_pass{i}")
            pass_results.append(spans)
            if errors:
                logger.warning(f"{label} S1 parse errors [{article.id}] pass {i}: {errors}")

    merged = merge_multipass(pass_results, article.text)
    for s in merged:
        s.stage_history.append("S1_merged")

    stage_outputs = {
        "stage1_raw": raw_outputs,
        "stage1_per_pass_spans": _serialise_pass_spans(pass_results),
        "stage1_per_pass_counts": [len(p) for p in pass_results],
        "stage1_merged_count": len(merged),
    }
    snapshot = [_snapshot_span(s) for s in merged]

    logger.info(
        f"{label} S1 [{article.id}]: "
        f"{sum(len(p) for p in pass_results)} total -> {len(merged)} merged"
    )
    return merged, stage_outputs, snapshot


# ASV: Stage 1 detection + Stage 2 adversarial critique.

async def _run_asv(
    article: Article, client: LLMClient, config: ExperimentConfig,
) -> Prediction:
    merged, stage_outputs, s1_snap = await _run_stage1(article, client, config, "ASV")
    stage_snapshots = {"after_s1": s1_snap}

    if not merged:
        return Prediction(
            article_id=article.id, spans=[], stage_outputs=stage_outputs,
            stage_snapshots=stage_snapshots, model=config.model, strategy="asv",
            eval_mode=config.eval_mode,
        )

    detections = [
        {"span_text": s.span_text, "technique": s.technique.value,
         "agreement_count": s.agreement_count}
        for s in merged
    ]
    s2_user = STAGE2_USER.format(
        text=article.text,
        detections_json=format_detections_for_stage2(detections),
    )

    with client.cost_tracker.stage("stage2_critique"):
        raw_s2 = await client.complete(
            system_prompt=STAGE2_SYSTEM, user_prompt=s2_user,
            temperature=0.0, max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            model_override=config.verify_model,
            seed=config.seed,
        )

    critiques = _parse_critiques(raw_s2)
    stage_outputs["stage2_raw"] = raw_s2
    stage_outputs["stage2_model"] = config.verify_model or config.model
    stage_outputs["stage2_critiques_count"] = len(critiques)
    stage_outputs["stage2_critique_parsed"] = critiques

    _apply_critique_verdicts(merged, critiques)
    for s in merged:
        s.stage_history.append(f"S2_{s.verdict}")
    stage_snapshots["after_s2"] = [_snapshot_span(s) for s in merged]

    confirmed = sum(1 for s in merged if s.verdict == "CONFIRMED")
    possible = sum(1 for s in merged if s.verdict == "POSSIBLE")
    rejected = sum(1 for s in merged if s.verdict == "REJECTED")
    logger.info(
        f"ASV final [{article.id}]: {confirmed} CONFIRMED, "
        f"{possible} POSSIBLE, {rejected} REJECTED"
    )

    return Prediction(
        article_id=article.id, spans=merged, stage_outputs=stage_outputs,
        stage_snapshots=stage_snapshots, model=config.model, strategy="asv",
        eval_mode=config.eval_mode,
    )


# Consolidation: Stage 1 detection + Stage 2 constructive consolidation.

async def _run_consol(
    article: Article, client: LLMClient, config: ExperimentConfig,
) -> Prediction:
    merged, stage_outputs, s1_snap = await _run_stage1(article, client, config, "Consol")
    stage_snapshots = {"after_s1": s1_snap}

    if not merged:
        return Prediction(
            article_id=article.id, spans=[], stage_outputs=stage_outputs,
            stage_snapshots=stage_snapshots, model=config.model, strategy="consol",
            eval_mode=config.eval_mode,
        )

    candidates = [
        {"span_text": s.span_text, "technique": s.technique.value,
         "agreement_count": s.agreement_count, "reasoning": s.reasoning or ""}
        for s in merged
    ]
    consol_user = CONSOL_USER.format(
        text=article.text, num_passes=config.asv_num_passes,
        candidates_json=format_candidates_for_consol(candidates),
    )

    with client.cost_tracker.stage("stage2_consolidation"):
        raw_s2 = await client.complete(
            system_prompt=CONSOL_SYSTEM, user_prompt=consol_user,
            temperature=0.0, max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            model_override=config.verify_model,
            seed=config.seed,
        )

    consol_annotations = _parse_annotations(raw_s2)
    stage_outputs["stage2_consol_raw"] = raw_s2
    stage_outputs["stage2_model"] = config.verify_model or config.model
    stage_outputs["stage2_consol_count"] = len(consol_annotations)
    stage_outputs["stage2_consol_parsed"] = consol_annotations

    _apply_consol_decisions(merged, consol_annotations, article.text)
    stage_snapshots["after_s2"] = [_snapshot_span(s) for s in merged]

    confirmed = sum(1 for s in merged if s.verdict == "CONFIRMED")
    rejected = sum(1 for s in merged if s.verdict == "REJECTED")
    possible = sum(1 for s in merged if s.verdict == "POSSIBLE")
    logger.info(
        f"Consol S2 [{article.id}]: {confirmed} CONFIRMED, "
        f"{rejected} REJECTED, {possible} POSSIBLE"
    )

    return Prediction(
        article_id=article.id, spans=merged, stage_outputs=stage_outputs,
        stage_snapshots=stage_snapshots, model=config.model, strategy="consol",
        eval_mode=config.eval_mode,
    )


# Hybrid: detect + adversarial critique (filters) + refinement (no drops).

async def _run_hybrid(
    article: Article, client: LLMClient, config: ExperimentConfig,
) -> Prediction:
    merged, stage_outputs, s1_snap = await _run_stage1(article, client, config, "Hybrid")
    stage_snapshots = {"after_s1": s1_snap}

    if not merged:
        return Prediction(
            article_id=article.id, spans=[], stage_outputs=stage_outputs,
            stage_snapshots=stage_snapshots, model=config.model, strategy="hybrid",
            eval_mode=config.eval_mode,
        )

    # Stage 2: adversarial critique
    detections = [
        {"span_text": s.span_text, "technique": s.technique.value,
         "agreement_count": s.agreement_count}
        for s in merged
    ]
    s2_user = STAGE2_USER.format(
        text=article.text,
        detections_json=format_detections_for_stage2(detections),
    )
    with client.cost_tracker.stage("stage2_critique"):
        raw_s2 = await client.complete(
            system_prompt=STAGE2_SYSTEM, user_prompt=s2_user,
            temperature=0.0, max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            model_override=config.verify_model,
            seed=config.seed,
        )

    critiques = _parse_critiques(raw_s2)
    stage_outputs["stage2_critique_raw"] = raw_s2
    stage_outputs["stage2_model"] = config.verify_model or config.model
    stage_outputs["stage2_critique_parsed"] = critiques
    _apply_critique_verdicts(merged, critiques)
    for s in merged:
        s.stage_history.append(f"S2_{s.verdict}")

    surviving = [s for s in merged if s.verdict in ("CONFIRMED", "POSSIBLE")]
    rejected_s2 = [s for s in merged if s.verdict == "REJECTED"]
    stage_snapshots["after_s2"] = [_snapshot_span(s) for s in merged]

    logger.info(
        f"Hybrid S2 [{article.id}]: {len(surviving)} surviving, "
        f"{len(rejected_s2)} rejected by critique"
    )

    if not surviving:
        return Prediction(
            article_id=article.id, spans=merged, stage_outputs=stage_outputs,
            stage_snapshots=stage_snapshots, model=config.model, strategy="hybrid",
            eval_mode=config.eval_mode,
        )

    # Stage 3: refinement (no drops allowed by prompt)
    candidates = [
        {"span_text": s.span_text, "technique": s.technique.value}
        for s in surviving
    ]
    refine_user = REFINE_USER.format(
        text=article.text,
        candidates_json=format_candidates_for_refine(candidates),
    )
    with client.cost_tracker.stage("stage3_refinement"):
        raw_s3 = await client.complete(
            system_prompt=REFINE_SYSTEM, user_prompt=refine_user,
            temperature=0.0, max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            model_override=config.verify_model,
            seed=config.seed,
        )

    refine_annotations = _parse_annotations(raw_s3)
    stage_outputs["stage3_refine_raw"] = raw_s3
    stage_outputs["stage3_model"] = config.verify_model or config.model
    stage_outputs["stage3_refine_parsed"] = refine_annotations

    disobey_count = _apply_refinement_decisions(surviving, refine_annotations, article.text)
    stage_outputs["stage3_refiner_disobeyed"] = disobey_count
    stage_snapshots["after_s3"] = [_snapshot_span(s) for s in merged]

    n_relabeled = sum(1 for s in surviving if s.was_relabeled)
    n_trimmed = sum(1 for s in surviving if s.was_trimmed)
    logger.info(
        f"Hybrid S3 [{article.id}]: {len(surviving)} kept "
        f"({n_relabeled} relabeled, {n_trimmed} trimmed, "
        f"{disobey_count} refiner-disobeys)"
    )

    return Prediction(
        article_id=article.id, spans=merged, stage_outputs=stage_outputs,
        stage_snapshots=stage_snapshots, model=config.model, strategy="hybrid",
        eval_mode=config.eval_mode,
    )


# Consolidation / refinement decision application.

def _snapshot_originals(spans: list[PredictedSpan]) -> None:
    for s in spans:
        if s.original_technique is None:
            s.original_technique = s.technique
            s.original_span_text = s.span_text
            s.original_start = s.start
            s.original_end = s.end


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip())


def _build_annotation_lookup(annotations: list[dict]) -> tuple[dict, list[tuple[str, dict]]]:
    """Build exact-match lookup and full annotation list for substring fallback."""
    exact_lookup: dict[str, dict] = {}
    all_anns: list[tuple[str, dict]] = []
    for ann in annotations:
        for key_field in ("original_text", "text"):
            raw = ann.get(key_field, "")
            if raw:
                norm = _norm_text(raw)
                if norm not in exact_lookup:
                    exact_lookup[norm] = ann
                all_anns.append((norm, ann))
    return exact_lookup, all_anns


def _find_annotation(
    span_norm: str,
    exact_lookup: dict,
    all_anns: list[tuple[str, dict]],
    used: set,
) -> dict | None:
    """Match a span to an annotation: exact first, then substring either direction."""
    if span_norm in exact_lookup:
        ann = exact_lookup[span_norm]
        if id(ann) not in used:
            used.add(id(ann))
            return ann
    for ann_norm, ann in all_anns:
        if id(ann) in used:
            continue
        if span_norm in ann_norm or ann_norm in span_norm:
            used.add(id(ann))
            return ann
    return None


def _apply_consol_decisions(
    spans: list[PredictedSpan],
    consol_annotations: list[dict],
    article_text: str,
) -> None:
    _snapshot_originals(spans)
    exact_lookup, all_anns = _build_annotation_lookup(consol_annotations)
    used: set = set()

    for span in spans:
        span_norm = _norm_text(span.span_text)
        match = _find_annotation(span_norm, exact_lookup, all_anns, used)

        if not match:
            span.verdict = span.verdict or "POSSIBLE"
            span.consol_action = "unmatched"
            span.stage_history.append("consol_unmatched")
            continue

        action = match.get("action", "kept").lower()
        span.consol_action = action
        span.stage_history.append(f"consol_{action}")

        if action == "dropped":
            drop_reason = match.get("drop_justification", match.get("reason", ""))
            span.reasoning = f"[CONSOL DROP] {drop_reason}"
            span.verdict = "REJECTED"
            continue

        new_type = match.get("type", "")
        if new_type:
            corrected = normalise_technique(new_type)
            if corrected and corrected != span.technique:
                span.technique = corrected
                span.was_relabeled = True

        new_text = match.get("text", "")
        if new_text and new_text != span.span_text:
            new_start = _find_closest_occurrence(article_text, new_text, anchor=span.start)
            if new_start >= 0:
                span.span_text = new_text
                span.start = new_start
                span.end = new_start + len(new_text)
                span.was_trimmed = True

        span.verdict = "CONFIRMED"


def _apply_refinement_decisions(
    spans: list[PredictedSpan],
    refine_annotations: list[dict],
    article_text: str,
) -> int:
    """Like consolidation but disallows drops. Returns count of disobey events."""
    _snapshot_originals(spans)
    exact_lookup, all_anns = _build_annotation_lookup(refine_annotations)
    used: set = set()
    disobey_count = 0

    for span in spans:
        span_norm = _norm_text(span.span_text)
        match = _find_annotation(span_norm, exact_lookup, all_anns, used)

        if not match:
            span.verdict = span.verdict or "CONFIRMED"
            span.consol_action = "unmatched"
            span.stage_history.append("refine_unmatched")
            continue

        action = match.get("action", "kept").lower()
        if action == "dropped":
            disobey_count += 1
            logger.warning(
                f"Refiner disobeyed 'no drops' instruction for span: "
                f"'{span.span_text[:40]}' — treating as kept"
            )
            action = "kept"

        span.consol_action = action
        span.stage_history.append(f"refine_{action}")

        new_type = match.get("type", "")
        if new_type:
            corrected = normalise_technique(new_type)
            if corrected and corrected != span.technique:
                span.technique = corrected
                span.was_relabeled = True

        new_text = match.get("text", "")
        if new_text and new_text != span.span_text:
            new_start = _find_closest_occurrence(article_text, new_text, anchor=span.start)
            if new_start >= 0:
                span.span_text = new_text
                span.start = new_start
                span.end = new_start + len(new_text)
                span.was_trimmed = True

        if span.verdict not in ("CONFIRMED", "POSSIBLE"):
            span.verdict = "CONFIRMED"

    return disobey_count


def _find_closest_occurrence(haystack: str, needle: str, anchor: int) -> int:
    """Find the occurrence of needle in haystack whose start is closest to anchor.

    Prevents first-occurrence ambiguity when a span's text appears multiple times.
    """
    if not needle:
        return -1

    def _all_matches(h: str, n: str) -> list[int]:
        out = []
        pos = 0
        while (idx := h.find(n, pos)) >= 0:
            out.append(idx)
            pos = idx + 1
        return out

    occurrences = _all_matches(haystack, needle)
    if not occurrences:
        occurrences = _all_matches(haystack.lower(), needle.lower())
    if not occurrences:
        return -1
    if anchor < 0:
        return occurrences[0]
    return min(occurrences, key=lambda o: abs(o - anchor))


# JSON parsers for stage outputs.

def _parse_annotations(raw: str) -> list[dict]:
    """Parse {'annotations': [...]} from consol/refine output."""
    obj = _extract_json(raw)
    if obj is None:
        logger.warning("Failed to parse consolidation/refinement output")
        return []
    anns = obj.get("annotations", [])
    return anns if isinstance(anns, list) else []


def _parse_critiques(raw: str) -> list[dict]:
    obj = _extract_json(raw)
    if obj is None:
        return []
    return obj.get("critiques", [])


def _apply_critique_verdicts(
    spans: list[PredictedSpan], critiques: list[dict],
) -> None:
    """CONCEDE -> CONFIRMED; CHALLENGE + agreement>=2 -> POSSIBLE; else REJECTED."""
    lookup: dict[tuple[str, str], dict] = {}
    for c in critiques:
        key = (
            _norm_text(c.get("span_text", "")),
            c.get("technique", "").lower().strip(),
        )
        lookup[key] = c

    for span in spans:
        key = (_norm_text(span.span_text), span.technique.value.lower().strip())
        c = lookup.get(key)
        if not c:
            span.verdict = "POSSIBLE"
            continue

        verdict = c.get("verdict", "").upper()
        if verdict == "CONCEDE":
            span.verdict = "CONFIRMED"
        elif verdict == "CHALLENGE":
            span.verdict = "POSSIBLE" if span.agreement_count >= 2 else "REJECTED"
        else:
            span.verdict = "POSSIBLE"


# Top-level orchestration.

async def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    logger.info(f"Starting experiment: {config.name} ({config.strategy}, {config.model})")

    articles = load_semeval(
        articles_dir=config.articles_dir, labels_dir=config.labels_path,
    )
    if config.max_articles is not None:
        articles = articles[:config.max_articles]
    logger.info(f"Loaded {len(articles)} articles")
    article_by_id = {a.id: a for a in articles}

    client = LLMClient(
        model=config.model,
        cache_dir=f"outputs/cache/{config.run_id()}",
        max_cost_usd=config.max_cost_usd,
    )

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
                f"Progress: {i+1}/{len(articles)}, "
                f"cost: {client.cost_tracker.summary()}"
            )

    elapsed = time.time() - start_time

    metrics = evaluate(articles, predictions)
    print_results(metrics, strategy=f"{config.strategy} ({config.model})")

    diagnostics = compute_all_diagnostics(articles, predictions)
    _print_diagnostic_summary(diagnostics, config.name)

    results = {
        "config": config.model_dump(),
        "metrics": metrics.to_dict(),
        "diagnostics": diagnostics,
        "cost": client.cost_tracker.to_dict(),
        "elapsed_seconds": round(elapsed, 1),
        "articles_processed": len(predictions),
        "predictions": _serialise_predictions(predictions, article_by_id),
        "run_metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "config_hash": config.run_id(),
            "strategy": config.strategy,
            "model": config.model,
            "verify_model": config.verify_model,
            "eval_mode": config.eval_mode,
            "seed": config.seed,
            "n_articles_processed": len(predictions),
            "n_articles_loaded": len(articles),
        },
    }

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.name}_{config.run_id()}.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved to {output_path}")

    diag_dir = Path("outputs/diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)
    chart_prefix = diag_dir / f"{config.name}_{config.run_id()}"
    chart_paths = generate_diagnostic_charts(
        diagnostics, chart_prefix,
        strategy_label=f"{config.strategy} / {config.model}",
    )
    if chart_paths:
        logger.info(f"Diagnostic charts: {len(chart_paths)} saved to {diag_dir}")

    return results


def _serialise_predictions(
    predictions: dict[str, Prediction], article_by_id: dict[str, Article],
) -> dict[str, dict]:
    return {
        aid: {
            "article_text": article_by_id[aid].text if aid in article_by_id else "",
            "gold_spans": [
                {"technique": g.technique.value, "start": g.start, "end": g.end}
                for g in (article_by_id[aid].gold_spans if aid in article_by_id else [])
            ],
            "all_spans": [
                {
                    "technique": s.technique.value,
                    "span_text": s.span_text,
                    "start": s.start, "end": s.end,
                    "reasoning": s.reasoning,
                    "agreement_count": s.agreement_count,
                    "verdict": s.verdict,
                    "pass_id": s.pass_id,
                    "original_technique": (
                        s.original_technique.value if s.original_technique else None
                    ),
                    "original_span_text": s.original_span_text,
                    "original_start": s.original_start,
                    "original_end": s.original_end,
                    "was_relabeled": s.was_relabeled,
                    "was_trimmed": s.was_trimmed,
                    "consol_action": s.consol_action,
                    "stage_history": s.stage_history,
                }
                for s in pred.spans
            ],
            "stage_snapshots": pred.stage_snapshots,
            "stage_outputs": pred.stage_outputs,
            "confirmed_count": len(pred.confirmed_spans),
            "total_count": len(pred.spans),
        }
        for aid, pred in predictions.items()
    }


def _print_diagnostic_summary(diagnostics: dict, name: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  DIAGNOSTICS - {name}")
    print(f"{'-' * 60}")

    stages = diagnostics.get("stage_by_stage_f1", {})
    if stages:
        print("  Stage-by-stage F1:")
        for stage in ["after_s1", "after_s2", "after_s3"]:
            if stage in stages:
                si = stages[stage]["si"]["f1"]
                tc = stages[stage]["tc"]["f1"]
                print(f"    {stage:12s}  SI={si:.4f}  TC={tc:.4f}")

    lucky = diagnostics.get("lucky_vs_systematic", {})
    if lucky.get("total_relabels", 0) > 0:
        ratio = lucky.get("overall_gain_ratio")
        net = lucky.get("net_f1_contribution", 0)
        ratio_s = f"{ratio:.2%}" if ratio is not None else "N/A"
        print(
            f"  Relabels: {lucky['total_relabels']} total "
            f"({lucky['total_gains']} gains, {lucky['total_losses']} losses, "
            f"{lucky['total_neutral']} neutral)"
        )
        print(f"    Gain ratio: {ratio_s}  |  Net F1 contribution: {net:+d}")

    drop_q = diagnostics.get("drop_quality", {})
    if drop_q.get("total_dropped", 0) > 0:
        fdr = drop_q["false_drop_rate"]
        print(
            f"  Drops: {drop_q['total_dropped']} total, "
            f"{drop_q['false_drops']} were gold -> false drop rate {fdr:.1%}"
        )

    disobey = diagnostics.get("refiner_disobedience", {})
    if disobey.get("total_disobey_events", 0) > 0:
        print(
            f"  Refiner disobeyed 'no drops' rule: "
            f"{disobey['total_disobey_events']} events across "
            f"{disobey['articles_with_disobey']}/{disobey['total_articles']} articles"
        )

    print(f"{'-' * 60}\n")