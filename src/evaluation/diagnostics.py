"""
Diagnostic metrics for multi-stage propaganda detection pipelines.

Computes:
1. Label-change log — every (original, new, action, agreement, correct_before/after)
2. 14x14 relabel matrix — confusion matrix of technique corrections
3. Lucky-vs-systematic analysis — gains vs losses per technique from relabelling
4. Drop quality — false drop rate (dropped candidates that were actually gold)
5. Stage-by-stage F1 — F1 computed from stage_snapshots at each pipeline stage
6. Agreement-vs-correctness — per-agreement-count precision

All diagnostics are computed from Prediction.stage_snapshots + gold labels.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from src.schemas import Article, GoldSpan, Prediction, Technique


# ── Helpers ────────────────────────────────────────────────────────────────

def _span_matches_gold(
    pred_technique: str,
    pred_start: int,
    pred_end: int,
    pred_text: str,
    gold_spans: list[GoldSpan],
    article_text: str,
    overlap_threshold: float = 0.25,
) -> tuple[bool, bool]:
    """Check if a predicted span matches gold.

    Returns (si_match, tc_match):
        si_match = span overlap >= threshold with any gold span (any technique)
        tc_match = overlap >= threshold AND same technique
    """
    # If pred_start unresolved, try to resolve from text
    if pred_start < 0:
        idx = article_text.find(pred_text)
        if idx >= 0:
            pred_start = idx
            pred_end = idx + len(pred_text)
        else:
            return False, False

    si_match = False
    tc_match = False
    for g in gold_spans:
        overlap_start = max(pred_start, g.start)
        overlap_end = min(pred_end, g.end)
        if overlap_end <= overlap_start:
            continue
        overlap = overlap_end - overlap_start
        pred_len = max(1, pred_end - pred_start)
        gold_len = max(1, g.end - g.start)
        min_len = min(pred_len, gold_len)
        if overlap / min_len >= overlap_threshold:
            si_match = True
            if g.technique.value == pred_technique:
                tc_match = True
                break
    return si_match, tc_match


# ── 1. Label change log ───────────────────────────────────────────────────

def build_label_change_log(
    articles: list[Article],
    predictions: dict[str, Prediction],
) -> list[dict]:
    """Extract every consolidation decision with gold correctness before/after."""
    log: list[dict] = []
    article_by_id = {a.id: a for a in articles}

    for aid, pred in predictions.items():
        article = article_by_id.get(aid)
        if not article:
            continue
        for span in pred.spans:
            if not span.consol_action or span.consol_action == "unmatched":
                continue

            orig_tech = span.original_technique.value if span.original_technique else span.technique.value
            orig_text = span.original_span_text or span.span_text
            orig_start = span.original_start if span.original_start >= 0 else span.start
            orig_end = span.original_end if span.original_end >= 0 else span.end

            _, tc_before = _span_matches_gold(
                orig_tech, orig_start, orig_end, orig_text,
                article.gold_spans, article.text,
            )
            _, tc_after = _span_matches_gold(
                span.technique.value, span.start, span.end, span.span_text,
                article.gold_spans, article.text,
            )
            si_after, _ = _span_matches_gold(
                span.technique.value, span.start, span.end, span.span_text,
                article.gold_spans, article.text,
            )

            log.append({
                "article_id": aid,
                "original_label": orig_tech,
                "new_label": span.technique.value,
                "action": span.consol_action,
                "agreement_count": span.agreement_count,
                "was_relabeled": span.was_relabeled,
                "was_trimmed": span.was_trimmed,
                "tc_correct_before": tc_before,
                "tc_correct_after": tc_after,
                "si_correct_after": si_after,
                "improved": tc_after and not tc_before,
                "worsened": tc_before and not tc_after,
                "verdict": span.verdict,
            })
    return log


# ── 2. 14x14 relabel confusion matrix ─────────────────────────────────────

def build_relabel_matrix(label_log: list[dict]) -> dict[str, Any]:
    """Build a 14x14 matrix: rows = original label, cols = new label.

    Also computes per-cell F1 delta (tc_correct_after - tc_correct_before).
    """
    techniques = [t.value for t in Technique]
    count_matrix = {o: {n: 0 for n in techniques} for o in techniques}
    improved_matrix = {o: {n: 0 for n in techniques} for o in techniques}
    worsened_matrix = {o: {n: 0 for n in techniques} for o in techniques}

    for entry in label_log:
        if not entry["was_relabeled"]:
            continue
        o = entry["original_label"]
        n = entry["new_label"]
        if o not in count_matrix or n not in count_matrix[o]:
            continue
        count_matrix[o][n] += 1
        if entry["improved"]:
            improved_matrix[o][n] += 1
        if entry["worsened"]:
            worsened_matrix[o][n] += 1

    return {
        "techniques": techniques,
        "counts": count_matrix,
        "improved": improved_matrix,
        "worsened": worsened_matrix,
    }


# ── 3. Lucky-vs-systematic analysis ───────────────────────────────────────

def lucky_vs_systematic(label_log: list[dict]) -> dict[str, Any]:
    """Per-technique breakdown: are relabel gains systematic or noisy?

    For each ORIGINAL technique label, count:
        gains   = relabels where tc_correct went False -> True
        losses  = relabels where tc_correct went True -> False
        neutral = relabels where tc_correct unchanged
    Ratio gains/(gains+losses) > 0.7 -> systematic, 0.4-0.7 -> mixed, <0.4 -> noisy.
    """
    per_tech: dict[str, dict] = defaultdict(lambda: {
        "relabels": 0, "gains": 0, "losses": 0, "neutral": 0,
    })
    for entry in label_log:
        if not entry["was_relabeled"]:
            continue
        o = entry["original_label"]
        per_tech[o]["relabels"] += 1
        if entry["improved"]:
            per_tech[o]["gains"] += 1
        elif entry["worsened"]:
            per_tech[o]["losses"] += 1
        else:
            per_tech[o]["neutral"] += 1

    for tech, d in per_tech.items():
        denom = d["gains"] + d["losses"]
        d["gain_ratio"] = d["gains"] / denom if denom > 0 else None
        if denom == 0:
            d["verdict"] = "no_signal"
        elif d["gain_ratio"] >= 0.7:
            d["verdict"] = "systematic"
        elif d["gain_ratio"] >= 0.4:
            d["verdict"] = "mixed"
        else:
            d["verdict"] = "noisy"

    total_gains = sum(d["gains"] for d in per_tech.values())
    total_losses = sum(d["losses"] for d in per_tech.values())
    total_neutral = sum(d["neutral"] for d in per_tech.values())
    total = total_gains + total_losses + total_neutral
    overall_ratio = total_gains / (total_gains + total_losses) if (total_gains + total_losses) > 0 else None

    return {
        "per_technique": dict(per_tech),
        "total_relabels": total,
        "total_gains": total_gains,
        "total_losses": total_losses,
        "total_neutral": total_neutral,
        "overall_gain_ratio": overall_ratio,
        "net_f1_contribution": total_gains - total_losses,
    }


# ── 4. Drop quality (false drop rate) ─────────────────────────────────────

def drop_quality(
    articles: list[Article],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Of the candidates that were DROPPED, how many were actually gold?"""
    article_by_id = {a.id: a for a in articles}
    total_dropped = 0
    false_drops = 0  # dropped but actually matched gold
    per_technique_dropped: dict[str, int] = defaultdict(int)
    per_technique_false: dict[str, int] = defaultdict(int)

    for aid, pred in predictions.items():
        article = article_by_id.get(aid)
        if not article:
            continue
        for span in pred.spans:
            if span.consol_action != "dropped" and span.verdict != "REJECTED":
                continue
            total_dropped += 1
            orig_tech = span.original_technique.value if span.original_technique else span.technique.value
            per_technique_dropped[orig_tech] += 1
            si_match, _ = _span_matches_gold(
                orig_tech, span.original_start if span.original_start >= 0 else span.start,
                span.original_end if span.original_end >= 0 else span.end,
                span.original_span_text or span.span_text,
                article.gold_spans, article.text,
            )
            if si_match:
                false_drops += 1
                per_technique_false[orig_tech] += 1

    fdr = false_drops / total_dropped if total_dropped > 0 else 0.0
    return {
        "total_dropped": total_dropped,
        "false_drops": false_drops,
        "false_drop_rate": fdr,
        "per_technique_dropped": dict(per_technique_dropped),
        "per_technique_false_drops": dict(per_technique_false),
    }


# ── 5. Stage-by-stage F1 ──────────────────────────────────────────────────

def stage_by_stage_f1(
    articles: list[Article],
    predictions: dict[str, Prediction],
    overlap_threshold: float = 0.25,
) -> dict[str, dict[str, float]]:
    """Compute SI/TC F1 from each stage snapshot using the official SemEval formula.

    Rebuilds a Prediction from each snapshot's spans and runs evaluate()
    so the numbers are directly comparable to the headline metrics.

    Filters each snapshot to only spans that would be "confirmed" at that stage:
        after_s1 → all spans (union of passes)
        after_s2 → verdict in CONFIRMED, POSSIBLE
        after_s3 → verdict in CONFIRMED, POSSIBLE
    """
    from src.evaluation.metrics import evaluate
    from src.schemas import PredictedSpan, normalise_technique

    article_by_id = {a.id: a for a in articles}
    stages = ["after_s1", "after_s2", "after_s3"]
    results: dict[str, dict[str, float]] = {}

    for stage in stages:
        stage_articles = []
        stage_predictions: dict[str, Prediction] = {}

        for aid, pred in predictions.items():
            article = article_by_id.get(aid)
            if not article or stage not in pred.stage_snapshots:
                continue

            snap = pred.stage_snapshots[stage]
            if stage == "after_s1":
                active = snap
            else:
                active = [
                    s for s in snap
                    if s.get("verdict") in ("CONFIRMED", "POSSIBLE")
                ]

            rebuilt_spans = []
            for s in active:
                tech = normalise_technique(s.get("technique", ""))
                if tech is None:
                    continue
                rebuilt_spans.append(PredictedSpan(
                    technique=tech,
                    span_text=s.get("span_text", ""),
                    start=s.get("start", -1),
                    end=s.get("end", -1),
                    verdict=s.get("verdict", "CONFIRMED"),
                    agreement_count=s.get("agreement_count", 1),
                    confidence=s.get("confidence", -1.0),
                ))

            stage_articles.append(article)
            stage_predictions[aid] = Prediction(
                article_id=aid,
                spans=rebuilt_spans,
                eval_mode=pred.eval_mode,
            )

        if not stage_articles:
            continue

        metrics = evaluate(stage_articles, stage_predictions)
        results[stage] = {
            "si": {
                "precision": metrics.si_precision,
                "recall": metrics.si_recall,
                "f1": metrics.si_f1,
            },
            "tc": {
                "precision": metrics.tc_precision,
                "recall": metrics.tc_recall,
                "f1": metrics.tc_f1,
            },
            "macro_f1": metrics.macro_f1,
            "per_technique": metrics.per_technique,
        }

    return results


# ── 6. Agreement vs correctness ───────────────────────────────────────────

def agreement_vs_correctness(
    articles: list[Article],
    predictions: dict[str, Prediction],
) -> dict[int, dict[str, float]]:
    """Per-agreement-count (1, 2, 3): precision for SI and TC.

    Answers: are high-agreement detections more likely to be correct?
    """
    article_by_id = {a.id: a for a in articles}
    buckets: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "si_correct": 0, "tc_correct": 0}
    )
    for aid, pred in predictions.items():
        article = article_by_id.get(aid)
        if not article:
            continue
        for span in pred.spans:
            if span.verdict == "REJECTED":
                continue
            ac = span.agreement_count
            buckets[ac]["total"] += 1
            si_m, tc_m = _span_matches_gold(
                span.technique.value, span.start, span.end, span.span_text,
                article.gold_spans, article.text,
            )
            if si_m:
                buckets[ac]["si_correct"] += 1
            if tc_m:
                buckets[ac]["tc_correct"] += 1

    return {
        ac: {
            "total": d["total"],
            "si_precision": d["si_correct"] / d["total"] if d["total"] > 0 else 0.0,
            "tc_precision": d["tc_correct"] / d["total"] if d["total"] > 0 else 0.0,
        }
        for ac, d in sorted(buckets.items())
    }


# ── 7. Refiner disobedience ────────────────────────────────────────────────

def refiner_disobedience(
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    """Count how often the refinement prompt was disobeyed across articles.

    Returns total disobey events and articles with at least one disobey.
    Only meaningful for the `hybrid` strategy.
    """
    total = 0
    articles_with_disobey = 0
    for pred in predictions.values():
        d = pred.stage_outputs.get("stage3_refiner_disobeyed", 0)
        if d > 0:
            articles_with_disobey += 1
        total += d
    return {
        "total_disobey_events": total,
        "articles_with_disobey": articles_with_disobey,
        "total_articles": len(predictions),
        "disobey_rate_per_article": (
            articles_with_disobey / len(predictions) if predictions else 0.0
        ),
    }


# ── Top-level: compute everything ────────────────────────────────────────

def compute_all_diagnostics(
    articles: list[Article],
    predictions: dict[str, Prediction],
) -> dict[str, Any]:
    label_log = build_label_change_log(articles, predictions)
    return {
        "label_change_log": label_log,
        "relabel_matrix": build_relabel_matrix(label_log),
        "lucky_vs_systematic": lucky_vs_systematic(label_log),
        "drop_quality": drop_quality(articles, predictions),
        "stage_by_stage_f1": stage_by_stage_f1(articles, predictions),
        "agreement_vs_correctness": agreement_vs_correctness(articles, predictions),
        "refiner_disobedience": refiner_disobedience(predictions),
    }