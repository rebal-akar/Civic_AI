"""
Evaluation metrics for multi-label propaganda detection.

Provides:
- Macro/micro/weighted F1, precision, recall
- Per-tactic breakdown
- Hamming loss
- Bootstrap confidence intervals
- McNemar's test for pairwise strategy comparison
- Confusion analysis (FP/FN buckets)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from src.schemas import (
    ALL_TACTICS,
    IDX_TO_TACTIC,
    NUM_TACTICS,
    TACTIC_TO_IDX,
    Prediction,
    Sample,
    Tactic,
)

logger = logging.getLogger(__name__)


# ── Core Metrics ─────────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    """Complete metrics report for an experiment."""
    # Aggregate
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    hamming: float = 0.0
    exact_match_ratio: float = 0.0

    # Per-tactic
    tactic_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    # Sample counts
    n_samples: int = 0
    n_positive_samples: int = 0  # samples with ≥1 label
    n_parse_failures: int = 0

    # Bootstrap CIs (populated separately)
    macro_f1_ci: tuple[float, float] | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "hamming_loss": round(self.hamming, 4),
            "exact_match_ratio": round(self.exact_match_ratio, 4),
            "n_samples": self.n_samples,
            "n_parse_failures": self.n_parse_failures,
            "macro_f1_ci_95": self.macro_f1_ci,
        }


def compute_metrics(
    samples: list[Sample],
    predictions: list[Prediction],
) -> MetricsReport:
    """
    Compute full metrics suite for multi-label classification.

    All metrics (F1, precision, recall) are computed only on the final predicted
    labels (e.g. for ASV, the verdict-stage output only; prosecution/defense
    are not used for scoring).

    Args:
        samples: Ground truth samples
        predictions: Model predictions — must align by index; each prediction's
            .tactics are the final labels used for scoring
    """
    assert len(samples) == len(predictions), (
        f"Mismatched lengths: {len(samples)} samples vs {len(predictions)} predictions"
    )

    y_true = np.stack([s.label_vector() for s in samples])
    y_pred = np.stack([p.label_vector() for p in predictions])
    n = len(samples)

    report = MetricsReport(n_samples=n)
    report.n_positive_samples = int((y_true.sum(axis=1) > 0).sum())
    report.n_parse_failures = sum(1 for p in predictions if not p.parse_success)

    # Aggregate metrics (zero_division=0 handles columns with no support)
    report.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report.micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    report.weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report.macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    report.macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    report.hamming = float(hamming_loss(y_true, y_pred))
    report.exact_match_ratio = float((y_true == y_pred).all(axis=1).mean())

    # Per-tactic metrics
    for idx, tactic in IDX_TO_TACTIC.items():
        col_true = y_true[:, idx]
        col_pred = y_pred[:, idx]
        support = int(col_true.sum())

        if support == 0 and col_pred.sum() == 0:
            continue

        tp = int((col_true & col_pred).sum())
        fp = int((~col_true.astype(bool) & col_pred.astype(bool)).sum())
        fn = int((col_true.astype(bool) & ~col_pred.astype(bool)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        report.tactic_scores[tactic] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    return report


# ── Bootstrap Confidence Intervals ───────────────────────────────────────────

def bootstrap_confidence_interval(
    samples: list[Sample],
    predictions: list[Prediction],
    metric_fn=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap CI for macro-F1 (or custom metric).
    
    Returns:
        (lower, upper) bounds at the specified confidence level
    """
    if metric_fn is None:
        def metric_fn(y_t, y_p):
            return f1_score(y_t, y_p, average="macro", zero_division=0)

    y_true = np.stack([s.label_vector() for s in samples])
    y_pred = np.stack([p.label_vector() for p in predictions])
    n = len(samples)
    rng = np.random.default_rng(seed)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return (round(lower, 4), round(upper, 4))


# ── Statistical Significance ─────────────────────────────────────────────────

def mcnemar_test(
    samples: list[Sample],
    preds_a: list[Prediction],
    preds_b: list[Prediction],
) -> dict[str, Any]:
    """
    McNemar's test comparing two strategies.
    
    Tests whether the strategies disagree in a systematic way:
    - b: cases where A is correct but B is wrong
    - c: cases where B is correct but A is wrong
    
    Returns dict with chi2 statistic, p-value, and interpretation.
    """
    b = 0  # A correct, B wrong
    c = 0  # B correct, A wrong

    for sample, pred_a, pred_b in zip(samples, preds_a, preds_b):
        gt = set(sample.labels)
        a_set = set(pred_a.tactics)
        b_set = set(pred_b.tactics)

        a_correct = (a_set == gt)
        b_correct = (b_set == gt)

        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1

    # McNemar's test with continuity correction
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "b": b,
        "c": c,
        "significant": p_value < 0.05,
        "interpretation": (
            f"Strategy comparison: b={b}, c={c}. "
            f"{'Significant' if p_value < 0.05 else 'Not significant'} "
            f"difference (p={p_value:.4f})."
        ),
    }


def bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


# ── Error Analysis ───────────────────────────────────────────────────────────

@dataclass
class ErrorBucket:
    """A bucket of errors of a specific type."""
    label: str
    sample_ids: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.sample_ids)


@dataclass
class ErrorAnalysis:
    """Complete error analysis for an experiment."""
    false_positives: dict[str, ErrorBucket] = field(default_factory=dict)  # by tactic
    false_negatives: dict[str, ErrorBucket] = field(default_factory=dict)  # by tactic
    tactic_confusions: dict[str, dict[str, int]] = field(default_factory=dict)
    total_fp: int = 0
    total_fn: int = 0
    total_correct: int = 0


def analyse_errors(
    samples: list[Sample],
    predictions: list[Prediction],
) -> ErrorAnalysis:
    """
    Detailed error analysis: FP/FN per tactic, confusion patterns.
    """
    analysis = ErrorAnalysis()

    for sample, pred in zip(samples, predictions):
        gt_set = set(sample.labels)
        pred_set = set(pred.tactics)

        fps = pred_set - gt_set
        fns = gt_set - pred_set
        correct = gt_set & pred_set

        analysis.total_correct += len(correct)
        analysis.total_fp += len(fps)
        analysis.total_fn += len(fns)

        for fp_tactic in fps:
            if fp_tactic not in analysis.false_positives:
                analysis.false_positives[fp_tactic] = ErrorBucket(label=fp_tactic)
            analysis.false_positives[fp_tactic].sample_ids.append(sample.id)
            analysis.false_positives[fp_tactic].details.append({
                "sample_id": sample.id,
                "text_preview": sample.text[:120],
                "ground_truth": sorted(gt_set),
                "predicted": sorted(pred_set),
            })

        for fn_tactic in fns:
            if fn_tactic not in analysis.false_negatives:
                analysis.false_negatives[fn_tactic] = ErrorBucket(label=fn_tactic)
            analysis.false_negatives[fn_tactic].sample_ids.append(sample.id)
            analysis.false_negatives[fn_tactic].details.append({
                "sample_id": sample.id,
                "text_preview": sample.text[:120],
                "ground_truth": sorted(gt_set),
                "predicted": sorted(pred_set),
            })

        # Track tactic confusions: for each FN, what was predicted instead?
        for fn_t in fns:
            if fn_t not in analysis.tactic_confusions:
                analysis.tactic_confusions[fn_t] = defaultdict(int)
            for fp_t in fps:
                analysis.tactic_confusions[fn_t][fp_t] += 1

    return analysis
