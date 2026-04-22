"""Post-hoc analysis functions that operate on saved result JSONs.

These functions read from outputs/results/*.json and produce derived
analyses without re-running any API calls. All functions are pure and
deterministic given their inputs.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from math import erf, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches  # noqa: F401 — used by callers

if TYPE_CHECKING:
    from src.schemas import Article, Prediction
import matplotlib.pyplot as plt
import numpy as np


def _load_results(paths: list[Path]) -> list[dict]:
    """Load result JSONs from disk and return as list of dicts."""
    loaded = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Result file not found: {p}")
        with open(p) as f:
            loaded.append(json.loads(f.read()))
    return loaded


def _verify_runs_comparable(results: list[dict]) -> None:
    """Sanity-check that a group of runs differ only in seed.

    Raises ValueError if the runs have mismatched strategy, model,
    eval_mode, or any other config field that should be identical.
    """
    if len(results) < 2:
        return

    ref = results[0]["config"]
    comparable_fields = [
        "strategy", "model", "verify_model", "eval_mode",
        "asv_num_passes", "asv_stages", "max_articles",
        "articles_dir", "labels_path",
    ]

    for i, r in enumerate(results[1:], start=1):
        other = r["config"]
        for fld in comparable_fields:
            if ref.get(fld) != other.get(fld):
                raise ValueError(
                    f"Runs differ in {fld}: "
                    f"run[0]={ref.get(fld)!r} vs run[{i}]={other.get(fld)!r}"
                )

    seeds = [r["config"].get("seed") for r in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"Duplicate seeds in input runs: {seeds}. "
            f"Multi-run aggregation requires distinct seeds."
        )


def _stats(values: list[float]) -> dict[str, Any]:
    """Compute descriptive stats for a list of numbers."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def aggregate_multi_run(
    result_paths: list[Path],
    verify: bool = True,
) -> dict[str, Any]:
    """Aggregate multiple runs of the same configuration with different seeds.

    Takes a list of result JSON paths (1+ runs, same config, different seeds)
    and returns a dict with mean, std, min, max for every numeric metric.

    Args:
        result_paths: List of paths to result JSON files.
        verify: If True, verify that all runs have the same config (except seed).
                Set False if you deliberately want to aggregate different configs.

    Returns:
        Dict with structure:
        {
            "n_runs": int,
            "seeds": [int, ...],
            "strategy": str,
            "model": str,
            "overall": {
                "si_f1": {"mean": float, "std": float, "min": float, "max": float, "values": [...]},
                ...
            },
            "per_technique": {
                "Loaded_Language": {"f1": {...}, "precision": {...}, "recall": {...}},
                ...
            },
            "cost": {
                "total_cost_usd": {...}, "total_input_tokens": {...}, ...
            },
            "by_stage_cost": {
                "stage1_detection": {"cost": {...}, "input": {...}, "output": {...}},
                ...
            },
        }
    """
    results = _load_results(result_paths)

    if verify:
        _verify_runs_comparable(results)

    if len(results) == 0:
        raise ValueError("No result files provided")

    # ── Overall metrics ─────────────────────────────────────────────────
    metric_keys = [
        "si_precision", "si_recall", "si_f1",
        "tc_precision", "tc_recall", "tc_f1",
        "f1_delta", "macro_f1",
    ]
    overall = {}
    for key in metric_keys:
        values = [r["metrics"].get(key, 0.0) for r in results]
        overall[key] = _stats(values)

    # ── Per-technique metrics ───────────────────────────────────────────
    per_technique: dict[str, dict] = {}
    all_techniques: set[str] = set()
    for r in results:
        all_techniques.update(r["metrics"].get("per_technique", {}).keys())

    for tech in sorted(all_techniques):
        per_technique[tech] = {}
        for metric in ["precision", "recall", "f1"]:
            values = [
                r["metrics"].get("per_technique", {}).get(tech, {}).get(metric, 0.0)
                for r in results
            ]
            per_technique[tech][metric] = _stats(values)

    # ── Cost aggregation ────────────────────────────────────────────────
    cost: dict[str, Any] = {}
    cost_keys = ["total_cost_usd", "total_input_tokens", "total_output_tokens"]
    for key in cost_keys:
        values = [r["cost"].get(key, 0.0) for r in results]
        cost[key] = _stats(values)

    # ── Per-stage cost ──────────────────────────────────────────────────
    stage_costs: dict[str, dict] = {}
    all_stages: set[str] = set()
    for r in results:
        all_stages.update(r["cost"].get("by_stage", {}).keys())

    for stage in sorted(all_stages):
        stage_costs[stage] = {}
        for metric in ["cost", "input", "output"]:
            values = [
                r["cost"].get("by_stage", {}).get(stage, {}).get(metric, 0.0)
                for r in results
            ]
            stage_costs[stage][metric] = _stats(values)

    return {
        "n_runs": len(results),
        "seeds": [r["config"]["seed"] for r in results],
        "strategy": results[0]["config"]["strategy"],
        "model": results[0]["config"]["model"],
        "eval_mode": results[0]["config"]["eval_mode"],
        "n_articles": results[0].get("articles_processed", 0),
        "overall": overall,
        "per_technique": per_technique,
        "cost": cost,
        "by_stage_cost": stage_costs,
    }


def format_multi_run_table(aggregated: dict[str, Any]) -> str:
    """Format a multi-run aggregation result as a human-readable table."""
    lines = []
    lines.append(f"Strategy: {aggregated['strategy']} ({aggregated['model']})")
    lines.append(f"N runs: {aggregated['n_runs']}, seeds: {aggregated['seeds']}")
    lines.append(f"Eval mode: {aggregated['eval_mode']}, articles: {aggregated['n_articles']}")
    lines.append("")
    lines.append(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 62)

    for key, stats in aggregated["overall"].items():
        lines.append(
            f"{key:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['min']:>10.4f} {stats['max']:>10.4f}"
        )

    lines.append("")
    lines.append("Cost:")
    for key, stats in aggregated["cost"].items():
        if "cost" in key:
            lines.append(
                f"  {key:<25} ${stats['mean']:.4f} ± ${stats['std']:.4f}"
            )
        else:
            lines.append(
                f"  {key:<25} {stats['mean']:>10.0f} ± {stats['std']:.0f}"
            )

    if aggregated.get("by_stage_cost"):
        lines.append("")
        lines.append("Per-stage cost:")
        for stage, metrics in aggregated["by_stage_cost"].items():
            cost_s = metrics.get("cost", {})
            lines.append(
                f"  {stage:<30} ${cost_s.get('mean', 0):.4f} ± ${cost_s.get('std', 0):.4f}"
            )

    return "\n".join(lines)


# ── Per-article character-overlap counts ────────────────────────────────────


def _reconstruct_per_article_counts(result: dict) -> list[dict]:
    """Rebuild per-article TP/FP/FN counts from saved predictions + gold.

    Returns a list of dicts, one per article, each containing:
        {
            "article_id": str,
            "si_tp": float, "si_fp": float, "si_fn": float,
            "tc_tp": float, "tc_fp": float, "tc_fn": float,
        }

    TP/FP/FN values are real-valued because character-overlap F1 uses
    proportional credit rather than binary match/no-match.
    """
    per_article = []
    eval_mode = result["config"].get("eval_mode", "permissive")

    for aid, pred in result["predictions"].items():
        article_text = pred.get("article_text", "")
        gold_spans = pred.get("gold_spans", [])
        all_spans = pred.get("all_spans", [])

        if eval_mode == "strict":
            pred_spans = [s for s in all_spans if s.get("verdict") == "CONFIRMED"]
        else:
            pred_spans = [
                s for s in all_spans
                if s.get("verdict") in ("CONFIRMED", "POSSIBLE") or not s.get("verdict")
            ]

        pred_spans = [
            s for s in pred_spans
            if s.get("start", -1) >= 0 and s.get("end", -1) > s.get("start", -1)
        ]

        counts = _compute_overlap_counts(pred_spans, gold_spans, len(article_text))
        counts["article_id"] = aid
        per_article.append(counts)

    return per_article


def _compute_overlap_counts(
    pred_spans: list[dict],
    gold_spans: list[dict],
    text_length: int,
) -> dict[str, float]:
    """Compute character-level TP/FP/FN for one article.

    Uses the SemEval-2020 Task 11 character-overlap metric:
    - SI (span identification): overlap regardless of technique label
    - TC (technique classification): overlap requires matching technique
    """
    if text_length == 0:
        return {"si_tp": 0.0, "si_fp": 0.0, "si_fn": 0.0,
                "tc_tp": 0.0, "tc_fp": 0.0, "tc_fn": 0.0}

    pred_chars_si = [False] * text_length
    gold_chars_si = [False] * text_length
    pred_chars_tc: dict[str, list[bool]] = {}
    gold_chars_tc: dict[str, list[bool]] = {}

    for span in pred_spans:
        start = max(0, span["start"])
        end = min(text_length, span["end"])
        if end <= start:
            continue
        for i in range(start, end):
            pred_chars_si[i] = True
        tech = span["technique"]
        if tech not in pred_chars_tc:
            pred_chars_tc[tech] = [False] * text_length
        for i in range(start, end):
            pred_chars_tc[tech][i] = True

    for span in gold_spans:
        start = max(0, span["start"])
        end = min(text_length, span["end"])
        if end <= start:
            continue
        for i in range(start, end):
            gold_chars_si[i] = True
        tech = span["technique"]
        if tech not in gold_chars_tc:
            gold_chars_tc[tech] = [False] * text_length
        for i in range(start, end):
            gold_chars_tc[tech][i] = True

    si_tp = sum(1 for i in range(text_length) if pred_chars_si[i] and gold_chars_si[i])
    si_fp = sum(1 for i in range(text_length) if pred_chars_si[i] and not gold_chars_si[i])
    si_fn = sum(1 for i in range(text_length) if not pred_chars_si[i] and gold_chars_si[i])

    tc_tp = 0
    tc_fp = 0
    tc_fn = 0
    all_techniques = set(pred_chars_tc.keys()) | set(gold_chars_tc.keys())
    for tech in all_techniques:
        pred_tech = pred_chars_tc.get(tech, [False] * text_length)
        gold_tech = gold_chars_tc.get(tech, [False] * text_length)
        for i in range(text_length):
            if pred_tech[i] and gold_tech[i]:
                tc_tp += 1
            elif pred_tech[i] and not gold_tech[i]:
                tc_fp += 1
            elif not pred_tech[i] and gold_tech[i]:
                tc_fn += 1

    return {
        "si_tp": float(si_tp),
        "si_fp": float(si_fp),
        "si_fn": float(si_fn),
        "tc_tp": float(tc_tp),
        "tc_fp": float(tc_fp),
        "tc_fn": float(tc_fn),
    }


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """Compute F1 from TP/FP/FN counts."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Reconstruct schema objects from result JSON ─────────────────────────────


def _reconstruct_from_result(
    result: dict,
) -> list[tuple["Article", "Prediction"]]:
    """Rebuild (Article, Prediction) pairs from a saved result JSON.

    Each tuple contains an Article with gold_spans and text, plus a Prediction
    with the all_spans reconstructed as PredictedSpan objects. This lets us
    call evaluate() directly on the reconstructed objects.
    """
    from src.schemas import (
        Article, GoldSpan, PredictedSpan, Prediction,
        Technique, normalise_technique,
    )

    eval_mode = result["config"].get("eval_mode", "permissive")
    pairs = []

    for aid, pred_dict in result["predictions"].items():
        article_text = pred_dict.get("article_text", "")

        gold_spans = []
        for gs in pred_dict.get("gold_spans", []):
            tech_str = gs["technique"]
            tech = normalise_technique(tech_str)
            if tech is None:
                continue
            gold_spans.append(GoldSpan(technique=tech, start=gs["start"], end=gs["end"]))

        article = Article(id=aid, text=article_text, gold_spans=gold_spans)

        pred_spans = []
        for s in pred_dict.get("all_spans", []):
            tech = normalise_technique(s.get("technique", ""))
            if tech is None:
                continue
            pred_spans.append(PredictedSpan(
                technique=tech,
                span_text=s.get("span_text", ""),
                reasoning=s.get("reasoning", ""),
                start=s.get("start", -1),
                end=s.get("end", -1),
                pass_id=s.get("pass_id", 0),
                agreement_count=s.get("agreement_count", 1),
                confidence=s.get("confidence", -1.0),
                verdict=s.get("verdict", ""),
            ))

        prediction = Prediction(
            article_id=aid,
            spans=pred_spans,
            eval_mode=eval_mode,
        )
        pairs.append((article, prediction))

    return pairs


def _evaluate_subset(
    pairs: list[tuple["Article", "Prediction"]],
    metric: str,
) -> float:
    """Run evaluate() on a list of (Article, Prediction) pairs and return one metric."""
    from src.evaluation.metrics import evaluate

    articles = [a for a, _ in pairs]
    predictions = {p.article_id: p for _, p in pairs}
    result = evaluate(articles, predictions)
    return getattr(result, metric, 0.0)


# ── Bootstrap confidence intervals ──────────────────────────────────────────


def compute_bootstrap_cis(
    result_path: Path,
    metric: str = "si_f1",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap CI using the official SemEval evaluate() function.

    This matches the formula used for headline metrics, so the point estimate
    from this function will match result['metrics'][metric] exactly.

    Bootstrap resamples articles with replacement, giving each duplicate a
    unique ID so evaluate()'s dict-keyed interface handles them correctly.
    """
    if metric not in ("si_f1", "tc_f1", "macro_f1"):
        raise ValueError(f"metric must be 'si_f1', 'tc_f1', or 'macro_f1', got {metric!r}")

    from src.schemas import Article, Prediction

    result = _load_results([result_path])[0]
    pairs = _reconstruct_from_result(result)

    if not pairs:
        raise ValueError(f"No articles found in {result_path}")

    n = len(pairs)
    point_estimate = _evaluate_subset(pairs, metric)

    rng = random.Random(seed)
    bootstrap_f1s = []

    for _ in range(n_bootstrap):
        sampled_indices = [rng.randint(0, n - 1) for _ in range(n)]
        sampled_pairs = []
        for rep_idx, orig_idx in enumerate(sampled_indices):
            article, prediction = pairs[orig_idx]
            uid = f"{article.id}__b{rep_idx}"
            sampled_pairs.append((
                Article(id=uid, text=article.text, gold_spans=article.gold_spans),
                Prediction(
                    article_id=uid,
                    spans=prediction.spans,
                    eval_mode=prediction.eval_mode,
                ),
            ))
        bootstrap_f1s.append(_evaluate_subset(sampled_pairs, metric))

    bootstrap_f1s.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return {
        "metric": metric,
        "point_estimate": point_estimate,
        "ci_lower": bootstrap_f1s[lower_idx],
        "ci_upper": bootstrap_f1s[upper_idx],
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_articles": n,
        "strategy": result["config"]["strategy"],
        "model": result["config"]["model"],
    }


# ── Pairwise bootstrap significance testing ─────────────────────────────────


def pairwise_bootstrap(
    result_path_a: Path,
    result_path_b: Path,
    metric: str = "tc_f1",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test: does strategy A beat strategy B on the same articles?

    Uses the official SemEval evaluate() function so F1 values match headlines.

    For each bootstrap iteration, samples the same article indices for both
    strategies, computes F1 for each, and records delta = F1_A - F1_B.
    The p-value is the fraction of iterations where A failed to beat B.
    """
    if metric not in ("si_f1", "tc_f1", "macro_f1"):
        raise ValueError(f"metric must be 'si_f1', 'tc_f1', or 'macro_f1', got {metric!r}")

    from src.schemas import Article, Prediction

    result_a = _load_results([result_path_a])[0]
    result_b = _load_results([result_path_b])[0]

    pairs_a = _reconstruct_from_result(result_a)
    pairs_b = _reconstruct_from_result(result_b)

    a_by_id = {a.id: (a, p) for a, p in pairs_a}
    b_by_id = {a.id: (a, p) for a, p in pairs_b}
    shared_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))

    if not shared_ids:
        raise ValueError("Runs A and B share no articles — cannot compare.")
    if len(shared_ids) < len(a_by_id) or len(shared_ids) < len(b_by_id):
        import logging
        logging.getLogger(__name__).warning(
            f"Pairwise bootstrap: using {len(shared_ids)} shared articles "
            f"(A has {len(a_by_id)}, B has {len(b_by_id)})"
        )

    aligned_a = [a_by_id[aid] for aid in shared_ids]
    aligned_b = [b_by_id[aid] for aid in shared_ids]
    n = len(shared_ids)

    f1_a_full = _evaluate_subset(aligned_a, metric)
    f1_b_full = _evaluate_subset(aligned_b, metric)
    delta_full = f1_a_full - f1_b_full

    rng = random.Random(seed)
    deltas = []

    for _ in range(n_bootstrap):
        sampled_idx = [rng.randint(0, n - 1) for _ in range(n)]

        boot_a = []
        boot_b = []
        for rep_idx, orig_idx in enumerate(sampled_idx):
            uid = f"{shared_ids[orig_idx]}__b{rep_idx}"
            art_a, pred_a = aligned_a[orig_idx]
            art_b, pred_b = aligned_b[orig_idx]
            boot_a.append((
                Article(id=uid, text=art_a.text, gold_spans=art_a.gold_spans),
                Prediction(article_id=uid, spans=pred_a.spans, eval_mode=pred_a.eval_mode),
            ))
            boot_b.append((
                Article(id=uid, text=art_b.text, gold_spans=art_b.gold_spans),
                Prediction(article_id=uid, spans=pred_b.spans, eval_mode=pred_b.eval_mode),
            ))

        f1_a = _evaluate_subset(boot_a, metric)
        f1_b = _evaluate_subset(boot_b, metric)
        deltas.append(f1_a - f1_b)

    n_not_beating = sum(1 for d in deltas if d <= 0)
    p_value = n_not_beating / n_bootstrap

    return {
        "strategy_a": result_a["config"]["strategy"],
        "strategy_b": result_b["config"]["strategy"],
        "metric": metric,
        "f1_a": f1_a_full,
        "f1_b": f1_b_full,
        "delta": delta_full,
        "p_value_one_sided": p_value,
        "n_bootstrap": n_bootstrap,
        "n_articles": n,
    }


def pairwise_matrix_with_holm(
    result_paths: list[Path],
    metric: str = "tc_f1",
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Build full pairwise significance matrix with Holm-Bonferroni correction.

    For each pair (A, B) where A is numerically higher than B, compute the
    paired bootstrap p-value for "A > B", then apply Holm-Bonferroni
    correction across all such directional comparisons.
    """
    if metric not in ("si_f1", "tc_f1", "macro_f1"):
        raise ValueError(f"Unsupported metric: {metric!r}")

    n = len(result_paths)
    if n < 2:
        raise ValueError("Need at least 2 result files for pairwise comparison")

    results = _load_results(result_paths)
    strategies = [r["config"]["strategy"] for r in results]

    raw_p_values: dict[tuple[str, str], float] = {}
    delta_values: dict[tuple[str, str], float] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                pair = pairwise_bootstrap(
                    result_paths[i], result_paths[j],
                    metric=metric, n_bootstrap=n_bootstrap, seed=seed,
                )
                key = (strategies[i], strategies[j])
                raw_p_values[key] = pair["p_value_one_sided"]
                delta_values[key] = pair["delta"]
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Pairwise {strategies[i]} vs {strategies[j]} failed: {e}"
                )
                continue

    directional_pairs = [
        (key, p) for key, p in raw_p_values.items()
        if delta_values[key] > 0
    ]
    directional_pairs.sort(key=lambda x: x[1])

    m = len(directional_pairs)
    corrected: dict[tuple[str, str], float] = {}
    significant: dict[tuple[str, str], bool] = {}
    for rank, (key, p) in enumerate(directional_pairs):
        holm_threshold = alpha / (m - rank)
        corrected[key] = min(1.0, p * (m - rank))
        significant[key] = p < holm_threshold

    for key in raw_p_values:
        if key not in corrected:
            corrected[key] = 1.0
            significant[key] = False

    return {
        "strategies": strategies,
        "metric": metric,
        "n_comparisons_corrected": m,
        "raw_p_values": {f"{a}_vs_{b}": p for (a, b), p in raw_p_values.items()},
        "deltas": {f"{a}_vs_{b}": d for (a, b), d in delta_values.items()},
        "holm_corrected_p": {f"{a}_vs_{b}": p for (a, b), p in corrected.items()},
        "significant_after_holm": {f"{a}_vs_{b}": s for (a, b), s in significant.items()},
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


# ── Per-pass F1 analysis ────────────────────────────────────────────────────


def compute_per_pass_f1(
    result_path: Path,
    metric: str = "si_f1",
) -> dict[str, Any]:
    """Compute F1 for each individual detection pass + the merged union.

    Uses the official SemEval evaluate() function so numbers match headlines.
    Only works for strategies that saved stage1_per_pass_spans (asv, consol,
    hybrid). Raises ValueError for baselines.

    Returns:
        {
            "strategy": str,
            "metric": "si_f1",
            "per_pass_f1": [0.31, 0.35, 0.29],
            "merged_f1": 0.42,
            "n_passes": 3,
            "n_articles": 5,
        }
    """
    if metric not in ("si_f1", "tc_f1"):
        raise ValueError(f"metric must be 'si_f1' or 'tc_f1', got {metric!r}")

    from src.schemas import (
        Article, GoldSpan, PredictedSpan, Prediction, normalise_technique,
    )

    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    eval_mode = result["config"].get("eval_mode", "permissive")

    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(
            f"Strategy {strategy!r} does not have multi-pass detection. "
            f"per-pass F1 only applies to asv, consol, hybrid."
        )

    first_pred = next(iter(result["predictions"].values()))
    per_pass_spans = first_pred.get("stage_outputs", {}).get("stage1_per_pass_spans", [])
    if not per_pass_spans:
        raise ValueError(
            f"Result {result_path} has no stage1_per_pass_spans. "
            f"The schema patch may not have been applied when this run was produced."
        )
    n_passes = len(per_pass_spans)

    def _build_article(aid: str, pred_dict: dict) -> Article:
        gold_spans = []
        for gs in pred_dict.get("gold_spans", []):
            tech = normalise_technique(gs["technique"])
            if tech is not None:
                gold_spans.append(GoldSpan(technique=tech, start=gs["start"], end=gs["end"]))
        return Article(id=aid, text=pred_dict.get("article_text", ""), gold_spans=gold_spans)

    def _build_pred_spans(spans_dicts: list[dict]) -> list[PredictedSpan]:
        out = []
        for s in spans_dicts:
            if s.get("start", -1) < 0 or s.get("end", -1) <= s.get("start", -1):
                continue
            tech = normalise_technique(s.get("technique", ""))
            if tech is None:
                continue
            out.append(PredictedSpan(
                technique=tech,
                span_text=s.get("span_text", ""),
                start=s["start"],
                end=s["end"],
                verdict=s.get("verdict", "CONFIRMED"),
            ))
        return out

    per_pass_pairs: list[list[tuple[Article, Prediction]]] = [[] for _ in range(n_passes)]
    merged_pairs: list[tuple[Article, Prediction]] = []

    for aid, pred_dict in result["predictions"].items():
        article = _build_article(aid, pred_dict)
        pass_data = pred_dict.get("stage_outputs", {}).get("stage1_per_pass_spans", [])

        for k in range(n_passes):
            if k < len(pass_data):
                spans = _build_pred_spans(pass_data[k])
                per_pass_pairs[k].append((
                    article,
                    Prediction(article_id=aid, spans=spans, eval_mode=eval_mode),
                ))

        merged_raw = pred_dict.get("stage_snapshots", {}).get("after_s1", [])
        merged_spans = _build_pred_spans(merged_raw)
        merged_pairs.append((
            article,
            Prediction(article_id=aid, spans=merged_spans, eval_mode=eval_mode),
        ))

    per_pass_f1 = [_evaluate_subset(per_pass_pairs[k], metric) for k in range(n_passes)]
    merged_f1 = _evaluate_subset(merged_pairs, metric)

    return {
        "strategy": strategy,
        "metric": metric,
        "per_pass_f1": per_pass_f1,
        "merged_f1": merged_f1,
        "n_passes": n_passes,
        "n_articles": len(result["predictions"]),
    }


# ── Stage-by-stage F1 (corrected, using official metric) ─────────────────────


def stage_f1_corrected(result_path: Path) -> dict[str, Any]:
    """Recompute stage-by-stage F1 using the official SemEval metric.

    Rebuilds Prediction objects from each stage snapshot and calls evaluate(),
    so the numbers are directly comparable to headline metrics.

    Returns:
        {
            "strategy": str,
            "after_s1": {"si_f1": ..., "tc_f1": ..., "si_precision": ..., ...},
            "after_s2": {...},
            "after_s3": {...},
        }
    """
    from src.evaluation.metrics import evaluate
    from src.schemas import (
        Article, GoldSpan, PredictedSpan, Prediction, normalise_technique,
    )

    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(f"Strategy {strategy} has no stage snapshots")

    eval_mode = result["config"].get("eval_mode", "permissive")
    out: dict[str, Any] = {"strategy": strategy}

    articles_by_id: dict[str, Article] = {}
    for aid, pred_dict in result["predictions"].items():
        article_text = pred_dict.get("article_text", "")
        gold_spans = []
        for gs in pred_dict.get("gold_spans", []):
            tech = normalise_technique(gs["technique"])
            if tech is not None:
                gold_spans.append(GoldSpan(technique=tech, start=gs["start"], end=gs["end"]))
        articles_by_id[aid] = Article(id=aid, text=article_text, gold_spans=gold_spans)

    stages = ["after_s1", "after_s2", "after_s3"]
    for stage in stages:
        stage_articles: list[Article] = []
        stage_predictions: dict[str, Prediction] = {}

        for aid, pred_dict in result["predictions"].items():
            snapshot = pred_dict.get("stage_snapshots", {}).get(stage, [])
            if not snapshot:
                continue

            spans = []
            for s in snapshot:
                tech = normalise_technique(s.get("technique", ""))
                if tech is None:
                    continue
                spans.append(PredictedSpan(
                    technique=tech,
                    span_text=s.get("span_text", ""),
                    start=s.get("start", -1),
                    end=s.get("end", -1),
                    verdict=s.get("verdict", "CONFIRMED"),
                    agreement_count=s.get("agreement_count", 1),
                    confidence=s.get("confidence", -1.0),
                ))

            stage_articles.append(articles_by_id[aid])
            stage_predictions[aid] = Prediction(
                article_id=aid,
                spans=spans,
                eval_mode=eval_mode,
            )

        if not stage_articles:
            continue

        metrics = evaluate(stage_articles, stage_predictions)
        out[stage] = {
            "si_precision": metrics.si_precision,
            "si_recall": metrics.si_recall,
            "si_f1": metrics.si_f1,
            "tc_precision": metrics.tc_precision,
            "tc_recall": metrics.tc_recall,
            "tc_f1": metrics.tc_f1,
            "macro_f1": metrics.macro_f1,
            "per_technique": metrics.per_technique,
        }

    return out


# ── Script 5: Pareto frontier (Figure 1.1) ─────────────────────────────────


def plot_pareto_frontier(
    result_paths: list[Path],
    output_path: Path,
    metric: str = "tc_f1",
    n_bootstrap: int = 1000,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate the cost vs F1 Pareto frontier scatter plot (Figure 1.1).

    For each strategy result, plots a point at (total_cost_usd, F1) with
    error bars from bootstrap CI. Identifies Pareto-optimal strategies
    (no other strategy has both lower cost and higher F1) and connects
    them with a frontier line.
    """
    if metric not in ("si_f1", "tc_f1"):
        raise ValueError(f"metric must be 'si_f1' or 'tc_f1', got {metric!r}")

    results = _load_results(result_paths)

    points = []
    for path, result in zip(result_paths, results):
        ci = compute_bootstrap_cis(path, metric=metric, n_bootstrap=n_bootstrap)
        points.append({
            "strategy": result["config"]["strategy"],
            "cost": result["cost"]["total_cost_usd"],
            "f1": ci["point_estimate"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
        })

    def _is_pareto(p: dict, others: list[dict]) -> bool:
        for o in others:
            if o is p:
                continue
            if o["cost"] <= p["cost"] and o["f1"] >= p["f1"]:
                if o["cost"] < p["cost"] or o["f1"] > p["f1"]:
                    return False
        return True

    for p in points:
        p["pareto_optimal"] = _is_pareto(p, points)

    fig, ax = plt.subplots(figsize=(9, 6))

    for p in points:
        yerr = [[p["f1"] - p["ci_lower"]], [p["ci_upper"] - p["f1"]]]
        color = "#2a9d8f" if p["pareto_optimal"] else "#94a3b8"
        marker = "o" if p["pareto_optimal"] else "s"
        ax.errorbar(
            p["cost"], p["f1"],
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=0.8,
            capsize=4,
            label=p["strategy"],
        )
        ax.annotate(
            p["strategy"],
            (p["cost"], p["f1"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
        )

    pareto_points = sorted(
        [p for p in points if p["pareto_optimal"]],
        key=lambda p: p["cost"],
    )
    if len(pareto_points) >= 2:
        xs = [p["cost"] for p in pareto_points]
        ys = [p["f1"] for p in pareto_points]
        ax.plot(xs, ys, "--", color="#2a9d8f", alpha=0.6, linewidth=1.5,
                label="Pareto frontier", zorder=1)

    ax.set_xlabel("Total cost (USD)", fontsize=11)
    ax.set_ylabel(f"{metric.upper().replace('_', '-')} (95% CI)", fontsize=11)
    ax.set_title(
        title or f"Cost-effectiveness: {metric.upper().replace('_', '-')} vs cost",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if unique:
        ax.legend(*zip(*unique), loc="lower right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "points": points,
        "pareto_optimal_strategies": [p["strategy"] for p in points if p["pareto_optimal"]],
        "output_path": str(output_path),
    }


# ── Script 6: TC confusion matrix (Figure 2.2) ────────────────────────────


def _span_iou(span_a: dict, span_b: dict) -> float:
    """Character-level IoU between two spans with start/end fields."""
    a_start, a_end = span_a.get("start", 0), span_a.get("end", 0)
    b_start, b_end = span_b.get("start", 0), span_b.get("end", 0)
    if a_end <= a_start or b_end <= b_start:
        return 0.0
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def plot_tc_confusion(
    result_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot 14x14 gold-vs-predicted technique confusion heatmap (Figure 2.2).

    For each predicted span, finds the gold span with the highest character
    IoU above the threshold and counts (gold_technique, predicted_technique).
    Unmatched predicted spans count as (NONE, predicted). Unmatched gold
    spans count as (gold, NONE). Normalised by row (gold).
    """
    result = _load_results([result_path])[0]

    from src.schemas import Technique
    techniques = [t.value for t in Technique]
    labels = techniques + ["NONE"]
    n = len(labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}

    counts = np.zeros((n, n), dtype=int)
    eval_mode = result["config"].get("eval_mode", "permissive")

    for aid, pred in result["predictions"].items():
        gold_spans = pred.get("gold_spans", [])
        all_spans = pred.get("all_spans", [])

        if eval_mode == "strict":
            pred_spans = [s for s in all_spans if s.get("verdict") == "CONFIRMED"]
        else:
            pred_spans = [
                s for s in all_spans
                if s.get("verdict") in ("CONFIRMED", "POSSIBLE") or not s.get("verdict")
            ]
        pred_spans = [
            s for s in pred_spans
            if s.get("start", -1) >= 0 and s.get("end", -1) > s.get("start", -1)
        ]

        matched_gold_indices: set[int] = set()
        for ps in pred_spans:
            best_iou = 0.0
            best_gold_idx = None
            for gi, gs in enumerate(gold_spans):
                iou = _span_iou(ps, gs)
                if iou > best_iou:
                    best_iou = iou
                    best_gold_idx = gi

            if best_gold_idx is not None and best_iou >= iou_threshold:
                gold_tech = gold_spans[best_gold_idx]["technique"]
                pred_tech = ps["technique"]
                matched_gold_indices.add(best_gold_idx)
                gi_idx = label_idx.get(gold_tech, label_idx["NONE"])
                pi_idx = label_idx.get(pred_tech, label_idx["NONE"])
                counts[gi_idx][pi_idx] += 1
            else:
                pred_tech = ps["technique"]
                pi_idx = label_idx.get(pred_tech, label_idx["NONE"])
                counts[label_idx["NONE"]][pi_idx] += 1

        for gi, gs in enumerate(gold_spans):
            if gi not in matched_gold_indices:
                gold_tech = gs["technique"]
                gli = label_idx.get(gold_tech, label_idx["NONE"])
                counts[gli][label_idx["NONE"]] += 1

    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(row_sums > 0, counts / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    short_labels = [lab.replace(",", ",\n").replace("_", " ") for lab in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Predicted technique", fontsize=11)
    ax.set_ylabel("Gold technique", fontsize=11)

    strategy = result["config"]["strategy"]
    ax.set_title(
        title or f"TC confusion matrix: {strategy} (row-normalised)", fontsize=12,
    )

    for i in range(n):
        for j in range(n):
            if counts[i][j] > 0:
                text_color = "white" if normalised[i][j] > 0.5 else "black"
                ax.text(
                    j, i, str(counts[i][j]),
                    ha="center", va="center",
                    fontsize=7, color=text_color,
                )

    fig.colorbar(im, ax=ax, label="Row-normalised frequency")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "strategy": strategy,
        "counts": counts.tolist(),
        "normalised": normalised.tolist(),
        "labels": labels,
        "output_path": str(output_path),
    }


# ── Script 7: Error classification (Table 6.1) ─────────────────────────────

ERROR_CATEGORIES = [
    "correct",
    "missed_detection",
    "false_alarm",
    "wrong_technique",
    "boundary_too_broad",
    "boundary_too_narrow",
    "boundary_offset",
]


def classify_errors(
    result_path: Path,
    iou_threshold: float = 0.3,
    boundary_ratio_threshold: float = 1.5,
    offset_threshold: float = 0.5,
) -> dict[str, Any]:
    """Classify every predicted and gold span into error categories (Table 6.1).

    Algorithm:
      1. For each predicted span, find the best-matching gold span by IoU.
      2. If IoU < iou_threshold: FALSE_ALARM.
      3. If techniques differ: WRONG_TECHNIQUE.
      4. If pred length / gold length > boundary_ratio_threshold: BOUNDARY_TOO_BROAD.
      5. If gold length / pred length > boundary_ratio_threshold: BOUNDARY_TOO_NARROW.
      6. If IoU >= offset_threshold but centers differ notably: BOUNDARY_OFFSET.
      7. Otherwise: CORRECT.
      8. Gold spans with no matching prediction: MISSED_DETECTION.
    """
    result = _load_results([result_path])[0]

    counts: dict[str, int] = {cat: 0 for cat in ERROR_CATEGORIES}
    error_records: list[dict] = []
    eval_mode = result["config"].get("eval_mode", "permissive")

    for aid, pred in result["predictions"].items():
        gold_spans = pred.get("gold_spans", [])
        all_spans = pred.get("all_spans", [])

        if eval_mode == "strict":
            pred_spans = [s for s in all_spans if s.get("verdict") == "CONFIRMED"]
        else:
            pred_spans = [
                s for s in all_spans
                if s.get("verdict") in ("CONFIRMED", "POSSIBLE") or not s.get("verdict")
            ]
        pred_spans = [
            s for s in pred_spans
            if s.get("start", -1) >= 0 and s.get("end", -1) > s.get("start", -1)
        ]

        matched_gold_indices: set[int] = set()

        for ps in pred_spans:
            best_iou = 0.0
            best_gi = None
            for gi, gs in enumerate(gold_spans):
                iou = _span_iou(ps, gs)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_gi is None or best_iou < iou_threshold:
                category = "false_alarm"
                gold_match = None
            else:
                gs = gold_spans[best_gi]
                matched_gold_indices.add(best_gi)
                pred_len = ps["end"] - ps["start"]
                gold_len = gs["end"] - gs["start"]

                if ps["technique"] != gs["technique"]:
                    category = "wrong_technique"
                elif pred_len / max(gold_len, 1) >= boundary_ratio_threshold:
                    category = "boundary_too_broad"
                elif gold_len / max(pred_len, 1) >= boundary_ratio_threshold:
                    category = "boundary_too_narrow"
                elif best_iou < offset_threshold:
                    category = "boundary_offset"
                else:
                    category = "correct"
                gold_match = gs

            counts[category] += 1
            error_records.append({
                "article_id": aid,
                "category": category,
                "predicted": {
                    "technique": ps["technique"],
                    "text": ps.get("span_text", ""),
                    "start": ps["start"],
                    "end": ps["end"],
                },
                "gold": {
                    "technique": gold_match["technique"] if gold_match else None,
                    "start": gold_match["start"] if gold_match else None,
                    "end": gold_match["end"] if gold_match else None,
                } if gold_match else None,
                "iou": best_iou,
            })

        for gi, gs in enumerate(gold_spans):
            if gi not in matched_gold_indices:
                counts["missed_detection"] += 1
                error_records.append({
                    "article_id": aid,
                    "category": "missed_detection",
                    "predicted": None,
                    "gold": {
                        "technique": gs["technique"],
                        "start": gs["start"],
                        "end": gs["end"],
                    },
                    "iou": 0.0,
                })

    total = sum(counts.values())
    percentages = {
        cat: (counts[cat] / total * 100) if total > 0 else 0.0
        for cat in ERROR_CATEGORIES
    }

    per_technique_errors: dict[str, dict[str, int]] = defaultdict(
        lambda: {cat: 0 for cat in ERROR_CATEGORIES}
    )
    for rec in error_records:
        tech = None
        if rec["gold"]:
            tech = rec["gold"]["technique"]
        elif rec["predicted"]:
            tech = rec["predicted"]["technique"]
        if tech:
            per_technique_errors[tech][rec["category"]] += 1

    return {
        "strategy": result["config"]["strategy"],
        "counts": counts,
        "percentages": percentages,
        "total": total,
        "per_technique": dict(per_technique_errors),
        "records": error_records,
    }


def classify_errors_multi(
    result_paths: list[Path],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run classify_errors on multiple result files and produce a comparison table.

    Returns a dict with one entry per strategy plus a combined table
    suitable for direct formatting into Table 6.1 of the dissertation.
    """
    per_strategy = {}
    for p in result_paths:
        result = classify_errors(p, iou_threshold=iou_threshold)
        per_strategy[result["strategy"]] = result

    strategies = list(per_strategy.keys())
    table: dict[str, Any] = {
        "categories": ERROR_CATEGORIES,
        "strategies": strategies,
        "counts": {},
        "percentages": {},
    }
    for cat in ERROR_CATEGORIES:
        table["counts"][cat] = [per_strategy[s]["counts"][cat] for s in strategies]
        table["percentages"][cat] = [per_strategy[s]["percentages"][cat] for s in strategies]

    return {
        "per_strategy": per_strategy,
        "comparison_table": table,
    }


# ── Script 8: Point-biserial agreement correlation (Section 7) ──────────────


def _student_t_cdf(t_val: float, df: int) -> float:
    """Approximate Student's t CDF without scipy.

    Uses standard normal approximation which is reasonable for df >= 30.
    For exact values in a dissertation, use scipy.stats.t.cdf instead.
    """
    if df < 1:
        return 0.5
    return 0.5 * (1 + erf(t_val / sqrt(2)))


def point_biserial_agreement(
    result_path: Path, iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Point-biserial correlation between agreement count and per-span correctness.

    Positive r means spans with higher multi-pass agreement are more likely
    to be correct — validates agreement as a meaningful confidence signal.

    Only meaningful for strategies with multi-pass detection (asv, consol, hybrid).
    """
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(f"Strategy {strategy} has no agreement signal")

    eval_mode = result["config"].get("eval_mode", "permissive")

    agreements: list[int] = []
    correctnesses: list[int] = []

    for aid, pred in result["predictions"].items():
        gold_spans = pred.get("gold_spans", [])
        all_spans = pred.get("all_spans", [])

        if eval_mode == "strict":
            pred_spans = [s for s in all_spans if s.get("verdict") == "CONFIRMED"]
        else:
            pred_spans = [
                s for s in all_spans
                if s.get("verdict") in ("CONFIRMED", "POSSIBLE") or not s.get("verdict")
            ]
        pred_spans = [
            s for s in pred_spans
            if s.get("start", -1) >= 0 and s.get("end", -1) > s.get("start", -1)
        ]

        for ps in pred_spans:
            best_iou = 0.0
            matching_tech = False
            for gs in gold_spans:
                iou = _span_iou(ps, gs)
                if iou > best_iou:
                    best_iou = iou
                    matching_tech = (ps["technique"] == gs["technique"])

            is_correct = int(best_iou >= iou_threshold and matching_tech)
            agreements.append(ps.get("agreement_count", 1))
            correctnesses.append(is_correct)

    if len(agreements) < 2:
        return {
            "strategy": strategy,
            "r_pb": None,
            "p_value": None,
            "n_spans": len(agreements),
            "note": "Insufficient spans for correlation",
        }

    agreements_arr = np.array(agreements, dtype=float)
    correct_arr = np.array(correctnesses, dtype=float)

    if correct_arr.std() == 0 or agreements_arr.std() == 0:
        return {
            "strategy": strategy,
            "r_pb": 0.0,
            "p_value": 1.0,
            "n_spans": len(agreements),
            "note": "No variance in correctness or agreement",
        }

    r = float(np.corrcoef(agreements_arr, correct_arr)[0, 1])

    n_spans = len(agreements)
    if abs(r) < 1.0:
        t_stat = r * np.sqrt((n_spans - 2) / (1 - r * r))
        p_approx = 2 * (1 - _student_t_cdf(abs(float(t_stat)), n_spans - 2))
    else:
        p_approx = 0.0

    return {
        "strategy": strategy,
        "r_pb": r,
        "p_value": float(p_approx),
        "n_spans": n_spans,
        "n_correct": int(correct_arr.sum()),
        "n_incorrect": int((1 - correct_arr).sum()),
    }


# ── Script 9: Per-stage F1 deltas (Table 3.1) ──────────────────────────────


def compute_stage_deltas(result_path: Path) -> dict[str, Any]:
    """Compute per-stage F1 deltas for multi-stage strategies (Table 3.1).

    Recomputes stage_by_stage_f1 from raw data using the official SemEval
    evaluate() function so numbers match headlines exactly.
    """
    from src.evaluation.diagnostics import stage_by_stage_f1
    from src.schemas import Article, GoldSpan, Prediction, PredictedSpan, normalise_technique

    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    pairs = _reconstruct_from_result(result)
    articles = [a for a, _ in pairs]
    predictions: dict[str, Prediction] = {}

    for aid, pred_dict in result["predictions"].items():
        article_text = pred_dict.get("article_text", "")
        eval_mode = result["config"].get("eval_mode", "permissive")

        pred_spans = []
        for s in pred_dict.get("all_spans", []):
            tech = normalise_technique(s.get("technique", ""))
            if tech is None:
                continue
            pred_spans.append(PredictedSpan(
                technique=tech,
                span_text=s.get("span_text", ""),
                start=s.get("start", -1),
                end=s.get("end", -1),
                pass_id=s.get("pass_id", 0),
                agreement_count=s.get("agreement_count", 1),
                confidence=s.get("confidence", -1.0),
                verdict=s.get("verdict", ""),
            ))

        snap_data = pred_dict.get("stage_snapshots", {})
        prediction = Prediction(
            article_id=aid,
            spans=pred_spans,
            stage_snapshots=snap_data,
            eval_mode=eval_mode,
        )
        predictions[aid] = prediction

    stage_f1 = stage_by_stage_f1(articles, predictions)

    if not stage_f1:
        raise ValueError(f"No stage snapshots found in {result_path}")

    def _extract(stage_key: str) -> dict[str, float]:
        s = stage_f1.get(stage_key, {})
        return {
            "si_p": s.get("si", {}).get("precision", 0),
            "si_r": s.get("si", {}).get("recall", 0),
            "si_f1": s.get("si", {}).get("f1", 0),
            "tc_p": s.get("tc", {}).get("precision", 0),
            "tc_r": s.get("tc", {}).get("recall", 0),
            "tc_f1": s.get("tc", {}).get("f1", 0),
        }

    s1 = _extract("after_s1")
    s2 = _extract("after_s2")
    s3 = _extract("after_s3")

    def _delta(a: dict, b: dict) -> dict:
        return {k: round(b[k] - a[k], 4) for k in a}

    return {
        "strategy": strategy,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "delta_s1_to_s2": _delta(s1, s2) if s2 else {},
        "delta_s2_to_s3": _delta(s2, s3) if s3 else {},
        "delta_s1_to_s3": _delta(s1, s3) if s3 else {},
    }


# ── Script 10: Stage lifecycle report ───────────────────────────────────────


def stage_lifecycle_report(result_path: Path) -> dict[str, Any]:
    """Build the unified per-stage lifecycle table combining F1, cost, and per-pass data.

    Rows are pipeline stages (including individual detection passes),
    columns are API calls / tokens / cost / SI-F1 / TC-F1 / delta-F1.

    Only meaningful for multi-stage strategies (asv, consol, hybrid).
    """
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(
            f"stage_lifecycle_report only supports multi-stage strategies, "
            f"got {strategy}"
        )

    cost_by_stage = result.get("cost", {}).get("by_stage", {})
    deltas = compute_stage_deltas(result_path)

    per_pass: dict[str, Any] | None = None
    try:
        per_pass_si = compute_per_pass_f1(result_path, metric="si_f1")
        per_pass_tc = compute_per_pass_f1(result_path, metric="tc_f1")
        per_pass = {
            "si_f1": per_pass_si["per_pass_f1"],
            "tc_f1": per_pass_tc["per_pass_f1"],
            "merged_si_f1": per_pass_si["merged_f1"],
            "merged_tc_f1": per_pass_tc["merged_f1"],
            "n_passes": per_pass_si["n_passes"],
        }
    except (ValueError, KeyError):
        pass

    rows = []

    if per_pass:
        prev_si = 0.0
        prev_tc = 0.0
        for k in range(per_pass["n_passes"]):
            si_k = per_pass["si_f1"][k]
            tc_k = per_pass["tc_f1"][k]
            rows.append({
                "stage": f"Pass {k}",
                "api_calls": "1 / article",
                "cost": None,
                "si_f1": si_k,
                "tc_f1": tc_k,
                "delta_si": si_k - prev_si if k > 0 else None,
                "delta_tc": tc_k - prev_tc if k > 0 else None,
            })
            prev_si = si_k
            prev_tc = tc_k

        rows.append({
            "stage": "Merged (after S1)",
            "api_calls": None,
            "cost": cost_by_stage.get("stage1_detection", {}).get("cost", 0),
            "si_f1": per_pass["merged_si_f1"],
            "tc_f1": per_pass["merged_tc_f1"],
            "delta_si": per_pass["merged_si_f1"] - max(per_pass["si_f1"]),
            "delta_tc": per_pass["merged_tc_f1"] - max(per_pass["tc_f1"]),
        })
    else:
        s1 = deltas["s1"]
        rows.append({
            "stage": "After S1 (merged detection)",
            "api_calls": cost_by_stage.get("stage1_detection", {}).get("calls", "?"),
            "cost": cost_by_stage.get("stage1_detection", {}).get("cost", 0),
            "si_f1": s1["si_f1"],
            "tc_f1": s1["tc_f1"],
            "delta_si": None,
            "delta_tc": None,
        })

    s2 = deltas["s2"]
    s2_cost_key = (
        "stage2_critique" if strategy in ("asv", "hybrid") else "stage2_consolidation"
    )
    s2_cost = cost_by_stage.get(s2_cost_key, {})
    if s2.get("si_f1", 0) > 0 or s2.get("tc_f1", 0) > 0:
        s2_label = (
            "After S2 (critique)" if strategy in ("asv", "hybrid")
            else "After S2 (consolidation)"
        )
        rows.append({
            "stage": s2_label,
            "api_calls": s2_cost.get("calls", "?"),
            "cost": s2_cost.get("cost", 0),
            "si_f1": s2["si_f1"],
            "tc_f1": s2["tc_f1"],
            "delta_si": deltas["delta_s1_to_s2"]["si_f1"],
            "delta_tc": deltas["delta_s1_to_s2"]["tc_f1"],
        })

    s3 = deltas["s3"]
    if strategy == "hybrid" and (s3.get("si_f1", 0) > 0 or s3.get("tc_f1", 0) > 0):
        s3_cost = cost_by_stage.get("stage3_refinement", {})
        rows.append({
            "stage": "After S3 (refinement)",
            "api_calls": s3_cost.get("calls", "?"),
            "cost": s3_cost.get("cost", 0),
            "si_f1": s3["si_f1"],
            "tc_f1": s3["tc_f1"],
            "delta_si": deltas["delta_s2_to_s3"]["si_f1"],
            "delta_tc": deltas["delta_s2_to_s3"]["tc_f1"],
        })

    return {
        "strategy": strategy,
        "rows": rows,
        "total_cost": result.get("cost", {}).get("total_cost_usd", 0),
    }


def analyze_relabel_quality(result_path: Path) -> dict[str, Any]:
    """Analyse relabel events: counts, directions, and correctness impact.

    Only meaningful for consol and hybrid (strategies that relabel).
    """
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    if strategy not in ("consol", "hybrid"):
        raise ValueError(
            f"Relabel analysis only meaningful for consol/hybrid; got {strategy!r}"
        )

    label_changes = result.get("diagnostics", {}).get("label_change_log", [])
    relabels = [e for e in label_changes if e.get("was_relabeled", False)]

    if not relabels:
        return {
            "strategy": strategy,
            "total_relabels": 0,
            "note": "No relabels found in this run",
        }

    improvements = 0
    regressions = 0
    lateral_correct = 0
    lateral_wrong = 0

    by_direction: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "count": 0, "improvements": 0, "regressions": 0,
            "lateral_correct": 0, "lateral_wrong": 0,
        }
    )

    per_original: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_relabels_from": 0, "improvements": 0, "regressions": 0,
            "relabel_targets": defaultdict(int),
        }
    )

    for entry in relabels:
        orig = entry["original_label"]
        new = entry["new_label"]
        correct_before = entry.get("tc_correct_before", False)
        correct_after = entry.get("tc_correct_after", False)

        direction = (orig, new)
        by_direction[direction]["count"] += 1

        if not correct_before and correct_after:
            category = "improvements"
            improvements += 1
        elif correct_before and not correct_after:
            category = "regressions"
            regressions += 1
        elif correct_before and correct_after:
            category = "lateral_correct"
            lateral_correct += 1
        else:
            category = "lateral_wrong"
            lateral_wrong += 1

        by_direction[direction][category] += 1
        per_original[orig]["total_relabels_from"] += 1
        if category == "improvements":
            per_original[orig]["improvements"] += 1
        elif category == "regressions":
            per_original[orig]["regressions"] += 1
        per_original[orig]["relabel_targets"][new] += 1

    total = len(relabels)

    per_original_final = {}
    for orig, data in per_original.items():
        targets = data["relabel_targets"]
        most_common = max(targets.items(), key=lambda x: x[1])[0] if targets else None
        per_original_final[orig] = {
            "total_relabels_from": data["total_relabels_from"],
            "improvements": data["improvements"],
            "regressions": data["regressions"],
            "net_improvement": data["improvements"] - data["regressions"],
            "most_common_new_label": most_common,
            "target_distribution": dict(targets),
        }

    return {
        "strategy": strategy,
        "total_relabels": total,
        "improvements": improvements,
        "regressions": regressions,
        "lateral_correct": lateral_correct,
        "lateral_wrong": lateral_wrong,
        "improvement_rate": improvements / total if total > 0 else 0,
        "regression_rate": regressions / total if total > 0 else 0,
        "net_improvement": improvements - regressions,
        "net_improvement_rate": (improvements - regressions) / total if total > 0 else 0,
        "by_direction": {f"{o}->{n}": d for (o, n), d in by_direction.items()},
        "by_direction_tuples": dict(by_direction),
        "per_original_technique": per_original_final,
    }


def plot_relabel_quality_matrix(
    result_path: Path,
    output_path: Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot 14x14 heatmap of relabels, coloured by net F1 impact per cell."""
    analysis = analyze_relabel_quality(result_path)
    strategy = analysis["strategy"]

    if analysis["total_relabels"] == 0:
        raise ValueError(f"No relabels to plot for {strategy}")

    from src.schemas import Technique
    techniques = [t.value for t in Technique]
    n = len(techniques)
    tech_idx = {t: i for i, t in enumerate(techniques)}

    counts = np.zeros((n, n), dtype=int)
    net_impact = np.zeros((n, n), dtype=float)

    for (orig, new), data in analysis["by_direction_tuples"].items():
        if orig not in tech_idx or new not in tech_idx:
            continue
        i = tech_idx[orig]
        j = tech_idx[new]
        counts[i][j] = data["count"]
        net_impact[i][j] = data["improvements"] - data["regressions"]

    fig, ax = plt.subplots(figsize=(11, 9))
    max_abs = max(abs(net_impact.min()), abs(net_impact.max()), 1)
    im = ax.imshow(
        net_impact, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto",
    )

    short_labels = [t.replace(",", ",\n").replace("_", " ") for t in techniques]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Relabeled to", fontsize=11)
    ax.set_ylabel("Original technique", fontsize=11)
    ax.set_title(
        title or f"Relabel matrix: {strategy}\n"
                 f"(colour = net F1 impact, cell value = count)",
        fontsize=12,
    )

    for i in range(n):
        for j in range(n):
            if counts[i][j] > 0:
                ax.text(
                    j, i, f"{counts[i][j]}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(net_impact[i][j]) < max_abs * 0.5 else "white",
                )

    fig.colorbar(im, ax=ax, label="Net impact (improvements \u2212 regressions)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "strategy": strategy,
        "output_path": str(output_path),
        "total_relabels": analysis["total_relabels"],
        "improvements": analysis["improvements"],
        "regressions": analysis["regressions"],
        "net_improvement": analysis["net_improvement"],
    }


def plot_per_pass_f1(
    result_path: Path,
    output_path: Path,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot per-pass F1 isolation: Pass 0 … Pass N, Merged."""
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(
            f"Per-pass F1 only applies to multi-pass strategies; got {strategy}"
        )

    si_data = compute_per_pass_f1(result_path, metric="si_f1")
    tc_data = compute_per_pass_f1(result_path, metric="tc_f1")

    n_passes = si_data["n_passes"]
    stages = [f"Pass {k}" for k in range(n_passes)] + ["Merged"]
    si_values = list(si_data["per_pass_f1"]) + [si_data["merged_f1"]]
    tc_values = list(tc_data["per_pass_f1"]) + [tc_data["merged_f1"]]

    x = np.arange(len(stages))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    si_color = "#2a9d8f"
    tc_color = "#e76f51"

    bars_si = ax.bar(
        x - width / 2, si_values, width,
        label="SI-F1", color=si_color, edgecolor="black", linewidth=0.6,
    )
    bars_tc = ax.bar(
        x + width / 2, tc_values, width,
        label="TC-F1", color=tc_color, edgecolor="black", linewidth=0.6,
    )

    for bar, val in zip(bars_si, si_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    for bar, val in zip(bars_tc, tc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )

    bars_si[-1].set_linewidth(2)
    bars_tc[-1].set_linewidth(2)

    best_si = max(si_data["per_pass_f1"])
    best_tc = max(tc_data["per_pass_f1"])
    ax.axhline(y=best_si, color=si_color, linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=best_tc, color=tc_color, linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_title(
        title or (
            f"Per-pass F1 isolation: {strategy}\n"
            f"(merged union gains {si_data['merged_f1'] - best_si:+.3f} SI-F1, "
            f"{tc_data['merged_f1'] - best_tc:+.3f} TC-F1 over best single pass)"
        ),
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(si_values), max(tc_values)) * 1.15)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "strategy": strategy,
        "n_passes": n_passes,
        "per_pass_si": si_data["per_pass_f1"],
        "per_pass_tc": tc_data["per_pass_f1"],
        "merged_si": si_data["merged_f1"],
        "merged_tc": tc_data["merged_f1"],
        "si_gain_over_best_pass": si_data["merged_f1"] - best_si,
        "tc_gain_over_best_pass": tc_data["merged_f1"] - best_tc,
        "output_path": str(output_path),
    }


def plot_error_distribution(
    result_paths: list[Path],
    output_path: Path,
    mode: str = "stacked_percentage",
    title: str | None = None,
) -> dict[str, Any]:
    """Plot six-category error distribution across strategies.

    Args:
        mode: "stacked_percentage" (bars sum to 100%), "stacked_count",
              or "grouped".
    """
    if mode not in ("stacked_percentage", "stacked_count", "grouped"):
        raise ValueError(f"mode must be stacked_percentage, stacked_count, or grouped")

    errors = classify_errors_multi(result_paths)
    strategies = list(errors["per_strategy"].keys())

    categories = [
        "correct", "missed_detection", "false_alarm", "wrong_technique",
        "boundary_too_broad", "boundary_too_narrow", "boundary_offset",
    ]
    colors = {
        "correct": "#2a9d8f",
        "missed_detection": "#d62828",
        "false_alarm": "#f77f00",
        "wrong_technique": "#fcbf49",
        "boundary_too_broad": "#9c6644",
        "boundary_too_narrow": "#7b5e57",
        "boundary_offset": "#5a4a42",
    }
    pretty = {
        "correct": "Correct",
        "missed_detection": "Missed detection",
        "false_alarm": "False alarm",
        "wrong_technique": "Wrong technique",
        "boundary_too_broad": "Boundary too broad",
        "boundary_too_narrow": "Boundary too narrow",
        "boundary_offset": "Boundary offset",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))

    if mode == "stacked_percentage":
        bottom = np.zeros(len(strategies))
        for cat in categories:
            values = np.array([
                errors["per_strategy"][s]["percentages"][cat] for s in strategies
            ])
            bars = ax.bar(
                x, values, bottom=bottom, width=0.65,
                color=colors[cat], edgecolor="black", linewidth=0.5,
                label=pretty[cat],
            )
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val >= 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[i] + val / 2,
                        f"{val:.1f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if val > 15 else "black",
                        fontweight="bold" if val > 15 else "normal",
                    )
            bottom += values
        ax.set_ylabel("Percentage of events", fontsize=11)
        ax.set_ylim(0, 100)

    elif mode == "stacked_count":
        bottom = np.zeros(len(strategies))
        for cat in categories:
            values = np.array([
                errors["per_strategy"][s]["counts"][cat] for s in strategies
            ])
            ax.bar(
                x, values, bottom=bottom, width=0.65,
                color=colors[cat], edgecolor="black", linewidth=0.5,
                label=pretty[cat],
            )
            bottom += values
        ax.set_ylabel("Number of events", fontsize=11)

    else:
        width = 0.11
        for i, cat in enumerate(categories):
            values = [errors["per_strategy"][s]["counts"][cat] for s in strategies]
            ax.bar(
                x + (i - len(categories) / 2) * width, values, width,
                color=colors[cat], edgecolor="black", linewidth=0.5,
                label=pretty[cat],
            )
        ax.set_ylabel("Number of events", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_xlabel("Strategy", fontsize=11)
    ax.set_title(
        title or "Six-category error distribution across strategies", fontsize=12,
    )
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "strategies": strategies,
        "categories": categories,
        "mode": mode,
        "output_path": str(output_path),
    }


def plot_per_technique_stage_delta(
    result_path: Path,
    output_path: Path,
    metric: str = "f1",
    from_stage: str = "after_s1",
    to_stage: str = "after_s2",
    title: str | None = None,
) -> dict[str, Any]:
    """Plot per-technique F1 delta from one stage to another.

    Horizontal bar chart sorted by delta, green for gains, red for losses.
    """
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(
            f"Stage delta only applies to multi-stage strategies; got {strategy}"
        )
    if metric not in ("f1", "precision", "recall"):
        raise ValueError(f"metric must be f1, precision, or recall; got {metric}")

    stage_data = stage_f1_corrected(result_path)

    if from_stage not in stage_data or to_stage not in stage_data:
        available = [k for k in stage_data if k.startswith("after_")]
        raise ValueError(f"Required stages not present. Have: {available}")

    from_tech = stage_data[from_stage].get("per_technique", {})
    to_tech = stage_data[to_stage].get("per_technique", {})

    from src.schemas import Technique
    techniques = [t.value for t in Technique]

    deltas = []
    for tech in techniques:
        from_val = from_tech.get(tech, {}).get(metric, 0)
        to_val = to_tech.get(tech, {}).get(metric, 0)
        gold_count = (
            from_tech.get(tech, {}).get("gold_count", 0)
            or to_tech.get(tech, {}).get("gold_count", 0)
        )
        if gold_count == 0:
            continue
        deltas.append({
            "technique": tech,
            "from_val": from_val,
            "to_val": to_val,
            "delta": to_val - from_val,
            "gold_count": gold_count,
        })

    deltas.sort(key=lambda d: d["delta"], reverse=True)
    if not deltas:
        raise ValueError("No techniques with gold instances found")

    fig, ax = plt.subplots(figsize=(10, max(6, len(deltas) * 0.4)))
    pos_color = "#2a9d8f"
    neg_color = "#d62828"
    zero_color = "#cccccc"

    technique_labels = [
        f"{d['technique'].replace('_', ' ').replace(',', ', ')}  (n={d['gold_count']})"
        for d in deltas
    ]
    delta_values = [d["delta"] for d in deltas]
    bar_colors = [
        pos_color if d > 0.001 else (neg_color if d < -0.001 else zero_color)
        for d in delta_values
    ]

    y = np.arange(len(deltas))
    bars = ax.barh(y, delta_values, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, d in zip(bars, deltas):
        val = d["delta"]
        if abs(val) < 0.001:
            label = "\u2014"
            x_offset = 0.002
        elif val > 0:
            label = f"+{val:.3f}"
            x_offset = val + 0.002
        else:
            label = f"{val:.3f}"
            x_offset = val - 0.002
        ax.text(
            x_offset, bar.get_y() + bar.get_height() / 2,
            label, va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(technique_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(
        f"\u0394 {metric.upper()} ({to_stage} \u2212 {from_stage})", fontsize=11,
    )

    n_improved = sum(1 for d in deltas if d["delta"] > 0.001)
    n_worsened = sum(1 for d in deltas if d["delta"] < -0.001)
    n_unchanged = len(deltas) - n_improved - n_worsened

    ax.set_title(
        title or (
            f"Per-technique {metric.upper()} change: {strategy}\n"
            f"{from_stage} \u2192 {to_stage}  "
            f"(\u2191 {n_improved} improved, \u2193 {n_worsened} worsened, "
            f"= {n_unchanged} unchanged)"
        ),
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "strategy": strategy,
        "metric": metric,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_unchanged": n_unchanged,
        "deltas": deltas,
        "output_path": str(output_path),
    }


def format_lifecycle_report(report: dict[str, Any]) -> str:
    """Format a stage_lifecycle_report result as a printable ASCII table."""
    lines = []
    lines.append(f"\nStage lifecycle: {report['strategy']}")
    lines.append("=" * 95)
    header = (
        f"{'Stage':<32} {'Cost':>10} {'SI-F1':>10} {'TC-F1':>10} "
        f"{'dSI':>10} {'dTC':>10}"
    )
    lines.append(header)
    lines.append("-" * 95)

    for row in report["rows"]:
        cost_str = f"${row['cost']:.4f}" if row["cost"] is not None else "-"
        si_str = f"{row['si_f1']:.4f}" if row["si_f1"] is not None else "-"
        tc_str = f"{row['tc_f1']:.4f}" if row["tc_f1"] is not None else "-"
        dsi_str = f"{row['delta_si']:+.4f}" if row["delta_si"] is not None else "-"
        dtc_str = f"{row['delta_tc']:+.4f}" if row["delta_tc"] is not None else "-"
        lines.append(
            f"{row['stage']:<32} {cost_str:>10} {si_str:>10} {tc_str:>10} "
            f"{dsi_str:>10} {dtc_str:>10}"
        )

    lines.append("-" * 95)
    lines.append(f"Total cost: ${report['total_cost']:.4f}")
    return "\n".join(lines)
