"""Post-hoc analysis functions operating on saved result JSONs.

All functions are pure and deterministic given their inputs — they read from
outputs/results/*.json and produce derived analyses without API calls.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.schemas import Article, Prediction

logger = logging.getLogger(__name__)


def _load_results(paths: list[Path]) -> list[dict]:
    loaded = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Result file not found: {p}")
        with open(p) as f:
            loaded.append(json.loads(f.read()))
    return loaded


def _reconstruct_per_article_counts(result: dict) -> list[dict]:
    """Rebuild per-article character-level TP/FP/FN for SI and TC."""
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
    """Character-level TP/FP/FN for SI (technique-agnostic) and TC (technique match)."""
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
        pred_chars_tc.setdefault(tech, [False] * text_length)
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
        gold_chars_tc.setdefault(tech, [False] * text_length)
        for i in range(start, end):
            gold_chars_tc[tech][i] = True

    si_tp = sum(1 for i in range(text_length) if pred_chars_si[i] and gold_chars_si[i])
    si_fp = sum(1 for i in range(text_length) if pred_chars_si[i] and not gold_chars_si[i])
    si_fn = sum(1 for i in range(text_length) if not pred_chars_si[i] and gold_chars_si[i])

    tc_tp = tc_fp = tc_fn = 0
    for tech in set(pred_chars_tc) | set(gold_chars_tc):
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
        "si_tp": float(si_tp), "si_fp": float(si_fp), "si_fn": float(si_fn),
        "tc_tp": float(tc_tp), "tc_fp": float(tc_fp), "tc_fn": float(tc_fn),
    }


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _reconstruct_from_result(result: dict) -> list[tuple["Article", "Prediction"]]:
    """Rebuild (Article, Prediction) pairs from a saved result JSON."""
    from src.schemas import (
        Article, GoldSpan, PredictedSpan, Prediction, normalise_technique,
    )

    eval_mode = result["config"].get("eval_mode", "permissive")
    pairs = []

    for aid, pred_dict in result["predictions"].items():
        article_text = pred_dict.get("article_text", "")

        gold_spans = []
        for gs in pred_dict.get("gold_spans", []):
            tech = normalise_technique(gs["technique"])
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
                verdict=s.get("verdict", ""),
            ))

        pairs.append((
            article,
            Prediction(article_id=aid, spans=pred_spans, eval_mode=eval_mode),
        ))

    return pairs


def _evaluate_subset(
    pairs: list[tuple["Article", "Prediction"]],
    metric: str,
) -> float:
    from src.evaluation.metrics import evaluate

    articles = [a for a, _ in pairs]
    predictions = {p.article_id: p for _, p in pairs}
    return getattr(evaluate(articles, predictions), metric, 0.0)


def compute_bootstrap_cis(
    result_path: Path,
    metric: str = "si_f1",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI by resampling articles with replacement and re-running evaluate().

    Each duplicate gets a unique ID so evaluate()'s dict-keyed interface handles it.
    """
    if metric not in ("si_f1", "tc_f1", "macro_f1"):
        raise ValueError(f"metric must be si_f1, tc_f1, or macro_f1, got {metric!r}")

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
        sampled_idx = [rng.randint(0, n - 1) for _ in range(n)]
        sampled_pairs = []
        for rep_idx, orig_idx in enumerate(sampled_idx):
            article, prediction = pairs[orig_idx]
            uid = f"{article.id}__b{rep_idx}"
            sampled_pairs.append((
                Article(id=uid, text=article.text, gold_spans=article.gold_spans),
                Prediction(
                    article_id=uid, spans=prediction.spans,
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


def pairwise_bootstrap(
    result_path_a: Path,
    result_path_b: Path,
    metric: str = "tc_f1",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap: for each iteration sample the same article indices for
    both strategies, compute F1_A - F1_B, return one-sided p-value for A > B.
    """
    if metric not in ("si_f1", "tc_f1", "macro_f1"):
        raise ValueError(f"metric must be si_f1, tc_f1, or macro_f1, got {metric!r}")

    from src.schemas import Article, Prediction

    result_a = _load_results([result_path_a])[0]
    result_b = _load_results([result_path_b])[0]

    pairs_a = _reconstruct_from_result(result_a)
    pairs_b = _reconstruct_from_result(result_b)

    a_by_id = {a.id: (a, p) for a, p in pairs_a}
    b_by_id = {a.id: (a, p) for a, p in pairs_b}
    shared_ids = sorted(set(a_by_id) & set(b_by_id))
    if not shared_ids:
        raise ValueError("Runs A and B share no articles.")
    if len(shared_ids) < len(a_by_id) or len(shared_ids) < len(b_by_id):
        logger.warning(
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
        deltas.append(_evaluate_subset(boot_a, metric) - _evaluate_subset(boot_b, metric))

    p_value = sum(1 for d in deltas if d <= 0) / n_bootstrap

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
    """Full pairwise p-value matrix with Holm-Bonferroni correction over
    directional comparisons (A > B where delta > 0).
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
                logger.warning(f"Pairwise {strategies[i]} vs {strategies[j]} failed: {e}")

    directional_pairs = [
        (key, p) for key, p in raw_p_values.items() if delta_values[key] > 0
    ]
    directional_pairs.sort(key=lambda x: x[1])

    m = len(directional_pairs)
    corrected: dict[tuple[str, str], float] = {}
    significant: dict[tuple[str, str], bool] = {}
    for rank, (key, p) in enumerate(directional_pairs):
        corrected[key] = min(1.0, p * (m - rank))
        significant[key] = p < alpha / (m - rank)

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


def compute_per_pass_f1(
    result_path: Path,
    metric: str = "si_f1",
) -> dict[str, Any]:
    """F1 for each individual detection pass and the merged union.

    Only applicable to multi-pass strategies (asv, consol, hybrid).
    """
    if metric not in ("si_f1", "tc_f1"):
        raise ValueError(f"metric must be si_f1 or tc_f1, got {metric!r}")

    from src.schemas import (
        Article, GoldSpan, PredictedSpan, Prediction, normalise_technique,
    )

    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    eval_mode = result["config"].get("eval_mode", "permissive")

    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(
            f"Strategy {strategy!r} has no multi-pass detection; "
            f"per-pass F1 only applies to asv, consol, hybrid."
        )

    first_pred = next(iter(result["predictions"].values()))
    per_pass_spans = first_pred.get("stage_outputs", {}).get("stage1_per_pass_spans", [])
    if not per_pass_spans:
        raise ValueError(f"Result {result_path} has no stage1_per_pass_spans.")
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

    return {
        "strategy": strategy,
        "metric": metric,
        "per_pass_f1": [_evaluate_subset(per_pass_pairs[k], metric) for k in range(n_passes)],
        "merged_f1": _evaluate_subset(merged_pairs, metric),
        "n_passes": n_passes,
        "n_articles": len(result["predictions"]),
    }


def stage_f1_corrected(result_path: Path) -> dict[str, Any]:
    """Stage-by-stage F1 recomputed via the official evaluate() function."""
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

    for stage in ("after_s1", "after_s2", "after_s3"):
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
                ))

            stage_articles.append(articles_by_id[aid])
            stage_predictions[aid] = Prediction(
                article_id=aid, spans=spans, eval_mode=eval_mode,
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


def _span_iou(span_a: dict, span_b: dict) -> float:
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
    """14x14 gold-vs-predicted technique confusion heatmap, row-normalised."""
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

        matched_gold: set[int] = set()
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
                matched_gold.add(best_gold_idx)
                counts[label_idx.get(gold_tech, label_idx["NONE"])][
                    label_idx.get(pred_tech, label_idx["NONE"])
                ] += 1
            else:
                counts[label_idx["NONE"]][
                    label_idx.get(ps["technique"], label_idx["NONE"])
                ] += 1

        for gi, gs in enumerate(gold_spans):
            if gi not in matched_gold:
                counts[label_idx.get(gs["technique"], label_idx["NONE"])][
                    label_idx["NONE"]
                ] += 1

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
    ax.set_title(title or f"TC confusion matrix: {strategy} (row-normalised)", fontsize=12)

    for i in range(n):
        for j in range(n):
            if counts[i][j] > 0:
                text_color = "white" if normalised[i][j] > 0.5 else "black"
                ax.text(j, i, str(counts[i][j]), ha="center", va="center",
                        fontsize=7, color=text_color)

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
    """Classify every predicted and gold span into one of the 7 error categories.

    1. pred span matched to best-IoU gold above iou_threshold
    2. IoU below threshold -> false_alarm
    3. different technique -> wrong_technique
    4. pred/gold length ratio above boundary_ratio_threshold -> boundary_too_broad
    5. gold/pred length ratio above threshold -> boundary_too_narrow
    6. IoU between iou_threshold and offset_threshold -> boundary_offset
    7. otherwise -> correct
    Gold spans with no matching prediction -> missed_detection.
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

        matched_gold: set[int] = set()

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
                matched_gold.add(best_gi)
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
            if gi not in matched_gold:
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
    """classify_errors across multiple runs, producing a comparison table."""
    per_strategy = {}
    for p in result_paths:
        r = classify_errors(p, iou_threshold=iou_threshold)
        per_strategy[r["strategy"]] = r

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

    return {"per_strategy": per_strategy, "comparison_table": table}


def compute_stage_deltas(result_path: Path) -> dict[str, Any]:
    """Per-stage F1 deltas (S1 -> S2 -> S3) using the official evaluate() function."""
    from src.evaluation.diagnostics import stage_by_stage_f1
    from src.schemas import PredictedSpan, Prediction, normalise_technique

    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]

    pairs = _reconstruct_from_result(result)
    articles = [a for a, _ in pairs]
    predictions: dict[str, Prediction] = {}

    for aid, pred_dict in result["predictions"].items():
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
                verdict=s.get("verdict", ""),
            ))

        predictions[aid] = Prediction(
            article_id=aid,
            spans=pred_spans,
            stage_snapshots=pred_dict.get("stage_snapshots", {}),
            eval_mode=eval_mode,
        )

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

    s1, s2, s3 = _extract("after_s1"), _extract("after_s2"), _extract("after_s3")

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


def stage_lifecycle_report(result_path: Path) -> dict[str, Any]:
    """Per-stage lifecycle table: API calls, cost, SI-F1, TC-F1, stage deltas.

    Rows are pipeline stages (including individual detection passes).
    Only applicable to multi-stage strategies (asv, consol, hybrid).
    """
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(f"stage_lifecycle_report requires multi-stage strategy, got {strategy}")

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
        prev_si = prev_tc = 0.0
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
            prev_si, prev_tc = si_k, tc_k

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
    s2_cost_key = "stage2_critique" if strategy in ("asv", "hybrid") else "stage2_consolidation"
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
    """Count relabel events and classify by F1 impact (improved / regressed / lateral)."""
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("consol", "hybrid"):
        raise ValueError(f"Relabel analysis requires consol/hybrid, got {strategy!r}")

    label_changes = result.get("diagnostics", {}).get("label_change_log", [])
    relabels = [e for e in label_changes if e.get("was_relabeled", False)]

    if not relabels:
        return {"strategy": strategy, "total_relabels": 0, "note": "No relabels found"}

    improvements = regressions = lateral_correct = lateral_wrong = 0
    by_direction: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"count": 0, "improvements": 0, "regressions": 0,
                 "lateral_correct": 0, "lateral_wrong": 0}
    )
    per_original: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_relabels_from": 0, "improvements": 0,
                 "regressions": 0, "relabel_targets": defaultdict(int)}
    )

    for entry in relabels:
        orig, new = entry["original_label"], entry["new_label"]
        correct_before = entry.get("tc_correct_before", False)
        correct_after = entry.get("tc_correct_after", False)

        by_direction[(orig, new)]["count"] += 1

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

        by_direction[(orig, new)][category] += 1
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
    """14x14 heatmap of relabels, coloured by net F1 impact per cell."""
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
        i, j = tech_idx[orig], tech_idx[new]
        counts[i][j] = data["count"]
        net_impact[i][j] = data["improvements"] - data["regressions"]

    fig, ax = plt.subplots(figsize=(11, 9))
    max_abs = max(abs(net_impact.min()), abs(net_impact.max()), 1)
    im = ax.imshow(net_impact, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")

    short_labels = [t.replace(",", ",\n").replace("_", " ") for t in techniques]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Relabeled to", fontsize=11)
    ax.set_ylabel("Original technique", fontsize=11)
    ax.set_title(
        title or f"Relabel matrix: {strategy}\n(colour = net F1 impact, cell = count)",
        fontsize=12,
    )

    for i in range(n):
        for j in range(n):
            if counts[i][j] > 0:
                ax.text(
                    j, i, f"{counts[i][j]}", ha="center", va="center", fontsize=8,
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
    """Per-pass F1 bars: Pass 0 … Pass N, then Merged. Multi-pass strategies only."""
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(f"Per-pass F1 requires multi-pass strategy, got {strategy}")

    si_data = compute_per_pass_f1(result_path, metric="si_f1")
    tc_data = compute_per_pass_f1(result_path, metric="tc_f1")

    n_passes = si_data["n_passes"]
    stages = [f"Pass {k}" for k in range(n_passes)] + ["Merged"]
    si_values = list(si_data["per_pass_f1"]) + [si_data["merged_f1"]]
    tc_values = list(tc_data["per_pass_f1"]) + [tc_data["merged_f1"]]

    x = np.arange(len(stages))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))

    si_color, tc_color = "#2a9d8f", "#e76f51"
    bars_si = ax.bar(x - width / 2, si_values, width, label="SI-F1",
                     color=si_color, edgecolor="black", linewidth=0.6)
    bars_tc = ax.bar(x + width / 2, tc_values, width, label="TC-F1",
                     color=tc_color, edgecolor="black", linewidth=0.6)

    for bar, val in zip(bars_si, si_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_tc, tc_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

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
            f"{tc_data['merged_f1'] - best_tc:+.3f} TC-F1 over best pass)"
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
    """Seven-category error distribution across strategies.

    mode: stacked_percentage (bars sum to 100), stacked_count, or grouped.
    """
    if mode not in ("stacked_percentage", "stacked_count", "grouped"):
        raise ValueError(f"mode must be stacked_percentage/stacked_count/grouped, got {mode}")

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
            values = np.array([errors["per_strategy"][s]["percentages"][cat] for s in strategies])
            bars = ax.bar(x, values, bottom=bottom, width=0.65,
                          color=colors[cat], edgecolor="black", linewidth=0.5,
                          label=pretty[cat])
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val >= 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bottom[i] + val / 2,
                        f"{val:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if val > 15 else "black",
                        fontweight="bold" if val > 15 else "normal",
                    )
            bottom += values
        ax.set_ylabel("Percentage of events", fontsize=11)
        ax.set_ylim(0, 100)

    elif mode == "stacked_count":
        bottom = np.zeros(len(strategies))
        for cat in categories:
            values = np.array([errors["per_strategy"][s]["counts"][cat] for s in strategies])
            ax.bar(x, values, bottom=bottom, width=0.65,
                   color=colors[cat], edgecolor="black", linewidth=0.5,
                   label=pretty[cat])
            bottom += values
        ax.set_ylabel("Number of events", fontsize=11)

    else:
        width = 0.11
        for i, cat in enumerate(categories):
            values = [errors["per_strategy"][s]["counts"][cat] for s in strategies]
            ax.bar(x + (i - len(categories) / 2) * width, values, width,
                   color=colors[cat], edgecolor="black", linewidth=0.5,
                   label=pretty[cat])
        ax.set_ylabel("Number of events", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_xlabel("Strategy", fontsize=11)
    ax.set_title(title or "Seven-category error distribution across strategies", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
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
    """Horizontal bar chart of per-technique F1 deltas between two stages."""
    result = _load_results([result_path])[0]
    strategy = result["config"]["strategy"]
    if strategy not in ("asv", "consol", "hybrid"):
        raise ValueError(f"Stage delta requires multi-stage strategy, got {strategy}")
    if metric not in ("f1", "precision", "recall"):
        raise ValueError(f"metric must be f1/precision/recall, got {metric}")

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
    pos_color, neg_color, zero_color = "#2a9d8f", "#d62828", "#cccccc"

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
            label, x_offset = "\u2014", 0.002
        elif val > 0:
            label, x_offset = f"+{val:.3f}", val + 0.002
        else:
            label, x_offset = f"{val:.3f}", val - 0.002
        ax.text(
            x_offset, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left" if val >= 0 else "right", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(technique_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(f"\u0394 {metric.upper()} ({to_stage} \u2212 {from_stage})", fontsize=11)

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

