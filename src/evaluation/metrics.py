"""
SemEval-2020 Task 11 evaluation metrics.

Implements the EXACT evaluation from Da San Martino et al. (2020):

Subtask SI (Span Identification) — Equations 1-3:
  P(S,T) = (1/|S|) * Σ_{s∈S,t∈T} |s ∩ t| / |t|
  R(S,T) = (1/|T|) * Σ_{s∈S,t∈T} |s ∩ t| / |s|
  F1 = harmonic mean of P and R

Key details from the paper:
  - P divides overlap by GOLD span length |t| (not predicted)
  - R divides overlap by PREDICTED span length |s| (not gold)
  - Overlapping PREDICTED spans are MERGED before scoring (regardless of technique)
  - SI metric is technique-agnostic (binary: propaganda or not)

Additionally computes:
  - Joint SI+TC metric (same formulas but only counts overlap when technique matches)
  - Per-technique breakdown
  - Macro-F1 across techniques
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.schemas import Article, GoldSpan, PredictedSpan, Prediction, Technique

logger = logging.getLogger(__name__)


@dataclass
class SpanMetrics:
    """Metrics for a single evaluation run."""
    # Official SemEval SI metric (technique-agnostic)
    si_precision: float = 0.0
    si_recall: float = 0.0
    si_f1: float = 0.0

    # Joint SI+TC metric (technique must match)
    tc_precision: float = 0.0
    tc_recall: float = 0.0
    tc_f1: float = 0.0

    # Delta between SI and TC F1 (shows technique classification error)
    f1_delta: float = 0.0

    # Per-technique TC F1
    per_technique: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0

    # Counts
    total_gold_spans: int = 0
    total_pred_spans: int = 0  # After merging overlaps
    total_pred_spans_raw: int = 0  # Before merging
    total_articles: int = 0
    unresolved_spans: int = 0

    # Error analysis
    technique_confusions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "si_precision": round(self.si_precision, 4),
            "si_recall": round(self.si_recall, 4),
            "si_f1": round(self.si_f1, 4),
            "tc_precision": round(self.tc_precision, 4),
            "tc_recall": round(self.tc_recall, 4),
            "tc_f1": round(self.tc_f1, 4),
            "f1_delta": round(self.f1_delta, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_technique": self.per_technique,
            "total_gold_spans": self.total_gold_spans,
            "total_pred_spans": self.total_pred_spans,
            "total_pred_spans_raw": self.total_pred_spans_raw,
            "total_articles": self.total_articles,
            "unresolved_spans": self.unresolved_spans,
        }


def evaluate(
    articles: list[Article],
    predictions: dict[str, Prediction],
) -> SpanMetrics:
    """Compute SemEval-2020 Task 11 metrics across all articles.

    Computes both the official SI metric (technique-agnostic, with span merging)
    and a joint SI+TC metric (technique must match, no merging).
    """
    # ── Official SI metric accumulators ───────────────────────────────────
    si_p_num = 0.0   # Σ |s∩t| / |t|
    si_p_den = 0     # |S| total merged predicted spans
    si_r_num = 0.0   # Σ |s∩t| / |s|
    si_r_den = 0     # |T| total gold spans

    # ── Joint SI+TC metric accumulators ───────────────────────────────────
    tc_p_num = 0.0
    tc_p_den = 0
    tc_r_num = 0.0
    tc_r_den = 0

    # Per-technique TC accumulators
    tech_p_num: dict[Technique, float] = defaultdict(float)
    tech_p_den: dict[Technique, int] = defaultdict(int)
    tech_r_num: dict[Technique, float] = defaultdict(float)
    tech_r_den: dict[Technique, int] = defaultdict(int)

    total_gold = 0
    total_pred_raw = 0
    total_pred_merged = 0
    total_unresolved = 0
    confusions: list[dict[str, Any]] = []

    for article in articles:
        pred = predictions.get(article.id)
        gold_spans = article.gold_spans

        if pred is None:
            si_r_den += len(gold_spans)
            tc_r_den += len(gold_spans)
            total_gold += len(gold_spans)
            for gs in gold_spans:
                tech_r_den[gs.technique] += 1
            continue

        # Get resolved predicted spans
        raw_pred_spans = [s for s in pred.confirmed_spans if s.resolved]
        unresolved = [s for s in pred.confirmed_spans if not s.resolved]
        total_unresolved += len(unresolved)
        total_pred_raw += len(raw_pred_spans)
        total_gold += len(gold_spans)

        # ── SI metric: merge overlapping predictions (per SemEval paper) ──
        merged_intervals = _merge_overlapping_spans(
            [(s.start, s.end) for s in raw_pred_spans]
        )
        total_pred_merged += len(merged_intervals)

        # SI Precision (Eq 1): for each merged predicted span s,
        #   find max over gold spans t of |s ∩ t| / |t|
        si_p_den += len(merged_intervals)
        for (ms, me) in merged_intervals:
            best = 0.0
            for gs in gold_spans:
                overlap = _char_overlap(ms, me, gs.start, gs.end)
                if overlap > 0 and gs.text_length > 0:
                    best = max(best, overlap / gs.text_length)
            si_p_num += best

        # SI Recall (Eq 2): for each gold span t,
        #   find max over merged predicted spans s of |s ∩ t| / |s|
        si_r_den += len(gold_spans)
        for gs in gold_spans:
            best = 0.0
            for (ms, me) in merged_intervals:
                m_len = me - ms
                overlap = _char_overlap(gs.start, gs.end, ms, me)
                if overlap > 0 and m_len > 0:
                    best = max(best, overlap / m_len)
            si_r_num += best

        # ── TC metric: per-span with technique matching (no merging) ──────
        # Uses same formula direction as SI (P divides by gold, R by pred)
        tc_p_den += len(raw_pred_spans)
        for ps in raw_pred_spans:
            tech_p_den[ps.technique] += 1
            ps_len = ps.end - ps.start
            best_tc = 0.0
            for gs in gold_spans:
                if ps.technique == gs.technique:
                    overlap = _char_overlap(ps.start, ps.end, gs.start, gs.end)
                    if overlap > 0 and gs.text_length > 0:
                        best_tc = max(best_tc, overlap / gs.text_length)
            tc_p_num += best_tc
            tech_p_num[ps.technique] += best_tc

        tc_r_den += len(gold_spans)
        for gs in gold_spans:
            tech_r_den[gs.technique] += 1
            best_tc = 0.0
            for ps in raw_pred_spans:
                if gs.technique == ps.technique:
                    ps_len = ps.end - ps.start
                    overlap = _char_overlap(gs.start, gs.end, ps.start, ps.end)
                    if overlap > 0 and ps_len > 0:
                        best_tc = max(best_tc, overlap / ps_len)
            tc_r_num += best_tc
            tech_r_num[gs.technique] += best_tc

            # Track technique confusions for error analysis
            if best_tc == 0.0:
                for ps in raw_pred_spans:
                    if ps.technique != gs.technique:
                        overlap = _char_overlap(gs.start, gs.end, ps.start, ps.end)
                        if overlap > 0:
                            confusions.append({
                                "article_id": article.id,
                                "gold_technique": gs.technique.value,
                                "pred_technique": ps.technique.value,
                                "gold_text": article.text[gs.start:gs.end][:80],
                                "pred_text": ps.span_text[:80],
                            })
                            break

    # Compute final metrics
    si_p = si_p_num / si_p_den if si_p_den > 0 else 0.0
    si_r = si_r_num / si_r_den if si_r_den > 0 else 0.0
    si_f1 = _f1(si_p, si_r)

    tc_p = tc_p_num / tc_p_den if tc_p_den > 0 else 0.0
    tc_r = tc_r_num / tc_r_den if tc_r_den > 0 else 0.0
    tc_f1 = _f1(tc_p, tc_r)

    # Per-technique F1
    per_technique: dict[str, dict[str, float]] = {}
    technique_f1s: list[float] = []

    for t in Technique:
        p = tech_p_num[t] / tech_p_den[t] if tech_p_den[t] > 0 else 0.0
        r = tech_r_num[t] / tech_r_den[t] if tech_r_den[t] > 0 else 0.0
        f1 = _f1(p, r)
        per_technique[t.value] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "gold_count": tech_r_den[t],
            "pred_count": tech_p_den[t],
        }
        if tech_r_den[t] > 0:
            technique_f1s.append(f1)

    macro_f1 = sum(technique_f1s) / len(technique_f1s) if technique_f1s else 0.0

    return SpanMetrics(
        si_precision=si_p,
        si_recall=si_r,
        si_f1=si_f1,
        tc_precision=tc_p,
        tc_recall=tc_r,
        tc_f1=tc_f1,
        f1_delta=si_f1 - tc_f1,
        per_technique=per_technique,
        macro_f1=macro_f1,
        total_gold_spans=total_gold,
        total_pred_spans=total_pred_merged,
        total_pred_spans_raw=total_pred_raw,
        total_articles=len(articles),
        unresolved_spans=total_unresolved,
        technique_confusions=confusions[:100],
    )


def _merge_overlapping_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping character spans into non-overlapping intervals.

    Per SemEval paper: "all overlapping annotations, independently of their
    techniques, are merged first" before computing SI precision/recall.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _char_overlap(start1: int, end1: int, start2: int, end2: int) -> int:
    """Compute character-level overlap between two spans."""
    return max(0, min(end1, end2) - max(start1, start2))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_results(metrics: SpanMetrics, strategy: str = "") -> None:
    """Pretty-print evaluation results for a single strategy."""
    header = f"Results: {strategy}" if strategy else "Results"
    w = 72
    print(f"\n{'='*w}")
    print(f"  {header}")
    print(f"{'='*w}")

    print(f"\n  {'Metric':<45} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  {'Span Identification (technique-agnostic)':<45} "
          f"{metrics.si_precision:>9.4f} {metrics.si_recall:>9.4f} {metrics.si_f1:>9.4f}")
    print(f"  {'Technique Classification (technique match)':<45} "
          f"{metrics.tc_precision:>9.4f} {metrics.tc_recall:>9.4f} {metrics.tc_f1:>9.4f}")
    print(f"  {'Macro-avg across techniques':<45} "
          f"{'':>9} {'':>9} {metrics.macro_f1:>9.4f}")

    print(f"\n  SI-TC Delta: {metrics.f1_delta:.4f}")
    print(f"  Gold spans: {metrics.total_gold_spans}  |  "
          f"Pred spans: {metrics.total_pred_spans_raw} (merged: {metrics.total_pred_spans})  |  "
          f"Unresolved: {metrics.unresolved_spans}")

    print(f"\n  {'Technique':<35} {'P':>7} {'R':>7} {'F1':>7} {'Gold':>6} {'Pred':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")
    for t_name, t_metrics in sorted(
        metrics.per_technique.items(),
        key=lambda x: x[1]["f1"],
        reverse=True,
    ):
        print(
            f"  {t_name:<35} "
            f"{t_metrics['precision']:>7.4f} "
            f"{t_metrics['recall']:>7.4f} "
            f"{t_metrics['f1']:>7.4f} "
            f"{t_metrics['gold_count']:>6} "
            f"{t_metrics['pred_count']:>6}"
        )
    print(f"{'='*w}\n")


def print_comparison_table(
    results_by_strategy: dict[str, dict[str, Any]],
    model: str = "",
    num_articles: int | None = None,
) -> None:
    """Print a side-by-side comparison table across strategies.

    Args:
        results_by_strategy: {strategy_name: experiment_results_dict}
        model: model name for the header
        num_articles: number of articles (for header)
    """
    strategies = [s for s in results_by_strategy if "error" not in results_by_strategy[s]]
    if not strategies:
        print("\n  No successful strategy results to compare.\n")
        return

    suffix = f" ({num_articles} articles)" if num_articles else ""
    header = f"STRATEGY COMPARISON: {model}{suffix}" if model else "STRATEGY COMPARISON"
    w = 105
    print(f"\n{'='*w}")
    print(f"  {header}")
    print(f"{'='*w}")
    print(
        f"  {'Strategy':<14} "
        f"{'SI P':>7} {'SI R':>7} {'SI F1':>7}  "
        f"{'TC P':>7} {'TC R':>7} {'TC F1':>7}  "
        f"{'Macro':>7}  "
        f"{'Cost':>8} {'Pred':>6}"
    )
    print(
        f"  {'-'*14} "
        f"{'-'*7} {'-'*7} {'-'*7}  "
        f"{'-'*7} {'-'*7} {'-'*7}  "
        f"{'-'*7}  "
        f"{'-'*8} {'-'*6}"
    )

    for strategy in strategies:
        r = results_by_strategy[strategy]
        m = r.get("metrics", {})
        c = r.get("cost", {})
        print(
            f"  {strategy:<14} "
            f"{m.get('si_precision', 0):>7.4f} "
            f"{m.get('si_recall', 0):>7.4f} "
            f"{m.get('si_f1', 0):>7.4f}  "
            f"{m.get('tc_precision', 0):>7.4f} "
            f"{m.get('tc_recall', 0):>7.4f} "
            f"{m.get('tc_f1', 0):>7.4f}  "
            f"{m.get('macro_f1', 0):>7.4f}  "
            f"${c.get('total_cost_usd', 0):>7.4f} "
            f"{m.get('total_pred_spans_raw', 0):>6}"
        )

    # Highlight best per column
    col_keys = [
        ("si_precision", "SI P"), ("si_recall", "SI R"), ("si_f1", "SI F1"),
        ("tc_precision", "TC P"), ("tc_recall", "TC R"), ("tc_f1", "TC F1"),
        ("macro_f1", "Macro"),
    ]
    best_row: dict[str, tuple[str, float]] = {}
    for key, label in col_keys:
        best_val, best_strat = -1.0, ""
        for s in strategies:
            v = results_by_strategy[s].get("metrics", {}).get(key, 0)
            if v > best_val:
                best_val, best_strat = v, s
        best_row[label] = (best_strat, best_val)

    print(f"\n  Best: ", end="")
    parts = [f"{label}={strat}({val:.4f})" for label, (strat, val) in best_row.items() if val > 0]
    print(", ".join(parts))
    print(f"{'='*w}\n")


def print_technique_comparison(
    results_by_strategy: dict[str, dict[str, Any]],
) -> None:
    """Print per-technique F1 comparison across strategies."""
    strategies = [s for s in results_by_strategy if "error" not in results_by_strategy[s]]
    if len(strategies) < 2:
        return

    all_techniques: set[str] = set()
    for s in strategies:
        m = results_by_strategy[s].get("metrics", {})
        all_techniques.update(m.get("per_technique", {}).keys())

    if not all_techniques:
        return

    strat_col_w = 9
    w = 37 + len(strategies) * (strat_col_w + 1) + 7
    print(f"{'='*w}")
    print(f"  PER-TECHNIQUE F1 COMPARISON")
    print(f"{'='*w}")

    hdr = f"  {'Technique':<35}"
    for s in strategies:
        hdr += f" {s:>{strat_col_w}}"
    hdr += f" {'Gold':>6}"
    print(hdr)

    sep = f"  {'-'*35}"
    for _ in strategies:
        sep += f" {'-'*strat_col_w}"
    sep += f" {'-'*6}"
    print(sep)

    technique_rows: list[tuple[str, dict[str, float], int]] = []
    for tech in sorted(all_techniques):
        f1s: dict[str, float] = {}
        gold = 0
        for s in strategies:
            pt = results_by_strategy[s].get("metrics", {}).get("per_technique", {}).get(tech, {})
            f1s[s] = pt.get("f1", 0.0)
            gold = max(gold, pt.get("gold_count", 0))
        technique_rows.append((tech, f1s, gold))

    technique_rows.sort(key=lambda x: max(x[1].values()), reverse=True)

    for tech, f1s, gold in technique_rows:
        row = f"  {tech:<35}"
        best_val = max(f1s.values())
        for s in strategies:
            val = f1s[s]
            marker = "*" if val == best_val and val > 0 and len(strategies) > 1 else " "
            row += f" {val:>{strat_col_w - 1}.4f}{marker}"
        row += f" {gold:>6}"
        print(row)

    print(f"{'='*w}\n")