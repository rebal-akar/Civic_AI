"""
Chart generation utilities for strategy comparison outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRATEGY_COLORS = {
    "zero_shot": "#4C72B0",
    "few_shot": "#55A868",
    "cot": "#C44E52",
    "asv": "#8172B2",
    "consol": "#CCB974",
    "hybrid": "#DD8452",
}


def _color_for(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#666666")


def _add_bar_labels(ax, bars, fmt=".3f"):
    for bar in bars:
        h = bar.get_height()
        if h > 0.005:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.008,
                f"{h:{fmt}}",
                ha="center", va="bottom", fontsize=8,
            )


def generate_comparison_charts(
    results_by_strategy: dict[str, dict[str, Any]],
    output_prefix: Path,
    num_articles: int | None = None,
) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    successful = {
        s: r for s, r in results_by_strategy.items()
        if isinstance(r, dict) and "error" not in r and isinstance(r.get("metrics"), dict)
    }
    if not successful:
        return []

    written: list[Path] = []
    strategies = list(successful.keys())

    fig1 = _plot_overall_f1(successful, strategies, num_articles)
    p1 = output_prefix.with_name(f"{output_prefix.name}_overall_f1.png")
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    written.append(p1)

    fig2 = _plot_recall_f1(successful, strategies)
    p2 = output_prefix.with_name(f"{output_prefix.name}_recall_f1.png")
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    written.append(p2)

    fig3 = _plot_per_technique_f1(successful, strategies, num_articles)
    p3 = output_prefix.with_name(f"{output_prefix.name}_technique_f1.png")
    fig3.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    written.append(p3)

    return written


def _plot_overall_f1(
    results_by_strategy: dict[str, dict[str, Any]],
    strategies: list[str],
    num_articles: int | None = None,
):
    """Chart 1: Main metrics — SI F1, TC F1, Macro F1 per strategy."""
    metric_keys = ["si_f1", "tc_f1", "macro_f1"]
    metric_labels = ["SI F1\n(span detection)", "TC F1\n(technique match)", "Macro F1\n(avg techniques)"]

    x = np.arange(len(metric_labels))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strategy in enumerate(strategies):
        vals = [results_by_strategy[strategy]["metrics"].get(k, 0.0) for k in metric_keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=strategy, color=_color_for(strategy),
        )
        _add_bar_labels(ax, bars)

    suffix = f" ({num_articles} articles)" if num_articles else ""
    ax.set_title(f"Overall Strategy Comparison{suffix}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def _plot_recall_f1(
    results_by_strategy: dict[str, dict[str, Any]],
    strategies: list[str],
):
    """Chart 2: Recall + F1 for both SI and TC."""
    metric_keys = ["si_recall", "si_f1", "tc_recall", "tc_f1"]
    metric_labels = ["SI Recall", "SI F1", "TC Recall", "TC F1"]

    x = np.arange(len(metric_labels))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, strategy in enumerate(strategies):
        vals = [results_by_strategy[strategy]["metrics"].get(k, 0.0) for k in metric_keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=strategy, color=_color_for(strategy),
        )
        _add_bar_labels(ax, bars)

    ax.set_title("Recall & F1 Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)

    ax.axvline(x=1.5, color="grey", linestyle="--", alpha=0.4)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "Span Identification",
            ha="center", fontsize=9, color="grey")
    ax.text(2.5, ax.get_ylim()[1] * 0.97, "Technique Classification",
            ha="center", fontsize=9, color="grey")

    fig.tight_layout()
    return fig


def _plot_per_technique_f1(
    results_by_strategy: dict[str, dict[str, Any]],
    strategies: list[str],
    num_articles: int | None = None,
):
    """Chart 3: Per-technique F1 for all techniques across strategies."""
    technique_names: set[str] = set()
    for s in strategies:
        technique_names.update(
            results_by_strategy[s]["metrics"].get("per_technique", {}).keys()
        )

    scored: list[tuple[str, float, int]] = []
    for t in technique_names:
        best = 0.0
        gold = 0
        for s in strategies:
            pt = (results_by_strategy[s]["metrics"]
                  .get("per_technique", {}).get(t, {}))
            best = max(best, float(pt.get("f1", 0.0)))
            gold = max(gold, int(pt.get("gold_count", 0)))
        scored.append((t, best, gold))
    scored.sort(key=lambda x: x[1], reverse=True)
    ordered_techniques = [t for t, _, _ in scored]

    x = np.arange(len(ordered_techniques))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, strategy in enumerate(strategies):
        per_tech = results_by_strategy[strategy]["metrics"].get("per_technique", {})
        vals = [float(per_tech.get(t, {}).get("f1", 0.0)) for t in ordered_techniques]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=strategy, color=_color_for(strategy),
        )
        _add_bar_labels(ax, bars, fmt=".2f")

    suffix = f" ({num_articles} articles)" if num_articles else ""
    ax.set_title(f"Per-Technique F1 Score by Strategy{suffix}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.1))
    ax.set_xticks(x)
    tech_labels = [f"{t}\n(n={scored[i][2]})" for i, t in enumerate(ordered_techniques)]
    ax.set_xticklabels(tech_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    return fig


def generate_diagnostic_charts(
    diagnostics: dict[str, Any],
    output_prefix: Path,
    strategy_label: str = "",
) -> list[Path]:
    """Generate diagnostic charts from compute_all_diagnostics output.

    Currently a minimal implementation that saves a stage-by-stage F1 chart
    when stage_snapshots data is available.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    stages = diagnostics.get("stage_by_stage_f1", {})
    if stages:
        fig, ax = plt.subplots(figsize=(8, 5))
        stage_names = []
        si_vals = []
        tc_vals = []
        for stage in ["after_s1", "after_s2", "after_s3"]:
            if stage not in stages:
                continue
            stage_names.append(stage.replace("after_", "").upper())
            si_vals.append(stages[stage]["si"]["f1"])
            tc_vals.append(stages[stage]["tc"]["f1"])
        if stage_names:
            x = np.arange(len(stage_names))
            w = 0.35
            bars_si = ax.bar(x - w / 2, si_vals, w, label="SI F1", color="#4C72B0")
            bars_tc = ax.bar(x + w / 2, tc_vals, w, label="TC F1", color="#C44E52")
            _add_bar_labels(ax, bars_si)
            _add_bar_labels(ax, bars_tc)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_names)
            ax.set_ylabel("F1 Score")
            ax.set_ylim(0, 1.0)
            title = "Stage-by-Stage F1"
            if strategy_label:
                title += f" — {strategy_label}"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = Path(f"{output_prefix}_stage_f1.png")
            fig.savefig(p, dpi=150)
            paths.append(p)
        plt.close(fig)

    return paths
