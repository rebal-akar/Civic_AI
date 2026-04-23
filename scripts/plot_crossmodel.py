"""Generate color-coded comparison bar charts for a cross-model run JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_COLORS: dict[str, str] = {
    "consol_xfam_same_4o":           "#4C72B0",
    "consol_xfam_same_sonnet":       "#55A868",
    "consol_xfam_cascade_4o_sonnet": "#C44E52",
    "consol_xfam_cascade_sonnet_4o": "#8172B2",
}

CONFIG_LABELS: dict[str, str] = {
    "consol_xfam_same_4o":           "gpt-4o → gpt-4o",
    "consol_xfam_same_sonnet":       "Sonnet → Sonnet",
    "consol_xfam_cascade_4o_sonnet": "gpt-4o → Sonnet",
    "consol_xfam_cascade_sonnet_4o": "Sonnet → gpt-4o",
}


def _color(config: str) -> str:
    return CONFIG_COLORS.get(config, "#999999")


def _label(config: str) -> str:
    return CONFIG_LABELS.get(config, config)


def _add_bar_labels(ax, bars, fmt: str = ".3f", min_height: float = 0.005) -> None:
    for bar in bars:
        h = bar.get_height()
        if h > min_height:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.008,
                f"{h:{fmt}}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _plot_overall_f1(results: dict[str, dict[str, Any]], configs: list[str]):
    keys = ["si_f1", "tc_f1", "macro_f1"]
    labels = ["SI F1\n(span detection)", "TC F1\n(technique match)", "Macro F1\n(avg techniques)"]

    x = np.arange(len(labels))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cfg in enumerate(configs):
        vals = [results[cfg]["metrics"].get(k, 0.0) for k in keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=_label(cfg), color=_color(cfg),
        )
        _add_bar_labels(ax, bars)

    ax.set_title("Cross-Model Family: Overall Comparison (122 articles)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, title="detect → verify")
    fig.tight_layout()
    return fig


def _plot_recall_f1(results: dict[str, dict[str, Any]], configs: list[str]):
    keys = ["si_recall", "si_f1", "tc_recall", "tc_f1"]
    labels = ["SI Recall", "SI F1", "TC Recall", "TC F1"]

    x = np.arange(len(labels))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, cfg in enumerate(configs):
        vals = [results[cfg]["metrics"].get(k, 0.0) for k in keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=_label(cfg), color=_color(cfg),
        )
        _add_bar_labels(ax, bars)

    ax.set_title("Cross-Model Family: Recall & F1 Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, title="detect → verify")

    ax.axvline(x=1.5, color="grey", linestyle="--", alpha=0.4)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "Span Identification",
            ha="center", fontsize=9, color="grey")
    ax.text(2.5, ax.get_ylim()[1] * 0.97, "Technique Classification",
            ha="center", fontsize=9, color="grey")

    fig.tight_layout()
    return fig


def _plot_per_technique_f1(results: dict[str, dict[str, Any]], configs: list[str]):
    techniques: set[str] = set()
    for cfg in configs:
        techniques.update(results[cfg]["metrics"].get("per_technique", {}).keys())

    scored: list[tuple[str, float, int]] = []
    for t in techniques:
        best = 0.0
        gold = 0
        for cfg in configs:
            pt = results[cfg]["metrics"].get("per_technique", {}).get(t, {})
            best = max(best, float(pt.get("f1", 0.0)))
            gold = max(gold, int(pt.get("gold_count", 0)))
        scored.append((t, best, gold))
    scored.sort(key=lambda x: x[1], reverse=True)
    ordered = [t for t, _, _ in scored]
    gold_by_tech = {t: g for t, _, g in scored}

    x = np.arange(len(ordered))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(figsize=(18, 8))

    for i, cfg in enumerate(configs):
        per_tech = results[cfg]["metrics"].get("per_technique", {})
        vals = [float(per_tech.get(t, {}).get("f1", 0.0)) for t in ordered]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=_label(cfg), color=_color(cfg),
        )
        _add_bar_labels(ax, bars, fmt=".2f", min_height=0.01)

    ax.set_title("Cross-Model Family: Per-Technique F1 (122 articles)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, min(1.0, (ax.get_ylim()[1] or 0.1) * 1.15))
    ax.set_xticks(x)
    tech_labels = [f"{t}\n(n={gold_by_tech[t]})" for t in ordered]
    ax.set_xticklabels(tech_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right", title="detect → verify")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cross-model comparison charts (SI F1, TC F1, per-technique)."
    )
    parser.add_argument(
        "--comparison-json",
        type=str,
        default="outputs/results/crossmodel_family_consol_permissive.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output prefix (without extension). Defaults to next to the JSON.",
    )
    args = parser.parse_args()

    json_path = Path(args.comparison_json)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    results = payload["results"] if isinstance(payload.get("results"), dict) else payload
    results = {
        k: v for k, v in results.items()
        if isinstance(v, dict) and "metrics" in v and "error" not in v
    }
    if not results:
        print("No successful configs found in JSON.")
        return

    preferred = [c for c in CONFIG_COLORS if c in results]
    extras = [c for c in results if c not in CONFIG_COLORS]
    configs = preferred + extras

    out_prefix = Path(args.output_prefix) if args.output_prefix else json_path.with_suffix("")

    fig1 = _plot_overall_f1(results, configs)
    p1 = out_prefix.with_name(f"{out_prefix.name}_overall_f1.png")
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    fig2 = _plot_recall_f1(results, configs)
    p2 = out_prefix.with_name(f"{out_prefix.name}_recall_f1.png")
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    fig3 = _plot_per_technique_f1(results, configs)
    p3 = out_prefix.with_name(f"{out_prefix.name}_technique_f1.png")
    fig3.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig3)

    print("Charts written:")
    for p in (p1, p2, p3):
        print(f"  {p}")


if __name__ == "__main__":
    main()
