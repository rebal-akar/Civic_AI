"""Chart generation for strategy comparison and diagnostic outputs."""
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

    for label, plot_fn in (
        ("overall_f1", lambda: _plot_overall_f1(successful, strategies, num_articles)),
        ("recall_f1", lambda: _plot_recall_f1(successful, strategies)),
        ("technique_f1", lambda: _plot_per_technique_f1(successful, strategies, num_articles)),
    ):
        fig = plot_fn()
        path = output_prefix.with_name(f"{output_prefix.name}_{label}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def _plot_overall_f1(
    results_by_strategy: dict[str, dict[str, Any]],
    strategies: list[str],
    num_articles: int | None = None,
):
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
            pt = results_by_strategy[s]["metrics"].get("per_technique", {}).get(t, {})
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


_XFAM_STYLE: dict[str, tuple[str, str]] = {
    "same_4o":               ("#4C72B0", "gpt-4o -> gpt-4o"),
    "same_sonnet":           ("#55A868", "Sonnet -> Sonnet"),
    "cascade_4o_to_sonnet":  ("#C44E52", "gpt-4o -> Sonnet"),
    "cascade_sonnet_to_4o":  ("#8172B2", "Sonnet -> gpt-4o"),
}


def _xfam_key(cfg: str) -> str | None:
    """Strip the `<strategy>_` or `<strategy>_xfam_` prefix off a config key."""
    for canonical in _XFAM_STYLE:
        if cfg.endswith(canonical):
            return canonical
    return None


def _xfam_color(cfg: str) -> str:
    k = _xfam_key(cfg)
    return _XFAM_STYLE[k][0] if k else "#999999"


def _xfam_label(cfg: str) -> str:
    k = _xfam_key(cfg)
    return _XFAM_STYLE[k][1] if k else cfg


def generate_crossmodel_charts(
    results_by_config: dict[str, dict[str, Any]],
    output_prefix: Path,
    num_articles: int | None = None,
) -> list[Path]:
    """Overall / recall-F1 / per-technique bar charts for a cross-family run."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    successful = {
        c: r for c, r in results_by_config.items()
        if isinstance(r, dict) and "error" not in r and isinstance(r.get("metrics"), dict)
    }
    if not successful:
        return []

    canonical_order = list(_XFAM_STYLE.keys())
    preferred = sorted(
        [c for c in successful if _xfam_key(c) is not None],
        key=lambda c: canonical_order.index(_xfam_key(c)),
    )
    extras = [c for c in successful if _xfam_key(c) is None]
    configs = preferred + extras

    written: list[Path] = []
    for label, plot_fn in (
        ("overall_f1", lambda: _plot_crossmodel_overall_f1(successful, configs, num_articles)),
        ("recall_f1",  lambda: _plot_crossmodel_recall_f1(successful, configs)),
        ("technique_f1", lambda: _plot_crossmodel_per_technique(successful, configs, num_articles)),
    ):
        fig = plot_fn()
        path = output_prefix.with_name(f"{output_prefix.name}_{label}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def _plot_crossmodel_overall_f1(
    results: dict[str, dict[str, Any]],
    configs: list[str],
    num_articles: int | None = None,
):
    keys = ["si_f1", "tc_f1", "macro_f1"]
    labels = ["SI F1\n(span detection)", "TC F1\n(technique match)", "Macro F1\n(avg techniques)"]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cfg in enumerate(configs):
        vals = [results[cfg]["metrics"].get(k, 0.0) for k in keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=_xfam_label(cfg), color=_xfam_color(cfg),
        )
        _add_bar_labels(ax, bars)

    suffix = f" ({num_articles} articles)" if num_articles else ""
    ax.set_title(f"Cross-Model Family: Overall Comparison{suffix}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, title="detect -> verify")
    fig.tight_layout()
    return fig


def _plot_crossmodel_recall_f1(
    results: dict[str, dict[str, Any]],
    configs: list[str],
):
    keys = ["si_recall", "si_f1", "tc_recall", "tc_f1"]
    labels = ["SI Recall", "SI F1", "TC Recall", "TC F1"]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(configs))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, cfg in enumerate(configs):
        vals = [results[cfg]["metrics"].get(k, 0.0) for k in keys]
        bars = ax.bar(
            x - 0.4 + (i + 0.5) * width, vals,
            width=width, label=_xfam_label(cfg), color=_xfam_color(cfg),
        )
        _add_bar_labels(ax, bars)

    ax.set_title("Cross-Model Family: Recall & F1 Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, title="detect -> verify")

    ax.axvline(x=1.5, color="grey", linestyle="--", alpha=0.4)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "Span Identification",
            ha="center", fontsize=9, color="grey")
    ax.text(2.5, ax.get_ylim()[1] * 0.97, "Technique Classification",
            ha="center", fontsize=9, color="grey")

    fig.tight_layout()
    return fig


def _plot_crossmodel_per_technique(
    results: dict[str, dict[str, Any]],
    configs: list[str],
    num_articles: int | None = None,
):
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
            width=width, label=_xfam_label(cfg), color=_xfam_color(cfg),
        )
        _add_bar_labels(ax, bars, fmt=".2f")

    suffix = f" ({num_articles} articles)" if num_articles else ""
    ax.set_title(f"Cross-Model Family: Per-Technique F1{suffix}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, min(1.0, (ax.get_ylim()[1] or 0.1) * 1.15))
    ax.set_xticks(x)
    tech_labels = [f"{t}\n(n={gold_by_tech[t]})" for t in ordered]
    ax.set_xticklabels(tech_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right", title="detect -> verify")
    fig.tight_layout()
    return fig


def generate_diagnostic_charts(
    diagnostics: dict[str, Any],
    output_prefix: Path,
    strategy_label: str = "",
) -> list[Path]:
    """Stage-by-stage F1 chart from compute_all_diagnostics output."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    stages = diagnostics.get("stage_by_stage_f1", {})
    if not stages:
        return paths

    fig, ax = plt.subplots(figsize=(8, 5))
    stage_names, si_vals, tc_vals = [], [], []
    for stage in ("after_s1", "after_s2", "after_s3"):
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
        path = Path(f"{output_prefix}_stage_f1.png")
        fig.savefig(path, dpi=150)
        paths.append(path)
    plt.close(fig)

    return paths


def _load_results_payload(path: Path) -> tuple[dict[str, dict], int | None]:
    """Accept both `{"results": {...}, "num_articles": N}` and flat shapes."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("results"), dict):
        block = payload["results"]
        num = payload.get("num_articles")
    else:
        block = payload
        num = None

    flat: dict[str, dict] = {}
    inferred: int | None = None
    for key, entry in block.items():
        if not isinstance(entry, dict) or "metrics" not in entry or "error" in entry:
            continue
        flat[key] = {"metrics": entry["metrics"]}
        processed = entry.get("articles_processed") or entry.get("articles_scored")
        if isinstance(processed, int):
            inferred = max(inferred or 0, processed)

    return flat, num or inferred


def main() -> None:
    """CLI: regenerate comparison or cross-family charts from a saved JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate F1 bar charts from a saved results JSON "
                    "(no API calls)."
    )
    parser.add_argument(
        "--mode",
        choices=["comparison", "crossmodel"],
        required=True,
        help="'comparison' for strategy bars (run_all output); "
             "'crossmodel' for detect->verify family bars (run_crossmodel output).",
    )
    parser.add_argument("--comparison-json", required=True, type=Path,
                        help="Path to the saved results JSON.")
    parser.add_argument("--output-prefix", required=True, type=Path,
                        help="Chart files will be named "
                             "<prefix>_overall_f1.png, <prefix>_recall_f1.png, "
                             "<prefix>_technique_f1.png.")
    parser.add_argument("--num-articles", type=int, default=None,
                        help="Override article count shown in chart titles.")
    args = parser.parse_args()

    results, inferred_n = _load_results_payload(args.comparison_json)
    if not results:
        raise SystemExit(f"No usable entries in {args.comparison_json}")

    n = args.num_articles or inferred_n
    if args.mode == "comparison":
        written = generate_comparison_charts(results, args.output_prefix, n)
    else:
        written = generate_crossmodel_charts(results, args.output_prefix, n)

    print("Charts written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()