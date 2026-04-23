"""Final dissertation analysis: produce all eight items in one file.

Output: outputs/dissertation_analysis/all_findings.txt

Run: uv run python -m scripts.generate_analysis
Skip slow bootstrap items: uv run python -m scripts.generate_analysis --start-from 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation.postrun import (
    analyze_relabel_quality,
    classify_errors_multi,
    compute_bootstrap_cis,
    compute_per_pass_f1,
    pairwise_matrix_with_holm,
    plot_error_distribution,
    plot_per_pass_f1,
    plot_per_technique_stage_delta,
    plot_relabel_quality_matrix,
    stage_f1_corrected,
)

RESULTS = Path("outputs/results")
OUT_DIR = Path("outputs/dissertation_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "all_findings.txt"


def header(title: str) -> str:
    return f"\n{'='*70}\n{title}\n{'='*70}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=1,
                        help="Item number to start from (1-8). Use 3 to skip bootstrap items.")
    args = parser.parse_args()
    start_from = args.start_from

    strategies = ["zero_shot", "few_shot", "asv", "consol", "hybrid"]
    result_paths: list[Path] = []
    missing: list[str] = []
    for s in strategies:
        files = sorted(RESULTS.glob(f"{s}_gpt-4o_n*.json"))
        if files:
            result_paths.append(files[-1])
        else:
            missing.append(s)

    if missing:
        print(f"WARNING: missing results for: {missing}", file=sys.stderr)

    print(f"\n{'='*50}", flush=True)
    print(f"  DISSERTATION ANALYSIS — {len(result_paths)} strategies", flush=True)
    print(f"  {[p.stem for p in result_paths]}", flush=True)
    print(f"{'='*50}\n", flush=True)

    lines: list[str] = []
    lines.append("Dissertation analysis output")
    lines.append(f"Strategies analysed: {[p.stem for p in result_paths]}")

    # === Item 1: Bootstrap CIs (n=10000) ===
    if start_from <= 1:
        lines.append(header("ITEM 1: Bootstrap 95% CIs (n_bootstrap=10000)"))
        lines.append(
            f"{'Strategy':<12} {'Metric':<10} "
            f"{'Estimate':>10} {'CI_Lower':>10} {'CI_Upper':>10}"
        )
        lines.append("-" * 56)
        ci_total = len(result_paths) * 3
        ci_done = 0
        for p in result_paths:
            for metric in ["si_f1", "tc_f1", "macro_f1"]:
                ci_done += 1
                strat_name = p.stem.split("_gpt")[0]
                print(f"  [Item 1] Bootstrap CI {ci_done}/{ci_total}: "
                      f"{strat_name} {metric} ...", flush=True)
                try:
                    ci = compute_bootstrap_cis(p, metric=metric, n_bootstrap=10000)
                    lines.append(
                        f"{ci['strategy']:<12} {metric:<10} "
                        f"{ci['point_estimate']:>10.4f} "
                        f"{ci['ci_lower']:>10.4f} "
                        f"{ci['ci_upper']:>10.4f}"
                    )
                    print(f"           -> {ci['point_estimate']:.4f} "
                          f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]", flush=True)
                except Exception as e:
                    lines.append(f"{p.name} [{metric}]: ERROR — {e}")
                    print(f"           -> ERROR: {e}", flush=True)
        print(f"  [Item 1] Done.\n", flush=True)
    else:
        print(f"  [Item 1] SKIPPED (--start-from {start_from})\n", flush=True)

    # === Item 2: Pairwise Holm-corrected significance matrix ===
    if start_from <= 2:
        lines.append(header("ITEM 2: Pairwise Holm-corrected significance"))
        pw_total = 3
        pw_done = 0
        for metric in ["si_f1", "tc_f1", "macro_f1"]:
            pw_done += 1
            print(f"  [Item 2] Pairwise matrix {pw_done}/{pw_total}: "
                  f"{metric} (this is the slowest part) ...", flush=True)
            lines.append(f"\n--- {metric} ---")
            try:
                matrix = pairwise_matrix_with_holm(
                    result_paths, metric=metric, n_bootstrap=10000,
                )
                lines.append(
                    f"Family size after restriction to delta>0 directional pairs: "
                    f"{matrix['n_comparisons_corrected']}"
                )
                lines.append(
                    f"{'Comparison':<28} {'Delta':>10} {'Raw_p':>10} "
                    f"{'Holm_p':>10} {'Sig':>6}"
                )
                lines.append("-" * 70)
                sorted_keys = sorted(
                    matrix["raw_p_values"].keys(),
                    key=lambda k: matrix["raw_p_values"][k],
                )
                for key in sorted_keys:
                    delta = matrix["deltas"][key]
                    raw = matrix["raw_p_values"][key]
                    holm = matrix["holm_corrected_p"][key]
                    sig = "***" if matrix["significant_after_holm"][key] else ""
                    lines.append(
                        f"{key:<28} {delta:>+10.4f} {raw:>10.4f} "
                        f"{holm:>10.4f} {sig:>6}"
                    )
            except Exception as e:
                lines.append(f"ERROR for {metric}: {e}")
                print(f"           -> ERROR: {e}", flush=True)
        print(f"  [Item 2] Done.\n", flush=True)
    else:
        print(f"  [Item 2] SKIPPED (--start-from {start_from})\n", flush=True)

    # === Item 3: Per-pass F1 isolation on consol ===
    print(f"  [Item 3] Per-pass F1 isolation (consol) ...", flush=True)
    lines.append(header("ITEM 3: Per-pass F1 isolation (consol)"))
    consol_paths = [p for p in result_paths if "consol" in p.name]
    if consol_paths:
        consol_path = consol_paths[0]
        for metric in ["si_f1", "tc_f1"]:
            try:
                pp = compute_per_pass_f1(consol_path, metric=metric)
                lines.append(f"\nConsol {metric}:")
                for k, f in enumerate(pp["per_pass_f1"]):
                    lines.append(f"  Pass {k}: {f:.4f}")
                lines.append(f"  Merged:  {pp['merged_f1']:.4f}")
                best_single = max(pp["per_pass_f1"])
                lines.append(
                    f"  Merge gain over best single pass: "
                    f"{pp['merged_f1'] - best_single:+.4f}"
                )
            except Exception as e:
                lines.append(f"ERROR for {metric}: {e}")
    else:
        lines.append("No consol result file found")
    print(f"  [Item 3] Done.\n", flush=True)

    # === Item 4: Stage-by-stage F1 (corrected) ===
    print(f"  [Item 4] Stage-by-stage F1 (corrected) ...", flush=True)
    lines.append(header("ITEM 4: Stage-by-stage F1 (using SemEval official formula)"))
    for si, name in enumerate(["asv", "consol", "hybrid"], 1):
        print(f"    {si}/3: {name} ...", flush=True)
        paths = [p for p in result_paths if name in p.name]
        if not paths:
            lines.append(f"\n{name}: NO RESULT FILE")
            continue
        try:
            result = stage_f1_corrected(paths[0])
            lines.append(f"\n{name}:")
            lines.append(
                f"  {'Stage':<14} {'SI-F1':>8} {'SI-P':>8} {'SI-R':>8} "
                f"{'TC-F1':>8} {'TC-P':>8} {'TC-R':>8}"
            )
            for stage in ["after_s1", "after_s2", "after_s3"]:
                if stage in result:
                    s = result[stage]
                    lines.append(
                        f"  {stage:<14} "
                        f"{s['si_f1']:>8.4f} {s['si_precision']:>8.4f} "
                        f"{s['si_recall']:>8.4f} "
                        f"{s['tc_f1']:>8.4f} {s['tc_precision']:>8.4f} "
                        f"{s['tc_recall']:>8.4f}"
                    )
        except Exception as e:
            lines.append(f"\n{name}: ERROR — {e}")
            print(f"    {name}: ERROR — {e}", flush=True)
    print(f"  [Item 4] Done.\n", flush=True)

    # === Item 4b: Per-technique & macro F1 at each stage (consol) ===
    print(f"  [Item 4b] Per-technique stage F1 (consol) ...", flush=True)
    lines.append(header("ITEM 4b: Stage-by-stage per-technique F1 (consol)"))
    consol_paths_4b = [p for p in result_paths if "consol" in p.name]
    if consol_paths_4b:
        try:
            result_4b = stage_f1_corrected(consol_paths_4b[0])
            available_stages = [
                st for st in ["after_s1", "after_s2", "after_s3"]
                if st in result_4b
            ]

            lines.append(f"\n  Macro F1 per stage:")
            stage_hdr = f"  {'Stage':<15} {'SI-F1':>8} {'TC-F1':>8} {'Macro-F1':>10}"
            lines.append(stage_hdr)
            lines.append(f"  {'-'*43}")
            for st in available_stages:
                sd = result_4b[st]
                lines.append(
                    f"  {st:<15} {sd['si_f1']:>8.4f} "
                    f"{sd['tc_f1']:>8.4f} {sd['macro_f1']:>10.4f}"
                )

            techniques = sorted(
                result_4b[available_stages[0]].get("per_technique", {}).keys()
            )
            if techniques:
                lines.append(f"\n  Per-technique TC-F1 by stage:")
                hdr = f"  {'Technique':<38}"
                for st in available_stages:
                    hdr += f" {st:>10}"
                hdr += f" {'delta':>8}"
                lines.append(hdr)
                lines.append(f"  {'-' * (38 + 11 * len(available_stages) + 9)}")

                first_stage = available_stages[0]
                final_stage = available_stages[-1]
                for t in techniques:
                    row = f"  {t:<38}"
                    s1_f1 = (
                        result_4b[first_stage]
                        .get("per_technique", {})
                        .get(t, {})
                        .get("f1", 0.0)
                    )
                    for st in available_stages:
                        f1 = (
                            result_4b[st]
                            .get("per_technique", {})
                            .get(t, {})
                            .get("f1", 0.0)
                        )
                        row += f" {f1:>10.4f}"
                    final_f1 = (
                        result_4b[final_stage]
                        .get("per_technique", {})
                        .get(t, {})
                        .get("f1", 0.0)
                    )
                    delta = final_f1 - s1_f1
                    row += f" {delta:>+8.4f}"
                    lines.append(row)

            print(f"  [Item 4b] Done.\n", flush=True)
        except Exception as e:
            lines.append(f"  ERROR — {e}")
            print(f"  [Item 4b] ERROR — {e}\n", flush=True)
    else:
        lines.append("  No consol result file found")
        print(f"  [Item 4b] No consol file.\n", flush=True)

    # === Item 5: Drop quality on consol (read from saved diagnostics) ===
    print(f"  [Item 5] Drop quality (consol) ...", flush=True)
    lines.append(header("ITEM 5: Drop quality (consol, from saved diagnostics)"))
    if consol_paths:
        with open(consol_paths[0]) as f:
            data = json.load(f)
        dq = data.get("diagnostics", {}).get("drop_quality", {})
        if dq:
            total = dq.get("total_dropped", 0)
            false_drops = dq.get("false_drops", 0)
            false_rate = dq.get("false_drop_rate", 0)
            lines.append(f"  Total dropped: {total}")
            lines.append(f"  False drops (matched gold): {false_drops}")
            lines.append(f"  False drop rate: {false_rate * 100:.1f}%")
            lines.append(f"  Correct drop rate: {(1 - false_rate) * 100:.1f}%")
            per_tech = dq.get("per_technique_dropped", {})
            if per_tech:
                lines.append(f"  Per-technique dropped: {per_tech}")
            per_tech_false = dq.get("per_technique_false_drops", {})
            if per_tech_false:
                lines.append(f"  Per-technique false drops: {per_tech_false}")
        else:
            lines.append("  No drop_quality in saved diagnostics")
    else:
        lines.append("No consol result file found")
    print(f"  [Item 5] Done.\n", flush=True)

    # === Item 5b: Relabel quality analysis ===
    print(f"  [Item 5b] Relabel quality (consol, hybrid) ...", flush=True)
    lines.append(header("ITEM 5b: Relabel quality (consol, hybrid)"))
    for name in ["consol", "hybrid"]:
        print(f"    {name} ...", flush=True)
        paths = [p for p in result_paths if name in p.name]
        if not paths:
            lines.append(f"\n{name}: NO RESULT FILE")
            continue
        try:
            analysis = analyze_relabel_quality(paths[0])

            if analysis["total_relabels"] == 0:
                lines.append(f"\n{name}: no relabels in this run")
                continue

            lines.append(f"\n{name}:")
            lines.append(f"  Total relabels: {analysis['total_relabels']}")
            lines.append(
                f"  Improvements (was wrong, now right): "
                f"{analysis['improvements']} ({analysis['improvement_rate']*100:.1f}%)"
            )
            lines.append(
                f"  Regressions (was right, now wrong): "
                f"{analysis['regressions']} ({analysis['regression_rate']*100:.1f}%)"
            )
            lines.append(f"  Lateral (both correct): {analysis['lateral_correct']}")
            lines.append(f"  Lateral (both wrong): {analysis['lateral_wrong']}")
            lines.append(
                f"  NET IMPROVEMENT: {analysis['net_improvement']:+d} "
                f"({analysis['net_improvement_rate']*100:+.1f}%)"
            )

            lines.append(f"\n  Top 10 relabel directions (by count):")
            by_dir = analysis["by_direction"]
            sorted_dirs = sorted(
                by_dir.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:10]
            for direction, data in sorted_dirs:
                improvs = data["improvements"]
                regs = data["regressions"]
                net = improvs - regs
                lines.append(
                    f"    {direction:<50} count={data['count']:>3}  "
                    f"improved={improvs:>2} regressed={regs:>2}  "
                    f"net={net:+d}"
                )

            lines.append(f"\n  Per-technique relabel impact:")
            per_orig = analysis["per_original_technique"]
            sorted_orig = sorted(
                per_orig.items(),
                key=lambda x: x[1]["net_improvement"],
                reverse=True,
            )
            for orig, odata in sorted_orig:
                if odata["total_relabels_from"] == 0:
                    continue
                most_common = odata["most_common_new_label"] or "-"
                lines.append(
                    f"    {orig:<32} "
                    f"relabels={odata['total_relabels_from']:>3}  "
                    f"improv={odata['improvements']:>2}  "
                    f"regress={odata['regressions']:>2}  "
                    f"net={odata['net_improvement']:+d}  "
                    f"most_common_target={most_common}"
                )

            plot_path = OUT_DIR / f"relabel_matrix_{name}.png"
            plot_info = plot_relabel_quality_matrix(paths[0], plot_path)
            lines.append(f"\n  Plot saved: {plot_info['output_path']}")
            print(f"    {name} done — {analysis['total_relabels']} relabels, "
                  f"net={analysis['net_improvement']:+d}", flush=True)

        except Exception as e:
            lines.append(f"\n{name}: ERROR — {e}")
            import traceback
            lines.append(traceback.format_exc())
            print(f"    {name}: ERROR — {e}", flush=True)
    print(f"  [Item 5b] Done.\n", flush=True)

    # === Item 7: Six-category error classification ===
    print(f"  [Item 7] Error classification ...", flush=True)
    lines.append(header("ITEM 7: Six-category error classification"))
    error_targets = [
        p for p in result_paths
        if any(s in p.name for s in ["zero_shot", "asv", "consol", "hybrid"])
    ]
    try:
        errors = classify_errors_multi(error_targets)
        for strategy, data in errors["per_strategy"].items():
            lines.append(f"\n{strategy}: {data['total']} events")
            categories = [
                "correct", "missed_detection", "false_alarm",
                "wrong_technique", "boundary_too_broad",
                "boundary_too_narrow", "boundary_offset",
            ]
            for cat in categories:
                count = data["counts"][cat]
                pct = data["percentages"][cat]
                lines.append(f"  {cat:<22} {count:>5}  ({pct:>5.1f}%)")
    except Exception as e:
        lines.append(f"ERROR: {e}")
    print(f"  [Item 7] Done.\n", flush=True)

    # === Item 8: Refiner disobedience (hybrid) ===
    print(f"  [Item 8] Refiner disobedience (hybrid) ...", flush=True)
    lines.append(header("ITEM 8: Refiner disobedience (hybrid)"))
    hybrid_paths = [p for p in result_paths if "hybrid" in p.name]
    if hybrid_paths:
        with open(hybrid_paths[0]) as f:
            data = json.load(f)
        rd = data.get("diagnostics", {}).get("refiner_disobedience", {})
        if rd:
            lines.append(
                f"  Total disobey events: {rd.get('total_disobey_events', 0)}"
            )
            lines.append(
                f"  Articles with disobey: {rd.get('articles_with_disobey', 0)}"
            )
            lines.append(
                f"  Total articles: {rd.get('total_articles', 'unknown')}"
            )
            lines.append(
                f"  Disobey rate per article: "
                f"{rd.get('disobey_rate_per_article', 0):.4f}"
            )
        else:
            lines.append("  No refiner_disobedience in saved diagnostics")
    else:
        lines.append("No hybrid result file found")
    print(f"  [Item 8] Done.\n", flush=True)

    # === Item 9: Per-technique F1 / P / R table across all strategies ===
    print(f"  [Item 9] Per-technique F1/P/R table ...", flush=True)
    lines.append(header("ITEM 9: Per-technique F1 across all strategies"))

    strat_names: list[str] = []
    per_tech_by_strat: dict[str, dict[str, dict[str, float]]] = {}
    for p in result_paths:
        with open(p) as f:
            data = json.load(f)
        sname = data["config"]["strategy"]
        strat_names.append(sname)
        per_tech_by_strat[sname] = data.get("metrics", {}).get("per_technique", {})

    from src.schemas import Technique
    techniques = [t.value for t in Technique]

    col_w = 10
    strat_hdr = "".join(f" {s:>{col_w}}" for s in strat_names)

    lines.append(f"\n  TC-F1 per technique:")
    lines.append(f"  {'Technique':<38}{strat_hdr}")
    lines.append(f"  {'-' * (38 + (col_w + 1) * len(strat_names))}")
    for t in techniques:
        row = f"  {t:<38}"
        for s in strat_names:
            f1 = per_tech_by_strat[s].get(t, {}).get("f1", 0.0)
            row += f" {f1:>{col_w}.4f}"
        lines.append(row)

    lines.append(f"\n  Precision per technique:")
    lines.append(f"  {'Technique':<38}{strat_hdr}")
    lines.append(f"  {'-' * (38 + (col_w + 1) * len(strat_names))}")
    for t in techniques:
        row = f"  {t:<38}"
        for s in strat_names:
            p_val = per_tech_by_strat[s].get(t, {}).get("precision", 0.0)
            row += f" {p_val:>{col_w}.4f}"
        lines.append(row)

    lines.append(f"\n  Recall per technique:")
    lines.append(f"  {'Technique':<38}{strat_hdr}")
    lines.append(f"  {'-' * (38 + (col_w + 1) * len(strat_names))}")
    for t in techniques:
        row = f"  {t:<38}"
        for s in strat_names:
            r_val = per_tech_by_strat[s].get(t, {}).get("recall", 0.0)
            row += f" {r_val:>{col_w}.4f}"
        lines.append(row)

    lines.append(f"\n  Gold/Pred counts per technique:")
    lines.append(
        f"  {'Technique':<38} {'Gold':>6}"
        + "".join(f" {'pred_' + s:>{col_w}}" for s in strat_names)
    )
    lines.append(f"  {'-' * (38 + 7 + (col_w + 1) * len(strat_names))}")
    for t in techniques:
        gold = 0
        row_parts = []
        for s in strat_names:
            td = per_tech_by_strat[s].get(t, {})
            gold = max(gold, td.get("gold_count", 0))
            row_parts.append(f" {td.get('pred_count', 0):>{col_w}}")
        lines.append(f"  {t:<38} {gold:>6}{''.join(row_parts)}")

    print(f"  [Item 9] Done.\n", flush=True)

    # === Item 10: Dissertation figures ===
    print(f"  [Item 10] Generating dissertation figures ...", flush=True)
    lines.append(header("ITEM 10: Dissertation figures"))

    # Figure: Error distribution (stacked percentage)
    print(f"    Error distribution ...", flush=True)
    error_targets = [
        p for p in result_paths
        if any(s in p.name for s in ["zero_shot", "asv", "consol", "hybrid"])
    ]
    try:
        fig_path = OUT_DIR / "fig_error_distribution.png"
        info = plot_error_distribution(
            error_targets, fig_path,
            mode="stacked_percentage",
            title="Six-category error distribution (% of events per strategy)",
        )
        lines.append(f"  Error distribution: {info['output_path']}")
    except Exception as e:
        lines.append(f"  Error distribution FAILED: {e}")
        print(f"    ERROR: {e}", flush=True)

    # Figure: Per-pass F1 for consol
    print(f"    Per-pass F1 (consol) ...", flush=True)
    consol_paths_fig = [p for p in result_paths if "consol" in p.name]
    if consol_paths_fig:
        try:
            fig_path = OUT_DIR / "fig_per_pass_f1_consol.png"
            info = plot_per_pass_f1(consol_paths_fig[0], fig_path)
            lines.append(f"  Per-pass F1 (consol): {info['output_path']}")
            lines.append(
                f"    Merged gain over best single pass: "
                f"SI-F1 {info['si_gain_over_best_pass']:+.4f}, "
                f"TC-F1 {info['tc_gain_over_best_pass']:+.4f}"
            )
        except Exception as e:
            lines.append(f"  Per-pass F1 (consol) FAILED: {e}")
            print(f"    ERROR: {e}", flush=True)

    # Figure: Per-technique S1->S2 delta for consol
    print(f"    Per-technique delta (consol s1->s2) ...", flush=True)
    if consol_paths_fig:
        try:
            fig_path = OUT_DIR / "fig_per_technique_delta_consol.png"
            info = plot_per_technique_stage_delta(
                consol_paths_fig[0], fig_path,
                metric="f1", from_stage="after_s1", to_stage="after_s2",
                title="Per-technique TC-F1 change (Stage 1 \u2192 Stage 2): consol",
            )
            lines.append(f"  Per-technique delta (consol): {info['output_path']}")
            lines.append(
                f"    {info['n_improved']} techniques improved, "
                f"{info['n_worsened']} worsened, "
                f"{info['n_unchanged']} unchanged"
            )
        except Exception as e:
            lines.append(f"  Per-technique delta (consol) FAILED: {e}")
            print(f"    ERROR: {e}", flush=True)

    # Appendix figures: per-pass F1 and per-technique delta for asv/hybrid
    for name in ["asv", "hybrid"]:
        app_paths = [p for p in result_paths if name in p.name]
        if not app_paths:
            continue
        print(f"    Appendix figures ({name}) ...", flush=True)
        try:
            fig_path = OUT_DIR / f"fig_per_pass_f1_{name}.png"
            info = plot_per_pass_f1(app_paths[0], fig_path)
            lines.append(f"  Per-pass F1 ({name}): {info['output_path']}")
        except Exception as e:
            lines.append(f"  Per-pass F1 ({name}) FAILED: {e}")
        try:
            fig_path = OUT_DIR / f"fig_per_technique_delta_{name}.png"
            info = plot_per_technique_stage_delta(
                app_paths[0], fig_path,
                from_stage="after_s1", to_stage="after_s2",
            )
            lines.append(f"  Per-technique delta ({name}): {info['output_path']}")
        except Exception as e:
            lines.append(f"  Per-technique delta ({name}) FAILED: {e}")

    print(f"  [Item 10] Done.\n", flush=True)

    lines.append(f"\n{'='*70}")
    lines.append("Analysis complete.")
    lines.append("=" * 70)

    output = "\n".join(lines)
    print(output)
    OUT_FILE.write_text(output, encoding="utf-8")
    print(f"\n[Saved to {OUT_FILE}]")


if __name__ == "__main__":
    main()
