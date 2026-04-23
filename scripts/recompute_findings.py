"""
Recompute items 3-8 of the dissertation analysis against fresh gold labels.

Pipeline:
  1. For each per-strategy result JSON under outputs/results/, rebuild
     Article + Prediction objects, swap in gold spans loaded from the RAW
     SemEval dataset (so Whataboutism,Straw_Men,Red_Herring is recognised),
     recompute `metrics` via `evaluate()` and `diagnostics` via
     `compute_all_diagnostics()`, and save a copy to
     outputs/results_rescored/.
  2. Run items 3, 4, 4b, 5, 5b, 6, 7, 8 of the dissertation analysis (the
     fast ones) against those rescored JSONs and write the output to
     outputs/dissertation_analysis/recomputed_all_findings.txt.

Items 1 (bootstrap CIs) and 2 (Holm-corrected pairwise matrix) are slow —
regenerate them separately with `generate_analysis.py` if you need them.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.data.loader import load_semeval
from src.evaluation.diagnostics import compute_all_diagnostics
from src.evaluation.metrics import evaluate
from src.evaluation.postrun import (
    analyze_relabel_quality,
    classify_errors_multi,
    compute_per_pass_f1,
    plot_relabel_quality_matrix,
    point_biserial_agreement,
    stage_f1_corrected,
)
from src.schemas import Article, GoldSpan, PredictedSpan, Prediction, normalise_technique

STRATEGIES = ["zero_shot", "few_shot", "asv", "consol", "hybrid"]


# ─── Rescore per-strategy JSON with fresh gold ─────────────────────────────


def _rebuild_predicted_span(raw: dict) -> PredictedSpan | None:
    technique = normalise_technique(raw.get("technique", ""))
    if technique is None:
        return None
    original_technique = None
    if raw.get("original_technique"):
        original_technique = normalise_technique(raw["original_technique"])
    return PredictedSpan(
        technique=technique,
        span_text=raw.get("span_text", ""),
        reasoning=raw.get("reasoning", ""),
        start=raw.get("start", -1),
        end=raw.get("end", -1),
        pass_id=raw.get("pass_id", 0),
        agreement_count=raw.get("agreement_count", 1),
        confidence=raw.get("confidence", -1.0),
        verdict=raw.get("verdict", ""),
        original_technique=original_technique,
        original_span_text=raw.get("original_span_text", ""),
        original_start=raw.get("original_start", -1),
        original_end=raw.get("original_end", -1),
        was_relabeled=raw.get("was_relabeled", False),
        was_trimmed=raw.get("was_trimmed", False),
        consol_action=raw.get("consol_action", ""),
        stage_history=raw.get("stage_history", []),
    )


def _build_gold_lookup(
    articles_dir: Path, labels_path: Path,
) -> dict[str, list[GoldSpan]]:
    articles_all = load_semeval(articles_dir, labels_path)
    return {a.id: a.gold_spans for a in articles_all}


def rescore_result_file(
    result_path: Path,
    gold_lookup: dict[str, list[GoldSpan]],
    output_path: Path,
) -> dict[str, Any]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    eval_mode = data.get("config", {}).get("eval_mode", "permissive")
    preds_raw: dict[str, dict] = data["predictions"]

    articles: list[Article] = []
    predictions: dict[str, Prediction] = {}
    missing_gold: list[str] = []

    for aid, pred_dict in preds_raw.items():
        fresh_gold = gold_lookup.get(aid)
        if fresh_gold is None:
            missing_gold.append(aid)
            fresh_gold = [
                GoldSpan(
                    technique=normalise_technique(g["technique"]),
                    start=g["start"],
                    end=g["end"],
                )
                for g in pred_dict.get("gold_spans", [])
                if normalise_technique(g["technique"]) is not None
            ]

        articles.append(Article(
            id=aid,
            text=pred_dict.get("article_text", ""),
            gold_spans=fresh_gold,
        ))

        spans: list[PredictedSpan] = []
        for raw_span in pred_dict.get("all_spans", []):
            s = _rebuild_predicted_span(raw_span)
            if s is not None:
                spans.append(s)
        predictions[aid] = Prediction(
            article_id=aid, spans=spans, eval_mode=eval_mode,
        )

        pred_dict["gold_spans"] = [
            {"technique": g.technique.value, "start": g.start, "end": g.end}
            for g in fresh_gold
        ]

    metrics = evaluate(articles, predictions)
    diagnostics = compute_all_diagnostics(articles, predictions)

    data["metrics"] = asdict(metrics)
    data["diagnostics"] = diagnostics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, default=str))

    return {
        "strategy": data.get("config", {}).get("strategy", ""),
        "articles": len(articles),
        "missing_gold": missing_gold,
        "si_f1": metrics.si_f1,
        "tc_f1": metrics.tc_f1,
        "macro_f1": metrics.macro_f1,
    }


# ─── Findings writer ───────────────────────────────────────────────────────


def _header(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


def build_findings(
    result_paths: list[Path],
    out_dir: Path,
) -> list[str]:
    lines: list[str] = []
    lines.append("Recomputed dissertation analysis (items 3-8)")
    lines.append(f"Strategies analysed: {[p.stem for p in result_paths]}")

    consol_paths = [p for p in result_paths if "consol" in p.name]
    hybrid_paths = [p for p in result_paths if "hybrid" in p.name]

    # Item 3
    lines.append(_header("ITEM 3: Per-pass F1 isolation (consol)"))
    if consol_paths:
        for metric in ["si_f1", "tc_f1"]:
            try:
                pp = compute_per_pass_f1(consol_paths[0], metric=metric)
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

    # Item 4
    lines.append(_header("ITEM 4: Stage-by-stage F1 (using SemEval official formula)"))
    for name in ["asv", "consol", "hybrid"]:
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
            lines.append(f"\n{name}: ERROR - {e}")

    # Item 4b
    lines.append(_header("ITEM 4b: Stage-by-stage per-technique F1 (consol)"))
    if consol_paths:
        try:
            result_4b = stage_f1_corrected(consol_paths[0])
            available = [
                st for st in ["after_s1", "after_s2", "after_s3"]
                if st in result_4b
            ]
            lines.append("\n  Macro F1 per stage:")
            lines.append(
                f"  {'Stage':<15} {'SI-F1':>8} {'TC-F1':>8} {'Macro-F1':>10}"
            )
            lines.append(f"  {'-' * 43}")
            for st in available:
                sd = result_4b[st]
                lines.append(
                    f"  {st:<15} {sd['si_f1']:>8.4f} "
                    f"{sd['tc_f1']:>8.4f} {sd['macro_f1']:>10.4f}"
                )

            techniques = sorted(
                result_4b[available[0]].get("per_technique", {}).keys()
            )
            if techniques:
                lines.append("\n  Per-technique TC-F1 by stage:")
                hdr = f"  {'Technique':<38}"
                for st in available:
                    hdr += f" {st:>10}"
                hdr += f" {'delta':>8}"
                lines.append(hdr)
                lines.append(f"  {'-' * (38 + 11 * len(available) + 9)}")
                first_stage, last_stage = available[0], available[-1]
                for t in techniques:
                    row = f"  {t:<38}"
                    for st in available:
                        f1 = (
                            result_4b[st]
                            .get("per_technique", {})
                            .get(t, {})
                            .get("f1", 0.0)
                        )
                        row += f" {f1:>10.4f}"
                    delta = (
                        result_4b[last_stage].get("per_technique", {})
                        .get(t, {}).get("f1", 0.0)
                        - result_4b[first_stage].get("per_technique", {})
                        .get(t, {}).get("f1", 0.0)
                    )
                    row += f" {delta:>+8.4f}"
                    lines.append(row)
        except Exception as e:
            lines.append(f"  ERROR - {e}")
    else:
        lines.append("  No consol result file found")

    # Item 5
    lines.append(_header("ITEM 5: Drop quality (consol, from recomputed diagnostics)"))
    if consol_paths:
        data = json.loads(consol_paths[0].read_text(encoding="utf-8"))
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
            lines.append("  No drop_quality in recomputed diagnostics")
    else:
        lines.append("No consol result file found")

    # Item 5b
    lines.append(_header("ITEM 5b: Relabel quality (consol, hybrid)"))
    for name in ["consol", "hybrid"]:
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
                f"{analysis['improvements']} "
                f"({analysis['improvement_rate']*100:.1f}%)"
            )
            lines.append(
                f"  Regressions (was right, now wrong): "
                f"{analysis['regressions']} "
                f"({analysis['regression_rate']*100:.1f}%)"
            )
            lines.append(f"  Lateral (both correct): {analysis['lateral_correct']}")
            lines.append(f"  Lateral (both wrong): {analysis['lateral_wrong']}")
            lines.append(
                f"  NET IMPROVEMENT: {analysis['net_improvement']:+d} "
                f"({analysis['net_improvement_rate']*100:+.1f}%)"
            )

            lines.append("\n  Top 10 relabel directions (by count):")
            sorted_dirs = sorted(
                analysis["by_direction"].items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:10]
            for direction, dinfo in sorted_dirs:
                improvs = dinfo["improvements"]
                regs = dinfo["regressions"]
                lines.append(
                    f"    {direction:<50} count={dinfo['count']:>3}  "
                    f"improved={improvs:>2} regressed={regs:>2}  "
                    f"net={improvs - regs:+d}"
                )

            lines.append("\n  Per-technique relabel impact:")
            per_orig = analysis["per_original_technique"]
            for orig, odata in sorted(
                per_orig.items(),
                key=lambda kv: kv[1]["net_improvement"],
                reverse=True,
            ):
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

            plot_path = out_dir / f"relabel_matrix_{name}_rescored.png"
            try:
                info = plot_relabel_quality_matrix(paths[0], plot_path)
                lines.append(f"\n  Plot saved: {info['output_path']}")
            except Exception as plot_err:
                lines.append(f"\n  Plot FAILED: {plot_err}")

        except Exception as e:
            lines.append(f"\n{name}: ERROR - {e}")

    # Item 6
    lines.append(_header("ITEM 6: Point-biserial correlation (agreement vs correctness)"))
    lines.append(f"{'Strategy':<12} {'r_pb':>10} {'p_value':>10} {'n_spans':>10}")
    lines.append("-" * 44)
    for name in ["asv", "consol", "hybrid"]:
        paths = [p for p in result_paths if name in p.name]
        if not paths:
            continue
        try:
            pb = point_biserial_agreement(paths[0])
            if pb["r_pb"] is not None:
                lines.append(
                    f"{pb['strategy']:<12} "
                    f"{pb['r_pb']:>+10.4f} "
                    f"{pb['p_value']:>10.4f} "
                    f"{pb['n_spans']:>10}"
                )
            else:
                lines.append(
                    f"{name}: insufficient data ({pb.get('note', 'N/A')})"
                )
        except Exception as e:
            lines.append(f"{name}: ERROR - {e}")

    # Item 7
    lines.append(_header("ITEM 7: Six-category error classification"))
    error_targets = [
        p for p in result_paths
        if any(s in p.name for s in ["zero_shot", "asv", "consol", "hybrid"])
    ]
    try:
        errors = classify_errors_multi(error_targets)
        for strategy, d in errors["per_strategy"].items():
            lines.append(f"\n{strategy}: {d['total']} events")
            categories = [
                "correct", "missed_detection", "false_alarm",
                "wrong_technique", "boundary_too_broad",
                "boundary_too_narrow", "boundary_offset",
            ]
            for cat in categories:
                lines.append(
                    f"  {cat:<22} {d['counts'][cat]:>5}  "
                    f"({d['percentages'][cat]:>5.1f}%)"
                )
    except Exception as e:
        lines.append(f"ERROR: {e}")

    # Item 8
    lines.append(_header("ITEM 8: Refiner disobedience (hybrid)"))
    if hybrid_paths:
        data = json.loads(hybrid_paths[0].read_text(encoding="utf-8"))
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
            lines.append("  No refiner_disobedience in recomputed diagnostics")
    else:
        lines.append("No hybrid result file found")

    lines.append(f"\n{'=' * 70}")
    lines.append("Recomputation complete.")
    lines.append("=" * 70)
    return lines


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--results-dir", type=Path, default=Path("outputs/results"),
    )
    parser.add_argument(
        "--rescored-dir", type=Path, default=Path("outputs/results_rescored"),
    )
    parser.add_argument(
        "--articles-dir", type=Path, default=Path("datasets/train-articles"),
    )
    parser.add_argument(
        "--labels-path", type=Path, default=Path("datasets/train-task2-TC.labels"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dissertation_analysis/recomputed_all_findings.txt"),
    )
    parser.add_argument(
        "--skip-rescore",
        action="store_true",
        help="Reuse existing rescored JSONs in --rescored-dir.",
    )
    args = parser.parse_args()

    args.rescored_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rescored_paths: list[Path] = []

    if not args.skip_rescore:
        print("Loading raw gold lookup ...")
        gold_lookup = _build_gold_lookup(args.articles_dir, args.labels_path)

    for strategy in STRATEGIES:
        matches = sorted(args.results_dir.glob(f"{strategy}_gpt-4o_n*.json"))
        matches = [m for m in matches if "_rescored" not in m.name]
        if not matches:
            print(f"  [{strategy}] no result file, skipping")
            continue
        src = matches[-1]
        dst = args.rescored_dir / src.name

        if args.skip_rescore and dst.exists():
            print(f"  [{strategy}] reusing {dst.name}")
        else:
            print(f"  [{strategy}] rescoring {src.name} -> {dst.name}")
            info = rescore_result_file(src, gold_lookup, dst)
            print(
                f"             articles={info['articles']}  "
                f"SI-F1={info['si_f1']:.4f}  TC-F1={info['tc_f1']:.4f}  "
                f"macro-F1={info['macro_f1']:.4f}"
            )
            if info["missing_gold"]:
                print(
                    f"             WARNING: {len(info['missing_gold'])} "
                    f"article id(s) not in raw dataset; kept stored gold."
                )
        rescored_paths.append(dst)

    if not rescored_paths:
        raise SystemExit("No rescored result files to analyse.")

    print("\nBuilding findings ...")
    lines = build_findings(rescored_paths, args.output.parent)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
