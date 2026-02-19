"""
Export utilities for dissertation — LaTeX tables, markdown reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.metrics import MetricsReport, ErrorAnalysis
from src.schemas import ALL_TACTICS


def metrics_to_latex(
    results: dict[str, dict[str, Any]],
    caption: str = "Comparison of prompting strategies",
    label: str = "tab:strategy_comparison",
) -> str:
    """
    Generate a LaTeX table comparing strategies.
    
    Args:
        results: {strategy_name: metrics_dict}
    """
    strategies = list(results.keys())
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * 5 + "}",
        r"\toprule",
        r"Strategy & Macro F1 & Micro F1 & Precision & Recall & Hamming \\",
        r"\midrule",
    ]

    # Find best macro F1 for bolding
    best_f1 = max(r.get("macro_f1", 0) for r in results.values())

    for strategy, metrics in results.items():
        f1 = metrics.get("macro_f1", 0)
        f1_str = rf"\textbf{{{f1:.3f}}}" if f1 == best_f1 else f"{f1:.3f}"
        
        lines.append(
            f"  {strategy.replace('_', ' ').title()} & "
            f"{f1_str} & "
            f"{metrics.get('micro_f1', 0):.3f} & "
            f"{metrics.get('macro_precision', 0):.3f} & "
            f"{metrics.get('macro_recall', 0):.3f} & "
            f"{metrics.get('hamming_loss', 0):.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def tactic_scores_to_latex(
    tactic_scores: dict[str, dict[str, float]],
    strategy_name: str = "ASV",
    caption: str | None = None,
    label: str = "tab:per_tactic",
) -> str:
    """Generate per-tactic breakdown LaTeX table."""
    if caption is None:
        caption = f"Per-tactic performance for {strategy_name}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Technique & Precision & Recall & F1 & Support \\",
        r"\midrule",
    ]

    for tactic in ALL_TACTICS:
        if tactic not in tactic_scores:
            continue
        s = tactic_scores[tactic]
        if s.get("support", 0) == 0:
            continue
        
        display_name = tactic.replace("_", " ").title()
        lines.append(
            f"  {display_name} & "
            f"{s['precision']:.3f} & {s['recall']:.3f} & {s['f1']:.3f} & {s['support']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_report(
    run_dir: str | Path,
) -> str:
    """Generate a markdown report from saved experiment outputs."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cost = json.loads((run_dir / "cost.json").read_text())
    errors = json.loads((run_dir / "error_analysis.json").read_text())

    tactic_scores = {}
    tactic_path = run_dir / "tactic_scores.json"
    if tactic_path.exists():
        tactic_scores = json.loads(tactic_path.read_text())

    report = f"""# Experiment Report: {config.get('name', 'N/A')}

## Configuration
| Parameter | Value |
|-----------|-------|
| Strategy | {config.get('strategy')} |
| Model | {config.get('model')} |
| Dataset | {config.get('dataset')} |
| Samples | {metrics.get('n_samples')} |
| Temperature | {config.get('temperature')} |
| Seed | {config.get('seed')} |

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| **Macro F1** | **{metrics.get('macro_f1', 0):.4f}** |
| Micro F1 | {metrics.get('micro_f1', 0):.4f} |
| Weighted F1 | {metrics.get('weighted_f1', 0):.4f} |
| Precision (macro) | {metrics.get('macro_precision', 0):.4f} |
| Recall (macro) | {metrics.get('macro_recall', 0):.4f} |
| Hamming Loss | {metrics.get('hamming_loss', 0):.4f} |
| Exact Match | {metrics.get('exact_match_ratio', 0):.4f} |
| 95% CI | {metrics.get('macro_f1_ci_95', 'N/A')} |

## Cost
| Metric | Value |
|--------|-------|
| API Calls | {cost.get('num_api_calls', 0)} |
| Prompt Tokens | {cost.get('total_prompt_tokens', 0):,} |
| Completion Tokens | {cost.get('total_completion_tokens', 0):,} |
| **Total Cost** | **${cost.get('total_cost_usd', 0):.4f}** |

## Error Summary
| | Count |
|--|-------|
| Correct | {errors.get('total_correct', 0)} |
| False Positives | {errors.get('total_fp', 0)} |
| False Negatives | {errors.get('total_fn', 0)} |

## Per-Tactic Performance
| Technique | Prec | Rec | F1 | Support |
|-----------|------|-----|----|---------| 
"""
    for tactic in ALL_TACTICS:
        if tactic in tactic_scores and tactic_scores[tactic].get("support", 0) > 0:
            s = tactic_scores[tactic]
            name = tactic.replace("_", " ").title()
            report += f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['support']} |\n"

    # Top FP/FN tactics
    fp_by_tactic = errors.get("fp_by_tactic", {})
    fn_by_tactic = errors.get("fn_by_tactic", {})
    if fp_by_tactic:
        report += "\n## Top False Positive Tactics\n"
        for t, c in sorted(fp_by_tactic.items(), key=lambda x: -x[1])[:5]:
            report += f"- {t.replace('_', ' ').title()}: {c}\n"
    if fn_by_tactic:
        report += "\n## Top False Negative Tactics\n"
        for t, c in sorted(fn_by_tactic.items(), key=lambda x: -x[1])[:5]:
            report += f"- {t.replace('_', ' ').title()}: {c}\n"

    return report
