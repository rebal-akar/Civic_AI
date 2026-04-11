"""
Generate matplotlib bar charts from a run_all comparison JSON.

Usage:
    uv run python -m scripts.plot_comparison \
        --comparison-json outputs/results/comparison_gpt-4o_train_zero_shot-asv_n1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.charts import generate_comparison_charts


def _extract_results(payload: dict) -> dict:
    # New run_all format
    if isinstance(payload.get("results"), dict):
        return payload["results"]

    # Backward compatibility: top-level strategy mapping
    result_like = {
        k: v for k, v in payload.items()
        if isinstance(v, dict) and ("metrics" in v or "error" in v)
    }
    return result_like


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strategy comparison charts")
    parser.add_argument(
        "--comparison-json",
        type=str,
        required=True,
        help="Path to comparison JSON produced by src.run_all",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix (without extension). Defaults near JSON file.",
    )
    args = parser.parse_args()

    json_path = Path(args.comparison_json)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results_by_strategy = _extract_results(payload)

    if args.output_prefix:
        out_prefix = Path(args.output_prefix)
    else:
        out_prefix = json_path.with_suffix("")

    chart_paths = generate_comparison_charts(results_by_strategy, out_prefix)
    if not chart_paths:
        print("No charts generated (no successful strategy results).")
        return

    print("Generated charts:")
    for p in chart_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

