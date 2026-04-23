"""Regenerate strategy-comparison charts from a saved comparison JSON.

Works for both the original `run_all` output shape (results under a "results"
key alongside metadata) and the rescored shape (a flat mapping of strategy
-> {metrics: ...}).

Example:
    python -m scripts.plot_comparison \
        --comparison-json outputs/results/comparison_gpt-4o_..._rescored.json \
        --output-prefix   outputs/results/comparison_gpt-4o_..._rescored
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.charts import generate_comparison_charts


def _flatten(payload: dict) -> tuple[dict[str, dict], int | None]:
    """Return a {strategy: {metrics: ...}} mapping plus article count if known."""
    if isinstance(payload.get("results"), dict):
        block = payload["results"]
        num = payload.get("num_articles")
    else:
        block = payload
        num = None

    flat: dict[str, dict] = {}
    inferred: int | None = None
    for strat, entry in block.items():
        if not isinstance(entry, dict) or "metrics" not in entry:
            continue
        flat[strat] = {"metrics": entry["metrics"]}
        processed = entry.get("articles_processed") or entry.get("articles_scored")
        if isinstance(processed, int):
            inferred = max(inferred or 0, processed)

    return flat, num or inferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison-json", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--num-articles",
        type=int,
        default=None,
        help="Override article count displayed in chart titles.",
    )
    args = parser.parse_args()

    payload = json.loads(args.comparison_json.read_text(encoding="utf-8"))
    results, inferred_n = _flatten(payload)

    if not results:
        raise SystemExit(f"No strategies with metrics found in {args.comparison_json}")

    paths = generate_comparison_charts(
        results, args.output_prefix, args.num_articles or inferred_n,
    )

    print("Charts written:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
