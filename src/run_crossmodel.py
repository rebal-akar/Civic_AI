"""Run cross-family verification: same-family and cascaded detect/verify pairs.

Four configurations tested in the dissertation:
    same_4o               : gpt-4o detects and consolidates
    same_sonnet           : claude-sonnet-4-6 detects and consolidates
    cascade_4o_to_sonnet  : gpt-4o detects, claude-sonnet-4-6 consolidates
    cascade_sonnet_to_4o  : claude-sonnet-4-6 detects, gpt-4o consolidates

Usage:
    python -m scripts.run_crossmodel --strategy consol \
        --articles-dir data/splits/eval/articles \
        --labels-path data/splits/eval/eval-task2-TC.labels \
        --max-cost 20.0
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.evaluation.charts import generate_crossmodel_charts
from src.experiment import run_experiment
from src.schemas import ExperimentConfig

CONFIGURATIONS = [
    # (suffix, detect_model, verify_model)
    ("same_4o", "gpt-4o", None),
    ("same_sonnet", "claude-sonnet-4-6", None),
    ("cascade_4o_to_sonnet", "gpt-4o", "claude-sonnet-4-6"),
    ("cascade_sonnet_to_4o", "claude-sonnet-4-6", "gpt-4o"),
]


async def run_crossmodel(args):
    all_results: dict[str, dict] = {}

    for suffix, detect_model, verify_model in CONFIGURATIONS:
        exp_name = f"{args.strategy}_{suffix}"
        verify_label = verify_model or detect_model

        print(f"\n{'='*60}")
        print(f"  {exp_name}: detect={detect_model}, verify={verify_label}")
        print(f"{'='*60}\n")

        config = ExperimentConfig(
            name=exp_name,
            strategy=args.strategy,
            model=detect_model,
            verify_model=verify_model,
            eval_mode=args.eval_mode,
            articles_dir=args.articles_dir,
            labels_path=args.labels_path,
            temperature=0.0,
            seed=args.seed,
            max_cost_usd=args.max_cost,
            max_articles=args.max_articles,
        )

        try:
            all_results[exp_name] = await run_experiment(config)
        except Exception as e:
            logging.error(f"Failed on {exp_name}: {e}")
            all_results[exp_name] = {"error": str(e)}

    print(f"\n{'='*100}")
    print(f"  CROSS-FAMILY COMPARISON: {args.strategy} ({args.eval_mode})")
    print(f"{'='*100}")
    print(f"  {'Config':<24} {'Detect':<20} {'Verify':<20} "
          f"{'SI F1':>7} {'TC F1':>7} {'Macro':>7} {'Delta':>7} {'Cost':>9}")
    print(f"  {'-'*24} {'-'*20} {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

    for suffix, detect_model, verify_model in CONFIGURATIONS:
        exp_name = f"{args.strategy}_{suffix}"
        r = all_results.get(exp_name, {})
        if "error" in r:
            print(f"  {suffix:<24}  ERROR: {r['error'][:60]}")
            continue
        m = r.get("metrics", {})
        c = r.get("cost", {})
        verify_label = verify_model or detect_model
        print(
            f"  {suffix:<24} {detect_model:<20} {verify_label:<20} "
            f"{m.get('si_f1', 0):>7.3f} {m.get('tc_f1', 0):>7.3f} "
            f"{m.get('macro_f1', 0):>7.3f} {m.get('f1_delta', 0):>7.3f} "
            f"${c.get('total_cost_usd', 0):>8.4f}"
        )
    print(f"{'='*100}\n")

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"crossmodel_{args.strategy}_{args.eval_mode}.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Results saved to {output_path}")

    chart_paths = generate_crossmodel_charts(
        all_results, output_path.with_suffix(""), args.max_articles,
    )
    if chart_paths:
        print("Charts written:")
        for p in chart_paths:
            print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(description="Run cross-family verification")
    parser.add_argument("--strategy", type=str, default="consol",
                        choices=["consol", "hybrid", "asv"])
    parser.add_argument("--articles-dir", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--max-cost", type=float, default=20.0)
    parser.add_argument("--max-articles", "--limit", dest="max_articles",
                        type=int, default=None)
    parser.add_argument("--eval-mode", choices=["permissive", "strict"],
                        default="permissive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"outputs/logs/crossmodel_{args.strategy}.log"),
        ],
    )

    asyncio.run(run_crossmodel(args))


if __name__ == "__main__":
    main()