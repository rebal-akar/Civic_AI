"""
Run all strategies and produce a comparison table.

Usage:
    python -m src.run_all --model gpt-4o-mini --strategies all --limit 10
    python -m src.run_all --model gpt-4o --strategies zero_shot,asv --limit 5 --asv-stages 2
    python -m src.run_all --model gpt-4o --articles-dir data/splits/eval/articles \
        --labels-path data/splits/eval/eval-task2-TC.labels --strategies asv --limit 10
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.data.loader import load_semeval
from src.evaluation.charts import generate_comparison_charts
from src.evaluation.metrics import (
    print_comparison_table,
    print_technique_comparison,
)
from src.experiment import run_experiment
from src.schemas import ExperimentConfig

STRATEGIES = ["zero_shot", "few_shot", "cot", "asv", "consol"]


async def run_all(args):
    all_results: dict[str, dict] = {}

    # Resolve requested strategies
    if args.strategies.strip().lower() == "all":
        selected_strategies = STRATEGIES
    elif args.strategies.strip().lower() == "none":
        selected_strategies = []
    else:
        selected_strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        invalid = [s for s in selected_strategies if s not in STRATEGIES]
        if invalid:
            raise ValueError(f"Invalid strategies: {invalid}. Valid: {STRATEGIES}")

    if not selected_strategies:
        print("No strategies selected. Exiting.")
        return

    # Pre-load articles once to guarantee identical set across strategies
    articles = load_semeval(
        articles_dir=args.articles_dir,
        labels_dir=args.labels_path,
    )
    if args.max_articles is not None:
        articles = articles[: args.max_articles]

    article_ids = [a.id for a in articles]
    logging.info(
        f"Using {len(articles)} articles for all strategies: "
        f"{article_ids[:5]}{'...' if len(article_ids) > 5 else ''}"
    )

    print(f"\n  Strategies : {', '.join(selected_strategies)}")
    print(f"  Model      : {args.model}")
    print(f"  Articles   : {len(articles)}")
    print(f"  ASV stages : {args.asv_stages}")
    print()

    for strategy in selected_strategies:
        print(f"\n{'='*60}")
        print(f"  Running: {strategy} ({args.model})")
        print(f"{'='*60}\n")

        if args.max_articles is not None:
            config_name = f"{strategy}_{args.model}_n{args.max_articles}"
        else:
            config_name = f"{strategy}_{args.model}"

        config = ExperimentConfig(
            name=config_name,
            strategy=strategy,
            model=args.model,
            articles_dir=args.articles_dir,
            labels_path=args.labels_path,
            asv_stages=args.asv_stages,
            temperature=0.0,
            seed=args.seed,
            max_cost_usd=args.max_cost,
            max_articles=args.max_articles,
        )

        try:
            results = await run_experiment(config)
            all_results[strategy] = results
        except Exception as e:
            logging.error(f"Failed on {strategy}: {e}")
            all_results[strategy] = {"error": str(e)}

    # ── Print comparison tables ───────────────────────────────────────────
    print_comparison_table(
        all_results,
        model=args.model,
        num_articles=len(articles),
    )
    print_technique_comparison(all_results)

    # ── Save combined results ─────────────────────────────────────────────
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_tag = Path(args.articles_dir).resolve().parent.name
    strategies_slug = (
        "all" if selected_strategies == STRATEGIES
        else "-".join(selected_strategies)
    )
    limit_slug = "all" if args.max_articles is None else f"n{args.max_articles}"
    output_path = (
        output_dir / f"comparison_{args.model}_{data_tag}_{strategies_slug}_{limit_slug}.json"
    )

    save_payload = {
        "articles_used": article_ids,
        "num_articles": len(articles),
        "model": args.model,
        "strategies": selected_strategies,
        "asv_stages": args.asv_stages,
        "results": all_results,
    }
    output_path.write_text(json.dumps(save_payload, indent=2, default=str))
    print(f"Combined results saved to {output_path}")

    # ── Save matplotlib bar charts ────────────────────────────────────────
    chart_prefix = output_dir / f"comparison_{args.model}_{data_tag}_{strategies_slug}_{limit_slug}"
    chart_paths = generate_comparison_charts(all_results, chart_prefix)
    if chart_paths:
        print("Charts saved:")
        for p in chart_paths:
            print(f"  - {p}")
    else:
        print("No charts generated (no successful strategy results).")


def main():
    parser = argparse.ArgumentParser(description="Run all strategies for comparison")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--articles-dir", type=str, default="data/train/articles")
    parser.add_argument("--labels-path", type=str, default="data/train/train-task2-TC.labels")
    parser.add_argument("--max-cost", type=float, default=30.0,
                        help="Max cost per strategy in USD")
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated strategies (e.g. 'asv' or 'zero_shot,asv'). Use 'all' for every strategy.",
    )
    parser.add_argument(
        "--max-articles",
        "--limit",
        dest="max_articles",
        type=int,
        default=None,
        help="Process only the first N articles per strategy (smoke test).",
    )
    parser.add_argument("--asv-stages", type=int, default=2, choices=[2, 3],
                        help="ASV stages: 2=detect+critique, 3=detect+critique+adjudicate")
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
            logging.FileHandler(f"outputs/logs/run_all_{args.model}.log"),
        ],
    )

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
