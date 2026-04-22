"""
Run all strategies and produce a comparison table.

Usage:
    python -m src.run_all --model gpt-4o-mini --strategies all --max-articles 10
    python -m src.run_all --model gpt-4o --strategies hybrid --verify-model gpt-4o --eval-mode strict --max-articles 50
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
    print_comparison_table, print_technique_comparison,
)
from src.experiment import run_experiment
from src.schemas import ExperimentConfig

STRATEGIES = ["zero_shot", "few_shot", "cot", "asv", "consol", "hybrid"]


async def run_all(args):
    all_results: dict[str, dict] = {}

    if args.strategies.strip().lower() == "all":
        selected = STRATEGIES
    elif args.strategies.strip().lower() == "none":
        selected = []
    else:
        selected = [s.strip() for s in args.strategies.split(",") if s.strip()]
        invalid = [s for s in selected if s not in STRATEGIES]
        if invalid:
            raise ValueError(f"Invalid strategies: {invalid}. Valid: {STRATEGIES}")

    if not selected:
        print("No strategies selected. Exiting.")
        return

    articles = load_semeval(
        articles_dir=args.articles_dir, labels_dir=args.labels_path,
    )
    if args.max_articles is not None:
        articles = articles[:args.max_articles]

    article_ids = [a.id for a in articles]
    logging.info(f"Using {len(articles)} articles for all strategies")

    print(f"\n  Strategies   : {', '.join(selected)}")
    print(f"  Model        : {args.model}")
    print(f"  Verify model : {args.verify_model or '(same)'}")
    print(f"  Eval mode    : {args.eval_mode}")
    print(f"  Articles     : {len(articles)}")
    print(f"  ASV stages   : {args.asv_stages}\n")

    for strategy in selected:
        print(f"\n{'='*60}")
        print(f"  Running: {strategy} ({args.model})")
        print(f"{'='*60}\n")

        if args.max_articles is not None:
            cfg_name = f"{strategy}_{args.model}_n{args.max_articles}"
        else:
            cfg_name = f"{strategy}_{args.model}"

        config = ExperimentConfig(
            name=cfg_name,
            strategy=strategy,
            model=args.model,
            articles_dir=args.articles_dir,
            labels_path=args.labels_path,
            asv_stages=args.asv_stages,
            verify_model=args.verify_model,
            eval_mode=args.eval_mode,
            temperature=0.0,
            seed=args.seed,
            max_cost_usd=args.max_cost,
            max_articles=args.max_articles,
        )

        try:
            all_results[strategy] = await run_experiment(config)
        except Exception as e:
            logging.error(f"Failed on {strategy}: {e}")
            all_results[strategy] = {"error": str(e)}

    print_comparison_table(
        all_results, model=args.model, num_articles=len(articles),
    )
    print_technique_comparison(all_results)

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_tag = Path(args.articles_dir).resolve().parent.name
    strategies_slug = "all" if selected == STRATEGIES else "-".join(selected)
    limit_slug = "all" if args.max_articles is None else f"n{args.max_articles}"
    eval_slug = args.eval_mode
    output_path = (
        output_dir
        / f"comparison_{args.model}_{data_tag}_{strategies_slug}_{limit_slug}_{eval_slug}.json"
    )

    save_payload = {
        "articles_used": article_ids,
        "num_articles": len(articles),
        "model": args.model,
        "verify_model": args.verify_model,
        "eval_mode": args.eval_mode,
        "strategies": selected,
        "asv_stages": args.asv_stages,
        "results": all_results,
    }
    output_path.write_text(json.dumps(save_payload, indent=2, default=str))
    print(f"Combined results saved to {output_path}")

    chart_prefix = output_path.with_suffix("")
    chart_paths = generate_comparison_charts(all_results, chart_prefix)
    if chart_paths:
        print("Comparison charts saved:")
        for p in chart_paths:
            print(f"  - {p}")


def main():
    parser = argparse.ArgumentParser(description="Run all strategies for comparison")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--articles-dir", type=str, default="data/train/articles")
    parser.add_argument("--labels-path", type=str,
                        default="data/train/train-task2-TC.labels")
    parser.add_argument("--max-cost", type=float, default=30.0,
                        help="Max cost per strategy in USD")
    parser.add_argument(
        "--strategies", type=str, default="all",
        help="Comma-separated strategies or 'all'",
    )
    parser.add_argument(
        "--max-articles", "--limit", dest="max_articles",
        type=int, default=None,
    )
    parser.add_argument("--asv-stages", type=int, default=2, choices=[2, 3])
    parser.add_argument("--verify-model", type=str, default=None,
                        help="Cross-model verification: model for Stage 2/3")
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
            logging.FileHandler(f"outputs/logs/run_all_{args.model}.log"),
        ],
    )

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()