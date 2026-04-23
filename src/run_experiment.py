"""Run a single experiment from the command line.

Usage:
    python -m scripts.run_experiment --strategy zero_shot --model gpt-4o-mini
    python -m scripts.run_experiment --strategy hybrid --model gpt-4o \
        --verify-model claude-sonnet-4-6 --max-articles 10
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.experiment import run_experiment
from src.schemas import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Run propaganda detection experiment")
    parser.add_argument("--strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "asv", "consol", "hybrid"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--articles-dir", type=str, default="data/train/articles")
    parser.add_argument("--labels-path", type=str,
                        default="data/train/train-task2-TC.labels")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--verify-model", type=str, default=None,
                        help="Use a different model for Stage 2/3 (cross-model verification)")
    parser.add_argument("--eval-mode", choices=["permissive", "strict"], default="permissive",
                        help="strict = CONFIRMED only; permissive = CONFIRMED+POSSIBLE")
    parser.add_argument("--max-articles", "--limit", dest="max_articles",
                        type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"outputs/logs/{args.strategy}_{args.model}.log"),
        ],
    )

    name = args.name or f"{args.strategy}_{args.model}"
    if args.max_articles is not None and not args.name:
        name += f"_n{args.max_articles}"

    config = ExperimentConfig(
        name=name,
        strategy=args.strategy,
        model=args.model,
        articles_dir=args.articles_dir,
        labels_path=args.labels_path,
        verify_model=args.verify_model,
        eval_mode=args.eval_mode,
        temperature=args.temperature,
        seed=args.seed,
        max_cost_usd=args.max_cost,
        max_articles=args.max_articles,
    )

    logging.info(f"Config: {config.model_dump_json(indent=2)}")
    results = asyncio.run(run_experiment(config))

    metrics = results["metrics"]
    cost = results["cost"]
    print(f"\n{'='*60}\n  DONE: {config.name}\n{'='*60}")
    print(f"  {'Metric':<30} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  {'Span Identification':<30} "
          f"{metrics['si_precision']:>9.4f} {metrics['si_recall']:>9.4f} "
          f"{metrics['si_f1']:>9.4f}")
    print(f"  {'Technique Classification':<30} "
          f"{metrics['tc_precision']:>9.4f} {metrics['tc_recall']:>9.4f} "
          f"{metrics['tc_f1']:>9.4f}")
    print(f"  {'Macro-F1':<30} {'':>9} {'':>9} {metrics['macro_f1']:>9.4f}")
    print(f"\n  Cost: ${cost['total_cost_usd']:.4f}  |  "
          f"Articles: {results['articles_processed']}  |  "
          f"Time: {results['elapsed_seconds']:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()