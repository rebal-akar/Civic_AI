#!/usr/bin/env python3
"""
CLI entry point for running experiments.

Usage:
    # Single experiment
    python run.py --strategy zero_shot --model gpt-4o-mini --dataset semeval_train --sample-size 50

    # ASV with asvS prompts on 2 articles, with trace tables
    python run.py --strategy asv --prompts asvS --num-articles 2 --data-root datasets --debug-asv

    # Full benchmark (all strategies)
    python run.py --benchmark --model gpt-4o-mini --dataset semeval_train

    # Compare two strategies
    python run.py --compare outputs/results/run_a outputs/results/run_b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path and load .env before any imports that need the API key
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
for _env_dir in (_project_root, Path.cwd()):
    _env_file = _env_dir / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        break
# So OpenAI client can use standard env var: copy CHATGPT_API_KEY -> OPENAI_API_KEY if needed
if "OPENAI_API_KEY" not in os.environ and "CHATGPT_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["CHATGPT_API_KEY"]

from src.runner import run_experiment
from src.schemas import ExperimentConfig
from src.metrics import mcnemar_test, bonferroni_correction
from src.data import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


ALL_STRATEGIES = ["zero_shot", "few_shot", "cot", "cot_sc", "hierarchical", "asv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manipulation Detection Experiments")
    
    # Mode
    parser.add_argument("--benchmark", action="store_true", help="Run all 6 strategies")
    parser.add_argument("--compare", nargs=2, metavar="DIR", help="Compare two result directories")

    # Experiment config
    parser.add_argument("--strategy", choices=ALL_STRATEGIES, default="zero_shot")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--dataset", default="semeval_train",
                        choices=["semeval_train", "semeval_dev", "creative"])
    parser.add_argument("--sample-size", type=int, default=None, help="Max number of samples (paragraphs)")
    parser.add_argument("--num-articles", type=int, default=None, help="Run on this many articles (all paragraphs from those articles)")
    parser.add_argument("--prompts", choices=["asv", "asvS"], default="asv", help="ASV prompt module: asv or asvS")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--debug-asv", action="store_true", help="Print prosecutor/defense/verdict trace tables (ASV only)")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--notes", default="", help="Notes for this run")

    # Paths
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/results")

    return parser.parse_args()


async def run_single(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    config = ExperimentConfig(
        name=args.name or f"{args.strategy}_{args.model}",
        strategy=args.strategy,
        model=args.model,
        temperature=args.temperature,
        dataset=args.dataset,
        sample_size=args.sample_size,
        num_articles=args.num_articles,
        seed=args.seed,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        debug_asv=args.debug_asv,
        prompts=args.prompts,
        notes=args.notes,
    )

    result = await run_experiment(config, args.data_root, args.output_dir)

    print("\n" + "=" * 60)
    print(f"RESULTS: {result.run_id}")
    print("=" * 60)
    print(json.dumps(result.metrics, indent=2))
    print(f"\nCost: ${result.cost.total_cost_usd:.4f}")
    print(f"Duration: {result.duration_seconds():.1f}s")

    out_path = (Path(args.output_dir) / result.run_id).resolve()
    print(f"\nResults written to: {out_path}")
    print(f"View report:        {out_path / 'report.txt'}")
    print(f"\nTo open the results folder in Explorer, run:")
    print(f"  explorer \"{out_path}\"")


async def run_benchmark(args: argparse.Namespace) -> None:
    """Run all strategies and compare."""
    results = {}

    for strategy in ALL_STRATEGIES:
        logger.info(f"\n{'='*60}\nRunning: {strategy}\n{'='*60}")
        args.strategy = strategy
        config = ExperimentConfig(
            name=f"benchmark_{strategy}_{args.model}",
            strategy=strategy,
            model=args.model,
            temperature=args.temperature,
            dataset=args.dataset,
            sample_size=args.sample_size,
            seed=args.seed,
            batch_size=args.batch_size,
            use_cache=not args.no_cache,
        )
        result = await run_experiment(config, args.data_root, args.output_dir)
        results[strategy] = result

    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<16} {'Macro F1':>10} {'Micro F1':>10} {'Precision':>10} {'Recall':>10} {'Cost':>10}")
    print("-" * 80)
    for strategy, result in results.items():
        m = result.metrics
        print(
            f"{strategy:<16} "
            f"{m.get('macro_f1', 0):>10.4f} "
            f"{m.get('micro_f1', 0):>10.4f} "
            f"{m.get('macro_precision', 0):>10.4f} "
            f"{m.get('macro_recall', 0):>10.4f} "
            f"${result.cost.total_cost_usd:>9.4f}"
        )

    # Save comparison
    comparison = {s: r.metrics for s, r in results.items()}
    out_path = Path(args.output_dir) / "benchmark_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to: {out_path}")


def run_compare(args: argparse.Namespace) -> None:
    """Compare two experiment runs with McNemar's test."""
    dir_a, dir_b = [Path(d) for d in args.compare]

    # Load predictions and configs
    config_a = json.loads((dir_a / "config.json").read_text())
    config_b = json.loads((dir_b / "config.json").read_text())
    metrics_a = json.loads((dir_a / "metrics.json").read_text())
    metrics_b = json.loads((dir_b / "metrics.json").read_text())

    print(f"\nComparing:")
    print(f"  A: {config_a.get('strategy')} ({config_a.get('model')})")
    print(f"  B: {config_b.get('strategy')} ({config_b.get('model')})")
    print(f"\n  A Macro F1: {metrics_a.get('macro_f1', 0):.4f}")
    print(f"  B Macro F1: {metrics_b.get('macro_f1', 0):.4f}")

    # Load predictions for McNemar's test
    from src.schemas import Prediction, Sample
    preds_a_raw = json.loads((dir_a / "predictions.json").read_text())
    preds_b_raw = json.loads((dir_b / "predictions.json").read_text())

    preds_a = [Prediction(**p) for p in preds_a_raw]
    preds_b = [Prediction(**p) for p in preds_b_raw]

    # Need ground truth — load from dataset
    dataset_name = config_a.get("dataset", "semeval_train")
    samples = load_dataset(dataset_name, args.data_root)

    if config_a.get("sample_size"):
        import numpy as np
        rng = np.random.default_rng(config_a.get("seed", 42))
        indices = rng.choice(len(samples), size=config_a["sample_size"], replace=False)
        samples = [samples[i] for i in sorted(indices)]

    result = mcnemar_test(samples, preds_a, preds_b)
    print(f"\nMcNemar's Test:")
    print(f"  {result['interpretation']}")


def main():
    args = parse_args()

    if args.compare:
        run_compare(args)
    elif args.benchmark:
        asyncio.run(run_benchmark(args))
    else:
        asyncio.run(run_single(args))


if __name__ == "__main__":
    main()
