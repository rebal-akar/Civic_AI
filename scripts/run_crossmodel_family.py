"""
Run cross-family verification experiments comparing OpenAI and Anthropic.

Tests four configurations for a single strategy:
- same_4o: detect + verify both gpt-4o
- same_sonnet: detect + verify both claude-sonnet-4-6
- cascade_4o_sonnet: detect with gpt-4o, verify with claude-sonnet-4-6
- cascade_sonnet_4o: detect with claude-sonnet-4-6, verify with gpt-4o

Usage:
    uv run python -m scripts.run_crossmodel_family --strategy consol \\
        --articles-dir data/splits/eval/articles \\
        --labels-path data/splits/eval/eval-task2-TC.labels \\
        --max-articles 122 \\
        --max-cost 30.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.experiment import run_experiment
from src.schemas import ExperimentConfig

OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

CONFIGURATIONS = [
    ("same_4o", OPENAI_MODEL, None),
    ("same_sonnet", ANTHROPIC_MODEL, None),
    ("cascade_4o_sonnet", OPENAI_MODEL, ANTHROPIC_MODEL),
    ("cascade_sonnet_4o", ANTHROPIC_MODEL, OPENAI_MODEL),
]


async def run_crossmodel_family(args: argparse.Namespace) -> None:
    all_results: dict[str, dict] = {}

    for suffix, detect_model, verify_model in CONFIGURATIONS:
        exp_name = f"{args.strategy}_xfam_{suffix}"
        verify_label = verify_model or detect_model

        print(f"\n{'='*70}")
        print(f"  {exp_name}")
        print(f"  detect={detect_model}, verify={verify_label}")
        print(f"{'='*70}\n")

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
            logging.error("Failed on %s: %s", exp_name, e)
            all_results[exp_name] = {"error": str(e)}

    print(f"\n{'='*110}")
    print(f"  CROSS-FAMILY VERIFICATION COMPARISON: {args.strategy} ({args.eval_mode})")
    print(f"{'='*110}")
    print(
        f"  {'Config':<22} {'Detect':<22} {'Verify':<22} "
        f"{'SI F1':>7} {'TC F1':>7} {'Macro':>7} {'Cost':>9}"
    )
    print(
        f"  {'-'*22} {'-'*22} {'-'*22} "
        f"{'-'*7} {'-'*7} {'-'*7} {'-'*9}"
    )

    for suffix, detect_model, verify_model in CONFIGURATIONS:
        exp_name = f"{args.strategy}_xfam_{suffix}"
        r = all_results.get(exp_name, {})
        if "error" in r:
            print(f"  {suffix:<22}  ERROR: {r['error'][:80]}")
            continue
        m = r.get("metrics", {})
        c = r.get("cost", {})
        verify_label = verify_model or detect_model
        print(
            f"  {suffix:<22} {detect_model:<22} {verify_label:<22} "
            f"{m.get('si_f1', 0):>7.3f} {m.get('tc_f1', 0):>7.3f} "
            f"{m.get('macro_f1', 0):>7.3f} ${c.get('total_cost_usd', 0):>8.4f}"
        )
    print(f"{'='*110}\n")

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"crossmodel_family_{args.strategy}_{args.eval_mode}.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cross-family verification (OpenAI x Anthropic)"
    )
    parser.add_argument(
        "--strategy", type=str, default="consol",
        choices=["consol", "hybrid", "asv"],
    )
    parser.add_argument("--articles-dir", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--max-cost", type=float, default=30.0)
    parser.add_argument(
        "--max-articles", "--limit", dest="max_articles",
        type=int, default=None,
    )
    parser.add_argument(
        "--eval-mode", choices=["permissive", "strict"],
        default="permissive",
    )
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
            logging.FileHandler(
                f"outputs/logs/crossmodel_family_{args.strategy}.log"
            ),
        ],
    )

    asyncio.run(run_crossmodel_family(args))


if __name__ == "__main__":
    main()
