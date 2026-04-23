"""
Build a per-article cost report for every (strategy, model) in the study.

Source of truth:
  - Scan outputs/results/ and outputs/old_results/ for any run that has
    `cost.total_cost_usd > 0`. Use the largest-n such run per
    (strategy, model) as the canonical per-article rate.
  - For targets without any uncached run on disk, optionally kick off a
    small N=3 measurement run (fresh seed + dedicated cache dir so no
    entries cache-hit) via `run_experiment`, average it, and record it.

Output:
  outputs/cost_report.json + a human-readable table on stdout.

Examples:

  # Report only (no API calls). Missing rates are flagged "NO DATA":
  uv run python -m scripts.cost_report

  # Measure missing rates with 3 fresh articles:
  uv run python -m scripts.cost_report --measure

  # Explicit target list and full-run sizes:
  uv run python -m scripts.cost_report --measure \
      --targets "zero_shot:gpt-4o:122,consol:gpt-4o:122,hybrid:gpt-4o:94"
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.experiment import run_experiment
from src.schemas import ExperimentConfig

# (strategy, model, run_size_for_extrapolation)
DEFAULT_TARGETS: list[tuple[str, str, int]] = [
    ("zero_shot", "gpt-4o", 122),
    ("few_shot", "gpt-4o", 122),
    ("asv", "gpt-4o", 122),
    ("consol", "gpt-4o", 122),
    ("hybrid", "gpt-4o", 94),
    ("zero_shot", "gpt-4o-mini", 80),
    ("consol", "gpt-4o-mini", 80),
]

RESULT_DIRS = [
    Path("outputs/results"),
    Path("outputs/old_results"),
]


@dataclass
class Rate:
    strategy: str
    model: str
    n_articles: int
    cost_per_article: float
    tokens_in_per_article: float
    tokens_out_per_article: float
    calls_per_article: float
    source_path: str

    def extrapolate(self, n: int) -> dict:
        return {
            "n_articles_target": n,
            "est_cost_usd": round(self.cost_per_article * n, 4),
            "est_tokens_in": int(self.tokens_in_per_article * n),
            "est_tokens_out": int(self.tokens_out_per_article * n),
            "est_calls": int(self.calls_per_article * n),
        }


# ─── Find existing rates ──────────────────────────────────────────────────


def _iter_result_blocks(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "results" in data and isinstance(data["results"], dict):
        for s, blk in data["results"].items():
            if isinstance(blk, dict):
                yield s, blk
    elif "cost" in data and "config" in data:
        yield data.get("config", {}).get("strategy", "unknown"), data


def _rate_from_block(blk: dict, source: Path) -> Rate | None:
    cost_obj = blk.get("cost") or {}
    if not cost_obj:
        return None
    total = float(cost_obj.get("total_cost_usd", 0) or 0)
    if total <= 0:
        return None
    n = int(
        blk.get("articles_processed")
        or blk.get("config", {}).get("max_articles")
        or 0
    )
    if n <= 0:
        return None
    cfg = blk.get("config") or {}
    strategy = cfg.get("strategy") or "?"
    model = cfg.get("model") or "?"
    # Skip cross-model runs: their aggregate cost mixes two models and
    # is not a clean per-article rate for the base model.
    verify = cfg.get("verify_model")
    if verify and verify != model:
        return None
    # Skip partial-cache runs where most calls came from cache: if the
    # average number of API calls per article is far below what the
    # strategy needs, the rate will be artificially low.
    min_calls_per_article = {
        "zero_shot": 0.9, "few_shot": 0.9, "cot": 0.9,
        "asv": 3.0, "consol": 3.0, "hybrid": 4.0,
    }.get(strategy, 0.9)
    num_calls = int(cost_obj.get("num_calls", 0) or 0)
    if num_calls / max(n, 1) < min_calls_per_article:
        return None
    return Rate(
        strategy=strategy,
        model=model,
        n_articles=n,
        cost_per_article=total / n,
        tokens_in_per_article=cost_obj.get("total_input_tokens", 0) / n,
        tokens_out_per_article=cost_obj.get("total_output_tokens", 0) / n,
        calls_per_article=cost_obj.get("num_calls", 0) / n,
        source_path=str(source),
    )


def find_existing_rates() -> dict[tuple[str, str], Rate]:
    best: dict[tuple[str, str], Rate] = {}
    for d in RESULT_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            for _, blk in _iter_result_blocks(p):
                rate = _rate_from_block(blk, p)
                if rate is None:
                    continue
                key = (rate.strategy, rate.model)
                existing = best.get(key)
                if existing is None or rate.n_articles > existing.n_articles:
                    best[key] = rate
    return best


# ─── Measurement run (N=3, fresh cache) ───────────────────────────────────


async def measure_one(
    strategy: str,
    model: str,
    n_measure: int,
    articles_dir: Path,
    labels_path: Path,
    seed: int | None = None,
) -> Rate:
    """Run run_experiment on a small subset with a fresh cache/seed to force
    real API calls, then derive a per-article rate.
    """
    if seed is None:
        seed = int(time.time()) % 1_000_000

    cfg_name = f"costprobe_{strategy}_{model}_n{n_measure}_s{seed}"
    config = ExperimentConfig(
        name=cfg_name,
        strategy=strategy,
        model=model,
        articles_dir=str(articles_dir),
        labels_path=str(labels_path),
        asv_stages=2,
        eval_mode="permissive",
        temperature=0.0,
        seed=seed,
        max_cost_usd=20.0,
        max_articles=n_measure,
    )

    print(f"  [measure] {strategy}/{model} N={n_measure} seed={seed} ...",
          flush=True)
    results = await run_experiment(config)

    out_path = Path("outputs/results") / (
        f"{cfg_name}_{hashlib.md5(cfg_name.encode()).hexdigest()[:8]}.json"
    )
    # run_experiment already wrote the results JSON; but we need to locate it.
    # Use the authoritative path produced by config.run_id().
    written = Path("outputs/results") / f"{cfg_name}_{config.run_id()}.json"
    src_path = written if written.exists() else out_path

    cost_obj = results.get("cost") or {}
    total = float(cost_obj.get("total_cost_usd", 0) or 0)
    n = int(results.get("articles_processed") or n_measure)
    return Rate(
        strategy=strategy,
        model=model,
        n_articles=n,
        cost_per_article=total / max(n, 1),
        tokens_in_per_article=cost_obj.get("total_input_tokens", 0) / max(n, 1),
        tokens_out_per_article=cost_obj.get("total_output_tokens", 0) / max(n, 1),
        calls_per_article=cost_obj.get("num_calls", 0) / max(n, 1),
        source_path=str(src_path),
    )


# ─── Main ─────────────────────────────────────────────────────────────────


def _parse_targets(s: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for part in s.split(","):
        strat, model, n = part.strip().split(":")
        out.append((strat, model, int(n)))
    return out


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Strategy':<12} {'Model':<15} {'N_target':>9} "
        f"{'$/article':>12} {'Est $':>10} {'Tokens in/art':>15} "
        f"{'Tokens out/art':>15} {'Source':<40}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if r.get("missing"):
            print(
                f"{r['strategy']:<12} {r['model']:<15} {r['n_target']:>9} "
                f"{'NO DATA':>12} {'-':>10} {'-':>15} {'-':>15} "
                f"{'-':<40}"
            )
            continue
        print(
            f"{r['strategy']:<12} {r['model']:<15} {r['n_target']:>9} "
            f"${r['per_article']:>11.4f} ${r['est_total']:>9.2f} "
            f"{r['tokens_in_per_article']:>15,.0f} "
            f"{r['tokens_out_per_article']:>15,.0f} "
            f"{r['source']:<40}"
        )


async def run(args) -> None:
    targets = (
        _parse_targets(args.targets) if args.targets else DEFAULT_TARGETS
    )
    existing = find_existing_rates()

    rows: list[dict] = []
    for strat, model, n_target in targets:
        rate = existing.get((strat, model))
        if rate is None and args.measure:
            rate = await measure_one(
                strat, model, args.n_measure,
                articles_dir=Path(args.articles_dir),
                labels_path=Path(args.labels_path),
            )

        if rate is None:
            rows.append({
                "strategy": strat, "model": model,
                "n_target": n_target, "missing": True,
            })
            continue

        extrap = rate.extrapolate(n_target)
        rows.append({
            "strategy": strat,
            "model": model,
            "n_target": n_target,
            "per_article": rate.cost_per_article,
            "tokens_in_per_article": rate.tokens_in_per_article,
            "tokens_out_per_article": rate.tokens_out_per_article,
            "calls_per_article": rate.calls_per_article,
            "est_total": extrap["est_cost_usd"],
            "est_tokens_in": extrap["est_tokens_in"],
            "est_tokens_out": extrap["est_tokens_out"],
            "est_calls": extrap["est_calls"],
            "source": Path(rate.source_path).name,
            "source_path": rate.source_path,
            "source_n": rate.n_articles,
        })

    _print_table(rows)
    out_path = Path("outputs/cost_report.json")
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--measure", action="store_true",
        help="For targets without cost data, run N=3 fresh to measure.",
    )
    parser.add_argument("--n-measure", type=int, default=3)
    parser.add_argument(
        "--articles-dir", type=str,
        default="data/splits/eval/articles",
        help="Article source used for measurement runs.",
    )
    parser.add_argument(
        "--labels-path", type=str,
        default="data/splits/eval/eval-task2-TC.labels",
        help="Label source used for measurement runs.",
    )
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated strategy:model:n list.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
