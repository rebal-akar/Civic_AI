"""
Stratified train/eval split for SemEval propaganda data.

Splits at the ARTICLE level (not span level) to avoid data leakage.
Stratification ensures rare techniques (Bandwagon, Whataboutism) appear
in both splits proportionally.

Usage:
    python -m scripts.split_data \
        --articles-dir data/train/articles \
        --labels-path data/train/train-task2-TC.labels \
        --output-dir data/splits \
        --eval-ratio 0.3 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.data.loader import load_semeval
from src.schemas import Article, Technique

logger = logging.getLogger(__name__)


def stratified_split(
    articles: list[Article],
    eval_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[Article], list[Article]]:
    """Split articles into train/eval preserving technique distribution.

    Strategy:
    1. Assign each article a "primary technique" = its rarest technique.
       This ensures rare techniques get priority in stratification.
    2. Group articles by primary technique.
    3. Within each group, split proportionally.
    4. Articles with no gold spans go into train (nothing to evaluate).

    Args:
        articles: All loaded articles with gold spans.
        eval_ratio: Fraction of articles for evaluation (default 0.3).
        seed: Random seed for reproducibility.

    Returns:
        (train_articles, eval_articles)
    """
    rng = random.Random(seed)

    # Count global technique frequency to determine rarity
    global_counts: Counter[Technique] = Counter()
    for article in articles:
        for span in article.gold_spans:
            global_counts[span.technique] += 1

    # Assign each article a primary technique (its rarest one)
    # Articles with no spans go to a special "none" bucket
    buckets: dict[str, list[Article]] = defaultdict(list)

    for article in articles:
        if not article.gold_spans:
            buckets["_no_spans"].append(article)
            continue

        techniques = {s.technique for s in article.gold_spans}
        # Primary = the rarest technique in this article
        rarest = min(techniques, key=lambda t: global_counts.get(t, 0))
        buckets[rarest.value].append(article)

    train_articles: list[Article] = []
    eval_articles: list[Article] = []

    # Split each bucket proportionally
    for bucket_name, bucket_articles in sorted(buckets.items()):
        rng.shuffle(bucket_articles)
        n_eval = max(1, round(len(bucket_articles) * eval_ratio))

        # Ensure at least 1 in eval for non-empty buckets (except _no_spans)
        if bucket_name == "_no_spans":
            train_articles.extend(bucket_articles)
            continue

        eval_articles.extend(bucket_articles[:n_eval])
        train_articles.extend(bucket_articles[n_eval:])

    # Shuffle final lists
    rng.shuffle(train_articles)
    rng.shuffle(eval_articles)

    return train_articles, eval_articles


def write_split(
    articles: list[Article],
    output_dir: Path,
    split_name: str,
) -> None:
    """Write a split to disk: articles dir + consolidated labels file."""
    articles_dir = output_dir / split_name / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    labels_lines: list[str] = []

    for article in articles:
        # Copy article text
        article_path = articles_dir / f"{article.id}.txt"
        article_path.write_text(article.text, encoding="utf-8")

        # Write label lines
        for span in article.gold_spans:
            labels_lines.append(
                f"{article.id}\t{span.technique.value}\t{span.start}\t{span.end}"
            )

    # Write consolidated labels file
    labels_path = output_dir / split_name / f"{split_name}-task2-TC.labels"
    labels_path.write_text("\n".join(labels_lines) + "\n", encoding="utf-8")


def print_split_stats(
    train: list[Article],
    eval_: list[Article],
) -> None:
    """Print distribution comparison between splits."""
    def technique_counts(articles: list[Article]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for a in articles:
            for s in a.gold_spans:
                counts[s.technique.value] += 1
        return counts

    train_counts = technique_counts(train)
    eval_counts = technique_counts(eval_)
    all_techniques = sorted(set(train_counts.keys()) | set(eval_counts.keys()))

    total_train = sum(train_counts.values())
    total_eval = sum(eval_counts.values())

    print(f"\n{'='*70}")
    print(f"  Split Statistics")
    print(f"{'='*70}")
    print(f"  Articles:  train={len(train)}  eval={len(eval_)}")
    print(f"  Spans:     train={total_train}  eval={total_eval}")
    print()
    print(f"  {'Technique':<40} {'Train':>6} {'%':>6} {'Eval':>6} {'%':>6}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for t in all_techniques:
        tc = train_counts.get(t, 0)
        ec = eval_counts.get(t, 0)
        tp = (tc / total_train * 100) if total_train > 0 else 0
        ep = (ec / total_eval * 100) if total_eval > 0 else 0
        print(f"  {t:<40} {tc:>6} {tp:>5.1f}% {ec:>6} {ep:>5.1f}%")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split SemEval train data into train/eval with stratification"
    )
    parser.add_argument("--articles-dir", type=str, required=True,
                        help="Path to articles directory")
    parser.add_argument("--labels-path", type=str, required=True,
                        help="Path to TC labels file")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                        help="Output directory for splits")
    parser.add_argument("--eval-ratio", type=float, default=0.3,
                        help="Fraction of articles for evaluation (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load all data
    print(f"Loading data from {args.articles_dir}...")
    articles = load_semeval(args.articles_dir, args.labels_path)
    print(f"Loaded {len(articles)} articles with "
          f"{sum(len(a.gold_spans) for a in articles)} gold spans")

    # Split
    train, eval_ = stratified_split(articles, args.eval_ratio, args.seed)

    # Print stats
    print_split_stats(train, eval_)

    # Write to disk
    output_dir = Path(args.output_dir)
    write_split(train, output_dir, "train")
    write_split(eval_, output_dir, "eval")

    print(f"Written to {output_dir}/")
    print(f"  {output_dir}/train/articles/  ({len(train)} articles)")
    print(f"  {output_dir}/train/train-task2-TC.labels")
    print(f"  {output_dir}/eval/articles/   ({len(eval_)} articles)")
    print(f"  {output_dir}/eval/eval-task2-TC.labels")

    # Save split metadata for reproducibility
    meta = {
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "source_articles_dir": args.articles_dir,
        "source_labels_path": args.labels_path,
        "train_article_ids": sorted(a.id for a in train),
        "eval_article_ids": sorted(a.id for a in eval_),
        "train_articles": len(train),
        "eval_articles": len(eval_),
        "train_spans": sum(len(a.gold_spans) for a in train),
        "eval_spans": sum(len(a.gold_spans) for a in eval_),
    }
    meta_path = output_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  {meta_path}  (for reproducibility)")


if __name__ == "__main__":
    main()