"""Stratified article-level train/eval split for SemEval data.

Stratification uses each article's rarest technique as the primary label,
ensuring that rare techniques (Bandwagon, Whataboutism) appear proportionally
in both splits. Articles with no gold spans go entirely to train.

Usage:
    python -m scripts.split_data \
        --articles-dir data/train/articles \
        --labels-path data/train/train-task2-TC.labels \
        --output-dir data/splits \
        --eval-ratio 0.3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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

    Each article's bucket is determined by its rarest technique (globally),
    and each bucket is split proportionally. Articles with no gold spans go
    into train (nothing to evaluate).
    """
    rng = random.Random(seed)

    global_counts: Counter[Technique] = Counter()
    for article in articles:
        for span in article.gold_spans:
            global_counts[span.technique] += 1

    buckets: dict[str, list[Article]] = defaultdict(list)
    for article in articles:
        if not article.gold_spans:
            buckets["_no_spans"].append(article)
            continue
        techniques = {s.technique for s in article.gold_spans}
        rarest = min(techniques, key=lambda t: global_counts.get(t, 0))
        buckets[rarest.value].append(article)

    train_articles: list[Article] = []
    eval_articles: list[Article] = []

    for bucket_name, bucket_articles in sorted(buckets.items()):
        rng.shuffle(bucket_articles)
        if bucket_name == "_no_spans":
            train_articles.extend(bucket_articles)
            continue
        n_eval = max(1, round(len(bucket_articles) * eval_ratio))
        eval_articles.extend(bucket_articles[:n_eval])
        train_articles.extend(bucket_articles[n_eval:])

    rng.shuffle(train_articles)
    rng.shuffle(eval_articles)
    return train_articles, eval_articles


def write_split(
    articles: list[Article],
    output_dir: Path,
    split_name: str,
) -> None:
    articles_dir = output_dir / split_name / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    labels_lines: list[str] = []
    for article in articles:
        (articles_dir / f"{article.id}.txt").write_text(article.text, encoding="utf-8")
        for span in article.gold_spans:
            labels_lines.append(
                f"{article.id}\t{span.technique.value}\t{span.start}\t{span.end}"
            )

    labels_path = output_dir / split_name / f"{split_name}-task2-TC.labels"
    labels_path.write_text("\n".join(labels_lines) + "\n", encoding="utf-8")


def print_split_stats(train: list[Article], eval_: list[Article]) -> None:
    def technique_counts(articles: list[Article]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for a in articles:
            for s in a.gold_spans:
                counts[s.technique.value] += 1
        return counts

    train_counts = technique_counts(train)
    eval_counts = technique_counts(eval_)
    all_techniques = sorted(set(train_counts) | set(eval_counts))

    total_train = sum(train_counts.values())
    total_eval = sum(eval_counts.values())

    print(f"\n{'='*70}\n  Split Statistics\n{'='*70}")
    print(f"  Articles:  train={len(train)}  eval={len(eval_)}")
    print(f"  Spans:     train={total_train}  eval={total_eval}\n")
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
    parser.add_argument("--articles-dir", type=str, required=True)
    parser.add_argument("--labels-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--eval-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading data from {args.articles_dir}...")
    articles = load_semeval(args.articles_dir, args.labels_path)
    print(f"Loaded {len(articles)} articles with "
          f"{sum(len(a.gold_spans) for a in articles)} gold spans")

    train, eval_ = stratified_split(articles, args.eval_ratio, args.seed)
    print_split_stats(train, eval_)

    output_dir = Path(args.output_dir)
    write_split(train, output_dir, "train")
    write_split(eval_, output_dir, "eval")

    print(f"Written to {output_dir}/")
    print(f"  {output_dir}/train/articles/  ({len(train)} articles)")
    print(f"  {output_dir}/train/train-task2-TC.labels")
    print(f"  {output_dir}/eval/articles/   ({len(eval_)} articles)")
    print(f"  {output_dir}/eval/eval-task2-TC.labels")

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
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()