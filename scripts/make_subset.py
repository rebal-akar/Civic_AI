"""
Build a smaller evaluation subset that guarantees coverage of every technique.

Greedily picks articles whose techniques extend the running coverage set,
then fills the remaining slots by random sampling. Writes articles + a
consolidated labels file in the same shape as scripts.split_data.

Example:
    python -m scripts.make_subset \
        --source-articles data/splits/eval/articles \
        --source-labels   data/splits/eval/eval-task2-TC.labels \
        --output-dir      data/splits/eval80 \
        --num-articles    80 \
        --seed            42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.data.loader import load_semeval
from src.schemas import Article, Technique

logger = logging.getLogger(__name__)


def select_subset(
    articles: list[Article],
    num_articles: int,
    seed: int = 42,
) -> list[Article]:
    """Pick `num_articles` articles so every Technique appears at least once."""
    if num_articles < len(Technique):
        raise ValueError(
            f"num_articles ({num_articles}) must be >= number of techniques "
            f"({len(Technique)}) to guarantee coverage."
        )

    rng = random.Random(seed)
    candidates = [a for a in articles if a.gold_spans]
    rng.shuffle(candidates)

    covered: set[Technique] = set()
    selected: list[Article] = []
    remaining = list(candidates)

    all_techniques = set(Technique)

    while covered != all_techniques and remaining:
        missing = all_techniques - covered

        def gain(article: Article) -> int:
            return len({s.technique for s in article.gold_spans} & missing)

        best = max(remaining, key=gain)
        if gain(best) == 0:
            break
        selected.append(best)
        remaining.remove(best)
        covered.update(s.technique for s in best.gold_spans)

    uncovered = all_techniques - covered
    if uncovered:
        names = sorted(t.value for t in uncovered)
        raise RuntimeError(
            f"Could not cover all techniques from source. Missing: {names}"
        )

    while len(selected) < num_articles and remaining:
        selected.append(remaining.pop())

    if len(selected) < num_articles:
        logger.warning(
            "Only %d articles with gold spans available; requested %d.",
            len(selected), num_articles,
        )

    rng.shuffle(selected)
    return selected


def write_subset(articles: list[Article], output_dir: Path, split_name: str) -> None:
    articles_dir = output_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for article in articles:
        (articles_dir / f"{article.id}.txt").write_text(article.text, encoding="utf-8")
        for span in article.gold_spans:
            lines.append(
                f"{article.id}\t{span.technique.value}\t{span.start}\t{span.end}"
            )

    labels_path = output_dir / f"{split_name}-task2-TC.labels"
    labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_stats(articles: list[Article]) -> None:
    counts: Counter[str] = Counter()
    for a in articles:
        for s in a.gold_spans:
            counts[s.technique.value] += 1

    total = sum(counts.values())
    print(f"\n  Subset: {len(articles)} articles, {total} spans")
    print(f"  {'Technique':<40} {'Spans':>6}")
    print(f"  {'-'*40} {'-'*6}")
    for t in sorted(Technique, key=lambda x: -counts.get(x.value, 0)):
        print(f"  {t.value:<40} {counts.get(t.value, 0):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source-articles", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--num-articles", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    articles = load_semeval(args.source_articles, args.source_labels)
    subset = select_subset(articles, args.num_articles, args.seed)
    print_stats(subset)

    output_dir = Path(args.output_dir)
    write_subset(subset, output_dir, args.split_name)

    meta = {
        "source_articles_dir": args.source_articles,
        "source_labels_path": args.source_labels,
        "num_articles": len(subset),
        "seed": args.seed,
        "article_ids": sorted(a.id for a in subset),
    }
    (output_dir / "subset_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  Written to {output_dir}/")
    print(f"    {output_dir}/articles/   ({len(subset)} articles)")
    print(f"    {output_dir}/{args.split_name}-task2-TC.labels")
    print(f"    {output_dir}/subset_metadata.json")


if __name__ == "__main__":
    main()
