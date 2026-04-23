"""
Data loader for SemEval-2020 Task 11.

Reads raw SemEval format:
  - articles/article_XXXXX.txt  (plain text)
  - labels/article_XXXXX.labels.tsv  (article_id \t technique \t start \t end)

Returns Article objects with character-level gold spans.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.schemas import Article, GoldSpan, Technique, normalise_technique

logger = logging.getLogger(__name__)


def load_semeval(
    articles_dir: str | Path,
    labels_dir: str | Path | None = None,
) -> list[Article]:
    """Load SemEval-2020 Task 11 articles with span-level annotations.

    Args:
        articles_dir: Path to directory containing article_XXXXX.txt files.
        labels_dir: Path to directory containing article_XXXXX.labels.tsv files.
                    If None, loads articles without gold labels (for test prediction).

    Returns:
        List of Article objects sorted by ID.
    """
    articles_path = Path(articles_dir)
    if not articles_path.exists():
        raise FileNotFoundError(
            f"Articles directory not found: {articles_dir}\n"
            "Download from: https://propaganda.qcri.org/semeval2020-task11/"
        )

    # Load all labels into a lookup: article_id -> list of (technique, start, end)
    label_lookup: dict[str, list[tuple[str, int, int]]] = {}
    if labels_dir is not None:
        labels_path = Path(labels_dir)
        if labels_path.is_dir():
            # Per-article label files: article_XXXXX.labels.tsv
            for label_file in sorted(labels_path.glob("*.labels.tsv")):
                article_id = label_file.stem.replace(".labels", "")
                label_lookup[article_id] = _parse_label_file(label_file)
        elif labels_path.is_file():
            # Single consolidated label file (train format)
            label_lookup = _parse_consolidated_labels(labels_path)

    # Load articles
    articles: list[Article] = []
    for article_file in sorted(articles_path.glob("article*.txt")):
        article_id = article_file.stem
        text = article_file.read_text(encoding="utf-8")

        gold_spans: list[GoldSpan] = []
        for raw_technique, start, end in label_lookup.get(article_id, []):
            technique = normalise_technique(raw_technique)
            if technique is None:
                logger.warning(
                    f"Unknown technique '{raw_technique}' in {article_id}, skipping"
                )
                continue
            gold_spans.append(GoldSpan(technique=technique, start=start, end=end))

        articles.append(Article(id=article_id, text=text, gold_spans=gold_spans))

    logger.info(
        f"Loaded {len(articles)} articles "
        f"({sum(len(a.gold_spans) for a in articles)} total gold spans)"
    )
    _log_technique_distribution(articles)
    return articles


def _parse_label_file(path: Path) -> list[tuple[str, int, int]]:
    """Parse a per-article .labels.tsv file."""
    spans = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            # Format: article_id \t technique \t start \t end
            spans.append((parts[1], int(parts[2]), int(parts[3])))
        elif len(parts) == 3:
            # Format: technique \t start \t end (no article_id prefix)
            spans.append((parts[0], int(parts[1]), int(parts[2])))
    return spans


def _parse_consolidated_labels(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Parse a consolidated labels file (all articles in one file)."""
    lookup: dict[str, list[tuple[str, int, int]]] = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            article_id = _normalise_article_id(parts[0])
            technique = parts[1]
            start = int(parts[2])
            end = int(parts[3])
            lookup.setdefault(article_id, []).append((technique, start, end))
    return lookup


def _normalise_article_id(raw: str) -> str:
    """Normalise article ID to match filename format (article_XXXXX)."""
    raw = raw.strip()
    if raw.startswith("article"):
        return raw
    # Numeric-only ID — prepend 'article'
    return f"article{raw}"


def _log_technique_distribution(articles: list[Article]) -> None:
    """Log the distribution of techniques across articles."""
    from collections import Counter
    counts: Counter[str] = Counter()
    for article in articles:
        for span in article.gold_spans:
            counts[span.technique.value] += 1

    if not counts:
        logger.info("No gold spans found (unlabeled data)")
        return

    total = sum(counts.values())
    logger.info(f"Technique distribution ({total} spans):")
    for technique, count in counts.most_common():
        logger.info(f"  {technique}: {count} ({count/total*100:.1f}%)")