"""
Convert span-level propaganda labels to document-level: one row per article
with the full text and the list of unique techniques identified anywhere in the article.
"""

import csv
import json
from pathlib import Path

# Paths relative to project root (run from Civic_AI/)
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
LABELS_FILE = DATASETS_DIR / "train-task2-TC.labels"
TRAIN_ARTICLES_DIR = DATASETS_DIR / "train-articles"
OUT_CSV = DATASETS_DIR / "train-document-techniques.csv"
OUT_JSON = DATASETS_DIR / "train-document-techniques.json"


def load_article_text(article_id: str) -> str:
    """Load full raw text of an article (title + blank line + body)."""
    path = TRAIN_ARTICLES_DIR / f"article{article_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> None:
    # 1. Parse labels: article_id -> set of techniques
    article_techniques: dict[str, set[str]] = {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            article_id, technique, _start, _end = parts
            article_techniques.setdefault(article_id, set()).add(technique)

    # 2. Build rows: article_id, text, techniques (sorted list for stable output)
    rows = []
    for article_id in sorted(article_techniques.keys(), key=int):
        techniques = sorted(article_techniques[article_id])
        text = load_article_text(article_id)
        rows.append(
            {
                "article_id": article_id,
                "text": text,
                "techniques": techniques,
            }
        )

    # 3. Write CSV (techniques as pipe-separated)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["article_id", "techniques", "text"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "article_id": r["article_id"],
                    "techniques": "|".join(r["techniques"]),
                    "text": r["text"],
                }
            )

    # 4. Write JSON (techniques as list)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            [{"article_id": r["article_id"], "text": r["text"], "techniques": r["techniques"]} for r in rows],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Converted {len(rows)} articles.")
    print(f"  CSV:  {OUT_CSV}")
    print(f"  JSON: {OUT_JSON}")
    print(f"  Example techniques for first article: {rows[0]['techniques']}")


if __name__ == "__main__":
    main()
