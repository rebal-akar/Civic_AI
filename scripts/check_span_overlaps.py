#!/usr/bin/env python3
"""
Check for overlapping spans in the training labels.
Two spans overlap if span1.start < span2.end AND span2.start < span1.end
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = PROJECT_ROOT / "datasets" / "train-articles"
LABELS_DIR = PROJECT_ROOT / "datasets" / "train-labels-task2-technique-classification"


def load_labels(article_id: str) -> list[tuple[int, int, str]]:
    """Load labels for an article. Returns list of (start, end, technique)."""
    path = LABELS_DIR / f"article{article_id}.task2-TC.labels"
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            _, technique, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            labels.append((start, end, technique))
    return labels


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if spans (start, end) overlap. End is exclusive."""
    a_start, a_end = a[0], a[1]
    b_start, b_end = b[0], b[1]
    return a_start < b_end and b_start < a_end


def find_overlaps(labels: list[tuple[int, int, str]]) -> list[tuple[int, int, str, int, int, str]]:
    """Return list of (start1, end1, tech1, start2, end2, tech2) for overlapping pairs."""
    result = []
    for i, (s1, e1, t1) in enumerate(labels):
        for j, (s2, e2, t2) in enumerate(labels):
            if i >= j:
                continue
            if overlaps((s1, e1), (s2, e2)):
                result.append((s1, e1, t1, s2, e2, t2))
    return result


def main():
    articles_with_overlaps = []
    total_overlap_pairs = 0

    for path in sorted(LABELS_DIR.glob("article*.task2-TC.labels")):
        article_id = path.stem.replace("article", "").replace(".task2-TC", "")
        labels = load_labels(article_id)
        if len(labels) < 2:
            continue

        overlap_pairs = find_overlaps(labels)
        if overlap_pairs:
            articles_with_overlaps.append((article_id, labels, overlap_pairs))
            total_overlap_pairs += len(overlap_pairs)

    # Report
    total_articles = sum(1 for _ in LABELS_DIR.glob("article*.task2-TC.labels"))
    print(f"Checked {total_articles} articles with labels")
    print(f"Articles with overlapping spans: {len(articles_with_overlaps)}")
    print(f"Total overlapping span pairs: {total_overlap_pairs}")
    print()

    if articles_with_overlaps:
        print("Examples (first 10 articles with overlaps):")
        print("-" * 80)
        for article_id, labels, overlap_pairs in articles_with_overlaps[:10]:
            article_path = ARTICLES_DIR / f"article{article_id}.txt"
            text = article_path.read_text(encoding="utf-8") if article_path.exists() else ""
            print(f"\nArticle {article_id}: {len(overlap_pairs)} overlap pair(s)")
            for s1, e1, t1, s2, e2, t2 in overlap_pairs[:3]:  # show first 3 per article
                span1_text = repr(text[s1:e1][:50] + ("..." if e1 - s1 > 50 else ""))
                span2_text = repr(text[s2:e2][:50] + ("..." if e2 - s2 > 50 else ""))
                print(f"  [{s1}:{e1}] {t1}: {span1_text}")
                print(f"  [{s2}:{e2}] {t2}: {span2_text}")
                print()
    else:
        print("No overlapping spans found.")


if __name__ == "__main__":
    main()
