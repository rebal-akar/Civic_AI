"""
Convert span-level labels to paragraph-level: split articles by double newline,
merge to min sentences, and assign techniques from overlapping spans.
Produces train (with labels) and dev (no labels, for evaluation) paragraph datasets.
"""

import csv
import json
import re
from pathlib import Path

# Paths relative to project root (run from Civic_AI/)
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
LABELS_FILE = DATASETS_DIR / "train-task2-TC.labels"
TRAIN_ARTICLES_DIR = DATASETS_DIR / "train-articles"
DEV_ARTICLES_DIR = DATASETS_DIR / "dev-articles"
OUT_TRAIN_CSV = DATASETS_DIR / "train-paragraph-techniques.csv"
OUT_TRAIN_JSON = DATASETS_DIR / "train-paragraph-techniques.json"
OUT_DEV_CSV = DATASETS_DIR / "dev-paragraph-techniques.csv"
OUT_DEV_JSON = DATASETS_DIR / "dev-paragraph-techniques.json"

MIN_SENTENCES = 4  # merge initial paragraphs until each has at least this many sentences


def _count_sentences(text: str) -> int:
    """Count sentences: one sentence per line (SemEval format), fallback to period-boundary count."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        return len(lines)
    # Fallback: split on sentence-ending punctuation
    return len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]) or 1


def _merge_paragraphs_min_sentences(
    paragraphs: list[tuple[int, int, str]],
    min_sentences: int,
) -> list[tuple[int, int, str]]:
    """
    Merge consecutive paragraphs until each resulting paragraph has at least
    min_sentences. Start from the first: if it has < min_sentences, merge with
    the next; repeat until >= min_sentences or run out, then move on.
    """
    if min_sentences <= 0 or not paragraphs:
        return paragraphs
    merged: list[tuple[int, int, str]] = []
    i = 0
    while i < len(paragraphs):
        p_start, p_end, p_text = paragraphs[i]
        j = i + 1
        while _count_sentences(p_text) < min_sentences and j < len(paragraphs):
            next_start, next_end, next_text = paragraphs[j]
            p_text = p_text + "\n\n" + next_text
            p_end = next_end
            j += 1
        merged.append((p_start, p_end, p_text))
        i = j
    return merged


def _normalize_technique(name: str) -> str:
    """Normalize SemEval technique name to snake_case (e.g. for schema alignment)."""
    s = name.lower().strip()
    s = re.sub(r"[-,\s]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_article_text(article_id: str, articles_dir: Path) -> str:
    """Load full raw text of an article (title + blank line + body)."""
    path = articles_dir / f"article{article_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _article_ids_from_dir(articles_dir: Path) -> list[str]:
    """Collect article IDs from article123.txt filenames."""
    ids = []
    for p in articles_dir.glob("article*.txt"):
        name = p.stem  # "article730093263"
        if name.startswith("article"):
            ids.append(name[7:])  # strip "article"
    return sorted(ids, key=lambda x: (int(x) if x.isdigit() else x))


def split_paragraphs_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """
    Split article text by double newline. Single newlines (e.g. inside a quoted
    block) stay part of the same paragraph.
    Returns list of (start_char, end_char, paragraph_text) in order.
    """
    if not text:
        return []
    paragraphs: list[tuple[int, int, str]] = []
    start = 0
    pos = 0
    n = len(text)
    while pos < n:
        # Find next double newline
        idx = text.find("\n\n", pos)
        if idx == -1:
            chunk = text[pos:].rstrip()
            if chunk:
                paragraphs.append((start, n, chunk))
            break
        chunk = text[pos:idx].rstrip()
        if chunk:
            paragraphs.append((start, idx, chunk))
        start = idx + 2  # skip \n\n
        pos = start
    return paragraphs


def build_paragraph_rows(
    articles_dir: Path,
    spans_by_article: dict[str, list[tuple[str, int, int]]],
    min_sentences: int,
) -> list[dict]:
    """Build paragraph-level rows. If spans_by_article is empty, use article IDs from articles_dir (e.g. dev)."""
    article_ids = (
        sorted(spans_by_article.keys(), key=lambda x: (int(x) if x.isdigit() else x))
        if spans_by_article
        else _article_ids_from_dir(articles_dir)
    )
    rows: list[dict] = []
    for article_id in article_ids:
        text = load_article_text(article_id, articles_dir)
        if not text:
            continue
        initial_paras = split_paragraphs_with_offsets(text)
        paras = _merge_paragraphs_min_sentences(initial_paras, min_sentences)
        spans = spans_by_article.get(article_id, [])

        for para_idx, (p_start, p_end, para_text) in enumerate(paras):
            techniques: set[str] = set()
            for technique, s_start, s_end in spans:
                if s_start < p_end and s_end > p_start:
                    techniques.add(_normalize_technique(technique))
            rows.append({
                "article_id": article_id,
                "paragraph_index": para_idx,
                "paragraph_text": para_text,
                "techniques": sorted(techniques),
            })
    return rows


def _write_paragraph_dataset(rows: list[dict], out_csv: Path, out_json: Path) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["article_id", "paragraph_index", "techniques", "paragraph_text"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "article_id": r["article_id"],
                "paragraph_index": r["paragraph_index"],
                "techniques": "|".join(r["techniques"]),
                "paragraph_text": r["paragraph_text"],
            })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "article_id": r["article_id"],
                    "paragraph_index": r["paragraph_index"],
                    "paragraph_text": r["paragraph_text"],
                    "techniques": r["techniques"],
                }
                for r in rows
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    # 1. Load train span-level labels
    spans_by_article: dict[str, list[tuple[str, int, int]]] = {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            article_id, technique, start_s, end_s = parts
            start, end = int(start_s), int(end_s)
            spans_by_article.setdefault(article_id, []).append((technique, start, end))

    # 2. Train paragraphs (with techniques)
    train_rows = build_paragraph_rows(TRAIN_ARTICLES_DIR, spans_by_article, MIN_SENTENCES)
    _write_paragraph_dataset(train_rows, OUT_TRAIN_CSV, OUT_TRAIN_JSON)
    n_train_with_labels = sum(1 for r in train_rows if r["techniques"])
    print(f"Train: {len(train_rows)} paragraphs from {len(spans_by_article)} articles.")
    print(f"  Paragraphs with >=1 technique: {n_train_with_labels}")
    print(f"  {OUT_TRAIN_CSV}")
    print(f"  {OUT_TRAIN_JSON}")

    # 3. Dev paragraphs (no gold techniques — for evaluation)
    dev_rows = build_paragraph_rows(DEV_ARTICLES_DIR, {}, MIN_SENTENCES)
    _write_paragraph_dataset(dev_rows, OUT_DEV_CSV, OUT_DEV_JSON)
    dev_article_ids = {r["article_id"] for r in dev_rows}
    print(f"Dev:   {len(dev_rows)} paragraphs from {len(dev_article_ids)} articles.")
    print(f"  {OUT_DEV_CSV}")
    print(f"  {OUT_DEV_JSON}")


if __name__ == "__main__":
    main()
