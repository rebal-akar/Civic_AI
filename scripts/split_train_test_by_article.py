"""
Split the train paragraph dataset into train/test by ARTICLE (not paragraph),
with stratified sampling so technique distribution is proportional.
~70% of articles -> train, ~30% -> test. All paragraphs of test articles go to test.
"""

import csv
import json
import re
from pathlib import Path
from random import Random

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
LABELS_FILE = DATASETS_DIR / "train-task2-TC.labels"
PARAGRAPH_JSON = DATASETS_DIR / "train-paragraph-techniques.json"
PARAGRAPH_CSV = DATASETS_DIR / "train-paragraph-techniques.csv"
OUT_TRAIN_JSON = DATASETS_DIR / "train-paragraph-train.json"
OUT_TRAIN_CSV = DATASETS_DIR / "train-paragraph-train.csv"
OUT_TEST_JSON = DATASETS_DIR / "train-paragraph-test.json"
OUT_TEST_CSV = DATASETS_DIR / "train-paragraph-test.csv"

TEST_FRACTION = 0.30
SEED = 42


def _normalize_technique(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[-,\s]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_article_techniques() -> dict[str, frozenset[str]]:
    """article_id -> set of technique names (normalized)."""
    out: dict[str, set[str]] = {}
    with open(LABELS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            article_id, technique, _start, _end = parts
            out.setdefault(article_id, set()).add(_normalize_technique(technique))
    return {aid: frozenset(techs) for aid, techs in out.items()}


def stratified_article_split(
    article_techniques: dict[str, frozenset[str]],
    test_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Split article IDs so that ~test_fraction go to test, stratified by
    technique set (articles with same techniques are split proportionally).
    Returns (train_article_ids, test_article_ids).
    """
    # Group by signature (sorted tuple of techniques)
    strata: dict[tuple, list[str]] = {}
    for aid, techs in article_techniques.items():
        sig = tuple(sorted(techs))
        strata.setdefault(sig, []).append(aid)
    rng = Random(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []
    for sig, aids in strata.items():
        rng.shuffle(aids)
        n_test = round(len(aids) * test_fraction)
        n_train = len(aids) - n_test
        train_ids.extend(aids[:n_train])
        test_ids.extend(aids[n_train:])
    # Sort for reproducible ordering
    train_ids.sort(key=lambda x: (int(x) if x.isdigit() else x))
    test_ids.sort(key=lambda x: (int(x) if x.isdigit() else x))
    return train_ids, test_ids


def main() -> None:
    article_techniques = load_article_techniques()
    train_article_ids, test_article_ids = stratified_article_split(
        article_techniques, TEST_FRACTION, SEED
    )
    test_set = set(test_article_ids)

    with open(PARAGRAPH_JSON, encoding="utf-8") as f:
        all_paragraphs = json.load(f)

    train_paragraphs = [p for p in all_paragraphs if p["article_id"] not in test_set]
    test_paragraphs = [p for p in all_paragraphs if p["article_id"] in test_set]

    # Write train
    with open(OUT_TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_paragraphs, f, indent=2, ensure_ascii=False)
    with open(OUT_TRAIN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["article_id", "paragraph_index", "techniques", "paragraph_text"],
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for p in train_paragraphs:
            w.writerow({
                "article_id": p["article_id"],
                "paragraph_index": p["paragraph_index"],
                "techniques": "|".join(p["techniques"]),
                "paragraph_text": p["paragraph_text"],
            })

    # Write test
    with open(OUT_TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_paragraphs, f, indent=2, ensure_ascii=False)
    with open(OUT_TEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["article_id", "paragraph_index", "techniques", "paragraph_text"],
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for p in test_paragraphs:
            w.writerow({
                "article_id": p["article_id"],
                "paragraph_index": p["paragraph_index"],
                "techniques": "|".join(p["techniques"]),
                "paragraph_text": p["paragraph_text"],
            })

    # Summary: technique proportions
    def technique_counts(paragraphs: list[dict]) -> dict[str, int]:
        c: dict[str, int] = {}
        for p in paragraphs:
            for t in p["techniques"]:
                c[t] = c.get(t, 0) + 1
        return c

    train_tech = technique_counts(train_paragraphs)
    test_tech = technique_counts(test_paragraphs)
    all_techs = sorted(set(train_tech) | set(test_tech))
    n_train_paras = len(train_paragraphs)
    n_test_paras = len(test_paragraphs)

    print(f"Split by article (stratified by technique set), test fraction = {TEST_FRACTION}, seed = {SEED}")
    print(f"  Train articles: {len(train_article_ids)}  ->  {n_train_paras} paragraphs")
    print(f"  Test  articles: {len(test_article_ids)}  ->  {n_test_paras} paragraphs")
    print(f"  {OUT_TRAIN_JSON}")
    print(f"  {OUT_TRAIN_CSV}")
    print(f"  {OUT_TEST_JSON}")
    print(f"  {OUT_TEST_CSV}")
    print("\nTechnique distribution (paragraph-level counts):")
    print(f"  {'Technique':<35} {'Train':>8} {'Test':>8}  Train%   Test%")
    for t in all_techs:
        tr = train_tech.get(t, 0)
        te = test_tech.get(t, 0)
        tr_pct = 100 * tr / n_train_paras if n_train_paras else 0
        te_pct = 100 * te / n_test_paras if n_test_paras else 0
        print(f"  {t:<35} {tr:>8} {te:>8}  {tr_pct:5.1f}%  {te_pct:5.1f}%")


if __name__ == "__main__":
    main()
