"""Tests for data splitting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import Article, GoldSpan, Technique
from scripts.split_data import stratified_split


def test_split_preserves_all_articles():
    """No articles lost during split."""
    articles = [
        Article(id=f"article{i}", text=f"text {i}", gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=5),
        ])
        for i in range(20)
    ]
    train, eval_ = stratified_split(articles, eval_ratio=0.3, seed=42)
    assert len(train) + len(eval_) == len(articles)
    all_ids = {a.id for a in train} | {a.id for a in eval_}
    assert all_ids == {a.id for a in articles}
    print("  ✓ all articles preserved")


def test_split_ratio():
    """Eval set is approximately the requested ratio."""
    articles = [
        Article(id=f"article{i}", text=f"text {i}", gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=5),
        ])
        for i in range(100)
    ]
    train, eval_ = stratified_split(articles, eval_ratio=0.3, seed=42)
    # Should be roughly 30% eval, allow some tolerance due to rounding
    assert 20 <= len(eval_) <= 40
    print(f"  ✓ split ratio: {len(eval_)}/{len(articles)} = {len(eval_)/len(articles):.2f}")


def test_split_rare_techniques_in_both():
    """Rare techniques appear in both train and eval."""
    articles = []
    # 50 articles with only loaded language (common)
    for i in range(50):
        articles.append(Article(id=f"article_common{i}", text=f"text {i}", gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=5),
        ]))
    # 5 articles with whataboutism (rare)
    for i in range(5):
        articles.append(Article(id=f"article_rare{i}", text=f"text {i}", gold_spans=[
            GoldSpan(technique=Technique.WHATABOUTISM, start=0, end=10),
        ]))

    train, eval_ = stratified_split(articles, eval_ratio=0.3, seed=42)

    train_techniques = {s.technique for a in train for s in a.gold_spans}
    eval_techniques = {s.technique for a in eval_ for s in a.gold_spans}

    assert Technique.WHATABOUTISM in eval_techniques, "Rare technique missing from eval"
    assert Technique.WHATABOUTISM in train_techniques, "Rare technique missing from train"
    print("  ✓ rare techniques in both splits")


def test_split_no_spans_go_to_train():
    """Articles without gold spans go to train (nothing to evaluate)."""
    articles = [
        Article(id="with_spans", text="text", gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=4),
        ]),
        Article(id="no_spans", text="clean text", gold_spans=[]),
    ]
    train, eval_ = stratified_split(articles, eval_ratio=0.5, seed=42)

    eval_ids = {a.id for a in eval_}
    train_ids = {a.id for a in train}
    assert "no_spans" in train_ids, "No-span article should be in train"
    assert "no_spans" not in eval_ids, "No-span article should not be in eval"
    print("  ✓ no-span articles go to train")


def test_split_deterministic():
    """Same seed produces same split."""
    articles = [
        Article(id=f"article{i}", text=f"text {i}", gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=5),
        ])
        for i in range(50)
    ]
    train1, eval1 = stratified_split(articles, eval_ratio=0.3, seed=42)
    train2, eval2 = stratified_split(articles, eval_ratio=0.3, seed=42)

    assert [a.id for a in train1] == [a.id for a in train2]
    assert [a.id for a in eval1] == [a.id for a in eval2]
    print("  ✓ deterministic with same seed")


if __name__ == "__main__":
    print("\nSplit tests:")
    test_split_preserves_all_articles()
    test_split_ratio()
    test_split_rare_techniques_in_both()
    test_split_no_spans_go_to_train()
    test_split_deterministic()
    print("\n🎉 All split tests passed!")