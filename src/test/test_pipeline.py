"""
Tests for core pipeline components.

Run: cd propaganda-detection && python -m pytest tests/ -v
Or:  cd propaganda-detection && python tests/test_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    Technique,
    normalise_technique,
    GoldSpan,
    PredictedSpan,
    Article,
    Prediction,
)
from src.parser import (
    parse_llm_response,
    merge_multipass,
    _resolve_offsets,
)
from src.evaluation.metrics import evaluate, _char_overlap, _merge_overlapping_spans


# ── Schema Tests ──────────────────────────────────────────────────────────────

def test_normalise_technique_canonical():
    """Canonical SemEval labels resolve correctly."""
    assert normalise_technique("Loaded_Language") == Technique.LOADED_LANGUAGE
    assert normalise_technique("Name_Calling,Labeling") == Technique.NAME_CALLING
    assert normalise_technique("Appeal_to_Fear-Prejudice") == Technique.APPEAL_TO_FEAR
    assert normalise_technique("Flag-Waving") == Technique.FLAG_WAVING
    assert normalise_technique("Black-and-White_Fallacy") == Technique.BLACK_AND_WHITE
    assert normalise_technique("Thought-terminating_Cliches") == Technique.THOUGHT_TERMINATING
    assert normalise_technique("Bandwagon,Reductio_ad_hitlerum") == Technique.BANDWAGON
    print("  ✓ canonical labels")


def test_normalise_technique_llm_variants():
    """Common LLM output variations resolve correctly."""
    assert normalise_technique("loaded language") == Technique.LOADED_LANGUAGE
    assert normalise_technique("Loaded Language") == Technique.LOADED_LANGUAGE
    assert normalise_technique("name calling") == Technique.NAME_CALLING
    assert normalise_technique("Appeal to Fear") == Technique.APPEAL_TO_FEAR
    assert normalise_technique("flag_waving") == Technique.FLAG_WAVING
    assert normalise_technique("false dilemma") == Technique.BLACK_AND_WHITE
    assert normalise_technique("False_Dilemma") == Technique.BLACK_AND_WHITE
    assert normalise_technique("thought terminating cliche") == Technique.THOUGHT_TERMINATING
    assert normalise_technique("exaggeration") == Technique.EXAGGERATION
    print("  ✓ LLM variant labels")


def test_normalise_technique_unknown():
    """Unknown labels return None."""
    assert normalise_technique("MadeUpThing") is None
    assert normalise_technique("") is None
    assert normalise_technique("   ") is None
    print("  ✓ unknown labels")


# ── Parser Tests ──────────────────────────────────────────────────────────────

def test_parse_clean_json():
    """Parse well-formed JSON response."""
    raw = '''{"annotations": [
        {"text": "corrupt elites", "type": "Name_Calling,Labeling", "reason": "Pejorative label"},
        {"text": "destroy our nation", "type": "Appeal_to_Fear-Prejudice", "reason": "Fear-based"}
    ]}'''
    original = "The corrupt elites want to destroy our nation forever"

    spans, errors = parse_llm_response(raw, original)
    assert len(spans) == 2
    assert len(errors) == 0
    assert spans[0].technique == Technique.NAME_CALLING
    assert spans[0].span_text == "corrupt elites"
    assert spans[0].start == 4  # "The " = 4 chars
    assert spans[0].end == 18
    assert spans[1].technique == Technique.APPEAL_TO_FEAR
    print("  ✓ clean JSON parse")


def test_parse_json_in_fences():
    """Parse JSON wrapped in markdown fences."""
    raw = '''Here's my analysis:

```json
{"annotations": [
    {"text": "radical mob", "type": "Name_Calling,Labeling", "reason": "Label"}
]}
```'''
    original = "The radical mob is coming"

    spans, errors = parse_llm_response(raw, original)
    assert len(spans) == 1
    assert spans[0].technique == Technique.NAME_CALLING
    print("  ✓ JSON in fences")


def test_parse_empty_annotations():
    """Parse response with no propaganda detected."""
    raw = '{"annotations": []}'
    spans, errors = parse_llm_response(raw, "Normal text here")
    assert len(spans) == 0
    assert len(errors) == 0
    print("  ✓ empty annotations")


def test_parse_reasoning_model():
    """Parse output with <think> blocks (DeepSeek-R1 style)."""
    raw = '''<think>
Let me analyze this text step by step...
The phrase "corrupt elites" is clearly name calling.
</think>
{"annotations": [
    {"text": "corrupt elites", "type": "Name_Calling,Labeling", "reason": "Label"}
]}'''
    spans, errors = parse_llm_response(raw, "The corrupt elites want power")
    assert len(spans) == 1
    assert spans[0].technique == Technique.NAME_CALLING
    print("  ✓ reasoning model output")


def test_parse_unknown_technique():
    """Unknown technique generates error but doesn't crash."""
    raw = '''{"annotations": [
        {"text": "some text", "type": "MadeUpTechnique", "reason": "test"},
        {"text": "corrupt elites", "type": "Name_Calling,Labeling", "reason": "real"}
    ]}'''
    spans, errors = parse_llm_response(raw, "some text by corrupt elites")
    assert len(spans) == 1  # Only the valid one
    assert len(errors) == 1  # One error for unknown technique
    print("  ✓ unknown technique handling")


def test_offset_resolution():
    """String matching offset resolution works correctly."""
    text = "The corrupt elites want to destroy our great nation"

    # Exact match
    start, end = _resolve_offsets("corrupt elites", text)
    assert start == 4
    assert end == 18

    # Case-insensitive
    start, end = _resolve_offsets("Corrupt Elites", text)
    assert start == 4

    # Unresolvable
    start, end = _resolve_offsets("nonexistent phrase", text)
    assert start == -1
    assert end == -1

    print("  ✓ offset resolution")


def test_multipass_merge():
    """Multi-pass union merge with agreement counting."""
    text = "The corrupt elites want to destroy our nation"

    pass0 = [
        PredictedSpan(technique=Technique.NAME_CALLING, span_text="corrupt elites",
                      start=4, end=18, pass_id=0),
        PredictedSpan(technique=Technique.APPEAL_TO_FEAR, span_text="destroy our nation",
                      start=27, end=45, pass_id=0),
    ]
    pass1 = [
        PredictedSpan(technique=Technique.NAME_CALLING, span_text="corrupt elites",
                      start=4, end=18, pass_id=1),
    ]
    pass2 = [
        PredictedSpan(technique=Technique.NAME_CALLING, span_text="corrupt elites",
                      start=4, end=18, pass_id=2),
        PredictedSpan(technique=Technique.FLAG_WAVING, span_text="our nation",
                      start=35, end=45, pass_id=2),
    ]

    merged = merge_multipass([pass0, pass1, pass2], text)

    # Should have 3 unique detections (union)
    assert len(merged) == 3

    nc = [s for s in merged if s.technique == Technique.NAME_CALLING][0]
    assert nc.agreement_count == 3  # All 3 passes

    af = [s for s in merged if s.technique == Technique.APPEAL_TO_FEAR][0]
    assert af.agreement_count == 1  # Only pass 0

    fw = [s for s in merged if s.technique == Technique.FLAG_WAVING][0]
    assert fw.agreement_count == 1  # Only pass 2

    print("  ✓ multi-pass merge")


# ── Evaluation Tests ──────────────────────────────────────────────────────────

def test_char_overlap():
    """Character overlap computation."""
    assert _char_overlap(0, 10, 5, 15) == 5
    assert _char_overlap(0, 10, 10, 20) == 0  # Adjacent, no overlap
    assert _char_overlap(0, 10, 0, 10) == 10  # Perfect overlap
    assert _char_overlap(0, 10, 20, 30) == 0  # No overlap
    assert _char_overlap(3, 7, 0, 10) == 4    # Contained
    print("  ✓ character overlap")


def test_evaluate_perfect():
    """Perfect prediction gives F1 = 1.0."""
    articles = [Article(
        id="test1",
        text="The corrupt elites want to destroy our nation",
        gold_spans=[
            GoldSpan(technique=Technique.NAME_CALLING, start=4, end=18),
        ],
    )]
    predictions = {
        "test1": Prediction(
            article_id="test1",
            spans=[
                PredictedSpan(
                    technique=Technique.NAME_CALLING,
                    span_text="corrupt elites",
                    start=4, end=18,
                ),
            ],
        ),
    }

    metrics = evaluate(articles, predictions)
    assert metrics.si_f1 == 1.0
    assert metrics.si_precision == 1.0
    assert metrics.si_recall == 1.0
    assert metrics.tc_f1 == 1.0
    print("  ✓ perfect prediction eval")


def test_evaluate_wrong_technique():
    """Correct span but wrong technique — SI detects it, TC does not."""
    articles = [Article(
        id="test1",
        text="The corrupt elites want power",
        gold_spans=[
            GoldSpan(technique=Technique.NAME_CALLING, start=4, end=18),
        ],
    )]
    predictions = {
        "test1": Prediction(
            article_id="test1",
            spans=[
                PredictedSpan(
                    technique=Technique.LOADED_LANGUAGE,  # Wrong technique!
                    span_text="corrupt elites",
                    start=4, end=18,
                ),
            ],
        ),
    }

    metrics = evaluate(articles, predictions)
    assert metrics.tc_f1 == 0.0   # Wrong technique — no TC credit
    assert metrics.si_f1 == 1.0   # But span is correct — SI gives full credit
    assert metrics.f1_delta > 0   # SI > TC
    print("  ✓ wrong technique eval (SI=1, TC=0)")


def test_evaluate_partial_overlap():
    """Partial span overlap gives partial credit per SemEval formulas."""
    articles = [Article(
        id="test1",
        text="The corrupt elites want to destroy our nation",
        gold_spans=[
            GoldSpan(technique=Technique.NAME_CALLING, start=4, end=18),  # "corrupt elites" len=14
        ],
    )]
    predictions = {
        "test1": Prediction(
            article_id="test1",
            spans=[
                PredictedSpan(
                    technique=Technique.NAME_CALLING,
                    span_text="The corrupt elites want",
                    start=0, end=23,  # Wider than gold, len=23
                ),
            ],
        ),
    }

    metrics = evaluate(articles, predictions)
    # SemEval Eq 1 (Precision): |s∩t|/|t| = 14/14 = 1.0
    #   (overlap=14, divided by GOLD length=14)
    assert metrics.si_precision == 1.0
    # SemEval Eq 2 (Recall): |s∩t|/|s| = 14/23 ≈ 0.609
    #   (overlap=14, divided by PREDICTED length=23)
    assert abs(metrics.si_recall - 14/23) < 0.001
    assert 0.0 < metrics.si_f1 < 1.0
    print("  ✓ partial overlap eval (SemEval formula direction)")


def test_evaluate_no_predictions():
    """No predictions gives F1 = 0."""
    articles = [Article(
        id="test1",
        text="Some text",
        gold_spans=[
            GoldSpan(technique=Technique.LOADED_LANGUAGE, start=0, end=4),
        ],
    )]
    predictions = {}  # No predictions at all

    metrics = evaluate(articles, predictions)
    assert metrics.si_f1 == 0.0
    assert metrics.si_recall == 0.0
    print("  ✓ no predictions eval")


def test_evaluate_span_merging():
    """Overlapping predicted spans are merged before SI scoring."""
    articles = [Article(
        id="test1",
        text="The corrupt elites want to destroy our nation",
        gold_spans=[
            GoldSpan(technique=Technique.NAME_CALLING, start=4, end=18),  # "corrupt elites"
        ],
    )]
    # Two overlapping predictions covering the same region
    predictions = {
        "test1": Prediction(
            article_id="test1",
            spans=[
                PredictedSpan(technique=Technique.NAME_CALLING,
                              span_text="corrupt", start=4, end=11),
                PredictedSpan(technique=Technique.LOADED_LANGUAGE,
                              span_text="corrupt elites", start=4, end=18),
            ],
        ),
    }

    metrics = evaluate(articles, predictions)
    # After merging, there's one merged interval [4,18]
    # SI should treat it as a single span
    assert metrics.total_pred_spans == 1  # Merged count
    assert metrics.total_pred_spans_raw == 2  # Raw count
    assert metrics.si_f1 == 1.0  # Perfect SI match after merging
    print("  ✓ span merging eval")


# ── Run All ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSchema tests:")
    test_normalise_technique_canonical()
    test_normalise_technique_llm_variants()
    test_normalise_technique_unknown()

    print("\nParser tests:")
    test_parse_clean_json()
    test_parse_json_in_fences()
    test_parse_empty_annotations()
    test_parse_reasoning_model()
    test_parse_unknown_technique()
    test_offset_resolution()
    test_multipass_merge()

    print("\nEvaluation tests:")
    test_char_overlap()
    test_evaluate_perfect()
    test_evaluate_wrong_technique()
    test_evaluate_partial_overlap()
    test_evaluate_no_predictions()
    test_evaluate_span_merging()

    print("\n🎉 All tests passed!")
