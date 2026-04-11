"""
Tests for ASV-specific components: critique parsing, verdict application.

Run: cd propaganda-detection && python tests/test_asv.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import Technique, PredictedSpan
from src.experiment import _parse_critiques, _parse_verdicts, _apply_verdicts


# ── Stage 2 Critique Parsing ──────────────────────────────────────────────────

def test_parse_critiques_clean():
    """Parse well-formed Stage 2 output."""
    raw = '''{
  "critiques": [
    {
      "span_text": "corrupt elites",
      "technique": "Name_Calling,Labeling",
      "substitution_test": "Removing the label leaves 'the [people] want power' — argument survives",
      "innocent_explanation": "This is a factually accurate label for people convicted of corruption",
      "weakness": "The label is used instead of engaging with their actual positions",
      "verdict": "CONCEDE"
    },
    {
      "span_text": "devastating earthquake",
      "technique": "Loaded_Language",
      "substitution_test": "Replacing with 'earthquake' — factual content preserved",
      "innocent_explanation": "Standard journalistic description of a genuinely severe event; proportionate language",
      "weakness": "The emotional language accompanies factual reporting, not replacing argument",
      "verdict": "CHALLENGE"
    }
  ]
}'''
    critiques = _parse_critiques(raw)
    assert len(critiques) == 2
    assert critiques[0]["verdict"] == "CONCEDE"
    assert critiques[1]["verdict"] == "CHALLENGE"
    assert "factually accurate" in critiques[0]["innocent_explanation"]
    print("  ✓ Stage 2 clean parse")


def test_parse_critiques_with_think():
    """Parse Stage 2 output wrapped in reasoning traces."""
    raw = '''<think>
Let me think about each detection carefully...
The first one is clearly propaganda but the second might not be.
</think>
{
  "critiques": [
    {
      "span_text": "radical mob",
      "technique": "Name_Calling,Labeling",
      "substitution_test": "test",
      "innocent_explanation": "none found",
      "weakness": "",
      "verdict": "CONCEDE"
    }
  ]
}'''
    critiques = _parse_critiques(raw)
    assert len(critiques) == 1
    assert critiques[0]["verdict"] == "CONCEDE"
    print("  ✓ Stage 2 parse with <think> block")


def test_parse_critiques_empty():
    """Handle empty or malformed critique output."""
    assert _parse_critiques("") == []
    assert _parse_critiques("not json at all") == []
    assert _parse_critiques('{"critiques": []}') == []
    print("  ✓ Stage 2 empty/malformed handling")


# ── Stage 3 Verdict Parsing ───────────────────────────────────────────────────

def test_parse_verdicts_clean():
    """Parse well-formed Stage 3 output."""
    raw = '''{
  "verdicts": [
    {
      "span_text": "corrupt elites",
      "technique": "Name_Calling,Labeling",
      "evidence_for": "Pejorative label used instead of substantive engagement",
      "evidence_against": "Could be accurate description",
      "critique_rating": "PLAUSIBLE",
      "confidence": 78,
      "verdict": "CONFIRMED"
    },
    {
      "span_text": "devastating earthquake",
      "technique": "Loaded_Language",
      "evidence_for": "Strong emotional language",
      "evidence_against": "Proportionate to actual severity of event",
      "critique_rating": "COMPELLING",
      "confidence": 35,
      "verdict": "REJECTED"
    }
  ]
}'''
    verdicts = _parse_verdicts(raw)
    assert len(verdicts) == 2
    assert verdicts[0]["verdict"] == "CONFIRMED"
    assert verdicts[0]["confidence"] == 78
    assert verdicts[1]["verdict"] == "REJECTED"
    print("  ✓ Stage 3 clean parse")


# ── Verdict Application ──────────────────────────────────────────────────────

def test_apply_verdicts():
    """Verdicts are correctly matched to spans and applied."""
    spans = [
        PredictedSpan(
            technique=Technique.NAME_CALLING,
            span_text="corrupt elites",
            start=4, end=18,
        ),
        PredictedSpan(
            technique=Technique.LOADED_LANGUAGE,
            span_text="devastating earthquake",
            start=30, end=52,
        ),
    ]

    verdicts = [
        {
            "span_text": "corrupt elites",
            "technique": "Name_Calling,Labeling",
            "confidence": 82,
            "verdict": "CONFIRMED",
        },
        {
            "span_text": "devastating earthquake",
            "technique": "Loaded_Language",
            "confidence": 35,
            "verdict": "REJECTED",
        },
    ]

    _apply_verdicts(spans, verdicts)

    assert spans[0].verdict == "CONFIRMED"
    assert spans[0].confidence == 82.0
    assert spans[1].verdict == "REJECTED"
    assert spans[1].confidence == 35.0
    print("  ✓ verdict application")


def test_apply_verdicts_case_insensitive():
    """Verdict matching handles case differences in span text."""
    spans = [
        PredictedSpan(
            technique=Technique.NAME_CALLING,
            span_text="The Corrupt Elites",
            start=0, end=18,
        ),
    ]

    verdicts = [
        {
            "span_text": "the corrupt elites",  # Different case
            "technique": "Name_Calling,Labeling",
            "confidence": 75,
            "verdict": "CONFIRMED",
        },
    ]

    _apply_verdicts(spans, verdicts)
    assert spans[0].verdict == "CONFIRMED"
    print("  ✓ verdict matching case-insensitive")


def test_confirmed_spans_filtering():
    """Prediction.confirmed_spans correctly filters by verdict."""
    from src.schemas import Prediction

    pred = Prediction(
        article_id="test",
        spans=[
            PredictedSpan(
                technique=Technique.NAME_CALLING,
                span_text="corrupt elites",
                start=4, end=18,
                verdict="CONFIRMED", confidence=85.0,
            ),
            PredictedSpan(
                technique=Technique.LOADED_LANGUAGE,
                span_text="devastating",
                start=30, end=41,
                verdict="REJECTED", confidence=30.0,
            ),
            PredictedSpan(
                technique=Technique.APPEAL_TO_FEAR,
                span_text="destroy everything",
                start=50, end=68,
                verdict="CONFIRMED", confidence=72.0,
            ),
        ],
    )

    confirmed = pred.confirmed_spans
    assert len(confirmed) == 2
    assert all(s.verdict == "CONFIRMED" for s in confirmed)
    assert confirmed[0].technique == Technique.NAME_CALLING
    assert confirmed[1].technique == Technique.APPEAL_TO_FEAR
    print("  ✓ confirmed_spans filtering")


# ── Run All ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStage 2 (Critique) tests:")
    test_parse_critiques_clean()
    test_parse_critiques_with_think()
    test_parse_critiques_empty()

    print("\nStage 3 (Adjudication) tests:")
    test_parse_verdicts_clean()

    print("\nVerdict application tests:")
    test_apply_verdicts()
    test_apply_verdicts_case_insensitive()
    test_confirmed_spans_filtering()

    print("\n🎉 All ASV tests passed!")
