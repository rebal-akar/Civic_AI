"""
ASV Stage 3: Adjudication — calibrated verdict.

The adjudicator receives both Stage 1 detections and Stage 2 critiques,
then makes the final call on each detection using structured evidence
evaluation and calibrated confidence.

Research grounding:
- Base rate anchoring (Xiong et al., 2024): explicit prior probabilities
  counteract LLM overdetection bias from RLHF training.
- Multi-step confidence decomposition: per-criterion confidence multiplied
  together naturally deflates inflated overall confidence.
- Technique-tier thresholds: Tier 3-4 techniques (pragmatic inference)
  require higher confidence for confirmation, reflecting empirically
  demonstrated LLM unreliability on these categories.
- Kasner et al. (2025): soft-hard F1 delta of 0.218 for best model on
  propaganda — models find spans but misclassify techniques. The
  adjudicator must specifically verify technique assignment.

Decision matrix (from research report):
| Agreement | Critique | Confidence | Verdict    |
|-----------|----------|------------|------------|
| 3/3       | CONCEDE  | ≥70%       | CONFIRMED  |
| 3/3       | CHALLENGE| ≥80%       | CONFIRMED  |
| 2/3       | CONCEDE  | ≥70%       | CONFIRMED  |
| 2/3       | CHALLENGE| ≥75%       | POSSIBLE   |
| 1/3       | CONCEDE  | ≥80%       | POSSIBLE   |
| 1/3       | CHALLENGE| any        | REJECTED   |
| any       | any      | <50%       | REJECTED   |
"""

STAGE3_SYSTEM = """You are an adjudicator for propaganda detection. You will receive:
1. A text passage
2. Spans flagged as potential propaganda (with technique labels and how many detection passes found them)
3. Adversarial critiques challenging each detection

Your job is to make the FINAL verdict on each detection.

For EACH detection, answer THREE questions:

QUESTION 1 — Does the span match the technique definition?
Look at the specific text. Does it actually do what the technique describes? Be concrete.

QUESTION 2 — Is the critique's innocent explanation convincing?
Rate it:
- WEAK: The innocent explanation is a stretch. The text really does look like propaganda.
- MODERATE: The innocent explanation is possible but the propaganda reading is stronger.
- STRONG: The innocent explanation is equally or more plausible than propaganda.

QUESTION 3 — Final verdict and confidence:
Based on your answers above, assign:
- confidence: a number from 0 to 100
- verdict: one of CONFIRMED, POSSIBLE, or REJECTED

Guidelines for verdict:
- CONFIRMED: The text clearly uses this propaganda technique. Confidence ≥ 65.
- POSSIBLE: Could be propaganda but uncertain. Confidence 40-64.
- REJECTED: The critique is convincing or the technique doesn't fit. Confidence < 40.

IMPORTANT: Detections found by all 3 detection passes (agreement_count=3) had strong consensus — give these the benefit of the doubt. Detections found by only 1 pass deserve more scrutiny.

Output a JSON object:
{
  "verdicts": [
    {
      "span_text": "the exact span text",
      "technique": "the technique label",
      "technique_match": "yes/no and brief reason",
      "critique_strength": "WEAK / MODERATE / STRONG",
      "confidence": 75,
      "verdict": "CONFIRMED / POSSIBLE / REJECTED"
    }
  ]
}
"""

STAGE3_USER = """Review the following detections and critiques, then deliver your final verdict on each.

ORIGINAL TEXT:
\"\"\"
{text}
\"\"\"

STAGE 1 DETECTIONS:
{detections_json}

STAGE 2 CRITIQUES:
{critiques_json}

Evaluate each detection and output your verdicts as JSON.
"""


def format_detections_for_stage3(detections: list[dict]) -> str:
    """Format detections for the adjudicator (includes agreement count)."""
    import json
    formatted = []
    for i, det in enumerate(detections, 1):
        formatted.append({
            "id": i,
            "span_text": det["span_text"],
            "technique": det["technique"],
            "agreement_count": det.get("agreement_count", 1),
        })
    return json.dumps(formatted, indent=2)


def format_critiques_for_stage3(critiques: list[dict]) -> str:
    """Format Stage 2 critiques for the adjudicator."""
    import json
    formatted = []
    for c in critiques:
        formatted.append({
            "span_text": c.get("span_text", ""),
            "technique": c.get("technique", ""),
            "innocent_explanation": c.get("innocent_explanation", ""),
            "weakness": c.get("weakness", ""),
            "verdict": c.get("verdict", ""),
        })
    return json.dumps(formatted, indent=2)