"""
ASV Stage 2: Critique — structured adversarial challenge.

The critique agent's sole job is to provide the strongest possible
INNOCENT explanation for each detection from Stage 1, to root out
false positives where the LLM flagged legitimate persuasion as propaganda.

Research grounding:
- CoVe independence principle (Dhuliawala et al., 2023): Stage 2 does NOT
  receive Stage 1's reasoning chains. It sees only the flagged spans +
  technique labels + original text. This prevents autoregressive copying
  of the detector's rationale — the mechanism that reduced hallucinated
  entities by 77% in CoVe's factored variant.
- Seven challenge questions derived from cross-framework synthesis:
  Walton (1992, 1995), Godber & Origgi (2023), pragma-dialectics
  (van Eemeren & Grootendorst), Jowett & O'Donnell, Gorin/Mills/Noggle.
- Decomposition principle (CoVe + Self-Verification): break holistic
  evaluation into narrow, answerable sub-questions. Models answer narrow
  questions more reliably than they make holistic judgments.
- Criterion-grounding principle (Constitutional AI): every critique must
  reference a specific, explicit criterion — not vague evaluation.
- Kasner et al. (2025) finding: CoT made Llama 3.3 "overthink" and dismiss
  valid detections. The critique must challenge WITHOUT dismissing everything.

Output format: same Kasner & Dušek JSON structure for consistency.
"""

STAGE2_SYSTEM = """You are a critical analyst specialising in distinguishing legitimate persuasion from propaganda. You have been given a list of text spans that were flagged as potential propaganda techniques by a detection system. Your job is to play DEVIL'S ADVOCATE — for each flagged span, construct the STRONGEST possible argument that it is NOT propaganda but rather legitimate persuasion, standard rhetoric, or neutral language.

You are NOT trying to confirm the detections. You are trying to CHALLENGE them. For each detection, you must:

1. Apply the SUBSTITUTION TEST: If the flagged rhetorical element were removed or neutralised, would the underlying argument collapse? If a substantive factual argument remains, the rhetoric may be supporting rather than replacing argument.

2. Apply the CONTEXTUAL FUNCTION TEST: In context, does the flagged text serve an informational purpose (describing, explaining, attributing) or a manipulative purpose (replacing argument, foreclosing questioning, directing toward a predetermined conclusion)?

3. Apply the PROPORTIONALITY TEST: Is the emotional/rhetorical intensity proportionate to the factual claims made? Emotional language describing a genuinely serious event may be proportionate, not propagandistic.

4. Provide the BEST INNOCENT EXPLANATION: What is the most plausible non-propagandistic reading? Consider:
   - Standard journalistic language conventions
   - Legitimate opinion expression
   - Direct quotation or attribution
   - Factual description that happens to be emotionally charged
   - Audience-appropriate simplification
   - Conventional genre expectations (editorial, campaign speech, opinion column)

5. Apply the TECHNIQUE SPECIFICITY TEST: Does this span match the SPECIFIC definition of the claimed technique, or does it merely share surface features? For example:
   - Emotional language is NOT automatically "Loaded_Language" — it must replace rather than accompany factual argument.
   - A patriotic reference is NOT automatically "Flag-Waving" — it must exploit group loyalty to justify an unrelated position.
   - A simplification is NOT automatically "Causal_Oversimplification" — it may be appropriate for the context.
   - A stock phrase is NOT automatically "Thought-terminating_Cliches" — it may be a genuine conversational marker.

For each detection, output your verdict:
- "CHALLENGE": You found a plausible innocent explanation. The detection may be a false positive.
- "CONCEDE": You could NOT find a plausible innocent explanation. The detection appears genuine.

Output a JSON object:
{
  "critiques": [
    {
      "span_text": "the exact flagged span text",
      "technique": "the claimed technique",
      "substitution_test": "What remains if the rhetorical element is removed?",
      "innocent_explanation": "The strongest non-propagandistic reading",
      "weakness": "The specific weakness in the original detection (if any)",
      "verdict": "CHALLENGE or CONCEDE"
    }
  ]
}

IMPORTANT: Be genuinely adversarial but intellectually honest. Do NOT:
- Challenge detections just to challenge them — if a span is clearly propaganda, CONCEDE.
- Dismiss all detections — you should concede on genuinely propagandistic spans.
- Provide vague challenges like "this could be interpreted differently" — be SPECIFIC about what the innocent reading is.
"""

STAGE2_USER = """The following text was analysed for propaganda techniques. A detection system flagged the spans listed below. For EACH flagged span, construct the strongest possible argument that it is NOT propaganda, then give your verdict (CHALLENGE or CONCEDE).

ORIGINAL TEXT:
\"\"\"
{text}
\"\"\"

FLAGGED DETECTIONS:
{detections_json}

For each detection above, provide your structured critique as JSON.
"""


def format_detections_for_stage2(detections: list[dict]) -> str:
    """Format Stage 1 detections for the Stage 2 prompt.

    Per the CoVe independence principle: includes ONLY span text,
    technique label, and agreement count. Does NOT include Stage 1's
    reasoning chains.
    """
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
