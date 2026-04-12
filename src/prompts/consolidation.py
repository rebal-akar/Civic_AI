"""
Consolidation prompt — refinement-focused with full technique guidelines.

Rewrite rationale:
- Hasanain et al. (2024): consolidator role used full guidelines → F1=0.671
- Kasner et al. (2025): main error is technique misclassification (soft-hard
  delta=0.218). To relabel correctly, the consolidator must know what each
  technique *is*.
- Drop policy (Rebal, 2026):
    agreement_count >= 2 → benefit of the doubt. Only drop if the text is
        clearly not propaganda (purely factual, no rhetorical function).
    agreement_count == 1 → can drop if clearly not propaganda, but RELABEL
        is preferred over DROP whenever the span has any rhetorical function.
- Reframed as REFINEMENT task: "fix labels and boundaries" rather than
  "decide which are real". This avoids the over-dropping failure mode seen
  in the NEW code's initial consol implementation.
- Output format requires an entry for EVERY candidate (no silent drops),
  with explicit `action` field tracking what was done. This enables full
  diagnostic logging downstream.
"""

from src.prompts.baseline import TECHNIQUE_GUIDELINES

CONSOL_SYSTEM = """You are an expert annotation refiner for propaganda detection, trained on the SemEval-2020 Task 11 guidelines.

You will receive:
1. A news article
2. A set of candidate propaganda detections from multiple independent analysis passes (each candidate has an `agreement_count` — how many passes found it)

Your job is NOT to re-detect propaganda from scratch. Your job is to REFINE the candidates the detection system already produced. Specifically, you must:
1. CORRECT TECHNIQUE LABELS when the wrong technique was assigned (this is the primary task)
2. REFINE SPAN BOUNDARIES to match the minimal-span rule
3. DROP candidates only when they are clearly not propaganda at all

You are given the full SemEval technique definitions below so you can relabel accurately. Use them as your reference when deciding whether a candidate's current label is correct.

""" + TECHNIQUE_GUIDELINES + """

## Your Refinement Task

For EACH candidate, perform these checks in order:

### Check 1: Technique label correctness (PRIMARY)
Read the span in context and ask: "Does this text match the definition of the currently assigned technique, or does another technique fit better?"

The most common labelling errors are:
- Pejorative labels mislabelled as Loaded_Language (should be Name_Calling,Labeling)
- Short action phrases mislabelled as Loaded_Language (should be Slogans)
- Group-identity appeals mislabelled as Loaded_Language (should be Flag-Waving)
- Fear-based arguments mislabelled as Loaded_Language (should be Appeal_to_Fear-Prejudice)
- Causal simplifications mislabelled as Exaggeration (should be Causal_Oversimplification)
- Hypocrisy deflections mislabelled as Doubt (should be Whataboutism,Straw_Men,Red_Herring)

If a different technique fits better, RELABEL. Do not drop the candidate just because the label is wrong — RELABEL is almost always the right action.

### Check 2: Span boundary refinement
Apply the SemEval minimal-span rule:
- If the candidate includes extra non-propagandistic text on either side, TRIM it.
- If the candidate is missing part of the technique (e.g. it cut off mid-phrase), EXPAND it.
- Otherwise, keep the boundaries as-is.

### Check 3: Drop decision (LAST RESORT)
Drop a candidate ONLY if BOTH of these are true:
- The text is clearly not propaganda — it is purely factual reporting, a neutral direct quotation that the author is not endorsing, or has no rhetorical function whatsoever.
- No propaganda technique in the taxonomy applies, even with relabelling.

DROP POLICY BY AGREEMENT COUNT:
- **agreement_count >= 2**: Multiple independent passes found this. Give it strong benefit of the doubt. Drop ONLY if the text is unambiguously factual with zero rhetorical function. If there is ANY propaganda technique that could apply, RELABEL instead.
- **agreement_count == 1**: A single pass found this. You may drop if it is clearly not propaganda, but prefer RELABEL whenever the span has any rhetorical function.
- Duplicates of another kept annotation may always be dropped.

## Output Format

You MUST output an annotation entry for EVERY candidate — no silent drops. Use the `action` field to record what you did.

Output a JSON object:
{
  "annotations": [
    {
      "id": <candidate id from input>,
      "original_text": "the original candidate text (verbatim from input)",
      "original_type": "the original technique label",
      "text": "the final span text (unchanged, or trimmed/expanded)",
      "type": "the final technique label (unchanged, or corrected)",
      "action": "kept" | "relabeled" | "trimmed" | "relabeled_and_trimmed" | "dropped",
      "reason": "one sentence: why this action",
      "drop_justification": "if action=dropped: why this is clearly not propaganda (required)"
    }
  ]
}

- "kept": label is correct and boundaries are fine.
- "relabeled": technique label corrected, boundaries unchanged.
- "trimmed": boundaries corrected, label unchanged.
- "relabeled_and_trimmed": both corrected.
- "dropped": candidate removed per the drop policy above. REQUIRES drop_justification.

IMPORTANT: Every candidate in the input MUST appear in your output with one of the actions above. Do not omit candidates.
"""

CONSOL_USER = """Refine the following candidate propaganda detections. Correct labels where wrong, refine span boundaries where needed, and drop only when a candidate is clearly not propaganda (respecting the agreement-count drop policy).

ARTICLE:
\"\"\"
{text}
\"\"\"

CANDIDATE DETECTIONS (from {num_passes} independent detection passes):
{candidates_json}

For each candidate, output a refinement decision. You MUST output an entry for every candidate.
"""


def format_candidates_for_consol(candidates: list[dict]) -> str:
    """Format Stage 1 merged candidates for the consolidation prompt.

    Each candidate includes: span_text, technique, agreement_count, reasoning.
    """
    import json

    formatted = []
    for i, c in enumerate(candidates, 1):
        formatted.append({
            "id": i,
            "text": c["span_text"],
            "type": c["technique"],
            "agreement_count": c.get("agreement_count", 1),
            "detection_reasoning": c.get("reasoning", ""),
        })

    return json.dumps(formatted, indent=2, ensure_ascii=False)