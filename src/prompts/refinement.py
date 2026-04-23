"""
Stage 3 refinement prompt for the `hybrid` strategy.

Distinct from `consolidation.py` in that it tells the LLM up front:
- You CANNOT drop candidates. Dropping is not an option.
- Your only jobs are technique label correction and span boundary refinement.
- Every candidate must come out the other side.

This avoids the "instruction then undo in Python" pattern used by the
no_drop=True flag. The LLM is told the truth about what it can do, which
is both cleaner experimentally and gives the model a more focused task.

Research grounding:
- Hasanain et al. (2024): LLMs excel at refining candidates when given
  a narrow, well-defined task. Removing the drop option narrows the task.
- Kasner et al. (2025): models "overthink" and second-guess valid
  detections when given adversarial framing. A refinement-only prompt
  removes this failure mode entirely.
- Rebal (2026) rationale: in the `hybrid` pipeline, Stage 2 (adversarial
  critique) has already handled filtering. Stage 3 should be a narrow
  refinement pass that polishes what Stage 2 kept, not a second filter.
"""

from src.prompts.detection import TECHNIQUE_GUIDELINES

REFINE_SYSTEM = """You are an expert annotation refiner for propaganda detection, trained on the SemEval-2020 Task 11 guidelines.

You will receive:
1. A news article
2. A set of propaganda detections that have ALREADY been validated by an adversarial review stage

Your job has TWO narrow tasks, and ONLY two tasks:

1. **Correct the technique label** if it is wrong.
2. **Refine the span boundaries** if they need trimming or expansion.

You MUST NOT drop, reject, or remove any candidate. Every candidate you receive will be kept. Your only decisions are: (a) is the label correct, and (b) are the boundaries correct. If a candidate seems weak or borderline to you, that is NOT your concern — it has already passed adversarial review. Your job is to polish it, not to re-litigate whether it belongs.

You are given the full SemEval technique definitions below so you can relabel accurately. Use them as your reference.

""" + TECHNIQUE_GUIDELINES + """

## Your Two Refinement Tasks

### Task 1: Technique label correctness (PRIMARY)
Read the span in context and ask: "Does this text match the definition of the currently assigned technique, or does a different technique from the 14 fit better?"

The most common labelling errors in detection output are:
- Pejorative labels mislabelled as Loaded_Language (should be Name_Calling,Labeling)
- Short action phrases mislabelled as Loaded_Language (should be Slogans)
- Group-identity appeals mislabelled as Loaded_Language (should be Flag-Waving)
- Fear-based arguments mislabelled as Loaded_Language (should be Appeal_to_Fear-Prejudice)
- Causal simplifications mislabelled as Exaggeration (should be Causal_Oversimplification)
- Hypocrisy deflections mislabelled as Doubt (should be Whataboutism,Straw_Men,Red_Herring)

If a different technique from the taxonomy fits better, RELABEL. If the current label is correct, leave it.

### Task 2: Span boundary refinement
Apply the SemEval minimal-span rule:
- If the candidate includes extra non-propagandistic text on either side, TRIM it.
- If the candidate is missing part of the technique (e.g. it cut off mid-phrase), EXPAND it.
- Otherwise, leave the boundaries alone.

## Output Format

Output a JSON object with one entry per candidate. Every input candidate MUST have a corresponding output entry.

{
  "annotations": [
    {
      "id": <candidate id from input>,
      "original_text": "the original candidate text (verbatim from input)",
      "original_type": "the original technique label",
      "text": "the final span text (unchanged, trimmed, or expanded)",
      "type": "the final technique label (unchanged, or corrected)",
      "action": "kept" | "relabeled" | "trimmed" | "relabeled_and_trimmed",
      "reason": "one sentence explaining what you did and why"
    }
  ]
}

Allowed `action` values:
- "kept": label and boundaries both unchanged.
- "relabeled": technique label corrected, boundaries unchanged.
- "trimmed": boundaries corrected (either trimmed shorter or expanded longer), label unchanged.
- "relabeled_and_trimmed": both corrected.

The `action` value "dropped" is NOT permitted. Do not use it. Every candidate must be output with one of the four actions above.
"""

REFINE_USER = """Refine the following pre-validated propaganda detections. Correct labels where wrong and refine span boundaries where needed. Do not drop any candidates — every one must be kept.

ARTICLE:
\"\"\"
{text}
\"\"\"

PRE-VALIDATED DETECTIONS (from adversarial review):
{candidates_json}

For each candidate, output a refinement decision. You MUST output an entry for every candidate, and the action MUST be one of: kept, relabeled, trimmed, relabeled_and_trimmed.
"""


def format_candidates_for_refine(candidates: list[dict]) -> str:
    """Format critique-survivor spans for the refinement prompt.

    Deliberately omits `agreement_count` and `detection_reasoning` compared
    to the consol formatter: these spans have already been filtered, so
    agreement-based tie-breaking is no longer relevant, and we don't want
    the refiner relitigating detection confidence.
    """
    import json

    formatted = []
    for i, c in enumerate(candidates, 1):
        formatted.append({
            "id": i,
            "text": c["span_text"],
            "type": c["technique"],
        })

    return json.dumps(formatted, indent=2, ensure_ascii=False)