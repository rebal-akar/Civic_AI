"""
Prompt formatting for all 6 strategies.

Each function takes a Sample and returns a list of message dicts 
ready for the OpenAI chat API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas import Sample


# ── Taxonomy Reference (shared across all strategies) ────────────────────────

TAXONOMY = """MANIPULATION TECHNIQUES (SemEval-2020 Task 11 — 14 techniques):

1. LOADED LANGUAGE — Words/phrases with strong emotional implications beyond literal meaning.
2. NAME CALLING / LABELING — Negative label applied to reject/condemn without evidence.
3. REPETITION — Same word/phrase repeated 3+ times for emphasis.
4. EXAGGERATION / MINIMIZATION — Overstating or trivialising claims beyond evidence.
5. DOUBT — Questioning credibility without providing evidence.
6. APPEAL TO FEAR / PREJUDICE — Instilling anxiety toward an alternative.
7. FLAG-WAVING — Playing on national/group identity feelings.
8. CAUSAL OVERSIMPLIFICATION — Single cause attributed to complex outcomes; scapegoating.
9. SLOGANS — Brief striking phrase as emotional appeal, designed to be memorable.
10. APPEAL TO AUTHORITY — Claim true because authority supports it, without other evidence.
11. BLACK-AND-WHITE FALLACY — Only two options presented when more exist.
12. THOUGHT-TERMINATING CLICHÉ — Phrases that discourage critical thought.
13. WHATABOUTISM / STRAW MAN / RED HERRING — Deflection, distortion, or diversion tactics.
14. BANDWAGON / REDUCTIO AD HITLERUM — Social proof appeal or Nazi/fascist comparison.

Use ONLY these exact technique names. Return them as snake_case values:
loaded_language, name_calling, repetition, exaggeration, doubt, appeal_to_fear,
flag_waving, causal_oversimplification, slogans, appeal_to_authority,
black_and_white, thought_terminating, whataboutism, bandwagon"""


OUTPUT_SCHEMA = """\nRespond with JSON only:
{
    "tactics": ["tactic_1", "tactic_2"],
    "confidence": 0.0 to 1.0,
    "reasoning": "<brief explanation>"
}
If no manipulation is present, return {"tactics": [], "confidence": ..., "reasoning": "..."}."""


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: ZERO-SHOT
# ═══════════════════════════════════════════════════════════════════════════════

def format_zero_shot(sample: Sample) -> list[dict]:
    system = f"""You are an expert in detecting propaganda and manipulation techniques in political text.

{TAXONOMY}

Analyse the text below and identify ALL manipulation techniques present.
{OUTPUT_SCHEMA}"""

    user = f"""Analyse this political advertisement for manipulation techniques:

---
{sample.text}
---

Respond with JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: FEW-SHOT (3 exemplars)
# ═══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "text": 'RADICAL socialists want to DEFUND the police and leave YOUR family defenseless. Vote Smith!',
        "output": '{"tactics": ["loaded_language", "name_calling", "appeal_to_fear", "exaggeration"], "confidence": 0.85, "reasoning": "RADICAL/DEFUND are loaded language with capitalisation for emphasis. \'socialists\' is name-calling. \'YOUR family defenseless\' is fear appeal. \'DEFUND the police\' exaggerates opponents\' positions."}',
    },
    {
        "text": 'Maria Rodriguez balanced the budget 6 consecutive years and reduced commute times by 12 minutes. Endorsed by the Firefighters Union. Vote Rodriguez for Mayor.',
        "output": '{"tactics": [], "confidence": 0.82, "reasoning": "Factual claims with specific numbers. Endorsement from a relevant organisation. No emotional manipulation or logical fallacies detected."}',
    },
    {
        "text": 'Everyone knows that career politician Johnson has been in Washington too long. Join millions of Americans who say: ENOUGH IS ENOUGH!',
        "output": '{"tactics": ["name_calling", "bandwagon", "slogans"], "confidence": 0.78, "reasoning": "\'career politician\' is a pejorative label (name-calling). \'Everyone knows\' and \'millions of Americans\' are bandwagon appeals. \'ENOUGH IS ENOUGH\' is a slogan."}',
    },
]


def format_few_shot(sample: Sample) -> list[dict]:
    messages: list[dict] = []

    system = f"""You are an expert in detecting propaganda and manipulation techniques in political text.

{TAXONOMY}
{OUTPUT_SCHEMA}"""
    messages.append({"role": "system", "content": system})

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Analyse this political text:\n\n---\n{ex['text']}\n---"})
        messages.append({"role": "assistant", "content": ex["output"]})

    messages.append({"role": "user", "content": f"Analyse this political advertisement:\n\n---\n{sample.text}\n---\n\nRespond with JSON only."})
    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: CHAIN-OF-THOUGHT (CoT)
# ═══════════════════════════════════════════════════════════════════════════════

def format_cot(sample: Sample) -> list[dict]:
    system = f"""You are an expert in detecting propaganda and manipulation techniques in political text.

{TAXONOMY}

ANALYSIS PROCESS — Follow these steps:
1. READ the text carefully. Note any emotional triggers, loaded terms, or logical gaps.
2. For each potential technique, identify the SPECIFIC text span that triggers it.
3. ASK: Does this cross from legitimate persuasion into manipulation? Persuasion respects rational agency; manipulation bypasses it.
4. CONCLUDE with your final list of confirmed techniques.
{OUTPUT_SCHEMA}"""

    user = f"""Analyse this political advertisement step by step:

---
{sample.text}
---

Think through each step, then provide your final JSON answer."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: CoT + SELF-CONSISTENCY (run N times, majority vote)
# ═══════════════════════════════════════════════════════════════════════════════

def format_cot_sc(sample: Sample) -> list[dict]:
    """Same prompt as CoT — caller runs it N times at temperature > 0 and aggregates."""
    return format_cot(sample)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: HIERARCHICAL (2-stage: category → tactic)
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "emotional_appeals": ["loaded_language", "appeal_to_fear", "flag_waving", "appeal_to_authority"],
    "logical_fallacies": ["causal_oversimplification", "black_and_white", "whataboutism"],
    "rhetorical_devices": ["name_calling", "repetition", "exaggeration", "doubt", "slogans", "bandwagon", "thought_terminating"],
}


def format_hierarchical_stage1(sample: Sample) -> list[dict]:
    system = f"""You are an expert in detecting manipulation in political text.

First, identify which HIGH-LEVEL CATEGORIES of manipulation are present:
1. EMOTIONAL APPEALS — techniques that exploit feelings (fear, patriotism, authority)
2. LOGICAL FALLACIES — techniques that use flawed reasoning
3. RHETORICAL DEVICES — techniques that use language tricks (labelling, repetition, exaggeration)

Respond with JSON:
{{"categories": ["emotional_appeals", "logical_fallacies", "rhetorical_devices"], "reasoning": "..."}}

Only include categories where you have evidence. Return {{"categories": [], ...}} if the text is clean."""

    user = f"""Which manipulation categories are present in this text?

---
{sample.text}
---"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_hierarchical_stage2(sample: Sample, categories: list[str]) -> list[dict]:
    relevant_tactics = []
    for cat in categories:
        if cat in CATEGORY_MAP:
            relevant_tactics.extend(CATEGORY_MAP[cat])

    tactic_list = ", ".join(sorted(set(relevant_tactics)))

    system = f"""You identified these manipulation categories: {', '.join(categories)}

Now identify the SPECIFIC techniques from this subset: {tactic_list}
{OUTPUT_SCHEMA}"""

    user = f"""Identify specific techniques in this text:

---
{sample.text}
---

Respond with JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6: ADVERSARIAL SELF-VERIFICATION (ASV) — 3-stage
# ═══════════════════════════════════════════════════════════════════════════════

# Load the full taxonomy reference for ASV (richer definitions with IS/IS NOT)
_ASV_TAXONOMY: str | None = None

def _get_asv_taxonomy() -> str:
    global _ASV_TAXONOMY
    if _ASV_TAXONOMY is not None:
        return _ASV_TAXONOMY

    # Try loading from file first
    taxonomy_path = Path("configs/taxonomy.txt")
    if taxonomy_path.exists():
        _ASV_TAXONOMY = taxonomy_path.read_text()
    else:
        _ASV_TAXONOMY = TAXONOMY  # fallback to compact version
    return _ASV_TAXONOMY


def format_asv_prosecution(sample: Sample) -> list[dict]:
    taxonomy = _get_asv_taxonomy()

    system = f"""You are the PROSECUTION in an adversarial analysis of political advertising manipulation.

YOUR OBJECTIVE: Identify ALL potential manipulation techniques in the advertisement below. Be thorough and err on the side of detection — it is better to flag a potential technique that may later be dismissed than to miss genuine manipulation.

ANALYTICAL PROCESS:
1. Read the advertisement carefully, noting emotional triggers, logical structure, and rhetorical devices
2. For each potential technique, identify the SPECIFIC text that constitutes evidence
3. Explain the manipulative MECHANISM — how does this text attempt to bypass rational evaluation?
4. Consider technique CO-OCCURRENCE — manipulation techniques frequently cluster

{taxonomy}

IMPORTANT:
- Quote EXACT text from the advertisement as evidence
- A single text span may exhibit multiple techniques
- Distinguish between PERSUASION (legitimate) and MANIPULATION (exploitative)

Respond with JSON only:
{{
    "prosecution_findings": [
        {{
            "tactic": "<snake_case technique name>",
            "evidence": "<exact quoted text>",
            "mechanism": "<how this manipulates the audience>",
            "confidence": "high" | "moderate" | "low"
        }}
    ],
    "co_occurrences": "<note any technique clusters>",
    "overall_case_strength": "strong" | "moderate" | "weak",
    "summary": "<2-3 sentence prosecution summary>"
}}

If no manipulation is found, return "prosecution_findings": [] with explanation."""

    user = f"""Build your prosecution case for this political advertisement:

---
{sample.text}
---

Respond with JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_asv_defense(sample: Sample, prosecution_output: dict) -> list[dict]:
    findings = prosecution_output.get("prosecution_findings", [])
    findings_text = _format_findings(findings)

    system = f"""You are the DEFENSE in an adversarial analysis of political advertising manipulation.

YOUR OBJECTIVE: Challenge EACH prosecution finding by arguing why the flagged text may NOT constitute manipulation but rather LEGITIMATE political communication.

THE PROSECUTION'S CASE:
{findings_text}

DEFENSE HEURISTICS — Apply these counter-arguments:
1. GENRE NORMS: Is this standard political advertising rhetoric?
2. FACTUAL BASIS: Could the language have a factual foundation?
3. PROPORTIONALITY: Is the emotional appeal proportionate to the issue's gravity?
4. AUDIENCE SOPHISTICATION: Would a reasonable voter feel deceived — or just disagree?
5. ALTERNATIVE INTERPRETATION: Is there a non-manipulative reading equally plausible?
6. THRESHOLD: Does this cross from passionate advocacy to deliberate exploitation?

Respond with JSON only:
{{
    "defense_arguments": [
        {{
            "responding_to": "<tactic name>",
            "counter_argument": "<specific reason this may not be manipulation>",
            "heuristic_applied": "<which heuristic>",
            "defense_strength": "strong" | "moderate" | "weak"
        }}
    ],
    "overall_defense_strength": "strong" | "moderate" | "weak",
    "summary": "<2-3 sentence defense summary>"
}}"""

    user = f"""The advertisement in question:

---
{sample.text}
---

The prosecution claims: {prosecution_output.get('summary', 'Manipulation detected.')}

Challenge each finding. Respond with JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_asv_verdict(
    sample: Sample,
    prosecution_output: dict,
    defense_output: dict,
) -> list[dict]:
    taxonomy = _get_asv_taxonomy()

    system = f"""You are the JUDGE delivering the final verdict on this political advertisement.

{taxonomy}

PROSECUTION CASE:
{prosecution_output.get('summary', 'N/A')}
Findings: {len(prosecution_output.get('prosecution_findings', []))} techniques alleged
Strength: {prosecution_output.get('overall_case_strength', 'unknown')}

DEFENSE CASE:
{defense_output.get('summary', 'N/A')}
Strength: {defense_output.get('overall_defense_strength', 'unknown')}

VERDICT CATEGORIES:
- "high_confidence": Prosecution evidence compelling; defense insufficient.
- "moderate_confidence": Evidence supports finding but defense raises legitimate points.
- "low_confidence": Plausibly present but substantial doubt.
- "not_proven": Defense prevails or evidence too weak.

CRITICAL: Only include "high_confidence" and "moderate_confidence" in the final "tactics" list.

Respond with JSON only:
{{
    "verdicts": [
        {{
            "tactic": "<snake_case name>",
            "verdict": "high_confidence" | "moderate_confidence" | "low_confidence" | "not_proven",
            "rationale": "<1-2 sentences>"
        }}
    ],
    "tactics": ["<only high/moderate confidence tactics>"],
    "confidence": 0.0 to 1.0,
    "reasoning": "<3-4 sentence overall judgment>"
}}

If no techniques meet the threshold, return "tactics": []."""

    user = f"""The advertisement:

---
{sample.text}
---

Deliver your final verdict. Respond with JSON only."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "The prosecution found no manipulation techniques."
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"Claim {i}: {f.get('tactic', 'Unknown')}\n"
            f"  Evidence: \"{f.get('evidence', 'N/A')}\"\n"
            f"  Mechanism: {f.get('mechanism', f.get('argument', 'N/A'))}"
        )
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "zero_shot": {"format": format_zero_shot, "stages": 1},
    "few_shot": {"format": format_few_shot, "stages": 1},
    "cot": {"format": format_cot, "stages": 1},
    "cot_sc": {"format": format_cot_sc, "stages": 1},  # caller handles N runs
    "hierarchical": {"format": format_hierarchical_stage1, "stages": 2},
    "asv": {"format": format_asv_prosecution, "stages": 3},
}
