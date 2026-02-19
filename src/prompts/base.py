"""Shared taxonomy loader and constants."""
from __future__ import annotations
import json
from pathlib import Path

# Load from Civic_AI/config/techniques.json (base.py is in src/prompts/, so root = parent.parent.parent)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TAXONOMY_PATH = _PROJECT_ROOT / "config" / "techniques.json"


def load_taxonomy(path: str | Path | None = None) -> str:
    taxonomy_path = Path(path) if path else _DEFAULT_TAXONOMY_PATH
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {taxonomy_path.absolute()}\n"
            f"Expected: config/techniques.json under project root ({_PROJECT_ROOT})"
        )
    with open(taxonomy_path) as f:
        techniques = json.load(f)
    lines = ["MANIPULATION TECHNIQUES (SemEval-2020 Task 11):\n"]
    for i, t in enumerate(techniques, 1):
        lines.append(f"{i}. {t['display'].upper()} — {t['definition']}")
    # Snake-case list from techniques.json so prompt stays in sync with config
    names = [t.get("name", "").strip() for t in techniques if t.get("name")]
    names_str = ", ".join(names)
    lines.append(f"""
Use ONLY these exact technique names as snake_case:
{names_str}""")
    return "\n".join(lines)


TAXONOMY = load_taxonomy()

OUTPUT_SCHEMA = """\nRespond with JSON only:
{
    "tactics": ["tactic_1", "tactic_2"],
    "confidence": 0.0 to 1.0,
    "reasoning": "<brief explanation>"
}
If no manipulation is present, return {"tactics": [], "confidence": ..., "reasoning": "..."}."""

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "The radical opposition wants to destroy everything we have built. "
            "Either you are with us, or you are against your own country. "
            "We the People will not stand for this assault on our values."
        ),
        "output": json.dumps({
            "tactics": ["loaded_language", "black_and_white", 
                       "flag_waving", "appeal_to_fear"],
            "confidence": 0.88,
            "reasoning": (
                "'Destroy everything we have built' uses loaded language. "
                "'Either with us or against your country' is black-and-white fallacy. "
                "'We the People' is flag-waving. 'Assault on our values' is appeal to fear."
            )
        })
    },
    {
        "text": (
            "The unemployment rate fell to 3.7% last quarter according to the "
            "Bureau of Labor Statistics. The bill passed with bipartisan support "
            "after three months of committee review."
        ),
        "output": json.dumps({
            "tactics": [],
            "confidence": 0.93,
            "reasoning": (
                "Specific statistics from named source, factual legislative "
                "information. No emotional manipulation or logical fallacies detected."
            )
        })
    },
    {
        "text": (
            "I do not really see any problems with the current policy. "
            "That is just how things have always worked. "
            "President Johnson, who himself accepted donations from the same "
            "corporations, keeps criticising our fundraising practices."
        ),
        "output": json.dumps({
            "tactics": ["thought_terminating", "whataboutism"],
            "confidence": 0.81,
            "reasoning": (
                "'I do not really see any problems' and 'that is just how things "
                "have always worked' are thought-terminating clichés. "
                "The Johnson sentence charges hypocrisy to deflect criticism — whataboutism."
            )
        })
    },
]
FEW_SHOT_EXAMPLES_ASV = [
    {
        "text": (
            "The radical opposition wants to destroy everything we have built. "
            "Either you are with us, or you are against your own country. "
            "We the People will not stand for this assault on our values."
        ),
        "output": json.dumps({
            "paragraph_id": "example_1",
            "detections": [
                {
                    "id": "c1",
                    "tactic": "loaded_language",
                    "evidence": "destroy everything we have built",
                    "reasoning": "Emotionally charged phrasing implying total annihilation, far beyond what the opposition's actual position warrants."
                },
                {
                    "id": "c2",
                    "tactic": "black_and_white",
                    "evidence": "Either you are with us, or you are against your own country",
                    "reasoning": "Presents only two mutually exclusive options, eliminating any middle ground or nuanced position."
                },
                {
                    "id": "c3",
                    "tactic": "flag_waving",
                    "evidence": "We the People",
                    "reasoning": "Invokes national identity and constitutional language to frame the speaker as representative of the entire nation."
                },
                {
                    "id": "c4",
                    "tactic": "appeal_to_fear",
                    "evidence": "assault on our values",
                    "reasoning": "Instils anxiety by framing opposition as an active attack on shared values without specifying what values or how they are threatened."
                }
            ]
        })
    },
    {
        "text": (
            "The unemployment rate fell to 3.7% last quarter according to the "
            "Bureau of Labor Statistics. The bill passed with bipartisan support "
            "after three months of committee review."
        ),
        "output": json.dumps({
            "paragraph_id": "example_2",
            "detections": []
        })
    },
    {
        "text": (
            "I do not really see any problems with the current policy. "
            "That is just how things have always worked. "
            "President Johnson, who himself accepted donations from the same "
            "corporations, keeps criticising our fundraising practices."
        ),
        "output": json.dumps({
            "paragraph_id": "example_3",
            "detections": [
                {
                    "id": "c1",
                    "tactic": "thought_terminating",
                    "evidence": "I do not really see any problems with the current policy",
                    "reasoning": "Dismisses criticism without engagement, discouraging further critical examination of the policy."
                },
                {
                    "id": "c2",
                    "tactic": "thought_terminating",
                    "evidence": "That is just how things have always worked",
                    "reasoning": "Appeals to tradition as a substitute for argument, shutting down discussion with a generic cliché."
                },
                {
                    "id": "c3",
                    "tactic": "whataboutism",
                    "evidence": "President Johnson, who himself accepted donations from the same corporations, keeps criticising our fundraising practices",
                    "reasoning": "Deflects criticism of own fundraising by charging the critic with hypocrisy, without addressing the original argument."
                }
            ]
        })
    },
]