"""
Response parser for LLM span detection output.

Parses structured JSON output following the Kasner & Dušek (2024) format:
{
  "annotations": [
    {"text": "exact span text", "type": "Technique_Name", "reason": "..."},
    ...
  ]
}

Resolves character offsets via string matching against the original text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.schemas import PredictedSpan, Technique, normalise_technique

logger = logging.getLogger(__name__)


def parse_llm_response(
    raw_output: str,
    original_text: str,
    pass_id: int = 0,
) -> tuple[list[PredictedSpan], list[str]]:
    """Parse structured JSON from LLM output into PredictedSpan objects.

    Handles:
    - Clean JSON
    - JSON wrapped in markdown fences
    - Trailing commas and single quotes
    - Reasoning model <think>...</think> blocks (stripped)

    Returns:
        (spans, errors) — parsed spans and any parse error messages.
    """
    errors: list[str] = []

    # Strip reasoning traces (DeepSeek-R1, etc.)
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

    # Extract JSON object
    json_obj = _extract_json(cleaned)
    if json_obj is None:
        errors.append(f"Pass {pass_id}: failed to extract JSON from response")
        return [], errors

    # Parse annotations array
    annotations = json_obj.get("annotations", [])
    if not isinstance(annotations, list):
        errors.append(f"Pass {pass_id}: 'annotations' is not a list")
        return [], errors

    spans: list[PredictedSpan] = []
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            errors.append(f"Pass {pass_id}: annotation {i} is not a dict")
            continue

        # Extract fields
        span_text = str(ann.get("text", "")).strip()
        raw_type = str(ann.get("type", "")).strip()
        reason = str(ann.get("reason", "")).strip()

        if not span_text:
            errors.append(f"Pass {pass_id}: annotation {i} has empty text")
            continue

        # Resolve technique
        technique = normalise_technique(raw_type)
        if technique is None:
            # Try the integer-index format from Kasner et al.
            technique = _resolve_type_index(raw_type)
        if technique is None:
            errors.append(
                f"Pass {pass_id}: unknown technique '{raw_type}' for span '{span_text[:40]}'"
            )
            continue

        # Resolve character offsets via string matching
        start, end = _resolve_offsets(span_text, original_text)

        spans.append(PredictedSpan(
            technique=technique,
            span_text=span_text,
            reasoning=reason,
            start=start,
            end=end,
            pass_id=pass_id,
        ))

    if not spans and not errors:
        # Legitimate empty — no propaganda detected
        pass

    return spans, errors


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling common formats."""
    # Try 1: Find JSON in markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return _safe_parse(fence_match.group(1))

    # Try 2: Find the last top-level JSON object (reasoning models put JSON at end)
    # Search from the end for the last complete {...}
    brace_depth = 0
    end_idx = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            if end_idx == -1:
                end_idx = i
            brace_depth += 1
        elif text[i] == "{":
            brace_depth -= 1
            if brace_depth == 0 and end_idx != -1:
                result = _safe_parse(text[i : end_idx + 1])
                if result is not None:
                    return result

    # Try 3: Direct parse of entire text
    return _safe_parse(text)


def _safe_parse(text: str) -> dict | None:
    """Attempt JSON parse with cleanup for common LLM issues."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Clean trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes
    cleaned = cleaned.replace("'", '"')
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    return None


# Index used by Kasner et al. paper for the propaganda task categories
_TYPE_INDEX_TO_TECHNIQUE: dict[int, Technique] = {
    0: Technique.APPEAL_TO_AUTHORITY,
    1: Technique.APPEAL_TO_FEAR,
    2: Technique.BANDWAGON,
    3: Technique.BLACK_AND_WHITE,
    4: Technique.CAUSAL_OVERSIMPLIFICATION,
    5: Technique.DOUBT,
    6: Technique.EXAGGERATION,
    7: Technique.FLAG_WAVING,
    8: Technique.LOADED_LANGUAGE,
    9: Technique.NAME_CALLING,
    # 10: Obfuscation (in 18-category but not in our 14)
    11: Technique.WHATABOUTISM,   # Red Herring → merged into Whataboutism
    12: Technique.BANDWAGON,      # Reductio ad hitlerum → merged into Bandwagon
    13: Technique.REPETITION,
    14: Technique.SLOGANS,
    15: Technique.WHATABOUTISM,   # Straw Men → merged into Whataboutism
    16: Technique.THOUGHT_TERMINATING,
    17: Technique.WHATABOUTISM,
}


def _resolve_type_index(raw: str) -> Technique | None:
    """Resolve an integer category index to a Technique."""
    try:
        idx = int(raw)
        return _TYPE_INDEX_TO_TECHNIQUE.get(idx)
    except (ValueError, TypeError):
        return None


def _resolve_offsets(span_text: str, original_text: str) -> tuple[int, int]:
    """Resolve character offsets for a span via string matching.

    Strategy (following Kasner & Dušek, 2024):
    1. Exact substring match
    2. Case-insensitive match
    3. If all fail, return (-1, -1) — unresolved

    Note: whitespace-normalised matching was removed because the normalised
    index maps to a different string than the original, producing incorrect
    character offsets that corrupt evaluation metrics.
    """
    # 1. Exact match
    idx = original_text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)

    # 2. Case-insensitive
    lower_text = original_text.lower()
    lower_span = span_text.lower()
    idx = lower_text.find(lower_span)
    if idx != -1:
        return idx, idx + len(span_text)

    # 3. Unresolved
    logger.debug(f"Unresolved span: '{span_text[:60]}...'")
    return -1, -1


def merge_multipass(
    pass_results: list[list[PredictedSpan]],
    original_text: str,
) -> list[PredictedSpan]:
    """Merge detections from multiple independent passes (union strategy).

    Groups detections by technique + overlapping text, sets agreement_count.
    """
    if len(pass_results) <= 1:
        spans = pass_results[0] if pass_results else []
        for s in spans:
            s.agreement_count = 1
        return spans

    # Flatten
    all_spans: list[PredictedSpan] = []
    for spans in pass_results:
        all_spans.extend(spans)

    if not all_spans:
        return []

    # Group by technique
    by_technique: dict[Technique, list[PredictedSpan]] = {}
    for s in all_spans:
        by_technique.setdefault(s.technique, []).append(s)

    merged: list[PredictedSpan] = []
    for technique, spans in by_technique.items():
        clusters = _cluster_spans(spans)
        for cluster in clusters:
            unique_passes = len(set(s.pass_id for s in cluster))
            # Pick the best representative (prefer resolved offsets)
            best = next((s for s in cluster if s.resolved), cluster[0])
            best.agreement_count = unique_passes
            # Keep the longest reasoning
            all_reasons = [s.reasoning for s in cluster if s.reasoning]
            if all_reasons:
                best.reasoning = max(all_reasons, key=len)
            merged.append(best)

    merged.sort(key=lambda s: s.start if s.resolved else float("inf"))
    return merged


def _cluster_spans(spans: list[PredictedSpan]) -> list[list[PredictedSpan]]:
    """Cluster overlapping spans of the same technique."""
    used = set()
    clusters: list[list[PredictedSpan]] = []

    for i, a in enumerate(spans):
        if i in used:
            continue
        cluster = [a]
        used.add(i)
        for j, b in enumerate(spans):
            if j in used:
                continue
            if _spans_match(a, b):
                cluster.append(b)
                used.add(j)
        clusters.append(cluster)

    return clusters


def _spans_match(a: PredictedSpan, b: PredictedSpan) -> bool:
    """Check if two spans refer to the same text region."""
    # Text match
    a_norm = re.sub(r"\s+", " ", a.span_text.lower().strip())
    b_norm = re.sub(r"\s+", " ", b.span_text.lower().strip())
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True

    # Offset overlap (>50%)
    if a.resolved and b.resolved:
        overlap_start = max(a.start, b.start)
        overlap_end = min(a.end, b.end)
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            min_len = min(a.end - a.start, b.end - b.start)
            if min_len > 0 and overlap / min_len > 0.5:
                return True

    return False
