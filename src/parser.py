"""Parse structured JSON output from LLM detection into PredictedSpan objects."""
from __future__ import annotations

import json
import logging
import re

from src.schemas import PredictedSpan, normalise_technique

logger = logging.getLogger(__name__)


def parse_llm_response(
    raw_output: str,
    original_text: str,
    pass_id: int = 0,
) -> tuple[list[PredictedSpan], list[str]]:
    """Parse LLM JSON output of shape {"annotations": [{"text", "type", "reason"}]}.

    Offsets are resolved via string matching against the original text.
    Returns (spans, errors).
    """
    errors: list[str] = []
    json_obj = _extract_json(raw_output)
    if json_obj is None:
        errors.append(f"Pass {pass_id}: failed to extract JSON from response")
        return [], errors

    annotations = json_obj.get("annotations", [])
    if not isinstance(annotations, list):
        errors.append(f"Pass {pass_id}: 'annotations' is not a list")
        return [], errors

    spans: list[PredictedSpan] = []
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            errors.append(f"Pass {pass_id}: annotation {i} is not a dict")
            continue

        span_text = str(ann.get("text", "")).strip()
        raw_type = str(ann.get("type", "")).strip()
        reason = str(ann.get("reason", "")).strip()

        if not span_text:
            errors.append(f"Pass {pass_id}: annotation {i} has empty text")
            continue

        technique = normalise_technique(raw_type)
        if technique is None:
            errors.append(
                f"Pass {pass_id}: unknown technique '{raw_type}' for '{span_text[:40]}'"
            )
            continue

        start, end = _resolve_offsets(span_text, original_text)
        spans.append(PredictedSpan(
            technique=technique,
            span_text=span_text,
            reasoning=reason,
            start=start,
            end=end,
            pass_id=pass_id,
        ))

    return spans, errors


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling markdown fences and suffix JSON."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return _safe_parse(fence_match.group(1))

    # Find the last top-level {...} block (for models that prepend reasoning)
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
                result = _safe_parse(text[i:end_idx + 1])
                if result is not None:
                    return result

    return _safe_parse(text)


def _safe_parse(text: str) -> dict | None:
    """Try strict JSON parse, then clean trailing commas and single quotes."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text).replace("'", '"')
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _resolve_offsets(span_text: str, original_text: str) -> tuple[int, int]:
    """Resolve character offsets via exact then case-insensitive string match."""
    idx = original_text.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)

    idx = original_text.lower().find(span_text.lower())
    if idx != -1:
        return idx, idx + len(span_text)

    return -1, -1


def merge_multipass(
    pass_results: list[list[PredictedSpan]],
    original_text: str,
) -> list[PredictedSpan]:
    """Union-merge detections from multiple passes; sets agreement_count per cluster."""
    if len(pass_results) <= 1:
        spans = pass_results[0] if pass_results else []
        for s in spans:
            s.agreement_count = 1
        return spans

    all_spans = [s for spans in pass_results for s in spans]
    if not all_spans:
        return []

    by_technique: dict = {}
    for s in all_spans:
        by_technique.setdefault(s.technique, []).append(s)

    merged: list[PredictedSpan] = []
    for spans in by_technique.values():
        for cluster in _cluster_spans(spans):
            unique_passes = len({s.pass_id for s in cluster})
            best = next((s for s in cluster if s.resolved), cluster[0])
            best.agreement_count = unique_passes
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
    """Two spans match if text equals/contains, or resolved offsets overlap >50%."""
    a_norm = re.sub(r"\s+", " ", a.span_text.lower().strip())
    b_norm = re.sub(r"\s+", " ", b.span_text.lower().strip())
    if a_norm == b_norm or a_norm in b_norm or b_norm in a_norm:
        return True

    if a.resolved and b.resolved:
        overlap_start = max(a.start, b.start)
        overlap_end = min(a.end, b.end)
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            min_len = min(a.end - a.start, b.end - b.start)
            if min_len > 0 and overlap / min_len > 0.5:
                return True

    return False