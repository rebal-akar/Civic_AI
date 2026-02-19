"""
Parse LLM responses into structured Prediction objects.

Handles:
- Well-formed JSON
- JSON embedded in markdown code fences
- Malformed JSON with recovery heuristics
- ASV multi-stage outputs
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.schemas import ALL_TACTICS, Prediction, _fuzzy_match_tactic

logger = logging.getLogger(__name__)


def parse_response(
    raw: str,
    sample_id: str,
    strategy: str = "zero_shot",
) -> Prediction:
    """
    Parse a raw LLM response string into a Prediction.
    
    Tries multiple parsing strategies in order:
    1. Direct JSON parse
    2. Extract JSON from markdown fences
    3. Regex extraction of tactic names
    """
    if not raw or not raw.strip():
        return Prediction(
            sample_id=sample_id,
            raw_response=raw,
            parse_success=False,
            error_message="Empty response",
        )

    # Try JSON parsing
    data = _try_parse_json(raw)

    if data is not None:
        tactics = _extract_tactics_from_json(data, strategy)
        confidence = _extract_confidence(data)
        return Prediction(
            sample_id=sample_id,
            tactics=tactics,
            raw_response=raw,
            confidence=confidence,
            stage_outputs=data if strategy == "asv" else {},
            parse_success=True,
        )

    # Fallback: regex extraction
    tactics = _regex_extract_tactics(raw)
    return Prediction(
        sample_id=sample_id,
        tactics=tactics,
        raw_response=raw,
        parse_success=len(tactics) > 0 or "none" in raw.lower(),
        error_message="JSON parse failed; used regex fallback" if not tactics else "",
    )


def parse_asv_verdict(raw: str, sample_id: str) -> Prediction:
    """Parse the final ASV verdict stage specifically."""
    data = _try_parse_json(raw)

    if data is None:
        return Prediction(
            sample_id=sample_id,
            raw_response=raw,
            parse_success=False,
            error_message="ASV verdict JSON parse failed",
        )

    # Extract only high/moderate confidence tactics from verdict
    tactics: list[str] = []

    # Method 1: "tactics" or "techniques" array (asv uses tactics, asvS uses techniques)
    raw_tactics = data.get("tactics") or data.get("techniques")
    if raw_tactics is not None:
        if isinstance(raw_tactics, list):
            for t in raw_tactics:
                t_str = str(t).lower().strip()
                if t_str in ("none", "[]", ""):
                    continue
                matched = _fuzzy_match_tactic(t_str) or (t_str if t_str in ALL_TACTICS else None)
                if matched:
                    tactics.append(matched)

    # Method 2: parse "verdicts" array and filter by confidence (asv style)
    if not tactics and "verdicts" in data:
        for v in data["verdicts"]:
            verdict = v.get("verdict", "").lower()
            if verdict in ("high_confidence", "moderate_confidence", "proven"):
                tactic_name = v.get("tactic", "").lower().strip()
                matched = _fuzzy_match_tactic(tactic_name) or (
                    tactic_name if tactic_name in ALL_TACTICS else None
                )
                if matched:
                    tactics.append(matched)

    confidence = _extract_confidence(data)

    return Prediction(
        sample_id=sample_id,
        tactics=sorted(set(tactics)),
        raw_response=raw,
        confidence=confidence,
        stage_outputs=data,
        parse_success=True,
    )


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _try_parse_json(raw: str) -> dict | None:
    """Try to parse JSON from raw string, handling code fences."""
    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract from markdown fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def _extract_tactics_from_json(data: dict, strategy: str) -> list[str]:
    """Extract tactic names from parsed JSON based on strategy format."""
    tactics: list[str] = []

    # Common keys to check
    for key in ("tactics", "detected_tactics", "labels", "techniques"):
        if key in data and isinstance(data[key], list):
            for t in data[key]:
                t_str = str(t).lower().strip()
                if t_str in ("none", "no_manipulation", ""):
                    continue
                matched = _fuzzy_match_tactic(t_str) or (t_str if t_str in ALL_TACTICS else None)
                if matched:
                    tactics.append(matched)
            break

    # For prosecution format
    if not tactics and "prosecution_findings" in data:
        for finding in data["prosecution_findings"]:
            tactic = finding.get("tactic", "").lower().strip()
            matched = _fuzzy_match_tactic(tactic) or (tactic if tactic in ALL_TACTICS else None)
            if matched:
                tactics.append(matched)

    return sorted(set(tactics))


def _extract_confidence(data: dict) -> float:
    """Extract confidence score from JSON if present."""
    if "confidence" in data:
        try:
            return float(data["confidence"])
        except (ValueError, TypeError):
            pass
    return 0.0


def _regex_extract_tactics(raw: str) -> list[str]:
    """Last-resort: scan response for known tactic names."""
    raw_lower = raw.lower()
    found: list[str] = []

    for tactic in ALL_TACTICS:
        # Match the tactic name (with underscores replaced by spaces)
        pattern = tactic.replace("_", r"[\s_-]")
        if re.search(pattern, raw_lower):
            found.append(tactic)

    return sorted(set(found))
