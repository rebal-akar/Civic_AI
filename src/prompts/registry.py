"""Strategy registry — maps strategy names to format functions."""
from src.prompts.asv import (
    format_zero_shot,
    format_few_shot,
    format_cot,
    format_cot_sc,
    format_hierarchical_stage1,
    format_hierarchical_stage2,
    format_asv_prosecution,
    format_asv_defense,
    format_asv_verdict,
)

STRATEGIES = {
    "zero_shot":    {"format": format_zero_shot,          "stages": 1},
    "few_shot":     {"format": format_few_shot,           "stages": 1},
    "cot":          {"format": format_cot,                "stages": 1},
    "cot_sc":       {"format": format_cot_sc,             "stages": 1},
    "hierarchical": {"format": format_hierarchical_stage1,"stages": 2},
    "asv":          {"format": format_asv_prosecution,    "stages": 3},
}