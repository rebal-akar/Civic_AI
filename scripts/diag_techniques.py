"""Diagnose technique labels across cross-family result JSONs.

Writes outputs/diagnostics/technique_diagnosis.txt with:
- Per-config technique frequency counts
- Which labels fail normalise_technique()
- Side-by-side comparison of gpt-4o vs Sonnet label distributions
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.schemas import normalise_technique

RESULTS_DIR = Path("outputs/results")
OUTPUT_PATH = Path("outputs/diagnostics/technique_diagnosis.txt")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    "consol_xfam_same_4o",
    "consol_xfam_same_sonnet",
    "consol_xfam_cascade_4o_sonnet",
    "consol_xfam_cascade_sonnet_4o",
]

lines: list[str] = []


def log(text: str = "") -> None:
    lines.append(text)
    print(text)


log("=" * 90)
log("TECHNIQUE LABEL DIAGNOSIS — cross-family consol results")
log("=" * 90)

all_configs: dict[str, dict] = {}

for config_prefix in CONFIGS:
    matches = sorted(RESULTS_DIR.glob(f"{config_prefix}_*.json"))
    if not matches:
        log(f"\n--- {config_prefix}: NO FILE FOUND ---")
        continue

    path = matches[-1]
    with open(path) as f:
        data = json.load(f)

    all_configs[config_prefix] = data

    log(f"\n{'='*90}")
    log(f"Config: {config_prefix}")
    log(f"File:   {path.name}")
    log(f"{'='*90}")

    tech_counter: Counter[str] = Counter()
    failed_labels: Counter[str] = Counter()
    ok_labels: Counter[str] = Counter()
    stage_techs: dict[str, Counter[str]] = {
        "all_spans": Counter(),
        "after_s1": Counter(),
        "after_s2": Counter(),
    }

    for aid, pred in data.get("predictions", {}).items():
        for span in pred.get("all_spans", []):
            raw = span.get("technique", "(none)")
            tech_counter[raw] += 1
            stage_techs["all_spans"][raw] += 1
            norm = normalise_technique(raw)
            if norm is None:
                failed_labels[raw] += 1
            else:
                ok_labels[raw] += 1

        for stage_key in ["after_s1", "after_s2"]:
            for span in pred.get("stage_snapshots", {}).get(stage_key, []):
                raw = span.get("technique", "(none)")
                stage_techs[stage_key][raw] += 1

    log(f"\nTotal spans (all_spans): {sum(tech_counter.values())}")
    log(f"Distinct technique labels: {len(tech_counter)}")

    log(f"\n--- All technique labels (frequency) ---")
    for tech, count in tech_counter.most_common():
        norm = normalise_technique(tech)
        status = f"-> {norm.value}" if norm else "*** FAIL ***"
        log(f"  {count:>5}x  {tech!r:50s} {status}")

    if failed_labels:
        log(f"\n--- FAILED normalisation ({sum(failed_labels.values())} spans) ---")
        for tech, count in failed_labels.most_common():
            log(f"  {count:>5}x  {tech!r}")
    else:
        log(f"\n--- All labels normalised successfully ---")

    for stage_key in ["after_s1", "after_s2"]:
        stage_data = stage_techs[stage_key]
        if stage_data:
            stage_failed = {t: c for t, c in stage_data.items()
                           if normalise_technique(t) is None}
            log(f"\n--- {stage_key}: {sum(stage_data.values())} spans, "
                f"{len(stage_data)} distinct labels, "
                f"{sum(stage_failed.values())} failed ---")
            if stage_failed:
                for tech, count in sorted(stage_failed.items(),
                                          key=lambda x: -x[1]):
                    log(f"  {count:>5}x  {tech!r}")

log(f"\n\n{'='*90}")
log("CROSS-CONFIG COMPARISON")
log("=" * 90)

if "consol_xfam_same_4o" in all_configs and "consol_xfam_same_sonnet" in all_configs:
    def _get_techs(cfg_data: dict) -> Counter[str]:
        c: Counter[str] = Counter()
        for pred in cfg_data.get("predictions", {}).values():
            for span in pred.get("all_spans", []):
                c[span.get("technique", "(none)")] += 1
        return c

    gpt_techs = _get_techs(all_configs["consol_xfam_same_4o"])
    sonnet_techs = _get_techs(all_configs["consol_xfam_same_sonnet"])
    all_labels = sorted(set(gpt_techs.keys()) | set(sonnet_techs.keys()))

    log(f"\n{'Label':<55} {'gpt-4o':>8} {'Sonnet':>8} {'Norm?':>8}")
    log("-" * 85)
    for label in all_labels:
        norm = normalise_technique(label)
        norm_str = "OK" if norm else "FAIL"
        log(f"  {label!r:<53} {gpt_techs.get(label, 0):>8} "
            f"{sonnet_techs.get(label, 0):>8} {norm_str:>8}")

    sonnet_only = set(sonnet_techs.keys()) - set(gpt_techs.keys())
    gpt_only = set(gpt_techs.keys()) - set(sonnet_techs.keys())
    if sonnet_only:
        log(f"\nLabels ONLY in Sonnet output: {sonnet_only}")
    if gpt_only:
        log(f"\nLabels ONLY in gpt-4o output: {gpt_only}")

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
log(f"\nDiagnosis written to: {OUTPUT_PATH.resolve()}")
