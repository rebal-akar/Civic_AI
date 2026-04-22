"""Diagnose span verdicts, gold overlap, and eval_mode filtering.

Writes outputs/diagnostics/span_verdict_diagnosis.txt with per-article
breakdowns for all 4 cross-family configs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("outputs/results")
OUTPUT_PATH = Path("outputs/diagnostics/span_verdict_diagnosis.txt")
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


log("=" * 100)
log("SPAN VERDICT & GOLD OVERLAP DIAGNOSIS — cross-family consol results")
log("=" * 100)

for config_prefix in CONFIGS:
    matches = sorted(RESULTS_DIR.glob(f"{config_prefix}_*.json"))
    if not matches:
        log(f"\n--- {config_prefix}: NO FILE FOUND ---")
        continue

    path = matches[-1]
    with open(path) as f:
        data = json.load(f)

    eval_mode = data.get("config", {}).get("eval_mode", "permissive")

    log(f"\n\n{'#'*100}")
    log(f"Config: {config_prefix}")
    log(f"File:   {path.name}")
    log(f"eval_mode: {eval_mode}")
    log(f"{'#'*100}")

    total_all = 0
    total_filtered = 0
    total_gold = 0
    global_verdict_counter: Counter[str] = Counter()
    global_technique_overlap = 0
    global_technique_total = 0

    for aid, pred in data.get("predictions", {}).items():
        log(f"\n{'='*80}")
        log(f"Article: {aid}")
        log(f"{'='*80}")

        gold = pred.get("gold_spans", [])
        gold_techs: Counter[str] = Counter(g["technique"] for g in gold)
        log(f"\nGold spans ({len(gold)}):")
        for t, c in gold_techs.most_common():
            log(f"  {c:>3}x  {t}")
        total_gold += len(gold)

        all_spans = pred.get("all_spans", [])
        pred_techs: Counter[str] = Counter(s["technique"] for s in all_spans)
        log(f"\nAll predicted spans ({len(all_spans)}):")
        for t, c in pred_techs.most_common():
            log(f"  {c:>3}x  {t}")
        total_all += len(all_spans)

        verdicts: Counter[str] = Counter(
            s.get("verdict", "(empty)") for s in all_spans
        )
        log(f"\nVerdict distribution:")
        for v, c in verdicts.most_common():
            log(f"  {c:>3}x  {v!r}")
        global_verdict_counter.update(verdicts)

        has_verdicts = any(s.get("verdict") for s in all_spans)
        if has_verdicts:
            if eval_mode == "strict":
                filtered = [s for s in all_spans
                            if s.get("verdict") == "CONFIRMED"]
            else:
                filtered = [s for s in all_spans
                            if s.get("verdict") in ("CONFIRMED", "POSSIBLE")]
        else:
            filtered = list(all_spans)
        total_filtered += len(filtered)

        log(f"\nSpans surviving {eval_mode} filter: "
            f"{len(filtered)} of {len(all_spans)}")

        gold_set = set(g["technique"] for g in gold)
        pred_set = set(s["technique"] for s in filtered)
        overlap = gold_set & pred_set
        log(f"\nGold techniques:      {sorted(gold_set)}")
        log(f"Pred techniques (filt): {sorted(pred_set)}")
        log(f"Overlap:                {sorted(overlap) if overlap else '(NONE)'}")

        global_technique_total += len(gold_set)
        global_technique_overlap += len(overlap)

        log(f"\n--- Span-level boundary detail (filtered preds vs gold) ---")
        for i, s in enumerate(filtered):
            best_iou = 0.0
            best_gold_idx = -1
            best_gold_tech = ""
            for gi, g in enumerate(gold):
                ov_start = max(s["start"], g["start"])
                ov_end = min(s["end"], g["end"])
                if ov_end > ov_start:
                    intersection = ov_end - ov_start
                    union = (s["end"] - s["start"]) + (g["end"] - g["start"]) - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_gold_idx = gi
                        best_gold_tech = g["technique"]

            tech_match = (s["technique"] == best_gold_tech) if best_gold_idx >= 0 else False
            log(f"  pred[{i}] [{s['start']:>5}:{s['end']:>5}] "
                f"{s['technique']:<35} "
                f"best_gold={best_gold_idx:>3} IoU={best_iou:.3f} "
                f"gold_tech={best_gold_tech!r:<35} "
                f"TC={'YES' if tech_match else 'NO '}")

    log(f"\n\n{'='*80}")
    log(f"SUMMARY for {config_prefix}")
    log(f"{'='*80}")
    log(f"Total gold spans:          {total_gold}")
    log(f"Total all_spans:           {total_all}")
    log(f"Total after {eval_mode} filter: {total_filtered}")
    log(f"Global verdict distribution: {dict(global_verdict_counter.most_common())}")
    log(f"Technique set overlap:     {global_technique_overlap}/{global_technique_total} "
        f"gold technique-sets had at least one matching pred technique")

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
log(f"\n\nDiagnosis written to: {OUTPUT_PATH.resolve()}")
