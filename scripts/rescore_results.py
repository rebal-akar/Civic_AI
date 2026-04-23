"""
Rescore a saved results JSON against fresh gold labels — no LLM calls.

Reconstructs `Prediction` objects from the `all_spans` stored per article,
loads current gold labels from disk, and recomputes SemEval SI/TC metrics.

Point `--articles-dir` / `--labels-path` at the RAW dataset by default,
not a split. A split only contains a subset of articles, so when the split
partition changes (e.g. after regenerating `data/splits/`) you can end up
scoring predictions against an articles directory that is missing their
gold labels — artificially tanking precision.

Example:
    python -m scripts.rescore_results \
        --results outputs/results/crossmodel_family_consol_permissive.json \
        --output outputs/results/crossmodel_family_consol_permissive_rescored.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.data.loader import load_semeval
from src.evaluation.metrics import evaluate
from src.schemas import (
    Article,
    PredictedSpan,
    Prediction,
    Technique,
    normalise_technique,
)

logger = logging.getLogger(__name__)


def _rebuild_predicted_span(raw: dict) -> PredictedSpan | None:
    technique = normalise_technique(raw.get("technique", ""))
    if technique is None:
        return None

    original_technique = None
    if raw.get("original_technique"):
        original_technique = normalise_technique(raw["original_technique"])

    return PredictedSpan(
        technique=technique,
        span_text=raw.get("span_text", ""),
        reasoning=raw.get("reasoning", ""),
        start=raw.get("start", -1),
        end=raw.get("end", -1),
        pass_id=raw.get("pass_id", 0),
        agreement_count=raw.get("agreement_count", 1),
        confidence=raw.get("confidence", -1.0),
        verdict=raw.get("verdict", ""),
        original_technique=original_technique,
        original_span_text=raw.get("original_span_text", ""),
        original_start=raw.get("original_start", -1),
        original_end=raw.get("original_end", -1),
        was_relabeled=raw.get("was_relabeled", False),
        was_trimmed=raw.get("was_trimmed", False),
        consol_action=raw.get("consol_action", ""),
        stage_history=raw.get("stage_history", []),
    )


def _rebuild_predictions(
    preds_raw: dict[str, dict],
    eval_mode: str,
) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for article_id, pred_raw in preds_raw.items():
        spans: list[PredictedSpan] = []
        for raw_span in pred_raw.get("all_spans", []):
            span = _rebuild_predicted_span(raw_span)
            if span is not None:
                spans.append(span)
        predictions[article_id] = Prediction(
            article_id=article_id,
            spans=spans,
            eval_mode=eval_mode,
        )
    return predictions


def _summarise_gold(articles: list[Article]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in articles:
        for s in a.gold_spans:
            counts[s.technique.value] = counts.get(s.technique.value, 0) + 1
    return counts


def rescore_config(
    cfg_name: str,
    cfg_block: dict,
    articles_dir: Path,
    labels_path: Path,
) -> dict:
    eval_mode = cfg_block.get("config", {}).get("eval_mode", "permissive")

    articles_all = load_semeval(articles_dir, labels_path)
    pred_ids = set(cfg_block.get("predictions", {}).keys())
    articles = [a for a in articles_all if a.id in pred_ids]

    missing = pred_ids - {a.id for a in articles}
    if missing:
        logger.warning(
            "%s: %d prediction ids not found on disk (%s...)",
            cfg_name, len(missing), sorted(missing)[:3],
        )

    predictions = _rebuild_predictions(cfg_block["predictions"], eval_mode)
    metrics = evaluate(articles, predictions)

    gold_summary = _summarise_gold(articles)

    return {
        "cfg_name": cfg_name,
        "articles_scored": len(articles),
        "eval_mode": eval_mode,
        "metrics": asdict(metrics),
        "gold_summary": gold_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=Path("datasets/train-articles"),
        help="Source of article text. Default: the raw SemEval dataset.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("datasets/train-task2-TC.labels"),
        help="Source of gold labels. Default: the raw SemEval labels file.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=None,
        help="Rescore a single config key only (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data = json.loads(args.results.read_text(encoding="utf-8"))

    results_block = data.get("results", data)
    configs = (
        {args.config: results_block[args.config]} if args.config else results_block
    )

    rescored: dict[str, dict] = {}
    for cfg_name, cfg_block in configs.items():
        if "predictions" not in cfg_block:
            logger.warning("%s has no predictions, skipping", cfg_name)
            continue
        print(f"\n--- Rescoring: {cfg_name} ---")
        result = rescore_config(
            cfg_name, cfg_block, args.articles_dir, args.labels_path,
        )
        m = result["metrics"]
        print(f"  Articles : {result['articles_scored']}")
        print(f"  SI F1    : {m['si_f1']:.4f}   "
              f"(P={m['si_precision']:.4f}  R={m['si_recall']:.4f})")
        print(f"  TC F1    : {m['tc_f1']:.4f}   "
              f"(P={m['tc_precision']:.4f}  R={m['tc_recall']:.4f})")
        print(f"  Macro F1 : {m['macro_f1']:.4f}")
        print("  Per-technique TC F1:")
        for t, stats in sorted(m["per_technique"].items()):
            print(
                f"    {t:<40} F1={stats['f1']:.3f}  "
                f"gold={stats['gold_count']}  pred={stats['pred_count']}"
            )
        rescored[cfg_name] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rescored, indent=2, default=str))
    print(f"\nWrote rescored metrics to {args.output}")


if __name__ == "__main__":
    main()
