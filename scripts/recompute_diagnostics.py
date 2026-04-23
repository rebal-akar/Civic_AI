"""
Recompute consolidation diagnostics against fresh gold labels.

Reads cached predictions from a comparison JSON, rebuilds `Prediction`
objects, and recomputes:
  - 14x14 TC confusion matrix (rows = gold technique, cols = predicted
    technique; diagonal = SI+TC hit, off-diagonal = SI hit with wrong tech;
    plus `(no_pred)` / `(no_gold)` sinks).
  - Relabel matrix + lucky-vs-systematic summary (including net F1
    contribution).

Example:
    python -m scripts.recompute_diagnostics \
        --results outputs/results/comparison_gpt-4o_..._n122_permissive.json \
        --config consol \
        --output outputs/results/consol_diagnostics_rescored.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_semeval
from src.evaluation.diagnostics import build_label_change_log, build_relabel_matrix, lucky_vs_systematic
from src.schemas import GoldSpan, PredictedSpan, Prediction, Technique, normalise_technique

from scripts.rescore_results import _rebuild_predictions


# ─── TC confusion matrix ──────────────────────────────────────────────────

_NO_PRED = "(no_pred)"
_NO_GOLD = "(no_gold)"


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def build_tc_confusion_matrix(
    articles: list,
    predictions: dict[str, Prediction],
    min_overlap_ratio: float = 0.25,
) -> dict[str, Any]:
    """Gold-centric TC confusion matrix.

    For every gold span we pick the SINGLE best overlapping prediction
    (by overlap ratio against the gold span), and record one vote for
    `matrix[gold_tech][pred_tech]`. If no prediction overlaps above the
    threshold, the gold span is counted under `(no_pred)`. Predicted
    spans that don't overlap any gold span are counted under `(no_gold)`.

    Row totals therefore equal gold-span counts per technique, and the
    bottom `(no_gold)` row counts predictions that had no overlapping
    gold (false positives).
    """
    techniques = [t.value for t in Technique]
    col_keys = techniques + [_NO_PRED]
    row_keys = techniques + [_NO_GOLD]
    matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in col_keys} for r in row_keys}

    article_by_id = {a.id: a for a in articles}
    for aid, pred in predictions.items():
        article = article_by_id.get(aid)
        if article is None:
            continue

        confirmed = [s for s in pred.confirmed_spans if s.resolved]
        pred_used: list[bool] = [False] * len(confirmed)

        for g in article.gold_spans:
            best_pi = -1
            best_ratio = 0.0
            for pi, p in enumerate(confirmed):
                ov = _overlap(g.start, g.end, p.start, p.end)
                if ov <= 0:
                    continue
                min_len = max(1, min(g.end - g.start, p.end - p.start))
                ratio = ov / min_len
                if ratio >= min_overlap_ratio and ratio > best_ratio:
                    best_ratio = ratio
                    best_pi = pi
            if best_pi >= 0:
                matrix[g.technique.value][confirmed[best_pi].technique.value] += 1
                pred_used[best_pi] = True
            else:
                matrix[g.technique.value][_NO_PRED] += 1

        for pi, p in enumerate(confirmed):
            if pred_used[pi]:
                continue
            has_overlap = any(
                _overlap(p.start, p.end, g.start, g.end) > 0
                for g in article.gold_spans
            )
            if not has_overlap:
                matrix[_NO_GOLD][p.technique.value] += 1

    row_totals = {r: sum(matrix[r].values()) for r in matrix}
    diagonal = sum(matrix[t][t] for t in techniques)
    si_hit = sum(matrix[gt][pt] for gt in techniques for pt in techniques)
    gold_only = sum(matrix[gt][_NO_PRED] for gt in techniques)
    pred_only = sum(matrix[_NO_GOLD][pt] for pt in techniques)

    return {
        "row_keys": row_keys,
        "col_keys": col_keys,
        "counts": matrix,
        "row_totals": row_totals,
        "summary": {
            "tc_hits_diagonal": diagonal,
            "si_hits_total": si_hit,
            "tc_miss_within_si": si_hit - diagonal,
            "gold_with_no_pred": gold_only,
            "pred_false_positive": pred_only,
        },
    }


# ─── Plotting ─────────────────────────────────────────────────────────────

def _short(name: str) -> str:
    return {
        "Loaded_Language": "Loaded Lang.",
        "Name_Calling,Labeling": "Name Calling",
        "Repetition": "Repetition",
        "Exaggeration,Minimisation": "Exagg./Min.",
        "Doubt": "Doubt",
        "Appeal_to_Fear-Prejudice": "Appeal Fear",
        "Flag-Waving": "Flag-Waving",
        "Causal_Oversimplification": "Causal Oversimp.",
        "Slogans": "Slogans",
        "Appeal_to_Authority": "Appeal Auth.",
        "Black-and-White_Fallacy": "Black-and-White",
        "Thought-terminating_Cliches": "Thought-term.",
        "Whataboutism,Straw_Men,Red_Herring": "Whataboutism+",
        "Bandwagon,Reductio_ad_hitlerum": "Bandwagon+",
        _NO_PRED: "(no pred)",
        _NO_GOLD: "(no gold)",
    }.get(name, name[:14])


def plot_confusion_matrix(
    matrix: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    row_keys = matrix["row_keys"]
    col_keys = matrix["col_keys"]
    counts = matrix["counts"]

    data = np.array(
        [[counts[r][c] for c in col_keys] for r in row_keys],
        dtype=float,
    )
    row_totals = data.sum(axis=1, keepdims=True)
    safe = np.where(row_totals > 0, row_totals, 1)
    row_norm = data / safe

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_keys)))
    ax.set_yticks(range(len(row_keys)))
    ax.set_xticklabels([_short(c) for c in col_keys], rotation=45, ha="right")
    ax.set_yticklabels([_short(r) for r in row_keys])
    ax.set_xlabel("Predicted technique")
    ax.set_ylabel("Gold technique")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            n = int(data[i, j])
            if n == 0:
                continue
            colour = "white" if row_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(n), ha="center", va="center",
                    color=colour, fontsize=8)

    plt.colorbar(im, ax=ax, label="Row-normalised share")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_relabel_matrix(
    relabel: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    techniques = relabel["techniques"]
    counts = relabel["counts"]
    improved = relabel["improved"]
    worsened = relabel["worsened"]

    net = np.array(
        [[improved[o][n] - worsened[o][n] for n in techniques] for o in techniques],
        dtype=float,
    )
    total = np.array(
        [[counts[o][n] for n in techniques] for o in techniques],
        dtype=float,
    )

    vmax = max(abs(net.min()), abs(net.max()), 1)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(net, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(techniques)))
    ax.set_yticks(range(len(techniques)))
    ax.set_xticklabels([_short(t) for t in techniques], rotation=45, ha="right")
    ax.set_yticklabels([_short(t) for t in techniques])
    ax.set_xlabel("New label")
    ax.set_ylabel("Original label")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(len(techniques)):
        for j in range(len(techniques)):
            n_total = int(total[i, j])
            n_net = int(net[i, j])
            if n_total == 0:
                continue
            shade = abs(net[i, j]) / vmax
            colour = "white" if shade > 0.5 else "black"
            ax.text(j, i, f"{n_total}\n({n_net:+d})",
                    ha="center", va="center", color=colour, fontsize=7)

    plt.colorbar(im, ax=ax, label="Net correctness (gains - losses)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Pretty-printers ───────────────────────────────────────────────────────

def _abbrev(name: str) -> str:
    if name in (_NO_PRED, _NO_GOLD):
        return name
    parts = name.split(",")[0].split("-")[0]
    return parts[:12]


def _print_matrix(matrix: dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===".encode("ascii", "replace").decode())
    row_keys = matrix["row_keys"]
    col_keys = matrix["col_keys"]
    counts = matrix["counts"]

    header = f"{'row \\ col':<14s}" + "".join(f"{_abbrev(c):>6s}" for c in col_keys) + f"{'total':>7s}"
    print(header)
    for r in row_keys:
        row_total = sum(counts[r].values())
        line = f"{_abbrev(r):<14s}" + "".join(f"{counts[r][c]:>6d}" for c in col_keys) + f"{row_total:>7d}"
        print(line)


def _print_relabel_summary(
    lvs: dict[str, Any], relabel: dict[str, Any]
) -> None:
    print("\n=== Relabel summary ===")
    print(f"  total relabels       : {lvs['total_relabels']}")
    print(f"  gains (wrong->right) : {lvs['total_gains']}")
    print(f"  losses (right->wrong): {lvs['total_losses']}")
    print(f"  neutral              : {lvs['total_neutral']}")
    print(f"  net F1 contribution  : {lvs['net_f1_contribution']:+d}")
    if lvs["overall_gain_ratio"] is not None:
        print(f"  overall gain ratio  : {lvs['overall_gain_ratio']:.3f}")

    print("\n  Per original label:")
    print(f"    {'technique':<40s} {'relabels':>8s} {'gains':>6s} {'losses':>7s} {'neutral':>8s} {'verdict':>11s}")
    for tech, d in sorted(lvs["per_technique"].items(), key=lambda kv: -kv[1]["relabels"]):
        ratio = "" if d["gain_ratio"] is None else f"{d['gain_ratio']:.2f}"
        print(
            f"    {tech:<40s} {d['relabels']:>8d} {d['gains']:>6d} "
            f"{d['losses']:>7d} {d['neutral']:>8d} {d['verdict']:>11s} {ratio}"
        )

    print("\n  Relabel flow (original -> new, counts):")
    counts = relabel["counts"]
    flows = []
    for o, cols in counts.items():
        for n, c in cols.items():
            if c > 0:
                flows.append((o, n, c))
    flows.sort(key=lambda x: -x[2])
    for o, n, c in flows[:20]:
        print(f"    {o:<40s} -> {n:<40s}  {c:>4d}")
    if len(flows) > 20:
        print(f"    ... and {len(flows) - 20} more entries (see JSON)")


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--config", default="consol",
                        help="Config key to analyse (e.g. consol, consol_xfam_same_4o).")
    parser.add_argument(
        "--articles-dir", type=Path, default=Path("datasets/train-articles"),
    )
    parser.add_argument(
        "--labels-path", type=Path, default=Path("datasets/train-task2-TC.labels"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    block = data.get("results", data)
    if args.config not in block:
        raise SystemExit(
            f"Config {args.config!r} not in JSON. Available: {list(block)}"
        )
    cfg = block[args.config]

    eval_mode = cfg.get("config", {}).get("eval_mode", "permissive")
    preds_raw = cfg["predictions"]
    pred_ids = set(preds_raw)

    articles_all = load_semeval(args.articles_dir, args.labels_path)
    articles = [a for a in articles_all if a.id in pred_ids]
    if not articles:
        raise SystemExit("No articles matched prediction ids.")

    predictions = _rebuild_predictions(preds_raw, eval_mode)

    print(f"Config       : {args.config}")
    print(f"Articles     : {len(articles)}")
    print(f"Predictions  : {sum(len(p.spans) for p in predictions.values())}")
    print(f"Gold spans   : {sum(len(a.gold_spans) for a in articles)}")
    print(f"Eval mode    : {eval_mode}")

    # TC confusion matrix
    confusion = build_tc_confusion_matrix(articles, predictions)
    _print_matrix(confusion, f"TC confusion matrix — {args.config}")
    s = confusion["summary"]
    print(f"\n  TC hits (diagonal)             : {s['tc_hits_diagonal']}")
    print(f"  SI hits with wrong technique   : {s['tc_miss_within_si']}")
    print(f"  Gold spans with no prediction  : {s['gold_with_no_pred']}")
    print(f"  Predictions with no gold (FP)  : {s['pred_false_positive']}")

    # Relabel matrix + lucky/systematic
    label_log = build_label_change_log(articles, predictions)
    relabel = build_relabel_matrix(label_log)
    lvs = lucky_vs_systematic(label_log)
    _print_relabel_summary(lvs, relabel)

    out = {
        "config": args.config,
        "articles": len(articles),
        "gold_spans": sum(len(a.gold_spans) for a in articles),
        "eval_mode": eval_mode,
        "tc_confusion_matrix": confusion,
        "relabel_matrix": relabel,
        "lucky_vs_systematic": lvs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.output}")

    stem = args.output.with_suffix("")
    confusion_png = stem.parent / f"{stem.name}_confusion.png"
    relabel_png = stem.parent / f"{stem.name}_relabel.png"
    plot_confusion_matrix(
        confusion,
        confusion_png,
        title=f"TC confusion matrix — {args.config} ({len(articles)} articles)",
    )
    plot_relabel_matrix(
        relabel,
        relabel_png,
        title=(
            f"Relabel matrix — {args.config} ({len(articles)} articles)\n"
            f"cell = total relabels (net gains - losses)"
        ),
    )
    print(f"Wrote {confusion_png}")
    print(f"Wrote {relabel_png}")


if __name__ == "__main__":
    main()
