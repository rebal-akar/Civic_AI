#!/usr/bin/env python3
"""
Visual side-by-side comparison of gold labels vs model predictions.

Left pane: gold span labels from a consolidated labels file.
Right pane: predicted spans from experiment results JSON.

Usage:
    python -m scripts.view_gold_vs_pred \
        --results outputs/results/asv_gpt-4o-mini_n10_ea4db2a6.json \
        --article-id article111111111
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared colors for techniques.
TECHNIQUE_COLORS = {
    "Loaded_Language": "#fef3c7",
    "Name_Calling,Labeling": "#fecaca",
    "Repetition": "#bbf7d0",
    "Exaggeration,Minimisation": "#fbcfe8",
    "Doubt": "#e0e7ff",
    "Appeal_to_Fear-Prejudice": "#fed7aa",
    "Flag-Waving": "#fde68a",
    "Causal_Oversimplification": "#c7d2fe",
    "Slogans": "#a5f3fc",
    "Appeal_to_Authority": "#d9f99d",
    "Black-and-White_Fallacy": "#e9d5ff",
    "Thought-terminating_Cliches": "#f5d0fe",
    "Whataboutism,Straw_Men,Red_Herring": "#ddd6fe",
    "Whataboutism": "#ddd6fe",
    "Bandwagon,Reductio_ad_hitlerum": "#fecaca",
}

VERDICT_BORDER = {
    "CONFIRMED": "#16a34a",
    "POSSIBLE": "#ca8a04",
    "REJECTED": "#dc2626",
    "": "#6b7280",
}


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _normalise_id(article_id: str) -> str:
    article_id = article_id.strip()
    if article_id.startswith("article"):
        return article_id
    return f"article{article_id}"


def load_article_text(articles_dir: Path, article_id: str) -> str:
    path = articles_dir / f"{article_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Article not found: {path}")
    return path.read_text(encoding="utf-8")


def load_gold_spans(labels_path: Path, article_id: str) -> list[dict]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    spans: list[dict] = []
    for raw in labels_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        aid, tech, start_s, end_s = parts[0], parts[1], parts[2], parts[3]
        if aid.strip() != article_id:
            continue
        try:
            start = int(start_s)
            end = int(end_s)
        except ValueError:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "technique": tech.strip(),
                "verdict": "",
                "source": "gold",
            }
        )
    return spans


def load_pred_spans(results_path: Path, article_id: str) -> list[dict]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    pred_entry = data.get("predictions", {}).get(article_id, {})
    raw_spans = pred_entry.get("all_spans", [])

    spans: list[dict] = []
    for s in raw_spans:
        start = int(s.get("start", -1))
        end = int(s.get("end", -1))
        if start < 0 or end <= start:
            # unresolved span; skip for visual char-offset rendering
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "technique": str(s.get("technique", "")).strip(),
                "verdict": str(s.get("verdict", "")).upper().strip(),
                "source": "pred",
            }
        )
    return spans


def _segment_text(text: str, spans: list[dict]) -> list[tuple[int, int, list[dict]]]:
    if not spans:
        return [(0, len(text), [])]

    boundaries = {0, len(text)}
    for s in spans:
        boundaries.add(max(0, min(len(text), s["start"])))
        boundaries.add(max(0, min(len(text), s["end"])))
    sorted_bounds = sorted(boundaries)

    segments: list[tuple[int, int, list[dict]]] = []
    for i in range(len(sorted_bounds) - 1):
        start = sorted_bounds[i]
        end = sorted_bounds[i + 1]
        if start >= end:
            continue
        covering = [s for s in spans if s["start"] <= start and s["end"] >= end]
        segments.append((start, end, covering))
    return segments


def _render_highlighted_text(text: str, spans: list[dict], show_verdict: bool) -> str:
    segments = _segment_text(text, spans)
    html_parts: list[str] = []

    for seg_start, seg_end, covering in segments:
        chunk = _escape(text[seg_start:seg_end])
        if not covering:
            html_parts.append(chunk)
            continue

        primary = covering[0]
        tech = primary["technique"] or "Unknown"
        verdict = primary.get("verdict", "")
        color = TECHNIQUE_COLORS.get(tech, "#e5e7eb")
        title_bits = [tech]
        if show_verdict and verdict:
            title_bits.append(f"verdict={verdict}")
        title = "; ".join(title_bits)

        border = VERDICT_BORDER.get(verdict, "#6b7280") if show_verdict else "#374151"
        html_parts.append(
            f'<span class="hl" style="background:{color};border-bottom:2px solid {border}" '
            f'title="{_escape(title)}">{chunk}</span>'
        )

    return "".join(html_parts)


def _legend_html(spans: list[dict], include_verdict: bool) -> str:
    techniques = sorted({s["technique"] for s in spans if s.get("technique")})
    if not techniques:
        return "<em>No spans</em>"
    tags = []
    for t in techniques:
        color = TECHNIQUE_COLORS.get(t, "#e5e7eb")
        tags.append(f'<span class="tag" style="background:{color}">{_escape(t)}</span>')

    verdict_block = ""
    if include_verdict:
        verdict_block = (
            '<div class="verdict-legend">'
            '<span class="vtag" style="border-color:#16a34a">CONFIRMED</span>'
            '<span class="vtag" style="border-color:#ca8a04">POSSIBLE</span>'
            '<span class="vtag" style="border-color:#dc2626">REJECTED</span>'
            "</div>"
        )
    return " ".join(tags) + verdict_block


def build_html(
    article_id: str,
    text: str,
    gold_spans: list[dict],
    pred_spans: list[dict],
    results_path: Path,
) -> str:
    title = _escape(text.splitlines()[0].strip() or article_id)
    left_body = _render_highlighted_text(text, gold_spans, show_verdict=False)
    right_body = _render_highlighted_text(text, pred_spans, show_verdict=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gold vs Pred - {article_id}</title>
  <style>
    body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 0; color: #111827; }}
    .wrap {{ max-width: 1400px; margin: 1rem auto; padding: 0 1rem; }}
    .meta {{ color: #6b7280; margin-bottom: 1rem; font-size: 0.95rem; }}
    .title {{ margin: 0.25rem 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }}
    .ph {{ background: #f9fafb; border-bottom: 1px solid #e5e7eb; padding: 10px 12px; font-weight: 600; }}
    .pc {{ padding: 14px; line-height: 1.6; white-space: normal; font-family: Georgia, 'Times New Roman', serif; }}
    .hl {{ border-radius: 2px; padding: 0 1px; cursor: help; }}
    .legend {{ margin-top: 8px; color: #4b5563; font-size: 0.85rem; }}
    .tag {{ display: inline-block; margin: 2px 6px 2px 0; padding: 2px 8px; border-radius: 999px; }}
    .verdict-legend {{ margin-top: 6px; }}
    .vtag {{ display:inline-block; margin-right:8px; padding:2px 8px; border:2px solid; border-radius:999px; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2 class="title">{title}</h2>
    <div class="meta">
      Article: <code>{article_id}</code> |
      Gold spans: <strong>{len(gold_spans)}</strong> |
      Pred spans: <strong>{len(pred_spans)}</strong> |
      Results: <code>{_escape(str(results_path))}</code>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="ph">Gold Labels</div>
        <div class="pc">{left_body}</div>
        <div class="pc legend">{_legend_html(gold_spans, include_verdict=False)}</div>
      </section>
      <section class="panel">
        <div class="ph">Model Predictions</div>
        <div class="pc">{right_body}</div>
        <div class="pc legend">{_legend_html(pred_spans, include_verdict=True)}</div>
      </section>
    </div>
  </div>
</body>
</html>"""


def choose_default_article_id(results_path: Path) -> str:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    predictions = data.get("predictions", {})
    if not predictions:
        raise ValueError(f"No predictions found in {results_path}")
    return sorted(predictions.keys())[0]


def infer_paths_from_results(results_path: Path) -> tuple[Path | None, Path | None]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    articles_dir = config.get("articles_dir")
    labels_path = config.get("labels_path")
    a = Path(articles_dir) if isinstance(articles_dir, str) and articles_dir else None
    l = Path(labels_path) if isinstance(labels_path, str) and labels_path else None
    return a, l


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize gold vs predicted spans side-by-side")
    parser.add_argument("--results", required=True, help="Path to experiment results JSON")
    parser.add_argument(
        "--articles-dir",
        default=None,
        help="Directory containing article .txt files (optional if present in results config)",
    )
    parser.add_argument(
        "--labels-path",
        default=None,
        help="Consolidated labels file for gold spans (optional if present in results config)",
    )
    parser.add_argument("--article-id", default=None, help="Article ID (e.g., article111111111 or 111111111)")
    parser.add_argument(
        "--out",
        default="outputs/results/gold_vs_pred.html",
        help="Output HTML path",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    results_path = Path(args.results)
    inferred_articles_dir, inferred_labels_path = infer_paths_from_results(results_path)
    articles_dir = Path(args.articles_dir) if args.articles_dir else inferred_articles_dir
    labels_path = Path(args.labels_path) if args.labels_path else inferred_labels_path
    if articles_dir is None or labels_path is None:
        raise ValueError(
            "Could not determine --articles-dir/--labels-path. "
            "Pass them explicitly or ensure they are present in results['config']."
        )
    out_path = Path(args.out)

    article_id = _normalise_id(args.article_id) if args.article_id else choose_default_article_id(results_path)
    text = load_article_text(articles_dir, article_id)
    gold_spans = load_gold_spans(labels_path, article_id)
    pred_spans = load_pred_spans(results_path, article_id)

    html = build_html(article_id, text, gold_spans, pred_spans, results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Article: {article_id} | gold={len(gold_spans)} | pred={len(pred_spans)}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
        print("Opened in browser.")


if __name__ == "__main__":
    main()

