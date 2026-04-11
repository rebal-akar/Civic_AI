#!/usr/bin/env python3
"""
View a training article with propaganda technique labels highlighted.
Generates an HTML file and opens it in the browser.
"""
from pathlib import Path
import webbrowser

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = PROJECT_ROOT / "datasets" / "train-articles"
LABELS_DIR = PROJECT_ROOT / "datasets" / "train-labels-task2-technique-classification"

# Distinct colors per technique (readable, print-friendly)
TECHNIQUE_COLORS = {
    "Loaded_Language": "#fef3c7",           # amber-100
    "Name_Calling,Labeling": "#fecaca",     # red-200
    "Repetition": "#bbf7d0",                # green-200
    "Exaggeration,Minimisation": "#fbcfe8", # pink-200
    "Doubt": "#e0e7ff",                     # indigo-100
    "Appeal_to_fear-prejudice": "#fed7aa",  # orange-200
    "Flag-Waving": "#fde68a",               # yellow-200
    "Causal_Oversimplification": "#c7d2fe", # indigo-200
    "Slogans": "#a5f3fc",                   # cyan-200
    "Appeal_to_Authority": "#d9f99d",      # lime-200
    "Black-and-White_Fallacy": "#e9d5ff",   # purple-200
    "Thought-terminating_Cliches": "#f5d0fe", # fuchsia-200
    "Whataboutism,Straw_Men,Red_Herring": "#ddd6fe", # violet-200
    "Bandwagon,Reductio_ad_Hitlerum": "#fecaca", # red-200 (reuse)
}


def load_article(article_id: str) -> str:
    """Load article text by ID."""
    path = ARTICLES_DIR / f"article{article_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Article not found: {path}")
    return path.read_text(encoding="utf-8")


def load_labels(article_id: str) -> list[tuple[int, int, str]]:
    """Load labels for an article. Returns list of (start, end, technique)."""
    path = LABELS_DIR / f"article{article_id}.task2-TC.labels"
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            _, technique, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            labels.append((start, end, technique))
    return labels


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def build_highlighted_html(text: str, labels: list[tuple[int, int, str]], article_id: str) -> str:
    """Build HTML with highlighted spans. Handles overlaps by segmenting."""
    if not labels:
        return _escape(text)

    # Build non-overlapping segments: (start, end) -> list of techniques
    boundaries = {0, len(text)}
    for start, end, _ in labels:
        boundaries.add(max(0, start))
        boundaries.add(min(len(text), end))
    boundaries = sorted(boundaries)

    segments = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        if seg_start >= seg_end:
            continue
        techniques = [
            tech for s, e, tech in labels
            if s <= seg_start and e >= seg_end
        ]
        segments.append((seg_start, seg_end, techniques))

    html_parts = []
    for seg_start, seg_end, techniques in segments:
        chunk = text[seg_start:seg_end]
        chunk = _escape(chunk)
        if techniques:
            # Use first technique's color; show all on hover
            tech_str = "; ".join(t.replace("_", " ") for t in techniques)
            color = TECHNIQUE_COLORS.get(techniques[0], "#e5e7eb")
            html_parts.append(
                f'<span class="highlight" style="background:{color};border-bottom:2px solid #374151" '
                f'title="{tech_str}">{chunk}</span>'
            )
        else:
            html_parts.append(chunk)

    # Use first line as title if available
    first_line = text.split("\n")[0].strip() or f"Article {article_id}"
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    first_line = first_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # Wrap in full HTML
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{first_line}</title>
  <style>
    body {{
      font-family: 'Georgia', 'Times New Roman', serif;
      max-width: 720px;
      margin: 2rem auto;
      padding: 0 1.5rem;
      line-height: 1.7;
      color: #1f2937;
    }}
    h1 {{
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #111827;
    }}
    .meta {{
      font-size: 0.875rem;
      color: #6b7280;
      margin-bottom: 1.5rem;
    }}
    .highlight {{
      cursor: help;
      padding: 0 1px;
      border-radius: 2px;
    }}
    .highlight:hover {{
      outline: 2px solid #6366f1;
      outline-offset: 1px;
    }}
    .legend {{
      margin-top: 2rem;
      padding-top: 1rem;
      border-top: 1px solid #e5e7eb;
      font-size: 0.8rem;
      color: #6b7280;
    }}
    .legend span {{
      display: inline-block;
      margin: 2px 8px 2px 0;
      padding: 2px 8px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>{first_line}</h1>
  <p class="meta">{len(labels)} propaganda spans labeled</p>
  <div class="content">
    {"".join(html_parts)}
  </div>
  <div class="legend">
    <strong>Techniques in this article:</strong><br>
    {" ".join(f'<span style="background:{TECHNIQUE_COLORS.get(t, "#e5e7eb")}">{t.replace("_", " ")}</span>' 
              for t in sorted(set(l[2] for l in labels)))}
  </div>
</body>
</html>"""


def list_articles_with_labels() -> list[str]:
    """Return article IDs that have both article and labels."""
    ids = set()
    for p in LABELS_DIR.glob("article*.task2-TC.labels"):
        aid = p.stem.replace("article", "").replace(".task2-TC", "")
        if (ARTICLES_DIR / f"article{aid}.txt").exists():
            ids.add(aid)
    return sorted(ids)


def main():
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--list":
        ids = list_articles_with_labels()
        print(f"{len(ids)} articles with labels. Examples: {', '.join(ids[:10])}...")
        return
    article_id = args[0] if args else "111111111"
    text = load_article(article_id)
    labels = load_labels(article_id)
    html = build_highlighted_html(text, labels, article_id)
    out_path = PROJECT_ROOT / "labeled_article.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")
    webbrowser.open(out_path.as_uri())
    print(f"Opened in browser. Article {article_id}: {len(labels)} labeled spans.")
    print("Usage: python scripts/view_labeled_article.py [ARTICLE_ID]  e.g. 111111112, 705035735")


if __name__ == "__main__":
    main()
