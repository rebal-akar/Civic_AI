"""Generate five HTML visualisations of the multi-pass detection + merge step.

Pick an illustrative article, then for each of 5 layout variants produce
a standalone HTML file that shows (a) what each of the 3 temperature
passes labelled and (b) the merged output produced by `merge_multipass`.

Run: uv run python -m scripts.visualize_multipass
Output: outputs/multipass_visualizations/variant_{1..5}.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────
ARTICLE_ID = "article723793978"
RESULT_PATH = Path("outputs/results/consol_gpt-4o_n122_9212d422.json")
OUT_DIR = Path("outputs/multipass_visualizations")
TEMPERATURES = [0.0, 0.3, 0.7]

# Distinct, high-contrast technique colours (used across all variants)
TECHNIQUE_COLORS = {
    "Loaded_Language":                   "#ffb347",
    "Name_Calling,Labeling":             "#c792ea",
    "Exaggeration,Minimisation":         "#ffd166",
    "Causal_Oversimplification":         "#82aaff",
    "Appeal_to_Fear-Prejudice":          "#f07178",
    "Doubt":                             "#89ddff",
    "Flag-Waving":                       "#a1e887",
    "Slogans":                           "#f78c6c",
    "Appeal_to_Authority":               "#b39ddb",
    "Black-and-White_Fallacy":           "#90a4ae",
    "Thought-terminating_Cliches":       "#ce93d8",
    "Whataboutism,Straw_Men,Red_Herring": "#a5d6a7",
    "Bandwagon,Reductio_ad_hitlerum":    "#ffab91",
    "Repetition":                        "#80deea",
}


def short_tech(tech: str) -> str:
    """Short label for badges."""
    mapping = {
        "Loaded_Language": "LL",
        "Name_Calling,Labeling": "NC",
        "Exaggeration,Minimisation": "EX",
        "Causal_Oversimplification": "CO",
        "Appeal_to_Fear-Prejudice": "AF",
        "Doubt": "DB",
        "Flag-Waving": "FW",
        "Slogans": "SL",
        "Appeal_to_Authority": "AA",
        "Black-and-White_Fallacy": "BW",
        "Thought-terminating_Cliches": "TC",
        "Whataboutism,Straw_Men,Red_Herring": "WH",
        "Bandwagon,Reductio_ad_hitlerum": "BR",
        "Repetition": "RP",
    }
    return mapping.get(tech, tech[:2].upper())


def pretty_tech(tech: str) -> str:
    return tech.replace("_", " ").replace(",", ", ")


# ── Data loading ───────────────────────────────────────────────────────

def load_data() -> dict:
    with open(RESULT_PATH, encoding="utf-8") as f:
        result = json.load(f)
    p = result["predictions"][ARTICLE_ID]
    text = p["article_text"]
    per_pass = p["stage_outputs"]["stage1_per_pass_spans"]
    merged = p["stage_snapshots"]["after_s1"]
    return {
        "article_id": ARTICLE_ID,
        "text": text,
        "passes": per_pass,
        "merged": merged,
    }


# ── Rendering helpers ──────────────────────────────────────────────────

def render_text_with_spans(text: str, spans: list[dict]) -> str:
    """Render article text with non-overlapping span highlights.

    For any overlapping spans in the same set, we render the span as a single
    outer wrapper whose technique takes the earliest-starting span; subsequent
    overlapping spans are shown as inline badges immediately after the wrapper.
    """
    if not spans:
        return f"<div class='text-body'>{html.escape(text)}</div>"

    events: list[tuple[int, int, int, dict]] = []
    for i, s in enumerate(spans):
        events.append((s["start"], 0, i, s))
        events.append((s["end"], 1, i, s))
    events.sort(key=lambda e: (e[0], e[1]))

    out: list[str] = []
    cursor = 0
    active: list[dict] = []

    boundaries = sorted({s["start"] for s in spans} | {s["end"] for s in spans})

    for b in boundaries:
        if b > cursor:
            frag = html.escape(text[cursor:b])
            if active:
                parts = []
                for s in active:
                    color = TECHNIQUE_COLORS.get(s["technique"], "#cccccc")
                    tag = short_tech(s["technique"])
                    parts.append(
                        f"<span class='tech-label' style='background:{color};' "
                        f"title='{html.escape(pretty_tech(s['technique']))}'>"
                        f"{tag}</span>"
                    )
                labels_html = "".join(parts)
                bg = TECHNIQUE_COLORS.get(active[0]["technique"], "#eee")
                out.append(
                    f"<span class='span-hl' "
                    f"style='background:{bg};'>{frag}{labels_html}</span>"
                )
            else:
                out.append(frag)
            cursor = b
        for s in spans:
            if s["start"] == b and s not in active:
                active.append(s)
        active = [s for s in active if s["end"] > b]

    if cursor < len(text):
        out.append(html.escape(text[cursor:]))

    return f"<div class='text-body'>{''.join(out)}</div>"


def legend_html(spans_used: set[str]) -> str:
    items = []
    for tech in spans_used:
        color = TECHNIQUE_COLORS.get(tech, "#ccc")
        items.append(
            f"<span class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color}'></span>"
            f"<span class='legend-label'>{html.escape(pretty_tech(tech))} "
            f"<span class='legend-short'>({short_tech(tech)})</span></span>"
            f"</span>"
        )
    return "<div class='legend'>" + "".join(items) + "</div>"


def collect_techniques(data: dict) -> set[str]:
    techs = set()
    for pass_spans in data["passes"]:
        for s in pass_spans:
            techs.add(s["technique"])
    for s in data["merged"]:
        techs.add(s["technique"])
    return techs


# ── Shared CSS ─────────────────────────────────────────────────────────

BASE_CSS = """
* { box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    background: #f7f7fa;
    color: #1a1a2e;
    margin: 0;
    padding: 32px;
    line-height: 1.55;
}
h1 { margin: 0 0 6px 0; font-size: 24px; font-weight: 700; }
h2 { margin: 24px 0 10px 0; font-size: 16px; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.04em; color: #555; }
h3 { margin: 14px 0 8px 0; font-size: 14px; font-weight: 600; color: #333; }
.subtitle { color: #666; font-size: 13px; margin-bottom: 24px; }
.text-body {
    font-family: 'Source Serif Pro', Georgia, serif;
    font-size: 14.5px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.span-hl {
    padding: 1px 2px;
    border-radius: 3px;
    box-shadow: inset 0 -2px 0 rgba(0,0,0,0.15);
}
.tech-label {
    display: inline-block;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 9.5px;
    font-weight: 700;
    color: #1a1a2e;
    padding: 1px 4px;
    margin: 0 1px;
    border-radius: 2px;
    vertical-align: 0.3em;
    border: 1px solid rgba(0,0,0,0.2);
}
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 12px;
    background: #fff;
    border: 1px solid #e0e0ea;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 12.5px;
}
.legend-item { display: inline-flex; align-items: center; gap: 5px; }
.legend-swatch {
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid rgba(0,0,0,0.15);
}
.legend-short { color: #888; font-size: 11px; }
.card {
    background: #fff;
    border: 1px solid #e0e0ea;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.meta {
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 11.5px;
    color: #888;
}
.agree {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 10px;
    font-size: 10.5px;
    font-weight: 700;
    font-family: 'JetBrains Mono', Consolas, monospace;
    color: #fff;
}
.agree-3 { background: #2a9d8f; }
.agree-2 { background: #e9c46a; color: #333; }
.agree-1 { background: #e76f51; }
"""


def html_shell(title: str, subtitle: str, body: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
{BASE_CSS}
{extra_css}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="subtitle">{html.escape(subtitle)}</div>
{body}
</body>
</html>
"""


# ── Variant 1: Side-by-side columns ─────────────────────────────────────

def variant1_columns(data: dict) -> str:
    techs = collect_techniques(data)
    css = """
.cols { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.col { background: #fff; border: 1px solid #e0e0ea; border-radius: 8px;
       padding: 14px; min-height: 300px; }
.col.merged { border: 2px solid #2a9d8f; background: #f0fdfb; }
.col-header { font-size: 13px; font-weight: 700; margin-bottom: 6px;
              display: flex; justify-content: space-between; align-items: center; }
.col-header .count { font-family: 'JetBrains Mono', Consolas, monospace;
                     font-size: 11px; color: #666; }
.pass-temp { color: #888; font-size: 11px; margin-bottom: 10px; }
.text-body { font-size: 12px; line-height: 1.55; }
@media (max-width: 1200px) { .cols { grid-template-columns: repeat(2, 1fr); } }
"""
    body_parts: list[str] = [legend_html(techs), "<div class='cols'>"]
    for i, pass_spans in enumerate(data["passes"]):
        body_parts.append(f"""
        <div class='col'>
          <div class='col-header'>
            <span>Pass {i}</span>
            <span class='count'>{len(pass_spans)} spans</span>
          </div>
          <div class='pass-temp'>temperature = {TEMPERATURES[i]}</div>
          {render_text_with_spans(data['text'], pass_spans)}
        </div>
        """)
    body_parts.append(f"""
    <div class='col merged'>
      <div class='col-header'>
        <span>Merged (after S1)</span>
        <span class='count'>{len(data['merged'])} spans</span>
      </div>
      <div class='pass-temp'>union of pass outputs, agreement counted</div>
      {render_text_with_spans(data['text'], data['merged'])}
    </div>
    """)
    body_parts.append("</div>")

    return html_shell(
        title="Multi-pass detection (3 passes → merged union)",
        subtitle=(
            f"Article {ARTICLE_ID}. Each pass re-runs Stage 1 with a different "
            f"sampling temperature. The merged column is the union of pass outputs, "
            f"with each span annotated by how many passes found it (agreement)."
        ),
        body="".join(body_parts),
        extra_css=css,
    )


# ── Variant 2: Vertical stack with agreement badges ─────────────────────

def variant2_stacked(data: dict) -> str:
    techs = collect_techniques(data)
    css = """
.stack { display: flex; flex-direction: column; gap: 16px; }
.row { background: #fff; border: 1px solid #e0e0ea; border-radius: 8px;
       padding: 18px 22px; }
.row.merged { border-left: 4px solid #2a9d8f; background: #f0fdfb; }
.row-header { display: flex; justify-content: space-between;
              align-items: baseline; margin-bottom: 10px; }
.row-title { font-size: 15px; font-weight: 700; }
.row-stat { font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 12px; color: #666; }
.merged-list { margin-top: 14px; }
.merged-item { display: flex; align-items: center; gap: 10px;
               padding: 5px 0; border-bottom: 1px solid #f0f0f4; font-size: 13px; }
.merged-item:last-child { border-bottom: none; }
.merged-item .tech-chip {
  font-family: 'JetBrains Mono', Consolas, monospace;
  padding: 2px 8px; border-radius: 4px; font-size: 11px;
  font-weight: 700; color: #1a1a2e; border: 1px solid rgba(0,0,0,0.15);
}
.merged-item .span-text { color: #333; font-style: italic; flex: 1;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.merged-item .pos { color: #888; font-family: 'JetBrains Mono', Consolas, monospace;
                    font-size: 11px; }
"""
    body: list[str] = [legend_html(techs), "<div class='stack'>"]
    for i, pass_spans in enumerate(data["passes"]):
        body.append(f"""
        <div class='row'>
          <div class='row-header'>
            <div class='row-title'>Pass {i} &nbsp; <span class='meta'>t={TEMPERATURES[i]}</span></div>
            <div class='row-stat'>{len(pass_spans)} spans</div>
          </div>
          {render_text_with_spans(data['text'], pass_spans)}
        </div>
        """)

    merged_list_html = ""
    for s in data["merged"]:
        color = TECHNIQUE_COLORS.get(s["technique"], "#ccc")
        agree = s.get("agreement_count", 1)
        agree_class = f"agree-{min(3, max(1, agree))}"
        merged_list_html += (
            f"<div class='merged-item'>"
            f"<span class='agree {agree_class}'>{agree}/3</span>"
            f"<span class='tech-chip' style='background:{color};'>"
            f"{short_tech(s['technique'])}</span>"
            f"<span class='pos'>[{s['start']}-{s['end']}]</span>"
            f"<span class='span-text'>"
            f"{html.escape(s['span_text'][:110])}</span>"
            f"</div>"
        )

    body.append(f"""
    <div class='row merged'>
      <div class='row-header'>
        <div class='row-title'>Merged union (after Stage 1)</div>
        <div class='row-stat'>{len(data['merged'])} spans</div>
      </div>
      {render_text_with_spans(data['text'], data['merged'])}
      <div class='merged-list'>
        <h3>Unique merged spans with agreement counts</h3>
        {merged_list_html}
      </div>
    </div>
    """)
    body.append("</div>")

    return html_shell(
        title="Multi-pass detection — passes stacked above merged union",
        subtitle=(
            f"Article {ARTICLE_ID}. The same text is shown under each of 3 "
            f"self-consistency passes. Agreement count (n/3) indicates how many "
            f"passes produced a given merged span."
        ),
        body="".join(body),
        extra_css=css,
    )


# ── Variant 3: Overlay view with pass-presence indicators ───────────────

def variant3_overlay(data: dict) -> str:
    techs = collect_techniques(data)

    unique_locations: list[dict] = []
    for s in data["merged"]:
        unique_locations.append({
            "start": s["start"],
            "end": s["end"],
            "technique": s["technique"],
            "span_text": s["span_text"],
            "agreement_count": s.get("agreement_count", 1),
        })

    def found_by(sstart: int, send: int, tech: str) -> list[int]:
        hits = []
        for i, pass_spans in enumerate(data["passes"]):
            for p in pass_spans:
                if p["start"] == sstart and p["end"] == send and p["technique"] == tech:
                    hits.append(i)
                    break
        return hits

    css = """
.overlay-wrap { background: #fff; border: 1px solid #e0e0ea;
                border-radius: 8px; padding: 20px 24px; }
.overlay-body .text-body { font-size: 15px; }
.span-matrix { margin-top: 18px; width: 100%; border-collapse: collapse; font-size: 13px; }
.span-matrix th, .span-matrix td { padding: 6px 10px; text-align: left;
                                    border-bottom: 1px solid #eee; }
.span-matrix th { background: #f5f5f9; font-size: 11px;
                  text-transform: uppercase; letter-spacing: 0.04em; color: #555; }
.presence { display: inline-flex; gap: 3px; }
.dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block;
       border: 1px solid rgba(0,0,0,0.2); font-size: 9px; line-height: 14px;
       text-align: center; font-weight: 700; color: #fff; }
.dot.on { background: #2a9d8f; }
.dot.off { background: #eceff1; color: #bbb; }
.span-snip { font-style: italic; color: #555; max-width: 260px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
"""

    body = [legend_html(techs)]
    body.append("<div class='overlay-wrap overlay-body'>")
    body.append("<h2>Merged article (union of 3 passes)</h2>")
    body.append(render_text_with_spans(data["text"], data["merged"]))

    rows = []
    for loc in sorted(unique_locations, key=lambda s: (s["start"], s["end"])):
        color = TECHNIQUE_COLORS.get(loc["technique"], "#ccc")
        hits = found_by(loc["start"], loc["end"], loc["technique"])
        dots = "".join(
            f"<span class='dot {'on' if i in hits else 'off'}'>{i}</span>"
            for i in range(3)
        )
        rows.append(
            f"<tr>"
            f"<td><span class='tech-chip' style='background:{color};padding:2px 8px;"
            f"border-radius:4px;font-family:monospace;font-size:11px;font-weight:700;"
            f"border:1px solid rgba(0,0,0,.15);'>{short_tech(loc['technique'])}</span></td>"
            f"<td class='meta'>[{loc['start']}-{loc['end']}]</td>"
            f"<td class='span-snip'>{html.escape(loc['span_text'][:120])}</td>"
            f"<td><span class='presence'>{dots}</span></td>"
            f"<td class='meta'>{loc['agreement_count']}/3</td>"
            f"</tr>"
        )

    body.append(f"""
    <h3>Per-span pass presence</h3>
    <table class='span-matrix'>
      <thead><tr>
        <th>Technique</th><th>Position</th><th>Snippet</th>
        <th>Found by passes</th><th>Agreement</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """)
    body.append("</div>")

    return html_shell(
        title="Multi-pass — merged text with per-span pass presence",
        subtitle=(
            f"Article {ARTICLE_ID}. One text with merged highlights; the table "
            f"below shows which of the 3 passes produced each final span (green "
            f"dot = found, grey dot = missed)."
        ),
        body="".join(body),
        extra_css=css,
    )


# ── Variant 4: Matrix / spreadsheet view ────────────────────────────────

def variant4_matrix(data: dict) -> str:
    techs = collect_techniques(data)

    keys: dict[tuple[int, int, str], dict] = {}
    for i, pass_spans in enumerate(data["passes"]):
        for s in pass_spans:
            k = (s["start"], s["end"], s["technique"])
            if k not in keys:
                keys[k] = {
                    "start": s["start"], "end": s["end"],
                    "technique": s["technique"], "span_text": s["span_text"],
                    "by_pass": [False, False, False],
                    "merged": False,
                    "agree": 0,
                }
            keys[k]["by_pass"][i] = True

    for s in data["merged"]:
        k = (s["start"], s["end"], s["technique"])
        if k in keys:
            keys[k]["merged"] = True
            keys[k]["agree"] = s.get("agreement_count", 1)

    sorted_keys = sorted(keys.values(), key=lambda x: (x["start"], x["end"]))

    css = """
.matrix-wrap { background: #fff; border: 1px solid #e0e0ea; border-radius: 8px;
               padding: 20px 24px; overflow-x: auto; }
.preview { margin-bottom: 22px; background: #fafafc; padding: 14px;
           border-radius: 6px; border: 1px solid #ececf2; }
.matrix { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.matrix th { background: #f5f5f9; padding: 8px 10px; text-align: left;
             border-bottom: 2px solid #e0e0ea; font-size: 11px;
             text-transform: uppercase; letter-spacing: 0.04em; color: #555; }
.matrix td { padding: 7px 10px; border-bottom: 1px solid #f0f0f4; }
.matrix tr:hover { background: #fafafe; }
.matrix tr.merged-row { background: #f0fdfb; }
.matrix tr.merged-row td { font-weight: 500; }
.cell-hit, .cell-miss { display: inline-block; width: 20px; height: 20px;
                        line-height: 20px; text-align: center; border-radius: 4px;
                        font-weight: 700; font-family: 'JetBrains Mono', Consolas, monospace;
                        font-size: 11px; }
.cell-hit  { background: #2a9d8f; color: #fff; }
.cell-miss { background: #eceff1; color: #b0bec5; }
.tech-chip { font-family: 'JetBrains Mono', Consolas, monospace;
             padding: 2px 8px; border-radius: 4px; font-size: 11px;
             font-weight: 700; border: 1px solid rgba(0,0,0,0.15); color: #1a1a2e; }
.span-snip { font-style: italic; color: #555; max-width: 340px;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
"""

    row_html = []
    for k in sorted_keys:
        color = TECHNIQUE_COLORS.get(k["technique"], "#ccc")
        cells = "".join(
            f"<td><span class='{'cell-hit' if hit else 'cell-miss'}'>"
            f"{'✓' if hit else '·'}</span></td>"
            for hit in k["by_pass"]
        )
        agree = k["agree"] if k["merged"] else sum(k["by_pass"])
        agree_class = f"agree-{min(3, max(1, agree))}"
        row_class = "merged-row" if k["merged"] else ""
        row_html.append(
            f"<tr class='{row_class}'>"
            f"<td class='meta'>[{k['start']}-{k['end']}]</td>"
            f"<td><span class='tech-chip' style='background:{color};'>"
            f"{short_tech(k['technique'])}</span></td>"
            f"<td class='span-snip'>{html.escape(k['span_text'][:120])}</td>"
            f"{cells}"
            f"<td><span class='agree {agree_class}'>{agree}/3</span></td>"
            f"</tr>"
        )

    body = [legend_html(techs), "<div class='matrix-wrap'>"]
    body.append("<h2>Article with merged highlights</h2>")
    body.append(f"<div class='preview'>{render_text_with_spans(data['text'], data['merged'])}</div>")

    body.append(f"""
    <h2>Span × Pass matrix ({len(sorted_keys)} unique span-technique pairs)</h2>
    <table class='matrix'>
      <thead>
        <tr>
          <th>Position</th><th>Technique</th><th>Span text</th>
          <th>Pass 0<br><span class='meta'>t={TEMPERATURES[0]}</span></th>
          <th>Pass 1<br><span class='meta'>t={TEMPERATURES[1]}</span></th>
          <th>Pass 2<br><span class='meta'>t={TEMPERATURES[2]}</span></th>
          <th>Agreement</th>
        </tr>
      </thead>
      <tbody>{''.join(row_html)}</tbody>
    </table>
    """)
    body.append("</div>")

    return html_shell(
        title="Multi-pass — span × pass presence matrix",
        subtitle=(
            f"Article {ARTICLE_ID}. Each row is a unique (span, technique) pair "
            f"found in at least one pass. Columns 4–6 show which passes produced "
            f"it. Green-tinted rows survive into the merged output."
        ),
        body="".join(body),
        extra_css=css,
    )


# ── Variant 5: Flow diagram (3 cards → arrow → merged card) ─────────────

def variant5_flow(data: dict) -> str:
    techs = collect_techniques(data)
    css = """
.flow { display: grid; grid-template-columns: 1fr 60px 1.2fr; gap: 14px;
        align-items: start; }
.flow-col { display: flex; flex-direction: column; gap: 10px; }
.flow-arrow { display: flex; align-items: center; justify-content: center;
              font-size: 32px; color: #2a9d8f; font-weight: 700;
              padding-top: 150px; }
.pass-card { background: #fff; border: 1px solid #e0e0ea; border-radius: 8px;
             padding: 12px 14px; }
.pass-card-header { display: flex; justify-content: space-between;
                    align-items: baseline; margin-bottom: 8px; }
.pass-card-title { font-weight: 700; font-size: 13.5px; }
.pass-card-meta { font-family: 'JetBrains Mono', Consolas, monospace;
                  font-size: 11px; color: #666; }
.pass-preview { font-size: 11.5px; font-family: Georgia, serif;
                max-height: 180px; overflow-y: auto; line-height: 1.45; }
.merge-card { background: #fff; border: 2px solid #2a9d8f; border-radius: 10px;
              padding: 18px 22px; }
.merge-card-title { font-size: 17px; font-weight: 700; color: #2a9d8f;
                    margin-bottom: 8px; }
.merge-counts { display: flex; gap: 12px; margin: 10px 0 14px 0; font-size: 12px; }
.merge-counts .pill { background: #f0fdfb; border: 1px solid #a7e9df;
                      padding: 3px 10px; border-radius: 20px;
                      font-family: 'JetBrains Mono', Consolas, monospace; }
.merge-items { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
.merge-item { display: flex; align-items: center; gap: 8px;
              padding: 5px 8px; background: #fafafc; border-radius: 5px;
              font-size: 12px; }
.merge-item .tech-chip { font-family: 'JetBrains Mono', Consolas, monospace;
                         padding: 2px 7px; border-radius: 4px; font-size: 10.5px;
                         font-weight: 700; border: 1px solid rgba(0,0,0,0.15);
                         color: #1a1a2e; }
.merge-item .snip { color: #555; font-style: italic; flex: 1;
                    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
@media (max-width: 1000px) { .flow { grid-template-columns: 1fr; }
                             .flow-arrow { padding: 10px 0; transform: rotate(90deg); } }
"""
    pass_cards = []
    for i, pass_spans in enumerate(data["passes"]):
        pass_cards.append(f"""
          <div class='pass-card'>
            <div class='pass-card-header'>
              <span class='pass-card-title'>Pass {i}</span>
              <span class='pass-card-meta'>t={TEMPERATURES[i]} &middot; {len(pass_spans)} spans</span>
            </div>
            <div class='pass-preview'>
              {render_text_with_spans(data['text'][:520] + '...', [s for s in pass_spans if s['start'] < 520])}
            </div>
          </div>
        """)

    merge_items = []
    for s in sorted(data["merged"], key=lambda x: (-x.get("agreement_count", 1), x["start"])):
        color = TECHNIQUE_COLORS.get(s["technique"], "#ccc")
        agree = s.get("agreement_count", 1)
        agree_class = f"agree-{min(3, max(1, agree))}"
        merge_items.append(
            f"<div class='merge-item'>"
            f"<span class='agree {agree_class}'>{agree}/3</span>"
            f"<span class='tech-chip' style='background:{color};'>"
            f"{short_tech(s['technique'])}</span>"
            f"<span class='meta'>[{s['start']}-{s['end']}]</span>"
            f"<span class='snip'>{html.escape(s['span_text'][:90])}</span>"
            f"</div>"
        )

    body = [legend_html(techs), f"""
    <div class='flow'>
      <div class='flow-col'>{''.join(pass_cards)}</div>
      <div class='flow-arrow'>&#10132;</div>
      <div class='merge-card'>
        <div class='merge-card-title'>Merged union (after Stage 1)</div>
        <div class='merge-counts'>
          <span class='pill'>{sum(len(p) for p in data['passes'])} total raw</span>
          <span class='pill'>{len(data['merged'])} unique merged</span>
        </div>
        <div>
          {render_text_with_spans(data['text'], data['merged'])}
        </div>
        <div class='merge-items'>{''.join(merge_items)}</div>
      </div>
    </div>
    """]

    return html_shell(
        title="Multi-pass detection — flow diagram (3 passes → merged)",
        subtitle=(
            f"Article {ARTICLE_ID}. Left: three independent detection passes with "
            f"different temperatures. Right: the merged union after spans are "
            f"deduplicated by (start, end, technique) and tagged with agreement counts."
        ),
        body="".join(body),
        extra_css=css,
    )


# ── Entrypoint ─────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    variants = [
        ("variant_1_columns.html",   variant1_columns),
        ("variant_2_stacked.html",   variant2_stacked),
        ("variant_3_overlay.html",   variant3_overlay),
        ("variant_4_matrix.html",    variant4_matrix),
        ("variant_5_flow.html",      variant5_flow),
    ]

    for fname, fn in variants:
        html_text = fn(data)
        path = OUT_DIR / fname
        path.write_text(html_text, encoding="utf-8")
        print(f"  wrote {path}")

    print(f"\nOpen any of them in your browser:")
    for fname, _ in variants:
        print(f"  {OUT_DIR / fname}")


if __name__ == "__main__":
    main()
