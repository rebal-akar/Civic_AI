# LLM-Based Propaganda Detection

Undergraduate dissertation codebase: comparing prompting strategies for end-to-end joint span annotation of propaganda techniques, using the SemEval-2020 Task 11 taxonomy (14 techniques).

Five strategies are evaluated: **zero-shot**, **few-shot**, **ASV** (detect + adversarial critique), **consol** (detect + consolidation), and **hybrid** (detect + critique + refinement). Evaluation uses the official SemEval character-overlap F1 metric.

---

## Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`
- An OpenAI API key (all experiments) and an Anthropic API key (cross-family experiments only)

## Setup

```bash
# 1. Install dependencies (uv handles the venv automatically)
uv sync

#    Or, with pip:
#    python -m venv .venv
#    .venv\Scripts\activate          # Windows (PowerShell)
#    source .venv/bin/activate       # macOS / Linux
#    pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Then edit .env and add:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...   (only for cross-family runs)

# 3. Download SemEval-2020 Task 11 data from
#    https://zenodo.org/records/3952415
#    Place the files so that these paths exist:
#       datasets/train-articles/          (article_XXXXX.txt files)
#       datasets/train-task2-TC.labels
```

All commands below use `uv run …`; drop the `uv run` prefix if you installed with pip and have the venv activated.

---

## Creating the train/eval split

Articles are split 70/30, stratified by each article's rarest technique so rare categories appear in both halves. Run this once:

```bash
uv run python -m scripts.split_data `
    --articles-dir datasets/train-articles `
    --labels-path  datasets/train-task2-TC.labels `
    --output-dir   data/splits `
    --eval-ratio   0.3 `
    --seed         42
```

*(PowerShell uses `` ` `` as the line-continuation character. On bash/zsh, replace `` ` `` with `\`.)*

This produces `data/splits/train/` and `data/splits/eval/`, plus `data/splits/split_metadata.json` recording exactly which articles went where. All experiments in the dissertation run on the 122-article eval split.

---

## Running experiments

Each run hits the OpenAI (and optionally Anthropic) API. Costs are tracked and capped per run via `--max-cost`. Response caching means re-runs with unchanged prompts are free.

### Single strategy, single model

```bash
uv run python -m src.run_experiment `
    --strategy zero_shot `
    --model gpt-4o `
    --articles-dir data/splits/eval/articles `
    --labels-path  data/splits/eval/eval-task2-TC.labels `
    --max-cost 5.0
```

Useful flags:
- `--strategy {zero_shot,few_shot,asv,consol,hybrid}`
- `--model` — any OpenAI or Anthropic model name (e.g. `gpt-4o`, `gpt-4o-mini`, `claude-sonnet-4-6`)
- `--verify-model` — a *different* model for Stage 2/3 (enables cross-model verification, e.g. gpt-4o detects, sonnet consolidates)
- `--max-articles N` (or `--limit N`) — run on the first *N* articles only
- `--eval-mode {permissive,strict}` — permissive = CONFIRMED+POSSIBLE, strict = CONFIRMED only

### Multi-strategy comparison — pick exactly what you want

`run_all` runs one or more strategies back-to-back on the same article set and produces a combined comparison JSON plus charts. You can narrow this by **strategies**, **model**, and **article count**.

```bash
# Full five-strategy comparison on gpt-4o over all 122 eval articles
uv run python -m src.run_all `
    --model gpt-4o `
    --articles-dir data/splits/eval/articles `
    --labels-path  data/splits/eval/eval-task2-TC.labels `
    --max-cost 10.0

# Only zero_shot and consol on gpt-4o-mini, first 80 articles
uv run python -m src.run_all `
    --model gpt-4o-mini `
    --strategies zero_shot,consol `
    --max-articles 80 `
    --articles-dir data/splits/eval/articles `
    --labels-path  data/splits/eval/eval-task2-TC.labels `
    --max-cost 2.0

# Cross-model: gpt-4o detects, sonnet verifies, hybrid strategy only, 20 articles
uv run python -m src.run_all `
    --model gpt-4o `
    --verify-model claude-sonnet-4-6 `
    --strategies hybrid `
    --max-articles 20 `
    --articles-dir data/splits/eval/articles `
    --labels-path  data/splits/eval/eval-task2-TC.labels `
    --max-cost 5.0
```

`run_all` flags:
- `--strategies` — comma-separated subset (e.g. `consol,hybrid`) or `all` (default). Valid: `zero_shot, few_shot, asv, consol, hybrid`.
- `--model` — main detection model.
- `--verify-model` — optional override for the Stage 2/3 (critique / consolidation / refinement) model.
- `--max-articles N` (or `--limit N`) — article count.
- `--max-cost` — per-strategy budget in USD.
- `--eval-mode {permissive,strict}`.

Approximate total cost for the full five-strategy run: **~$20 on gpt-4o**, **~$1.60 on gpt-4o-mini**. Results land in `outputs/results/comparison_{model}_{split}_{strategies}_n{N}_{eval_mode}.json`, and `run_all` **automatically generates the three comparison F1 charts** (overall, recall-vs-F1, per-technique) next to the JSON with the `_overall_f1.png`, `_recall_f1.png`, `_technique_f1.png` suffixes.

### Cross-family experiment

Runs the four-way cross-family comparison — same-family baselines and both cascade directions between gpt-4o and claude-sonnet-4-6. Model pairs are fixed by the script (this is the dissertation's exact cross-family design).

```bash
uv run python -m src.run_crossmodel `
    --strategy consol `
    --articles-dir data/splits/eval/articles `
    --labels-path  data/splits/eval/eval-task2-TC.labels `
    --max-articles 122 `
    --max-cost 20.0
```

The four configurations (hardcoded in `src/run_crossmodel.py`) are:

| Config | Detect | Verify |
|---|---|---|
| `same_4o` | gpt-4o | gpt-4o |
| `same_sonnet` | claude-sonnet-4-6 | claude-sonnet-4-6 |
| `cascade_4o_to_sonnet` | gpt-4o | claude-sonnet-4-6 |
| `cascade_sonnet_to_4o` | claude-sonnet-4-6 | gpt-4o |

Approximate total cost over all four configs on 122 articles: **~$30**. `run_crossmodel` also **automatically generates the three cross-family F1 charts** next to the output JSON (`_overall_f1.png`, `_recall_f1.png`, `_technique_f1.png`).

---

## Charts and analyses: what's auto-generated and what you run manually

Every `outputs/results/*.json` contains the article text, gold spans, every predicted span with its pipeline history, stage-by-stage snapshots, and per-stage costs. Most plots and every statistical analysis can be regenerated from these JSONs alone — no API calls, no keys, no cost.

### What the experiment runners emit automatically

| Runner | What it drops next to the JSON |
|---|---|
| `src.run_experiment` | Per-run diagnostic PNG: `outputs/diagnostics/{run_id}_stage_f1.png` (stage-by-stage SI/TC F1) |
| `src.run_all` | Comparison JSON + `*_overall_f1.png`, `*_recall_f1.png`, `*_technique_f1.png` in `outputs/results/` |
| `src.run_crossmodel` | Cross-family JSON + the same three F1 PNGs in `outputs/results/` |

So **the SI/TC/Macro + per-technique F1 bar charts for both comparison and cross-family runs are auto-plotted**. You only need the CLI below when you're regenerating them from an old / rescored / hand-edited JSON.

### Rebuilding F1 charts from a saved JSON

One CLI, two modes (`comparison` for `run_all` output, `crossmodel` for `run_crossmodel` output):

```bash
# Strategy comparison
uv run python -m src.evaluation.charts `
    --mode comparison `
    --comparison-json outputs/results/comparison_gpt-4o_eval_zero_shot-few_shot-asv-consol-hybrid_n122_permissive.json `
    --output-prefix   outputs/figures/comparison_gpt-4o

# Cross-family
uv run python -m src.evaluation.charts `
    --mode crossmodel `
    --comparison-json outputs/results/crossmodel_family_consol_permissive.json `
    --output-prefix   outputs/figures/crossmodel `
    --num-articles 122
```

Each invocation writes the three F1 PNGs under the given prefix.

### Full dissertation analysis (relabel matrix, error taxonomy, stage F1, per-pass F1, …)

This is the command for **everything beyond the headline F1 bars** — relabel quality matrix, per-pass (three detection passes) F1, per-technique stage delta, error classification, stage-by-stage F1 corrected, drop-quality, refiner-disobedience. It runs bootstrap CIs + Holm-corrected pairwise significance on top and writes a combined report:

```bash
# Everything (bootstrap items 1 & 2 are slow; 10k resamples x 5 strategies)
uv run python -m scripts.generate_analysis

# Skip the slow bootstrap items, just do items 3-8
uv run python -m scripts.generate_analysis --start-from 3
```

Output lands in `outputs/dissertation_analysis/`:

- `all_findings.txt` — textual summary of all findings
- `relabel_matrix_{strategy}.png` — original-label -> new-label matrix with net correctness deltas
- `fig_error_distribution.png` — six-category error breakdown
- `fig_per_pass_f1_{strategy}.png` — F1 per individual detection pass
- `fig_per_technique_delta_{strategy}.png` — per-technique gain/loss from consolidation

The script auto-discovers the most recent `*_gpt-4o_n*.json` for each of the five strategies under `outputs/results/`.

### TC confusion matrix (standalone)

Not part of `generate_analysis`; call it directly when you want a gold-vs-predicted technique matrix for a single run:

```bash
uv run python -c "from pathlib import Path; from src.evaluation.postrun import plot_tc_confusion; plot_tc_confusion(Path('outputs/results/consol_gpt-4o_n122_9212d422.json'), Path('outputs/figures/consol_confusion.png'))"
```

---

## Where outputs land

```
outputs/
├── results/       # Per-run JSON (metrics, predictions, gold, diagnostics, cost) + comparison JSONs
├── cache/         # API response cache (delete to force fresh calls)
├── logs/          # Run logs, one per experiment
├── diagnostics/   # Per-run stage-by-stage F1 / relabel charts (written by run_experiment)
└── figures/       # (Convention for manually regenerated charts via `python -m src.evaluation.charts`)
```

Each result JSON includes, for every article: the article text, gold spans, every predicted span with its pipeline history, stage-by-stage snapshots, and the per-stage cost breakdown. This is what the post-hoc analysis in `src/evaluation/postrun.py` and `scripts/generate_analysis.py` operate on.

---

## File tour

```
src/
├── schemas.py                 # Pydantic models: Technique enum, Article, Prediction, ExperimentConfig
├── parser.py                  # Parse LLM JSON into PredictedSpan; multi-pass union-merge
├── llm_client.py              # OpenAI / Anthropic async client, caching, cost tracking, retries
├── experiment.py              # Orchestrator for the 5 strategies + run_experiment()
├── run_experiment.py          # CLI: single strategy, single model
├── run_all.py                 # CLI: run multiple strategies in one invocation
├── run_crossmodel.py          # CLI: four-way cross-family (same/cascade) experiment
├── data/
│   └── loader.py              # SemEval data loader (article text + character-level spans)
├── evaluation/
│   ├── metrics.py             # Official SemEval SI / TC / macro F1
│   ├── diagnostics.py         # Per-stage F1, relabel matrix, drop quality, refiner disobedience
│   ├── charts.py              # F1 bar charts (auto-called by runners; also a CLI: `python -m src.evaluation.charts`)
│   └── postrun.py             # Post-hoc analysis: bootstrap CIs, Holm, per-pass F1, error taxonomy, TC confusion
└── prompts/
    ├── detection.py           # Zero-shot, few-shot, and Stage-1 multi-pass detection prompts
    ├── critique.py            # ASV Stage 2 adversarial critique
    ├── consolidation.py       # Consol Stage 2 constructive consolidation
    └── refinement.py          # Hybrid Stage 3 refinement (no-drop)

scripts/
├── split_data.py              # Stratified train/eval split
└── generate_analysis.py       # Full dissertation analysis (items 1-8) into all_findings.txt
```
