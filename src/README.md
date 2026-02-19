# Manipulation Detection in Political Advertising

BSc Artificial Intelligence Dissertation — King's College London

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Place data files
#    data/raw/semeval/train-articles/   ← .txt article files
#    data/raw/semeval/train-labels-TC.txt
#    data/raw/semeval/dev-articles/
#    data/raw/semeval/dev-labels-TC.txt
#    data/raw/creative/political_ads.json  ← your annotated ads

# 4. Run a single experiment
python run.py --strategy zero_shot --model gpt-4o-mini --dataset semeval_train --sample-size 50

# 5. Run the full benchmark (all 6 strategies)
python run.py --benchmark --model gpt-4o-mini --dataset semeval_train --sample-size 100

# 6. Compare two runs
python run.py --compare outputs/results/run_a outputs/results/run_b
```

## Project Structure

```
├── run.py                  # CLI entry point
├── requirements.txt
├── configs/
│   └── taxonomy.txt        # Full ASV taxonomy (optional, fallback built-in)
├── data/
│   ├── raw/
│   │   ├── semeval/        # SemEval-2020 Task 11 data
│   │   └── creative/       # Your annotated political ads
│   ├── processed/          # Auto-generated cached data
│   └── cache/              # LLM response cache (saves $$$)
├── src/
│   ├── schemas.py          # Pydantic data models, taxonomy
│   ├── data.py             # Dataset loading (SemEval + CREATIVE)
│   ├── llm.py              # OpenAI client with cache + retry
│   ├── prompts.py          # All 6 prompt strategies
│   ├── parser.py           # LLM response → Prediction parsing
│   ├── runner.py           # Experiment orchestrator
│   ├── metrics.py          # Evaluation: F1, bootstrap CI, McNemar's
│   └── export.py           # LaTeX tables, markdown reports
├── outputs/
│   └── results/            # Each run gets its own directory
│       └── {run_id}/
│           ├── config.json
│           ├── metrics.json
│           ├── tactic_scores.json
│           ├── predictions.json
│           ├── cost.json
│           ├── error_analysis.json
│           ├── fp_details.json
│           └── fn_details.json
└── tests/
```

## 6 Prompting Strategies

| # | Strategy | Stages | Description |
|---|----------|--------|-------------|
| 1 | `zero_shot` | 1 | Direct classification with taxonomy |
| 2 | `few_shot` | 1 | 3 exemplars (positive, negative, mixed) |
| 3 | `cot` | 1 | Chain-of-thought step-by-step reasoning |
| 4 | `cot_sc` | 1×3 | CoT with self-consistency (majority vote over 3 runs) |
| 5 | `hierarchical` | 2 | Category detection → specific tactic identification |
| 6 | `asv` | 3 | **Novel** Adversarial Self-Verification (Prosecution → Defense → Verdict) |

## Key Metrics

- **Macro F1** (primary): Treats all 14 tactics equally — critical for rare techniques
- **Micro F1**: Weighted by instance count
- **Per-tactic F1**: Individual technique performance
- **Explicit vs Implicit F1**: Aggregate performance on surface-level vs pragmatic techniques
- **Bootstrap 95% CI**: Confidence interval for macro F1
- **McNemar's test**: Pairwise statistical comparison between strategies
- **Hamming loss**: Proportion of incorrect individual labels

## Caching

LLM responses are cached by default in `data/cache/`. Cache key = hash(model + temperature + messages). This means:
- Re-running the same experiment costs $0
- Changing the prompt invalidates the cache for that sample
- Use `--no-cache` to force fresh API calls

## Cost Tracking

Every run tracks tokens and estimated cost. At GPT-4o-mini rates:
- Single-stage strategies: ~$0.001/sample → 500 samples ≈ $0.50
- ASV (3 stages): ~$0.003/sample → 500 samples ≈ $1.50
- Full benchmark (6 strategies, 500 samples): ~$5-8

## Generating LaTeX Tables

```python
from src.export import metrics_to_latex, generate_report
import json

# Load benchmark results
results = json.loads(open("outputs/results/benchmark_comparison.json").read())

# Generate LaTeX
latex = metrics_to_latex(results)
print(latex)

# Generate markdown report for a single run
report = generate_report("outputs/results/your_run_id")
print(report)
```
