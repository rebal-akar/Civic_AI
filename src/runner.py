"""
Experiment runner — orchestrates data loading, inference, parsing, and evaluation.

Usage:
    from src.runner import run_experiment
    result = await run_experiment(config)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm.asyncio import tqdm as atqdm

# Load .env from project root (Civic_AI/) so CHATGPT_API_KEY / OPENAI_API_KEY are set
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from src.data import load_dataset
from src.llm import LLMClient
from src.metrics import (
    MetricsReport,
    bootstrap_confidence_interval,
    compute_metrics,
    analyse_errors,
    ErrorAnalysis,
)
from src.parser import parse_asv_verdict, parse_response
from src.prompts import (
    STRATEGIES,
    format_asv_defense,
    format_asv_prosecution,
    format_asv_verdict,
    format_cot,
    format_hierarchical_stage1,
    format_hierarchical_stage2,
)
from src.schemas import (
    ALL_TACTICS,
    CostSummary,
    ExperimentConfig,
    ExperimentResult,
    LLMResponse,
    Prediction,
    Sample,
    _fuzzy_match_tactic,
)

logger = logging.getLogger(__name__)


def _article_id(sample: Sample) -> str:
    """Extract article id from sample (e.g. article111111111_p0 -> article111111111)."""
    aid = (sample.metadata or {}).get("article_id")
    if aid:
        return aid
    if "_p" in sample.id:
        return sample.id.rsplit("_p", 1)[0]
    return sample.id


def _sample_by_articles(samples: list[Sample], num_articles: int, seed: int) -> list[Sample]:
    """Keep only samples from num_articles randomly chosen articles."""
    import numpy as np
    from collections import defaultdict
    by_article: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_article[_article_id(s)].append(s)
    article_ids = sorted(by_article.keys())
    rng = np.random.default_rng(seed)
    n = min(num_articles, len(article_ids))
    chosen = rng.choice(article_ids, size=n, replace=False)
    out: list[Sample] = []
    for aid in sorted(chosen):
        out.extend(by_article[aid])
    return out


# ── Main Entry Point ─────────────────────────────────────────────────────────

async def run_experiment(
    config: ExperimentConfig,
    data_root: str | Path = "data/raw",
    output_dir: str | Path = "outputs/results",
) -> ExperimentResult:
    """
    Run a complete experiment: load data → infer → parse → evaluate → save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = config.run_id()
    logger.info(f"Starting experiment: {run_id}")
    logger.info(f"  Strategy: {config.strategy}, Model: {config.model}")

    # ── 1. Load data ─────────────────────────────────────────────────────
    samples = load_dataset(config.dataset, data_root)

    if getattr(config, "num_articles", None) and config.num_articles > 0:
        import numpy as np
        samples = _sample_by_articles(samples, config.num_articles, config.seed)
        logger.info(f"  Selected {config.num_articles} articles -> {len(samples)} samples")
    elif config.sample_size and config.sample_size < len(samples):
        import numpy as np
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(len(samples), size=config.sample_size, replace=False)
        samples = [samples[i] for i in sorted(indices)]
        logger.info(f"  Sampled {config.sample_size} from {len(samples)} total")

    logger.info(f"  Dataset: {config.dataset}, Samples: {len(samples)}")

    # ── 2. Initialise LLM client ─────────────────────────────────────────
    # Re-load .env from project root in case run.py was invoked from another cwd
    _runner_root = Path(__file__).resolve().parent.parent
    load_dotenv(_runner_root / ".env")
    api_key = (os.environ.get("CHATGPT_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("No CHATGPT_API_KEY or OPENAI_API_KEY in env — set one in Civic_AI/.env")
        raise ValueError(
            "OpenAI API key required. Add to Civic_AI/.env: OPENAI_API_KEY=sk-... or CHATGPT_API_KEY=sk-..."
        )
    client = LLMClient(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        use_cache=config.use_cache,
        api_key=api_key,
    )

    # ── 3. Run inference ─────────────────────────────────────────────────
    started_at = datetime.now()
    predictions: list[Prediction] = []

    if config.strategy == "asv":
        predictions = await _run_asv(client, samples, config)
    elif config.strategy == "hierarchical":
        predictions = await _run_hierarchical(client, samples, config)
    elif config.strategy == "cot_sc":
        predictions = await _run_cot_sc(client, samples, config, n_samples=3)
    else:
        predictions = await _run_single_stage(client, samples, config)

    finished_at = datetime.now()

    # ── 3b. ASV trace: write prosecutor/defense/verdict tables to file (not terminal) ─
    if config.strategy == "asv" and getattr(config, "debug_asv", False):
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_asv_trace(samples, predictions, run_dir / "asv_trace.txt")

    # ── 4. Compute metrics (only on final predicted labels, e.g. ASV verdict output) ─
    metrics_report = compute_metrics(samples, predictions)
    logger.info(f"  Macro F1: {metrics_report.macro_f1:.4f}")
    logger.info(f"  Micro F1: {metrics_report.micro_f1:.4f}")

    # ASV stage metrics: prosecutor (high recall), defense (filter FPs), verdict (final)
    asv_stage_metrics: dict[str, Any] | None = None
    if config.strategy == "asv":
        prosecutor_preds, defense_preds = _asv_stage_predictions(samples, predictions)
        prosecutor_metrics = compute_metrics(samples, prosecutor_preds)
        logger.info(f"  Prosecutor: F1={prosecutor_metrics.macro_f1:.4f} P={prosecutor_metrics.macro_precision:.4f} R={prosecutor_metrics.macro_recall:.4f}")
        defense_metrics: MetricsReport | None = None
        if defense_preds is not None:
            defense_metrics = compute_metrics(samples, defense_preds)
            logger.info(f"  Defense:    F1={defense_metrics.macro_f1:.4f} P={defense_metrics.macro_precision:.4f} R={defense_metrics.macro_recall:.4f}")
        logger.info(f"  Verdict:    F1={metrics_report.macro_f1:.4f} P={metrics_report.macro_precision:.4f} R={metrics_report.macro_recall:.4f}")
        asv_stage_metrics = {
            "prosecutor": {
                "macro_f1": prosecutor_metrics.macro_f1,
                "macro_precision": prosecutor_metrics.macro_precision,
                "macro_recall": prosecutor_metrics.macro_recall,
                "micro_f1": prosecutor_metrics.micro_f1,
                "per_tactic": prosecutor_metrics.tactic_scores,
            },
            "defense": {
                "macro_f1": defense_metrics.macro_f1,
                "macro_precision": defense_metrics.macro_precision,
                "macro_recall": defense_metrics.macro_recall,
                "micro_f1": defense_metrics.micro_f1,
                "per_tactic": defense_metrics.tactic_scores,
            } if defense_metrics else None,
            "verdict": {
                "macro_f1": metrics_report.macro_f1,
                "macro_precision": metrics_report.macro_precision,
                "macro_recall": metrics_report.macro_recall,
                "micro_f1": metrics_report.micro_f1,
                "per_tactic": metrics_report.tactic_scores,
            },
        }

    # Bootstrap CI
    ci = bootstrap_confidence_interval(samples, predictions, seed=config.seed)
    metrics_report.macro_f1_ci = ci
    logger.info(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Error analysis
    error_analysis = analyse_errors(samples, predictions)

    # ── 5. Build result ──────────────────────────────────────────────────
    result = ExperimentResult(
        config=config,
        run_id=run_id,
        predictions=predictions,
        metrics=metrics_report.summary_dict(),
        cost=client.cost,
        started_at=started_at,
        finished_at=finished_at,
    )

    # ── 6. Save outputs ─────────────────────────────────────────────────
    _save_results(result, metrics_report, error_analysis, output_dir, run_id, samples, asv_stage_metrics=asv_stage_metrics)

    logger.info(f"  Cost: ${client.cost.total_cost_usd:.4f} ({client.cost.num_api_calls} calls)")
    logger.info(f"  Duration: {result.duration_seconds():.1f}s")
    logger.info(f"  Results saved to: {output_dir / run_id}")

    return result


# ── Strategy Runners ─────────────────────────────────────────────────────────

async def _run_single_stage(
    client: LLMClient,
    samples: list[Sample],
    config: ExperimentConfig,
) -> list[Prediction]:
    """Run zero-shot, few-shot, or CoT (single API call per sample)."""
    format_fn = STRATEGIES[config.strategy]["format"]
    predictions: list[Prediction] = []
    sem = asyncio.Semaphore(config.batch_size)

    async def process_one(sample: Sample) -> Prediction:
        async with sem:
            messages = format_fn(sample)
            response = await client.complete(messages)
            return parse_response(response.content, sample.id, config.strategy)

    tasks = [process_one(s) for s in samples]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=config.strategy):
        pred = await coro
        predictions.append(pred)

    # Re-sort by sample order
    pred_map = {p.sample_id: p for p in predictions}
    return [pred_map.get(s.id, Prediction(sample_id=s.id, parse_success=False)) for s in samples]


def _get_asv_formatters(config: ExperimentConfig):
    """Return (format_prosecution, format_defense, format_verdict) from asv or asvS."""
    if getattr(config, "prompts", "asv") == "asvS":
        from src.prompts.asvS import (
            format_asv_defense as fmt_defense,
            format_asv_prosecution as fmt_prosecution,
            format_asv_verdict as fmt_verdict,
        )
        return fmt_prosecution, fmt_defense, fmt_verdict
    return format_asv_prosecution, format_asv_defense, format_asv_verdict


async def _run_asv(
    client: LLMClient,
    samples: list[Sample],
    config: ExperimentConfig,
) -> list[Prediction]:
    """Run 3-stage ASV pipeline (uses asv or asvS prompts per config.prompts)."""
    fmt_prosecution, fmt_defense, fmt_verdict = _get_asv_formatters(config)
    predictions: list[Prediction] = []
    sem = asyncio.Semaphore(config.batch_size)

    async def process_one(sample: Sample) -> Prediction:
        async with sem:
            try:
                # Stage 1: Prosecution
                pros_msgs = fmt_prosecution(sample)
                pros_resp = await client.complete(pros_msgs)
                pros_data = _safe_json_parse(pros_resp.content)

                if pros_data is None:
                    return Prediction(
                        sample_id=sample.id, parse_success=False,
                        error_message="Prosecution stage JSON parse failed",
                        raw_response=pros_resp.content,
                    )

                # Stage 2: Defense
                def_msgs = fmt_defense(sample, pros_data)
                def_resp = await client.complete(def_msgs)
                def_data = _safe_json_parse(def_resp.content)

                if def_data is None:
                    # Fall back to prosecution-only
                    return parse_response(pros_resp.content, sample.id, "asv")

                # Stage 3: Verdict
                verd_msgs = fmt_verdict(sample, pros_data, def_data)
                verd_resp = await client.complete(verd_msgs)

                pred = parse_asv_verdict(verd_resp.content, sample.id)
                pred.stage_outputs = {
                    "prosecution": pros_data,
                    "defense": def_data,
                    "verdict_raw": verd_resp.content,
                }
                return pred

            except Exception as e:
                logger.error(f"ASV failed for {sample.id}: {e}")
                return Prediction(
                    sample_id=sample.id, parse_success=False,
                    error_message=str(e),
                )

    tasks = [process_one(s) for s in samples]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="asv"):
        pred = await coro
        predictions.append(pred)

    pred_map = {p.sample_id: p for p in predictions}
    return [pred_map.get(s.id, Prediction(sample_id=s.id, parse_success=False)) for s in samples]


async def _run_hierarchical(
    client: LLMClient,
    samples: list[Sample],
    config: ExperimentConfig,
) -> list[Prediction]:
    """Run 2-stage hierarchical pipeline."""
    predictions: list[Prediction] = []
    sem = asyncio.Semaphore(config.batch_size)

    async def process_one(sample: Sample) -> Prediction:
        async with sem:
            try:
                # Stage 1: Category detection
                s1_msgs = format_hierarchical_stage1(sample)
                s1_resp = await client.complete(s1_msgs)
                s1_data = _safe_json_parse(s1_resp.content)

                categories = s1_data.get("categories", []) if s1_data else []

                if not categories:
                    return Prediction(sample_id=sample.id, tactics=[], parse_success=True)

                # Stage 2: Specific tactics
                s2_msgs = format_hierarchical_stage2(sample, categories)
                s2_resp = await client.complete(s2_msgs)
                return parse_response(s2_resp.content, sample.id, "hierarchical")

            except Exception as e:
                logger.error(f"Hierarchical failed for {sample.id}: {e}")
                return Prediction(sample_id=sample.id, parse_success=False, error_message=str(e))

    tasks = [process_one(s) for s in samples]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="hierarchical"):
        pred = await coro
        predictions.append(pred)

    pred_map = {p.sample_id: p for p in predictions}
    return [pred_map.get(s.id, Prediction(sample_id=s.id, parse_success=False)) for s in samples]


async def _run_cot_sc(
    client: LLMClient,
    samples: list[Sample],
    config: ExperimentConfig,
    n_samples: int = 3,
) -> list[Prediction]:
    """Run CoT with self-consistency (majority vote over N runs at temp > 0)."""
    # Temporarily override temperature
    original_temp = client.temperature
    client.temperature = 0.7
    client.use_cache = False  # Need different responses each time

    predictions: list[Prediction] = []
    sem = asyncio.Semaphore(config.batch_size)

    async def process_one(sample: Sample) -> Prediction:
        async with sem:
            all_tactics: list[list[str]] = []
            for _ in range(n_samples):
                messages = format_cot(sample)
                response = await client.complete(messages)
                pred = parse_response(response.content, sample.id, "cot_sc")
                all_tactics.append(pred.tactics)

            # Majority vote: include tactic if it appears in > half the runs
            counter: Counter = Counter()
            for tactics in all_tactics:
                for t in tactics:
                    counter[t] += 1

            threshold = n_samples / 2
            final_tactics = sorted([t for t, c in counter.items() if c > threshold])

            return Prediction(
                sample_id=sample.id,
                tactics=final_tactics,
                parse_success=True,
                stage_outputs={"individual_runs": all_tactics, "vote_counts": dict(counter)},
            )

    tasks = [process_one(s) for s in samples]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="cot_sc"):
        pred = await coro
        predictions.append(pred)

    # Restore settings
    client.temperature = original_temp
    client.use_cache = True

    pred_map = {p.sample_id: p for p in predictions}
    return [pred_map.get(s.id, Prediction(sample_id=s.id, parse_success=False)) for s in samples]


# ── ASV Debug Trace ──────────────────────────────────────────────────────────

def _truncate(s: str, max_len: int = 300, suffix: str = "...") -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)].rsplit(maxsplit=1)[0] + suffix


def _render_asv_trace(
    console: Console,
    samples: list[Sample],
    predictions: list[Prediction],
) -> None:
    """Render prosecutor, defense, and verdict stage I/O as tables to the given console (terminal or file)."""
    pred_map = {p.sample_id: p for p in predictions}
    for sample in samples:
        pred = pred_map.get(sample.id)
        console.print()
        console.rule(f"[bold blue]Sample: {sample.id}[/]", style="blue")
        console.print("[dim]Input text (truncated):[/]")
        console.print(_truncate(sample.text, 400))
        console.print()
        if not pred or not pred.stage_outputs:
            console.print("[yellow]No stage outputs (parse failed or not ASV).[/]")
            continue
        pros = pred.stage_outputs.get("prosecution") or {}
        defe = pred.stage_outputs.get("defense") or {}
        # Prosecution table (asv: prosecution_findings; asvS: detections)
        findings = pros.get("prosecution_findings") or pros.get("detections") or []
        if findings:
            t = Table(title="[bold]1. Prosecutor[/]", show_header=True, header_style="magenta")
            t.add_column("Tactic", style="cyan")
            t.add_column("Evidence", max_width=50, overflow="fold")
            t.add_column("Mechanism", max_width=45, overflow="fold")
            t.add_column("Confidence")
            for f in findings:
                # asv: mechanism, confidence; asvS: reasoning only
                mech = f.get("mechanism") or f.get("reasoning", "")
                t.add_row(
                    str(f.get("tactic", "")),
                    str(f.get("evidence", ""))[:80],
                    str(mech)[:80],
                    str(f.get("confidence", "")),
                )
            console.print(t)
            console.print(f"[dim]Summary: {pros.get('summary', 'N/A')}[/]")
        else:
            console.print("[dim]Prosecutor: no findings.[/]")
        console.print()
        # Defense table (asv: defense_arguments; asvS: reviews)
        args_list = defe.get("defense_arguments") or defe.get("reviews") or []
        if args_list:
            t = Table(title="[bold]2. Defense[/]", show_header=True, header_style="green")
            # asv has responding_to, counter_argument, heuristic_applied, defense_strength; asvS has id, verdict, reasoning
            if args_list and "verdict" in args_list[0]:
                t.add_column("Id", style="cyan")
                t.add_column("Verdict")
                t.add_column("Reasoning", max_width=60, overflow="fold")
                for a in args_list:
                    t.add_row(
                        str(a.get("id", "")),
                        str(a.get("verdict", "")),
                        str(a.get("reasoning", ""))[:100],
                    )
            else:
                t.add_column("Responding to", style="cyan")
                t.add_column("Counter-argument", max_width=50, overflow="fold")
                t.add_column("Heuristic")
                t.add_column("Strength")
                for a in args_list:
                    t.add_row(
                        str(a.get("responding_to", "")),
                        str(a.get("counter_argument", ""))[:80],
                        str(a.get("heuristic_applied", "")),
                        str(a.get("defense_strength", "")),
                    )
            console.print(t)
            console.print(f"[dim]Summary: {defe.get('summary', 'N/A')}[/]")
        else:
            console.print("[dim]Defense: no arguments.[/]")
        console.print()
        # Verdict: parse verdict_raw if it's stored as string, or we might have verdict in stage_outputs
        verdict_raw = pred.stage_outputs.get("verdict_raw")
        verdict_data: dict | None = None
        if isinstance(verdict_raw, str):
            verdict_data = _safe_json_parse(verdict_raw)
        elif isinstance(verdict_raw, dict):
            verdict_data = verdict_raw
        if verdict_data:
            verdicts_list = verdict_data.get("verdicts") or []
            final_tactics_key = verdict_data.get("tactics") or verdict_data.get("techniques") or pred.tactics
            if verdicts_list:
                t = Table(title="[bold]3. Verdict[/]", show_header=True, header_style="yellow")
                t.add_column("Tactic", style="cyan")
                t.add_column("Verdict")
                t.add_column("Rationale", max_width=55, overflow="fold")
                for v in verdicts_list:
                    t.add_row(
                        str(v.get("tactic", "")),
                        str(v.get("verdict", "")),
                        str(v.get("rationale", ""))[:100],
                    )
                console.print(t)
            console.print(f"[bold]Final tactics:[/] {final_tactics_key}")
            console.print(f"[dim]Reasoning: {verdict_data.get('reasoning', 'N/A')}[/]")
        else:
            console.print(f"[bold]Final tactics:[/] {pred.tactics}")
    console.print()


def _print_asv_trace(samples: list[Sample], predictions: list[Prediction]) -> None:
    """Print ASV trace tables to terminal."""
    _render_asv_trace(Console(), samples, predictions)


def _write_asv_trace(
    samples: list[Sample],
    predictions: list[Prediction],
    path: Path,
) -> None:
    """Write ASV trace tables to a file (same content as terminal, for review)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        console = Console(file=f, force_terminal=False)
        _render_asv_trace(console, samples, predictions)
    logger.info(f"  ASV trace written to {path}")


# ── ASV stage metrics (prosecutor / defense / verdict) ────────────────────────

def _normalise_tactic(t: str) -> str | None:
    """Return snake_case tactic in ALL_TACTICS or None."""
    out = _fuzzy_match_tactic(t) or (t if t in ALL_TACTICS else None)
    return out


def _asv_prosecutor_tactics(pred: Prediction) -> list[str]:
    """Extract set of tactics from prosecution stage (asv or asvS). High recall expected."""
    if not pred.stage_outputs:
        return []
    pros = pred.stage_outputs.get("prosecution") or {}
    # asvS: detections; asv: prosecution_findings
    findings = pros.get("detections") or pros.get("prosecution_findings") or []
    tactics: list[str] = []
    for f in findings:
        t = (f.get("tactic") or f.get("technique") or "").strip()
        if not t:
            continue
        norm = _normalise_tactic(t)
        if norm and norm not in tactics:
            tactics.append(norm)
    return sorted(tactics)


def _asv_defense_tactics(pred: Prediction) -> list[str] | None:
    """
    Extract tactics that defense UPHOLD (asvS: id→verdict; asv has no per-claim verdict → None).
    More conservative than prosecutor; filters false positives.
    """
    if not pred.stage_outputs:
        return None
    pros = pred.stage_outputs.get("prosecution") or {}
    defe = pred.stage_outputs.get("defense") or {}
    findings = pros.get("detections") or pros.get("prosecution_findings") or []
    # asvS: reviews with id, verdict (UPHOLD/WEAKEN/REJECT)
    reviews = defe.get("reviews") or []
    if not reviews or "verdict" not in (reviews[0] if reviews else {}):
        return None  # asv format or no reviews
    upheld_ids = {r.get("id") for r in reviews if str(r.get("verdict", "")).upper() == "UPHOLD"}
    tactics: list[str] = []
    for f in findings:
        cid = f.get("id")
        if cid not in upheld_ids:
            continue
        t = (f.get("tactic") or f.get("technique") or "").strip()
        if not t:
            continue
        norm = _normalise_tactic(t)
        if norm and norm not in tactics:
            tactics.append(norm)
    return sorted(tactics)


def _asv_stage_predictions(
    samples: list[Sample],
    predictions: list[Prediction],
) -> tuple[list[Prediction], list[Prediction] | None]:
    """
    Build prosecutor and defense prediction lists (same order as samples).
    Defense is None if format doesn't support per-claim verdict (e.g. asv).
    """
    pred_map = {p.sample_id: p for p in predictions}
    prosecutor_preds: list[Prediction] = []
    defense_preds: list[Prediction] = []
    defense_ok = True
    for s in samples:
        pred = pred_map.get(s.id)
        if not pred:
            prosecutor_preds.append(Prediction(sample_id=s.id, tactics=[], parse_success=False))
            defense_preds.append(Prediction(sample_id=s.id, tactics=[], parse_success=False))
            continue
        pt = _asv_prosecutor_tactics(pred)
        prosecutor_preds.append(Prediction(sample_id=s.id, tactics=pt, parse_success=pred.parse_success))
        dt = _asv_defense_tactics(pred)
        if dt is None:
            defense_ok = False
            defense_preds.append(Prediction(sample_id=s.id, tactics=[], parse_success=False))
        else:
            defense_preds.append(Prediction(sample_id=s.id, tactics=dt, parse_success=pred.parse_success))
    return prosecutor_preds, defense_preds if defense_ok else None


# ── Output Saving ────────────────────────────────────────────────────────────

def _save_results(
    result: ExperimentResult,
    metrics: MetricsReport,
    errors: ErrorAnalysis,
    output_dir: Path,
    run_id: str,
    samples: list[Sample] | None = None,
    asv_stage_metrics: dict[str, Any] | None = None,
) -> None:
    """Save all experiment outputs to disk."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = samples or []

    # Config
    (run_dir / "config.json").write_text(result.config.model_dump_json(indent=2))

    # Metrics: one file with both summary and per-tactic F1/precision/recall
    full_metrics = {
        **metrics.summary_dict(),
        "per_tactic": metrics.tactic_scores,
    }
    (run_dir / "metrics.json").write_text(json.dumps(full_metrics, indent=2))

    # ASV stage metrics (prosecutor / defense / verdict) when strategy is asv
    if asv_stage_metrics:
        (run_dir / "asv_stage_metrics.json").write_text(
            json.dumps(asv_stage_metrics, indent=2, default=str)
        )

    # Per-tactic scores (standalone for scripts)
    (run_dir / "tactic_scores.json").write_text(json.dumps(metrics.tactic_scores, indent=2))

    # Predictions (full detail)
    preds_data = [p.model_dump() for p in result.predictions]
    (run_dir / "predictions.json").write_text(json.dumps(preds_data, indent=2, default=str))

    # Cost
    (run_dir / "cost.json").write_text(result.cost.model_dump_json(indent=2))

    # Error analysis summary
    error_summary = {
        "total_fp": errors.total_fp,
        "total_fn": errors.total_fn,
        "total_correct": errors.total_correct,
        "fp_by_tactic": {k: v.count for k, v in errors.false_positives.items()},
        "fn_by_tactic": {k: v.count for k, v in errors.false_negatives.items()},
        "tactic_confusions": {
            k: dict(v) for k, v in errors.tactic_confusions.items()
        },
    }
    (run_dir / "error_analysis.json").write_text(json.dumps(error_summary, indent=2))

    # FP/FN detail for qualitative analysis
    fp_details = {k: v.details[:20] for k, v in errors.false_positives.items()}
    fn_details = {k: v.details[:20] for k, v in errors.false_negatives.items()}
    (run_dir / "fp_details.json").write_text(json.dumps(fp_details, indent=2))
    (run_dir / "fn_details.json").write_text(json.dumps(fn_details, indent=2))

    # Human-readable report (open this file to view results)
    summary = metrics.summary_dict()
    report_lines = [
        "=" * 60,
        f"EXPERIMENT: {run_id}",
        "=" * 60,
        "",
        "All metrics (F1, precision, recall) are computed only on final predicted labels.",
        "(e.g. ASV: verdict output only; prosecution/defense are not used for scoring.)",
        "",
        f"Strategy:    {result.config.strategy}",
        f"Model:       {result.config.model}",
        f"Samples:     {summary.get('n_samples', len(result.predictions))}",
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Macro F1:    {summary.get('macro_f1', 0):.4f}",
        f"Micro F1:    {summary.get('micro_f1', 0):.4f}",
        f"Precision:   {summary.get('macro_precision', 0):.4f}",
        f"Recall:      {summary.get('macro_recall', 0):.4f}",
        f"Exact match: {summary.get('exact_match_ratio', 0):.4f}",
        "",
    ]
    if asv_stage_metrics:
        report_lines.extend([
            "ASV STAGE METRICS (Prosecutor → Defense → Verdict)",
            "-" * 40,
            "  Prosecutor (expect high recall):",
            f"    Macro F1: {asv_stage_metrics['prosecutor']['macro_f1']:.4f}  "
            f"Precision: {asv_stage_metrics['prosecutor']['macro_precision']:.4f}  "
            f"Recall: {asv_stage_metrics['prosecutor']['macro_recall']:.4f}",
            "  Defense (filters FPs, more conservative):",
        ])
        if asv_stage_metrics.get("defense"):
            report_lines.append(
                f"    Macro F1: {asv_stage_metrics['defense']['macro_f1']:.4f}  "
                f"Precision: {asv_stage_metrics['defense']['macro_precision']:.4f}  "
                f"Recall: {asv_stage_metrics['defense']['macro_recall']:.4f}"
            )
        else:
            report_lines.append("    (not available — defense format has no per-claim verdict)")
        report_lines.extend([
            "  Verdict (final output):",
            f"    Macro F1: {asv_stage_metrics['verdict']['macro_f1']:.4f}  "
            f"Precision: {asv_stage_metrics['verdict']['macro_precision']:.4f}  "
            f"Recall: {asv_stage_metrics['verdict']['macro_recall']:.4f}",
            "",
        ])
    report_lines.extend([
        "PER-TACTIC F1 / PRECISION / RECALL",
        "-" * 40,
    ])
    # Table header (all 14 tactics; missing ones get 0)
    report_lines.append(f"{'Tactic':<32} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Support':>8}")
    report_lines.append("-" * 66)
    for tactic in sorted(ALL_TACTICS):
        scores = metrics.tactic_scores.get(tactic, {})
        f1 = scores.get("f1", 0.0)
        prec = scores.get("precision", 0.0)
        rec = scores.get("recall", 0.0)
        sup = scores.get("support", 0)
        report_lines.append(f"{tactic:<32} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {sup:>8}")
    report_lines.extend([
        "",
        f"Cost:        ${result.cost.total_cost_usd:.4f} ({result.cost.num_api_calls} calls)",
        f"Duration:    {result.duration_seconds():.1f}s",
        "",
        "FILES IN THIS FOLDER",
        "-" * 40,
        "  config.json        - experiment config",
        "  metrics.json       - full metrics (includes per_tactic)",
        "  tactic_scores.json - per-tactic F1/precision/recall (JSON)",
        "  report.txt         - this file",
        "  cases.txt         - case-by-case review (ground truth + final predicted labels)",
        "  asv_stage_metrics.json - prosecutor/defense/verdict F1,P,R (when strategy=asv)",
        "  asv_trace.txt     - ASV prosecutor/defense/verdict tables (when --debug-asv)",
        "  predictions.json   - per-sample predictions (full JSON)",
        "  cost.json          - API cost",
        "  error_analysis.json, fp_details.json, fn_details.json",
        "",
    ])
    (run_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # Case-by-case review file: ground truth + final predicted labels only, easy to compare
    pred_map = {p.sample_id: p for p in result.predictions}
    case_lines = [
        "CASE-BY-CASE REVIEW",
        "Ground truth = labels for this paragraph. Metrics (F1, P, R) use final predicted labels only.",
        "=" * 72,
        "",
    ]
    for i, sample in enumerate(samples):
        pred = pred_map.get(sample.id)
        gold = sorted(set(sample.labels))
        pred_tactics = sorted(set(pred.tactics)) if pred else []
        missed = [t for t in gold if t not in pred_tactics]
        extra = [t for t in pred_tactics if t not in gold]
        match = "YES" if set(gold) == set(pred_tactics) else "NO"
        para_text = sample.text.strip().replace("\r\n", "\n").replace("\n", " ")
        if len(para_text) > 600:
            para_text = para_text[:597] + "..."
        sep = "-" * 72
        case_lines.extend([
            "",
            f"  CASE {i + 1}  │  {sample.id}",
            sep,
            "",
            "  GROUND TRUTH (this paragraph)",
            "  " + " ".join(gold) if gold else "  (none)",
            "",
            "  FINAL MODEL OUTPUT (predicted labels)",
            "  " + " ".join(pred_tactics) if pred_tactics else "  (none)",
            "",
            "  COMPARISON",
            "  " + sep,
            f"  Match (exact)?   {match}",
            f"  Missed (FN):     {', '.join(missed) if missed else '(none)'}",
            f"  Extra (FP):      {', '.join(extra) if extra else '(none)'}",
            "  " + sep,
            "",
            "  PARAGRAPH TEXT",
            "  " + sep,
        ])
        for line in textwrap.wrap(para_text, width=68):
            case_lines.append("  " + line)
        case_lines.extend(["", sep, ""])
    (run_dir / "cases.txt").write_text("\n".join(case_lines), encoding="utf-8")

    logger.info(f"  Saved results to {run_dir}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json_parse(raw: str) -> dict | None:
    """Parse JSON with fallback for code fences."""
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1:
        try:
            return json.loads(raw[first:last + 1])
        except json.JSONDecodeError:
            pass
    return None
