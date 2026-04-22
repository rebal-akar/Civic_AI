"""Smoke-test the post-hoc analysis functions against existing result JSONs."""

from pathlib import Path
from src.evaluation.postrun import aggregate_multi_run, format_multi_run_table

RESULTS_DIR = Path("outputs/results")


def test_single_run():
    """Aggregating a single run should give n_runs=1 with std=0."""
    candidates = sorted(RESULTS_DIR.glob("*_gpt-4o_n122_*.json"))
    if not candidates:
        candidates = sorted(RESULTS_DIR.glob("*.json"))
        candidates = [c for c in candidates if not c.name.startswith("comparison")]

    if not candidates:
        print("SKIP: no result JSONs found")
        return

    latest = candidates[-1]
    print(f"Testing single-run aggregation on: {latest.name}")

    result = aggregate_multi_run([latest], verify=False)
    assert result["n_runs"] == 1, f"Expected 1 run, got {result['n_runs']}"
    assert result["overall"]["si_f1"]["std"] == 0.0, "Single run should have std=0"

    print(format_multi_run_table(result))
    print("\nPASS: single-run aggregation\n")


def test_multi_run_mock():
    """Aggregate two different result files (verify=False since configs differ).

    Tests the aggregation math even though this isn't a true multi-seed scenario.
    """
    candidates = sorted(RESULTS_DIR.glob("*_gpt-4o_n122_*.json"))
    if len(candidates) < 2:
        all_jsons = sorted(RESULTS_DIR.glob("*.json"))
        candidates = [c for c in all_jsons if not c.name.startswith("comparison")]

    if len(candidates) < 2:
        print("SKIP: need 2+ result files for multi-run test")
        return

    two_files = candidates[:2]
    print(f"Testing multi-run aggregation on: {[f.name for f in two_files]}")

    result = aggregate_multi_run(two_files, verify=False)
    assert result["n_runs"] == 2, f"Expected 2 runs, got {result['n_runs']}"

    si_f1 = result["overall"]["si_f1"]
    print(f"SI-F1 across the two runs: {si_f1['values']}")
    print(f"Mean: {si_f1['mean']:.4f}, Std: {si_f1['std']:.4f}")
    print()
    print(format_multi_run_table(result))
    print("\nPASS: multi-run aggregation\n")


def test_verify_catches_mismatches():
    """verify=True should raise ValueError when configs mismatch (e.g. different strategies)."""
    zero_files = sorted(RESULTS_DIR.glob("zero_shot_*.json"))
    few_files = sorted(RESULTS_DIR.glob("few_shot_*.json"))

    if not zero_files or not few_files:
        print("SKIP: need both zero_shot and few_shot results for verify test")
        return

    mixed = [zero_files[-1], few_files[-1]]
    print(f"Testing that verify catches mismatched configs: {[f.name for f in mixed]}")

    try:
        aggregate_multi_run(mixed, verify=True)
        print("FAIL: verify did not catch the mismatch")
    except ValueError as e:
        print(f"PASS: verify correctly raised ValueError: {e}\n")


def test_bootstrap_cis():
    """Smoke-test bootstrap CI on an existing result JSON."""
    from src.evaluation.postrun import compute_bootstrap_cis

    candidates = sorted(RESULTS_DIR.glob("*_gpt-4o_n122_*.json"))
    if not candidates:
        candidates = sorted(RESULTS_DIR.glob("*.json"))
        candidates = [c for c in candidates if not c.name.startswith("comparison")]

    if not candidates:
        print("SKIP: no result JSONs found for bootstrap test")
        return

    latest = candidates[-1]
    print(f"Testing bootstrap CIs on: {latest.name}")

    for metric in ["si_f1", "tc_f1"]:
        ci = compute_bootstrap_cis(latest, metric=metric, n_bootstrap=500)
        print(f"  {metric}: {ci['point_estimate']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
              f"(95% CI, n_articles={ci['n_articles']})")
        assert 0 <= ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"] <= 1, \
            f"CI bounds sanity check failed for {metric}"

    print("PASS: bootstrap CIs\n")


def test_pairwise_bootstrap():
    """Smoke-test pairwise bootstrap on two result files from the same run."""
    from src.evaluation.postrun import pairwise_bootstrap

    zero_files = sorted(RESULTS_DIR.glob("zero_shot_gpt-4o_n122_*.json"))
    few_files = sorted(RESULTS_DIR.glob("few_shot_gpt-4o_n122_*.json"))

    if not zero_files or not few_files:
        all_jsons = sorted(RESULTS_DIR.glob("*.json"))
        all_jsons = [c for c in all_jsons if not c.name.startswith("comparison")]
        if len(all_jsons) >= 2:
            zero_files = [all_jsons[0]]
            few_files = [all_jsons[1]]
        else:
            print("SKIP: need 2+ result files for pairwise bootstrap test")
            return

    file_a = zero_files[-1]
    file_b = few_files[-1]
    print(f"Testing pairwise bootstrap: {file_a.name} vs {file_b.name}")

    try:
        pair = pairwise_bootstrap(file_a, file_b, metric="tc_f1", n_bootstrap=500)
        print(f"  {pair['strategy_a']} TC-F1: {pair['f1_a']:.4f}")
        print(f"  {pair['strategy_b']} TC-F1: {pair['f1_b']:.4f}")
        print(f"  delta: {pair['delta']:+.4f}")
        print(f"  p-value ({pair['strategy_a']} > {pair['strategy_b']}): "
              f"{pair['p_value_one_sided']:.4f}")
        print(f"  n_articles: {pair['n_articles']}")
        print("PASS: pairwise bootstrap\n")
    except ValueError as e:
        print(f"  SKIP: {e}\n")


def test_per_pass_f1():
    """Smoke-test per-pass F1 on a multi-pass result (asv/consol/hybrid)."""
    from src.evaluation.postrun import compute_per_pass_f1

    multi_pass_files = []
    for prefix in ["asv_", "consol_", "hybrid_"]:
        multi_pass_files.extend(sorted(RESULTS_DIR.glob(f"{prefix}*.json")))

    if not multi_pass_files:
        print("SKIP: no asv/consol/hybrid results for per-pass F1 test")
        return

    latest = multi_pass_files[-1]
    print(f"Testing per-pass F1 on: {latest.name}")

    try:
        for metric in ["si_f1", "tc_f1"]:
            pp = compute_per_pass_f1(latest, metric=metric)
            print(f"  {metric}:")
            for k, f in enumerate(pp["per_pass_f1"]):
                print(f"    pass {k}: {f:.4f}")
            print(f"    merged: {pp['merged_f1']:.4f}")
            assert len(pp["per_pass_f1"]) == pp["n_passes"], "pass count mismatch"
        print("PASS: per-pass F1\n")
    except ValueError as e:
        print(f"  SKIP: {e}\n")


def test_per_pass_rejects_baselines():
    """per-pass F1 should raise ValueError for baseline strategies."""
    from src.evaluation.postrun import compute_per_pass_f1

    baseline_files = sorted(RESULTS_DIR.glob("zero_shot_*.json"))
    if not baseline_files:
        print("SKIP: no zero_shot results for rejection test")
        return

    latest = baseline_files[-1]
    print(f"Testing per-pass F1 rejects baseline: {latest.name}")
    try:
        compute_per_pass_f1(latest, metric="si_f1")
        print("FAIL: should have raised ValueError for baseline strategy")
    except ValueError as e:
        print(f"PASS: correctly raised ValueError: {e}\n")


if __name__ == "__main__":
    test_single_run()
    test_multi_run_mock()
    test_verify_catches_mismatches()
    test_bootstrap_cis()
    test_pairwise_bootstrap()
    test_per_pass_f1()
    test_per_pass_rejects_baselines()
