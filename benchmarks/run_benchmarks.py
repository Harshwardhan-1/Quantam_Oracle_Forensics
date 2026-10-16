"""
benchmarks/run_benchmarks.py - Rigorous, reproducible benchmarking harness for Quantum Oracle Forensics.

Evaluates:
  1. Proposed Adaptive Solver (solver.py)
  2. Baseline 1: Classical-Only
  3. Baseline 2: Single-Algorithm
  4. Baseline 3: Random Guessing
  5. Baseline 4: Non-Adaptive Strategy

Measures:
  - Overall, per-family, and worst-family accuracy
  - Secret recovery accuracy vs family identification accuracy
  - Normal noise (4% misfire, 2% readout flip) vs Stress noise (8% misfire, 4% readout flip)
  - Query saving: mean(1 - queries / 60) over correct tasks
  - Accuracy as a function of budget caps: 20, 30, 40, 50, 60
  - Confusion matrix over all 8 families
  - Wilson score confidence intervals (95%)
  - Runtime and timeout tracking

Saves raw results to logs/benchmarks_summary.json and logs/benchmarks_detailed.csv.
"""

import os
import sys
import time
import json
import csv
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver import solve
from oracles import generate_oracle, OracleFunction
from simulator import OracleSimulator
from benchmarks.baselines import (
    solve_classical,
    solve_single_algorithm,
    solve_random_guessing,
    solve_non_adaptive,
)

ALL_FAMILIES = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2.0 * total)) / denom
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * total)) / total)) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return float(lower), float(upper)


def is_task_correct(f: OracleFunction, pred_family: str, pred_secret: Any) -> Tuple[bool, bool]:
    """
    Returns (family_correct, full_correct).
    A task is fully correct only if family is right AND secret is exactly right.
    """
    family_correct = (pred_family == f.family)
    if not family_correct:
        return False, False

    if f.secret is None:
        return True, (pred_secret is None)

    if f.family == "linear":
        secret_correct = (isinstance(pred_secret, (list, tuple)) and
                          len(pred_secret) == 2 and
                          list(pred_secret) == list(f.secret))
        return True, secret_correct

    elif f.family == "simon2":
        secret_correct = (isinstance(pred_secret, (list, tuple)) and
                          sorted(list(pred_secret)) == sorted(list(f.secret)))
        return True, secret_correct

    else:
        # constant, grover, simon
        secret_correct = (pred_secret == f.secret)
        return True, secret_correct


def run_benchmark_suite(
    instances_per_family: int = 50,
    seed_base: int = 2026,
    budget_caps: List[int] = [20, 30, 40, 50, 60],
) -> Dict[str, Any]:
    """Run comprehensive evaluation across all conditions and baselines."""
    print("=" * 80)
    print("QUANTUM ORACLE FORENSICS: COMPETITION BENCHMARK SUITE")
    print(f"Sample size: {instances_per_family} instances per family ({instances_per_family * 8} per condition)")
    print("=" * 80)

    solvers = {
        "Proposed Adaptive": solve,
        "Baseline 1 (Classical-Only)": solve_classical,
        "Baseline 2 (Single-Algorithm)": solve_single_algorithm,
        "Baseline 3 (Random Guessing)": solve_random_guessing,
        "Baseline 4 (Non-Adaptive)": solve_non_adaptive,
    }

    noise_conditions = {
        "Normal (4% misfire, 2% readout)": {"misfire": 0.04, "readout": 0.02},
        "Stress (8% misfire, 4% readout)": {"misfire": 0.08, "readout": 0.04},
    }

    all_detailed_records = []
    summary_results: Dict[str, Any] = {}

    # Pre-generate independent test instances for normal and stress conditions
    for noise_label, noise_params in noise_conditions.items():
        print(f"\n>>> Running condition: {noise_label}")
        summary_results[noise_label] = {}

        for solver_name, solver_fn in solvers.items():
            print(f"  Evaluating {solver_name}...", end="", flush=True)
            start_time = time.time()

            correct_tasks = 0
            correct_families = 0
            per_family_stats: Dict[str, Dict[str, int]] = {
                fam: {"total": 0, "family_correct": 0, "full_correct": 0, "queries_sum": 0}
                for fam in ALL_FAMILIES
            }
            queries_used_list = []
            query_savings_list = []
            confusion_matrix = {true_f: {pred_f: 0 for pred_f in ALL_FAMILIES} for true_f in ALL_FAMILIES}

            for fam_idx, true_family in enumerate(ALL_FAMILIES):
                for inst_idx in range(instances_per_family):
                    seed = seed_base + (1000 * fam_idx) + inst_idx
                    rng = np.random.default_rng(seed)
                    oracle = generate_oracle(true_family, rng=rng)
                    sim = OracleSimulator(
                        oracle,
                        misfire_rate=noise_params["misfire"],
                        readout_flip_rate=noise_params["readout"],
                        seed=seed + 50000,
                    )

                    t0 = time.time()
                    try:
                        pred_family, pred_secret = solver_fn(
                            oracle.n, oracle.m, 60, ALL_FAMILIES, sim.run
                        )
                    except Exception as err:
                        pred_family, pred_secret = "error", None
                    elapsed = time.time() - t0

                    fam_corr, full_corr = is_task_correct(oracle, pred_family, pred_secret)
                    q_used = sim.total_queries_used

                    if pred_family in confusion_matrix[true_family]:
                        confusion_matrix[true_family][pred_family] += 1

                    per_family_stats[true_family]["total"] += 1
                    if fam_corr:
                        per_family_stats[true_family]["family_correct"] += 1
                        correct_families += 1
                    if full_corr:
                        per_family_stats[true_family]["full_correct"] += 1
                        correct_tasks += 1
                        queries_used_list.append(q_used)
                        query_savings_list.append(max(0.0, 1.0 - (q_used / 60.0)))

                    per_family_stats[true_family]["queries_sum"] += q_used

                    all_detailed_records.append({
                        "noise_condition": noise_label,
                        "solver": solver_name,
                        "true_family": true_family,
                        "pred_family": pred_family,
                        "family_correct": fam_corr,
                        "full_correct": full_corr,
                        "queries_used": q_used,
                        "runtime_sec": elapsed,
                        "seed": seed,
                    })

            total_tasks = instances_per_family * len(ALL_FAMILIES)
            overall_acc = correct_tasks / total_tasks
            ci_low, ci_high = wilson_score_interval(correct_tasks, total_tasks)
            worst_fam_acc = min(
                stats["full_correct"] / stats["total"] for stats in per_family_stats.values()
            )
            mean_query_saving = float(np.mean(query_savings_list)) if query_savings_list else 0.0
            avg_queries_all = float(np.mean([r["queries_used"] for r in all_detailed_records if r["solver"] == solver_name and r["noise_condition"] == noise_label]))
            avg_runtime = (time.time() - start_time) / total_tasks

            summary_results[noise_label][solver_name] = {
                "overall_accuracy": overall_acc,
                "accuracy_ci_95": [ci_low, ci_high],
                "worst_family_accuracy": worst_fam_acc,
                "family_identification_accuracy": correct_families / total_tasks,
                "query_saving": mean_query_saving,
                "average_queries": avg_queries_all,
                "average_runtime_sec": avg_runtime,
                "per_family_accuracy": {
                    fam: stats["full_correct"] / stats["total"]
                    for fam, stats in per_family_stats.items()
                },
                "confusion_matrix": confusion_matrix,
            }

            print(f" Done ({avg_runtime:.3f}s/oracle). Acc: {overall_acc:.1%} [{ci_low:.1%}, {ci_high:.1%}], Worst: {worst_fam_acc:.1%}, QuerySaving: {mean_query_saving:.1%}")

    # -----------------------------------------------------------------------
    # Budget Cap Analysis for Proposed Solver
    # -----------------------------------------------------------------------
    print("\n>>> Evaluating Accuracy vs Budget Caps for Proposed Adaptive Solver...")
    budget_cap_results: Dict[int, Dict[str, float]] = {}

    for cap in budget_caps:
        cap_correct = 0
        cap_total = 0
        cap_queries = []
        for fam_idx, true_family in enumerate(ALL_FAMILIES):
            for inst_idx in range(instances_per_family):
                seed = seed_base + (2000 * fam_idx) + inst_idx + 7777
                rng = np.random.default_rng(seed)
                oracle = generate_oracle(true_family, rng=rng)
                sim = OracleSimulator(oracle, misfire_rate=0.04, readout_flip_rate=0.02, seed=seed + 90000)

                pred_family, pred_secret = solve(oracle.n, oracle.m, cap, ALL_FAMILIES, sim.run)
                _, full_corr = is_task_correct(oracle, pred_family, pred_secret)
                cap_total += 1
                if full_corr:
                    cap_correct += 1
                cap_queries.append(sim.total_queries_used)

        acc = cap_correct / cap_total
        ci_l, ci_u = wilson_score_interval(cap_correct, cap_total)
        avg_q = float(np.mean(cap_queries))
        budget_cap_results[cap] = {
            "accuracy": acc,
            "accuracy_ci_95": [ci_l, ci_u],
            "average_queries": avg_q,
        }
        print(f"  Budget Cap {cap:2d}: Accuracy = {acc:.1%} [{ci_l:.1%}, {ci_u:.1%}], Avg Queries = {avg_q:.1f}")

    summary_results["budget_cap_analysis"] = budget_cap_results

    # -----------------------------------------------------------------------
    # Save Results to disk
    # -----------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    summary_path = os.path.join("logs", "benchmarks_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[+] Saved benchmark summary to {summary_path}")

    csv_path = os.path.join("logs", "benchmarks_detailed.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_detailed_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_detailed_records)
    print(f"[+] Saved detailed per-instance outcomes to {csv_path}")

    return summary_results


if __name__ == "__main__":
    # Run evaluation with 50 instances per family (400 per condition)
    # Fast, rigorous, and statistically well-powered
    run_benchmark_suite(instances_per_family=50, seed_base=2026)
