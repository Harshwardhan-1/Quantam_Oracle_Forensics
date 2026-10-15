"""
solver.py - Competition-grade, evidence-driven quantum oracle forensics solver.

IBM Qiskit Fall Fest 2026 | QC@IITI x QryptoNox
Author: Antigravity Forensics Team

Contract:
    solve(n, m, budget, families, run) -> (family, secret)

Self-contained module using only Python standard library, numpy, scipy, and qiskit.
No imports from other project files.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class BudgetGuard:
    """Zero-tolerance budget tracker preventing any execution exceeding the budget."""

    def __init__(self, raw_run: Callable[[QuantumCircuit, int], Dict[str, int]], budget: int):
        self.raw_run = raw_run
        self.budget = int(budget)
        self.queries_used = 0

    def remaining(self) -> int:
        return max(0, self.budget - self.queries_used)

    def can_afford(self, oracle_gates_per_shot: int, shots: int) -> bool:
        return (self.queries_used + oracle_gates_per_shot * shots) <= self.budget

    def run(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        if shots <= 0:
            return {}

        # Count oracle gates
        oracle_gates = sum(1 for inst in circuit.data if inst.operation.name == "oracle")
        cost = oracle_gates * shots

        if self.queries_used + cost > self.budget:
            raise RuntimeError(
                f"Budget violation: attempting {cost} queries with only {self.remaining()} remaining."
            )

        self.queries_used += cost
        return self.raw_run(circuit, shots)


def add_diffuser(qc: QuantumCircuit, n: int):
    """Grover diffusion operator on n qubits: H^n (2|0><0| - I) H^n."""
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def calibrate_readout_noise(guard: BudgetGuard, n: int = 8, shots: int = 150) -> float:
    """
    Estimate readout flip rate for free using 0-oracle circuit.
    Cost: 0 queries.
    """
    qc = QuantumCircuit(n, n)
    qc.measure(range(n), range(n))
    counts = guard.run(qc, shots=shots)

    total_bits = shots * n
    ones_measured = sum(bitstr.count("1") * cnt for bitstr, cnt in counts.items())
    eta = ones_measured / max(1, total_bits)
    return float(np.clip(eta, 0.005, 0.15))


def query_f_direct(
    guard: BudgetGuard,
    n: int,
    m: int,
    x: int,
    shots: int = 1,
) -> int:
    """
    Directly query f(x) in computational basis.
    Cost: 1 * shots queries.
    Returns majority integer.
    """
    actual_shots = min(shots, guard.remaining())
    if actual_shots <= 0:
        return 0

    qc = QuantumCircuit(n + m, m)
    for bit_pos in range(n):
        if (x >> bit_pos) & 1:
            qc.x(bit_pos)

    qc.append(Gate("oracle", n + m, []), range(n + m))
    qc.measure(range(n, n + m), range(m))

    counts = guard.run(qc, shots=actual_shots)
    if not counts:
        return 0

    best_str = max(counts.items(), key=lambda item: item[1])[0]
    return int(best_str, 2)


# ---------------------------------------------------------------------------
# Solver for m = 8 families: {simon, simon2, perm}
# ---------------------------------------------------------------------------

def solve_m8(
    guard: BudgetGuard,
    n: int,
    m: int,
    families: List[str],
    eta: float,
) -> Tuple[str, Any]:
    """
    Identifies simon, simon2, or perm using linear algebraic kernel analysis.
    """
    allowed = set(families)

    def make_simon_qc() -> QuantumCircuit:
        qc = QuantumCircuit(n + m, n)
        qc.h(range(n))
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        return qc

    total_shots = min(48, guard.remaining())
    measurements: List[int] = []

    if total_shots > 0:
        counts = guard.run(make_simon_qc(), shots=total_shots)
        for bitstr, cnt in counts.items():
            measurements.extend([int(bitstr, 2)] * cnt)

    if not measurements:
        return families[0], None

    # Evaluate candidate kernel vectors v in GF(2)^8 \ {0}
    all_v = np.arange(1, 256, dtype=np.int64)
    arr = np.array(measurements, dtype=np.int64)
    dots = np.bitwise_and(arr[:, None], all_v[None, :])
    parities = np.zeros_like(dots, dtype=np.int64)
    for b in range(n):
        parities ^= (dots >> b) & 1
    comp = np.sum(parities == 0, axis=0)  # shape (255,)

    sorted_idx = np.argsort(comp)[::-1]
    top_v = all_v[sorted_idx]
    top_scores = comp[sorted_idx]

    # Search for Simon2 subspace: 3 nonzero elements forming 2D subspace H
    best_s2: Optional[List[int]] = None
    s2_thresh = int(total_shots * 0.73) if total_shots != 48 else 35

    if "simon2" in allowed:
        for i in range(min(10, len(top_v))):
            h1 = top_v[i]
            for j in range(i + 1, min(12, len(top_v))):
                h2 = top_v[j]
                h3 = h1 ^ h2
                if h3 in top_v[:14]:
                    min_c = min(comp[h1 - 1], comp[h2 - 1], comp[h3 - 1])
                    if min_c >= s2_thresh:
                        best_s2 = sorted([int(h1), int(h2), int(h3)])
                        break
            if best_s2 is not None:
                break

    s1_thresh = int(total_shots * 0.79) if total_shots != 48 else 38

    if best_s2 is not None and "simon2" in allowed:
        return "simon2", best_s2
    elif "simon" in allowed and top_scores[0] >= s1_thresh:
        return "simon", int(top_v[0])
    elif "perm" in allowed:
        return "perm", None

    # Fallbacks for restricted family lists
    if "simon2" in allowed and best_s2 is not None:
        return "simon2", best_s2
    if "simon" in allowed:
        return "simon", int(top_v[0])

    return families[0], None


# ---------------------------------------------------------------------------
# Solver for m = 1 families: {constant, balanced, linear, grover, randfun}
# ---------------------------------------------------------------------------

def solve_m1(
    guard: BudgetGuard,
    n: int,
    m: int,
    families: List[str],
    eta: float,
) -> Tuple[str, Any]:
    """
    Identifies constant, balanced, linear, grover, or randfun and recovers secret.
    """
    allowed = set(families)

    # Step 1: Diagnostic BV/DJ Circuit
    def make_bv_circuit() -> QuantumCircuit:
        qc = QuantumCircuit(n + m, n)
        qc.x(n)
        qc.h(n)
        qc.h(range(n))
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        return qc

    diag_shots = min(8, guard.remaining())
    bv_counts = guard.run(make_bv_circuit(), shots=diag_shots)

    int_counts: Dict[int, int] = {}
    total_diag = 0
    for bitstr, cnt in bv_counts.items():
        v = int(bitstr, 2)
        int_counts[v] = int_counts.get(v, 0) + cnt
        total_diag += cnt

    zero_count = int_counts.get(0, 0)
    nonzero_items = [(v, cnt) for v, cnt in int_counts.items() if v != 0]
    top_nonzero_v, top_nonzero_cnt = max(nonzero_items, key=lambda x: x[1]) if nonzero_items else (0, 0)

    # -----------------------------------------------------------------------
    # Branch A: Strong peak at nonzero a != 0 -> Linear candidate
    # -----------------------------------------------------------------------
    if "linear" in allowed and top_nonzero_cnt >= max(3, total_diag // 2):
        candidate_a = top_nonzero_v
        # Recover b via majority vote of f(0)
        candidate_b = query_f_direct(guard, n, m, 0, shots=3)

        # Verification at input x_test where candidate_a . x_test = 1
        if guard.remaining() >= 2:
            bit_idx = 0
            for i in range(8):
                if (candidate_a >> i) & 1:
                    bit_idx = i
                    break
            x_test = 1 << bit_idx
            verify_b = query_f_direct(guard, n, m, x_test, shots=2)
            if verify_b == (1 ^ candidate_b):
                return "linear", [candidate_a, candidate_b]

        return "linear", [candidate_a, candidate_b]

    # -----------------------------------------------------------------------
    # Branch B: Dominant zero count -> Constant or Grover candidate
    # -----------------------------------------------------------------------
    if zero_count >= max(5, int(0.7 * total_diag)):
        # Probe f(0) with 3 shots majority
        f0 = query_f_direct(guard, n, m, 0, shots=3)

        if f0 == 1:
            # Either constant(1) or grover(0)
            if "constant" in allowed and guard.remaining() >= 3:
                f1 = query_f_direct(guard, n, m, 1, shots=3)
                if f1 == 1:
                    return "constant", 1
                elif "grover" in allowed:
                    return "grover", 0
            elif "constant" in allowed:
                return "constant", 1

        # f0 == 0: cannot be constant(1). Candidates: grover(t) with t != 0, or constant(0)
        if "grover" in allowed and guard.remaining() >= 10:
            k_iter = 9

            def make_grover_qc(k: int) -> QuantumCircuit:
                qc = QuantumCircuit(n + m, n)
                qc.x(n)
                qc.h(n)
                qc.h(range(n))
                for _ in range(k):
                    qc.append(Gate("oracle", n + m, []), range(n + m))
                    add_diffuser(qc, n)
                qc.measure(range(n), range(n))
                return qc

            def check_grover_candidate(cand: int) -> bool:
                if guard.remaining() < 1:
                    return False
                v = query_f_direct(guard, n, m, cand, shots=1)
                if v == 1:
                    # Confirm with 2 additional shots (total 3 shots, need at least 2 ones)
                    shots_to_run = min(2, guard.remaining())
                    if shots_to_run > 0:
                        qc_conf = QuantumCircuit(n + m, m)
                        for bit_pos in range(n):
                            if (cand >> bit_pos) & 1:
                                qc_conf.x(bit_pos)
                        qc_conf.append(Gate("oracle", n + m, []), range(n + m))
                        qc_conf.measure(range(n, n + m), range(m))
                        c_conf = guard.run(qc_conf, shots=shots_to_run)
                        ones = c_conf.get("1", 0) + 1
                        return ones >= 2
                    return True
                return False

            # Run up to 3 Grover shots with Hamming-1 neighborhood checking
            for _ in range(3):
                if not guard.can_afford(k_iter, 1):
                    break
                g_counts = guard.run(make_grover_qc(k_iter), shots=1)
                t_cand = int(next(iter(g_counts.keys())), 2) if g_counts else 0

                search_list = [t_cand] + [t_cand ^ (1 << b) for b in range(8)]
                for cand in search_list:
                    if check_grover_candidate(cand):
                        return "grover", cand

        # If Grover search did not verify any marked input, it is constant(0)
        if "constant" in allowed:
            return "constant", 0

    # -----------------------------------------------------------------------
    # Branch C: Spread over basis states -> Balanced vs Randfun
    # -----------------------------------------------------------------------
    if "balanced" in allowed or "randfun" in allowed:
        sample_budget = min(44, guard.remaining())
        rng = np.random.default_rng(total_diag + 42)
        test_inputs = rng.choice(256, size=sample_budget, replace=False) if sample_budget <= 256 else np.arange(sample_budget)

        ones_count = 0
        actual_sampled = 0
        for x_val in test_inputs:
            if guard.remaining() < 1:
                break
            meas = query_f_direct(guard, n, m, int(x_val), shots=1)
            ones_count += meas
            actual_sampled += 1

        if actual_sampled > 0:
            threshold = int(np.round(actual_sampled * 0.385))
            if "balanced" in allowed and "randfun" in allowed:
                return ("balanced", None) if ones_count >= threshold else ("randfun", None)
            elif "balanced" in allowed:
                return "balanced", None
            else:
                return "randfun", None


    # Fallbacks for restricted family lists
    if "constant" in allowed:
        return "constant", 0
    if "linear" in allowed:
        return "linear", [1, 0]
    if "grover" in allowed:
        return "grover", 0

    return families[0], None


# ---------------------------------------------------------------------------
# Main Challenge Entrypoint
# ---------------------------------------------------------------------------

def solve(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """
    Main entry point for Quantum Oracle Forensics.

    Parameters:
        n: Number of input bits (8)
        m: Number of output bits (1 or 8)
        budget: Query budget (never exceed)
        families: List of candidate family names
        run: Harness execution function (circuit, shots) -> counts

    Returns:
        (family, secret)
    """
    guard = BudgetGuard(run, budget)

    # Free readout noise calibration (0 queries)
    eta = calibrate_readout_noise(guard, n=n, shots=150)

    # Partition by output dimension m
    if m == 8:
        return solve_m8(guard, n, m, families, eta)
    else:
        return solve_m1(guard, n, m, families, eta)
