"""
benchmarks/baselines.py - Baseline strategies for Quantum Oracle Forensics benchmark comparison.

Includes:
  1. Classical-only strategy (queries f only in computational basis, never uses superposition)
  2. Single-algorithm strategy (always runs standard fixed circuit: BV for m=1, Simon for m=8)
  3. Random guessing baseline
  4. Non-adaptive quantum strategy (fixed static query allocation without adaptive early stopping or verification)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from solver import BudgetGuard


# ---------------------------------------------------------------------------
# Baseline 1: Classical-Only Strategy
# ---------------------------------------------------------------------------

def solve_classical(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """
    Classical-only strategy: queries f(x) strictly in the computational basis.
    Never uses superposition gates (H, diffusers, etc.).
    """
    guard = BudgetGuard(run, budget)
    allowed = set(families)

    def query_classical(x: int) -> int:
        if guard.remaining() < 1:
            return 0
        qc = QuantumCircuit(n + m, m)
        for b in range(n):
            if (x >> b) & 1:
                qc.x(b)
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.measure(range(n, n + m), range(m))
        counts = guard.run(qc, shots=1)
        if not counts:
            return 0
        return int(max(counts.items(), key=lambda i: i[1])[0], 2)

    if m == 8:
        # Classical birthday search for collisions:
        # Sample distinct random x and record (x, f(x))
        seen: Dict[int, int] = {}
        found_collision: Optional[Tuple[int, int]] = None
        sample_limit = min(budget, 58)
        rng = np.random.default_rng(12345)
        inputs = rng.choice(256, size=sample_limit, replace=False)

        for x in inputs:
            fx = query_classical(int(x))
            if fx in seen and seen[fx] != x:
                found_collision = (seen[fx], int(x))
                break
            seen[fx] = int(x)

        if found_collision is not None:
            s_cand = found_collision[0] ^ found_collision[1]
            if "simon" in allowed:
                return "simon", s_cand
            elif "simon2" in allowed:
                return "simon2", [s_cand, 2, s_cand ^ 2]
        if "perm" in allowed:
            return "perm", None
        return families[0], None

    else:
        # m = 1 classical exploration:
        # Query f(0), f(1), f(2), ...
        sample_limit = min(budget, 55)
        rng = np.random.default_rng(54321)
        inputs = rng.choice(256, size=sample_limit, replace=False)
        outcomes: Dict[int, int] = {}
        for x in inputs:
            outcomes[int(x)] = query_classical(int(x))

        vals = list(outcomes.values())
        ones_count = sum(vals)

        # Constant check
        if ones_count == 0 and "constant" in allowed:
            return "constant", 0
        if ones_count == len(vals) and "constant" in allowed:
            return "constant", 1

        # Grover check: exactly one 1 found
        if ones_count == 1 and "grover" in allowed:
            t_cand = [x for x, val in outcomes.items() if val == 1][0]
            return "grover", t_cand

        # Balanced vs randfun check
        if len(vals) > 0:
            p_meas = ones_count / len(vals)
            if "balanced" in allowed and "randfun" in allowed:
                return ("balanced", None) if p_meas >= 0.40 else ("randfun", None)
            elif "balanced" in allowed:
                return "balanced", None
            elif "randfun" in allowed:
                return "randfun", None

        if "linear" in allowed:
            return "linear", [1, outcomes.get(0, 0)]

        return families[0], None


# ---------------------------------------------------------------------------
# Baseline 2: Single-Algorithm Strategy (Always BV for m=1, Always Simon for m=8)
# ---------------------------------------------------------------------------

def solve_single_algorithm(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """
    Always applies the exact same textbook quantum algorithm.
    m=1: Always runs Bernstein-Vazirani.
    m=8: Always runs textbook Simon.
    No branching, no Grover iterations, no direct verifications.
    """
    guard = BudgetGuard(run, budget)
    allowed = set(families)

    if m == 8:
        # Always run Simon circuit with all available budget
        shots = min(budget, 50)
        qc = QuantumCircuit(n + m, n)
        qc.h(range(n))
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        counts = guard.run(qc, shots=shots)

        samples = []
        for bstr, cnt in counts.items():
            samples.extend([int(bstr, 2)] * cnt)

        if not samples:
            return families[0], None

        all_v = np.arange(1, 256)
        arr = np.array(samples)
        dots = np.bitwise_and(arr[:, None], all_v[None, :])
        parities = np.zeros_like(dots)
        for b in range(n):
            parities ^= (dots >> b) & 1
        comp = np.sum(parities == 0, axis=0)
        best_v = int(all_v[np.argmax(comp)])

        if "simon" in allowed:
            return "simon", best_v
        elif "simon2" in allowed:
            return "simon2", [best_v, 2, best_v ^ 2]
        return families[0], None

    else:
        # Always run Bernstein-Vazirani
        shots = min(budget, 50)
        qc = QuantumCircuit(n + m, n)
        qc.x(n); qc.h(n); qc.h(range(n))
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        counts = guard.run(qc, shots=shots)

        if not counts:
            return families[0], None

        best_str = max(counts.items(), key=lambda i: i[1])[0]
        v = int(best_str, 2)
        if v == 0 and "constant" in allowed:
            return "constant", 0
        elif "linear" in allowed and v != 0:
            return "linear", [v, 0]
        elif "balanced" in allowed:
            return "balanced", None
        return families[0], None


# ---------------------------------------------------------------------------
# Baseline 3: Random Guessing Strategy
# ---------------------------------------------------------------------------

def solve_random_guessing(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """Randomly selects a family from allowed candidates and generates a random secret."""
    rng = np.random.default_rng()
    # Filter candidates by m
    if m == 8:
        valid_fams = [f for f in families if f in ("simon", "simon2", "perm")]
    else:
        valid_fams = [f for f in families if f in ("constant", "balanced", "linear", "grover", "randfun")]

    if not valid_fams:
        valid_fams = families

    fam = str(rng.choice(valid_fams))
    if fam == "constant":
        return fam, int(rng.integers(0, 2))
    elif fam == "linear":
        return fam, [int(rng.integers(1, 256)), int(rng.integers(0, 2))]
    elif fam == "grover":
        return fam, int(rng.integers(0, 256))
    elif fam == "simon":
        return fam, int(rng.integers(1, 256))
    elif fam == "simon2":
        h1 = int(rng.integers(1, 256))
        h2 = int(rng.integers(1, 256))
        while h2 == h1 or h2 == 0:
            h2 = int(rng.integers(1, 256))
        return fam, sorted([h1, h2, h1 ^ h2])
    else:
        return fam, None


# ---------------------------------------------------------------------------
# Baseline 4: Non-Adaptive Strategy
# ---------------------------------------------------------------------------

def solve_non_adaptive(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """
    Fixed-allocation quantum strategy without dynamic early stopping,
    without noise calibration, and using static fixed budget slices.
    """
    guard = BudgetGuard(run, budget)
    allowed = set(families)

    if m == 8:
        # Fixed 40 shots of Simon, static uncalibrated threshold
        shots = min(40, guard.remaining())
        qc = QuantumCircuit(n + m, n)
        qc.h(range(n)); qc.append(Gate("oracle", n + m, []), range(n + m)); qc.h(range(n)); qc.measure(range(n), range(n))
        counts = guard.run(qc, shots=shots)
        samples = []
        for bstr, cnt in counts.items(): samples.extend([int(bstr, 2)] * cnt)
        all_v = np.arange(1, 256)
        arr = np.array(samples)
        dots = np.bitwise_and(arr[:, None], all_v[None, :])
        parities = np.zeros_like(dots)
        for b in range(n): parities ^= (dots >> b) & 1
        comp = np.sum(parities == 0, axis=0)
        sorted_idx = np.argsort(comp)[::-1]
        top_v = all_v[sorted_idx]
        top_scores = comp[sorted_idx]

        # Static uncalibrated threshold
        if top_scores[0] >= 34 and "simon" in allowed:
            return "simon", int(top_v[0])
        elif "perm" in allowed:
            return "perm", None
        return families[0], None

    else:
        # Fixed 20 shots BV, 20 shots sampling
        shots_bv = min(20, guard.remaining())
        qc_bv = QuantumCircuit(n + m, n)
        qc_bv.x(n); qc_bv.h(n); qc_bv.h(range(n)); qc_bv.append(Gate("oracle", n + m, []), range(n + m)); qc_bv.h(range(n)); qc_bv.measure(range(n), range(n))
        counts_bv = guard.run(qc_bv, shots=shots_bv)

        best_str = max(counts_bv.items(), key=lambda i: i[1])[0]
        v = int(best_str, 2)
        if v != 0 and "linear" in allowed:
            return "linear", [v, 0]
        elif v == 0 and "constant" in allowed:
            return "constant", 0
        elif "balanced" in allowed:
            return "balanced", None
        return families[0], None
