from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class BudgetGuard:
    """Track oracle query usage and prevent budget violations."""

    def __init__(
        self,
        raw_run: Callable[[QuantumCircuit, int], Dict[str, int]],
        budget: int,
    ):
        self.raw_run = raw_run
        self.budget = int(budget)
        self.queries_used = 0

    def remaining(self) -> int:
        return max(0, self.budget - self.queries_used)

    def can_afford(self, oracle_gates_per_shot: int, shots: int) -> bool:
        return (
            self.queries_used + oracle_gates_per_shot * shots
            <= self.budget
        )

    def run(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        if shots <= 0:
            return {}
        oracle_gates = sum(
            1 for inst in circuit.data
            if inst.operation.name == "oracle"
        )
        cost = oracle_gates * shots
        if self.queries_used + cost > self.budget:
            raise RuntimeError(
                f"Budget violation: attempting {cost} queries "
                f"with only {self.remaining()} remaining."
            )
        self.queries_used += cost
        return self.raw_run(circuit, shots)


def add_diffuser(qc: QuantumCircuit, n: int):
    """Grover diffusion operator."""
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def calibrate_readout_noise(
    guard: BudgetGuard,
    n: int = 8,
    shots: int = 150,
) -> float:
    """Estimate readout noise using a zero-oracle circuit (0 query cost)."""
    qc = QuantumCircuit(n, n)
    qc.measure(range(n), range(n))
    counts = guard.run(qc, shots=shots)
    total_bits = shots * n
    ones_measured = sum(
        bitstr.count("1") * count
        for bitstr, count in counts.items()
    )
    eta = ones_measured / max(1, total_bits)
    return float(np.clip(eta, 0.005, 0.15))


def query_f_direct(
    guard: BudgetGuard,
    n: int,
    m: int,
    x: int,
    shots: int = 1,
) -> int:
    """Query f(x) directly in the computational basis."""
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


def solve_m8(
    guard: BudgetGuard,
    n: int,
    m: int,
    families: List[str],
    eta: float,
) -> Tuple[str, Any]:
    """Identify simon, simon2, or perm."""
    allowed = set(families)

    def make_simon_qc() -> QuantumCircuit:
        qc = QuantumCircuit(n + m, n)
        qc.h(range(n))
        qc.append(Gate("oracle", n + m, []), range(n + m))
        qc.h(range(n))
        qc.measure(range(n), range(n))
        return qc

    # Use up to 48 shots (saves 20% query budget while maintaining high accuracy)
    total_shots = min(48, guard.remaining())
    measurements: List[int] = []
    if total_shots > 0:
        counts = guard.run(make_simon_qc(), shots=total_shots)
        for bitstr, count in counts.items():
            measurements.extend([int(bitstr, 2)] * count)

    if not measurements:
        return families[0], None

    all_v = np.arange(1, 256, dtype=np.int64)
    arr = np.array(measurements, dtype=np.int64)
    dots = np.bitwise_and(arr[:, None], all_v[None, :])
    parities = np.zeros_like(dots, dtype=np.int64)
    for bit in range(n):
        parities ^= (dots >> bit) & 1

    comp = np.sum(parities == 0, axis=0)
    sorted_idx = np.argsort(comp)[::-1]
    top_v = all_v[sorted_idx]
    top_scores = comp[sorted_idx]

    best_s2: Optional[List[int]] = None
    s2_thresh = (
        int(total_shots * 0.75)
        if total_shots != 48
        else 36
    )
    if "simon2" in allowed:
        for i in range(min(8, len(top_v))):
            h1 = int(top_v[i])
            for j in range(i + 1, min(10, len(top_v))):
                h2 = int(top_v[j])
                h3 = h1 ^ h2
                if h3 in top_v[:12]:
                    min_count = min(
                        comp[h1 - 1],
                        comp[h2 - 1],
                        comp[h3 - 1],
                    )
                    if min_count >= max(
                        s2_thresh,
                        int(top_scores[0]) - 6,
                    ):
                        best_s2 = sorted([h1, h2, h3])
                        break
            if best_s2 is not None:
                break

    s1_thresh = (
        int(total_shots * 0.79)
        if total_shots != 48
        else 38
    )
    if best_s2 is not None and "simon2" in allowed:
        return "simon2", best_s2
    if "simon" in allowed and top_scores[0] >= s1_thresh:
        return "simon", int(top_v[0])
    if "perm" in allowed:
        return "perm", None
    if "simon2" in allowed and best_s2 is not None:
        return "simon2", best_s2
    if "simon" in allowed:
        return "simon", int(top_v[0])
    return families[0], None


def solve_m1(
    guard: BudgetGuard,
    n: int,
    m: int,
    families: List[str],
    eta: float,
) -> Tuple[str, Any]:
    """Identify m=1 oracle families."""
    allowed = set(families)

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
    near_zero_count = 0
    total_diag = 0
    for bitstr, count in bv_counts.items():
        value = int(bitstr, 2)
        int_counts[value] = int_counts.get(value, 0) + count
        if bitstr.count("1") <= 1:
            near_zero_count += count
        total_diag += count

    zero_count = int_counts.get(0, 0)
    nonzero_items = [
        (value, count)
        for value, count in int_counts.items()
        if value != 0
    ]
    top_nonzero_v, top_nonzero_cnt = (
        max(nonzero_items, key=lambda item: item[1])
        if nonzero_items
        else (0, 0)
    )

    # Branch A: linear candidate.
    if "linear" in allowed and top_nonzero_cnt >= 3:
        candidate_a = top_nonzero_v
        candidate_b = query_f_direct(guard, n, m, 0, shots=3)
        if guard.remaining() >= 2:
            bit_idx = 0
            for bit in range(8):
                if (candidate_a >> bit) & 1:
                    bit_idx = bit
                    break
            x_test = 1 << bit_idx
            verify_b = query_f_direct(
                guard, n, m, x_test, shots=2
            )
            if verify_b == (1 ^ candidate_b):
                return "linear", [candidate_a, candidate_b]
        return "linear", [candidate_a, candidate_b]

    # Branch B: constant or Grover.
    if zero_count >= max(4, int(0.5 * total_diag)) or (eta >= 0.03 and near_zero_count >= 7):
        f0 = query_f_direct(guard, n, m, 0, shots=3)
        if f0 == 1:
            if "constant" in allowed and guard.remaining() >= 3:
                f1 = query_f_direct(guard, n, m, 1, shots=3)
                if f1 == 1:
                    return "constant", 1
                if "grover" in allowed:
                    return "grover", 0
            elif "constant" in allowed:
                return "constant", 1

        if "grover" in allowed and guard.remaining() >= 10:
            k_iter = 8 if eta >= 0.03 else 9

            def make_grover_qc(k: int) -> QuantumCircuit:
                qc = QuantumCircuit(n + m, n)
                qc.x(n)
                qc.h(n)
                qc.h(range(n))
                for _ in range(k):
                    qc.append(
                        Gate("oracle", n + m, []),
                        range(n + m),
                    )
                    add_diffuser(qc, n)
                qc.measure(range(n), range(n))
                return qc

            def check_grover_candidate(candidate: int) -> bool:
                if guard.remaining() < 1:
                    return False
                value = query_f_direct(
                    guard, n, m, candidate, shots=1
                )
                if value == 1:
                    shots_to_run = min(2, guard.remaining())
                    if shots_to_run > 0:
                        qc_conf = QuantumCircuit(n + m, m)
                        for bit_pos in range(n):
                            if (candidate >> bit_pos) & 1:
                                qc_conf.x(bit_pos)
                        qc_conf.append(
                            Gate("oracle", n + m, []),
                            range(n + m),
                        )
                        qc_conf.measure(
                            range(n, n + m),
                            range(m),
                        )
                        conf_counts = guard.run(
                            qc_conf, shots=shots_to_run
                        )
                        ones = conf_counts.get("1", 0) + 1
                        return ones >= 2
                    return True
                return False

            for _ in range(3):
                if not guard.can_afford(k_iter, 1):
                    break
                grover_counts = guard.run(
                    make_grover_qc(k_iter), shots=1
                )
                candidate = (
                    int(next(iter(grover_counts.keys())), 2)
                    if grover_counts
                    else 0
                )
                candidates = [candidate] + [
                    candidate ^ (1 << bit)
                    for bit in range(8)
                ]
                for item in candidates:
                    if check_grover_candidate(item):
                        return "grover", item

        if "constant" in allowed:
            return "constant", 0

    # Branch C: balanced versus randfun.
    if "balanced" in allowed or "randfun" in allowed:
        sample_budget = guard.remaining()
        rng = np.random.default_rng(total_diag + 42)
        if sample_budget <= 256:
            test_inputs = rng.choice(
                256,
                size=sample_budget,
                replace=False,
            )
        else:
            test_inputs = np.arange(sample_budget)

        ones_count = 0
        actual_sampled = 0
        for x_value in test_inputs:
            if guard.remaining() < 1:
                break
            measurement = query_f_direct(
                guard, n, m, int(x_value), shots=1
            )
            ones_count += measurement
            actual_sampled += 1

        if actual_sampled > 0:
            if "constant" in allowed and actual_sampled >= 20:
                if ones_count <= int(actual_sampled * 0.10):
                    return "constant", 0
                if ones_count >= int(actual_sampled * 0.90):
                    return "constant", 1

            eta_c = float(np.clip(eta, 0.005, 0.10))
            p_mis = min(2.0 * eta_c, 0.20)
            p_bal = 0.5 * (1.0 - p_mis) * (1.0 - 2.0 * eta_c) + eta_c
            p_ran = 0.3 * (1.0 - p_mis) * (1.0 - 2.0 * eta_c) + eta_c

            # In BV diagnostic, balanced has exact 0 amplitude on 0^n (only appears via misfire),
            # whereas randfun has (1 - 2*0.3)^2 = 0.16 probability per shot.
            bv_prior_llr = -0.65 * zero_count

            llr = (
                ones_count * np.log(p_bal / p_ran)
                + (actual_sampled - ones_count)
                * np.log((1.0 - p_bal) / (1.0 - p_ran))
                + bv_prior_llr
            )
            if "balanced" in allowed and "randfun" in allowed:
                if llr >= 0.35:
                    return "balanced", None
                return "randfun", None
            if "balanced" in allowed:
                return "balanced", None
            return "randfun", None

    # Fallbacks.
    if "constant" in allowed:
        return "constant", 0
    if "linear" in allowed:
        return "linear", [1, 0]
    if "grover" in allowed:
        return "grover", 0
    return families[0], None


def solve(
    n: int,
    m: int,
    budget: int,
    families: List[str],
    run: Callable[[QuantumCircuit, int], Dict[str, int]],
) -> Tuple[str, Any]:
    """Main entry point for Quantum Oracle Forensics."""
    guard = BudgetGuard(run, budget)
    eta = calibrate_readout_noise(
        guard,
        n=n,
        shots=150,
    )
    if m == 8:
        return solve_m8(guard, n, m, families, eta)
    return solve_m1(guard, n, m, families, eta)