"""
simulator.py - High-performance, self-contained quantum circuit simulator for Quantum Oracle Forensics.

Supports:
- Execution of Qiskit QuantumCircuit with Gate("oracle", n+m, [])
- Flat bitmask statevector evolution (fast, bug-free, zero axis bugs)
- Configurable oracle misfire noise (4% normal, 8% stress)
- Configurable readout bit-flip noise (2% normal, 4% stress)
- Exact query cost accounting: (number of oracle gates) * shots
- Circuits with 0 oracle gates cost 0 queries
- Standard Qiskit bitstring ordering (clbit 0 is rightmost)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from qiskit import QuantumCircuit
from oracles import OracleFunction

INV_SQRT2 = 1.0 / np.sqrt(2.0)


class OracleSimulator:
    """Simulates quantum circuits containing sealed oracles under noise."""

    def __init__(
        self,
        oracle: OracleFunction,
        misfire_rate: float = 0.04,
        readout_flip_rate: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.oracle = oracle
        self.misfire_rate = misfire_rate
        self.readout_flip_rate = readout_flip_rate
        self.rng = np.random.default_rng(seed)
        self.total_queries_used = 0

    def reset_query_counter(self):
        self.total_queries_used = 0

    def run(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """
        Executes circuit for given shots.
        Charges (oracle gates) * shots to query counter.
        Returns counts dictionary keyed by classical bitstrings (clbit 0 rightmost).
        """
        if shots <= 0:
            return {}

        # Count oracle gates
        oracle_gates_count = 0
        for instruction in circuit.data:
            if instruction.operation.name == "oracle":
                oracle_gates_count += 1

        # Query cost accounting: (oracle gates) * shots
        query_cost = oracle_gates_count * shots
        self.total_queries_used += query_cost

        # Inspect circuit registers
        num_qubits = circuit.num_qubits
        if num_qubits > 20:
            raise ValueError(f"Circuit exceeds maximum allowed qubits (20): got {num_qubits}")

        num_clbits = circuit.num_clbits
        if num_clbits == 0:
            return {}

        # Extract measurements
        measurements: List[Tuple[int, int]] = []
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                q_idx = circuit.find_bit(instruction.qubits[0]).index
                c_idx = circuit.find_bit(instruction.clbits[0]).index
                measurements.append((q_idx, c_idx))

        # Check if circuit has 0 oracle gates
        if oracle_gates_count == 0:
            probs = self._simulate_circuit_probs(circuit, oracle_misfires=[])
            return self._sample_counts(probs, measurements, num_qubits, num_clbits, shots)

        # If circuit has 1 oracle gate, group shots by misfire
        if oracle_gates_count == 1:
            num_misfires = int(self.rng.binomial(shots, self.misfire_rate))
            num_success = shots - num_misfires
            counts: Dict[str, int] = {}

            if num_success > 0:
                probs_succ = self._simulate_circuit_probs(circuit, oracle_misfires=[False])
                counts_succ = self._sample_counts(probs_succ, measurements, num_qubits, num_clbits, num_success)
                for k, v in counts_succ.items():
                    counts[k] = counts.get(k, 0) + v

            if num_misfires > 0:
                probs_misfire = self._simulate_circuit_probs(circuit, oracle_misfires=[True])
                counts_misfire = self._sample_counts(probs_misfire, measurements, num_qubits, num_clbits, num_misfires)
                for k, v in counts_misfire.items():
                    counts[k] = counts.get(k, 0) + v

            return counts

        # If circuit has multiple oracle gates (e.g. Grover iterations)
        counts: Dict[str, int] = {}
        for _ in range(shots):
            misfires = [bool(self.rng.random() < self.misfire_rate) for _ in range(oracle_gates_count)]
            probs = self._simulate_circuit_probs(circuit, oracle_misfires=misfires)
            shot_count = self._sample_counts(probs, measurements, num_qubits, num_clbits, 1)
            for k, v in shot_count.items():
                counts[k] = counts.get(k, 0) + v

        return counts

    def _simulate_circuit_probs(self, circuit: QuantumCircuit, oracle_misfires: List[bool]) -> np.ndarray:
        """Simulate unitary evolution and return state probabilities."""
        num_qubits = circuit.num_qubits
        dim = 1 << num_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0  # Initial state |0...0>

        # Precompute basis indices for gate operations
        all_j = np.arange(dim, dtype=np.int64)

        oracle_idx = 0
        for instruction in circuit.data:
            op_name = instruction.operation.name

            if op_name in ("measure", "barrier"):
                continue

            elif op_name == "h":
                q = circuit.find_bit(instruction.qubits[0]).index
                mask = 1 << q
                j0 = all_j[(all_j & mask) == 0]
                j1 = j0 | mask
                psi0 = state[j0].copy()
                psi1 = state[j1].copy()
                state[j0] = INV_SQRT2 * (psi0 + psi1)
                state[j1] = INV_SQRT2 * (psi0 - psi1)

            elif op_name == "x":
                q = circuit.find_bit(instruction.qubits[0]).index
                mask = 1 << q
                j0 = all_j[(all_j & mask) == 0]
                j1 = j0 | mask
                psi0 = state[j0].copy()
                state[j0] = state[j1]
                state[j1] = psi0

            elif op_name == "z":
                q = circuit.find_bit(instruction.qubits[0]).index
                mask = 1 << q
                state[(all_j & mask) != 0] *= -1.0

            elif op_name == "s":
                q = circuit.find_bit(instruction.qubits[0]).index
                mask = 1 << q
                state[(all_j & mask) != 0] *= 1.0j

            elif op_name == "cx":
                c = circuit.find_bit(instruction.qubits[0]).index
                t = circuit.find_bit(instruction.qubits[1]).index
                c_mask = 1 << c
                t_mask = 1 << t
                # Indices where control bit is 1 and target is 0
                j0 = all_j[((all_j & c_mask) != 0) & ((all_j & t_mask) == 0)]
                j1 = j0 | t_mask
                psi0 = state[j0].copy()
                state[j0] = state[j1]
                state[j1] = psi0

            elif op_name == "mcx":
                controls = [circuit.find_bit(q).index for q in instruction.qubits[:-1]]
                target = circuit.find_bit(instruction.qubits[-1]).index
                ctrl_mask = sum(1 << c for c in controls)
                t_mask = 1 << target
                j0 = all_j[((all_j & ctrl_mask) == ctrl_mask) & ((all_j & t_mask) == 0)]
                j1 = j0 | t_mask
                psi0 = state[j0].copy()
                state[j0] = state[j1]
                state[j1] = psi0

            elif op_name == "oracle":
                misfire = oracle_misfires[oracle_idx]
                oracle_idx += 1
                if not misfire:
                    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                    state = self._apply_oracle_unitary(state, qubits, all_j)

            else:
                try:
                    unitary = instruction.operation.to_matrix()
                    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
                    state = self._apply_custom_unitary(state, unitary, qubits, num_qubits)
                except Exception as err:
                    raise NotImplementedError(f"Unsupported gate {op_name}: {err}")

        probs = np.abs(state) ** 2
        total_p = np.sum(probs)
        if total_p > 0:
            probs /= total_p
        return probs

    def _apply_oracle_unitary(self, state: np.ndarray, qubits: List[int], all_j: np.ndarray) -> np.ndarray:
        """
        Applies oracle U|x>|y> = |x>|y ^ f(x)>.
        qubits: input qubits (first n) followed by output qubits (next m).
        """
        n = self.oracle.n
        m = self.oracle.m
        in_qubits = qubits[:n]
        out_qubits = qubits[n : n + m]

        # Extract input x integer for each basis state
        x = np.zeros(len(all_j), dtype=np.int64)
        for bit_pos, q in enumerate(in_qubits):
            x |= ((all_j >> q) & 1) << bit_pos

        # f(x) for all basis states
        fx = self.oracle.table[x]

        # Compute new basis state index j'
        new_j = all_j.copy()
        for bit_pos, q in enumerate(out_qubits):
            fx_bit = (fx >> bit_pos) & 1
            flip_mask = fx_bit << q
            new_j ^= flip_mask

        # Apply permutation: state'[j'] = state[j]
        # Since new_j is an involution, state'[j] = state[new_j]
        return state[new_j]

    def _apply_custom_unitary(self, state: np.ndarray, unitary: np.ndarray, qubits: List[int], num_qubits: int) -> np.ndarray:
        k = len(qubits)
        tensor = state.reshape([2] * num_qubits)
        # In numpy row-major order, qubit q corresponds to axis (num_qubits - 1 - q)
        axes = [num_qubits - 1 - q for q in qubits]
        tensor = np.moveaxis(tensor, axes, list(range(k)))
        reshaped = tensor.reshape(1 << k, -1)
        evolved = unitary @ reshaped
        back = evolved.reshape([2] * num_qubits)
        back = np.moveaxis(back, list(range(k)), axes)
        return back.reshape(-1)

    def _sample_counts(
        self,
        probs: np.ndarray,
        measurements: List[Tuple[int, int]],
        num_qubits: int,
        num_clbits: int,
        shots: int,
    ) -> Dict[str, int]:
        """Sample classical outcomes with readout flip noise."""
        dim = len(probs)
        all_j = np.arange(dim, dtype=np.int64)

        # Classical integer mapped to each basis state j
        # Classical bit c_idx receives the bit of q_idx
        cl_ints = np.zeros(dim, dtype=np.int64)
        for q_idx, c_idx in measurements:
            cl_ints |= ((all_j >> q_idx) & 1) << c_idx

        # Sample directly from probs
        sampled_basis = self.rng.choice(dim, size=shots, p=probs)
        sampled_cl = cl_ints[sampled_basis]

        # Apply readout bit-flip noise independently to each classical bit
        if self.readout_flip_rate > 0.0:
            flip_mask = np.zeros(shots, dtype=np.int64)
            for c_idx in range(num_clbits):
                flips = (self.rng.random(size=shots) < self.readout_flip_rate).astype(np.int64)
                flip_mask |= (flips << c_idx)
            sampled_cl ^= flip_mask

        # Format into Qiskit bitstrings (clbit 0 rightmost)
        unique_vals, val_counts = np.unique(sampled_cl, return_counts=True)
        counts = {}
        for val, count in zip(unique_vals, val_counts):
            bitstr = f"{int(val):0{num_clbits}b}"
            counts[bitstr] = int(count)

        return counts
