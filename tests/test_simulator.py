"""
tests/test_simulator.py - Unit tests verifying simulator correctness, cost accounting, noise channels, and bit-ordering.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
from simulator import OracleSimulator
from oracles import (
    make_public_linear,
    make_public_constant,
    make_public_simon,
    generate_linear,
)


def test_zero_oracle_circuit_cost_and_readout():
    """Circuit with 0 oracle gates must cost 0 queries and measure state |0>."""
    f = make_public_constant()
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=0.0, seed=42)

    qc = QuantumCircuit(4, 4)
    qc.measure(range(4), range(4))

    counts = sim.run(qc, shots=50)
    assert sim.total_queries_used == 0
    assert counts == {"0000": 50}


def test_query_cost_accounting():
    """Verify that (oracle gates) * shots is tracked accurately."""
    f = make_public_linear()
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=0.0, seed=42)

    # 1 oracle gate, 10 shots -> cost 10
    qc1 = QuantumCircuit(9, 1)
    qc1.append(Gate("oracle", 9, []), range(9))
    qc1.measure(8, 0)
    sim.run(qc1, shots=10)
    assert sim.total_queries_used == 10

    # 2 oracle gates, 5 shots -> cost 10
    qc2 = QuantumCircuit(9, 1)
    qc2.append(Gate("oracle", 9, []), range(9))
    qc2.append(Gate("oracle", 9, []), range(9))
    qc2.measure(8, 0)
    sim.run(qc2, shots=5)
    assert sim.total_queries_used == 20


def test_bernstein_vazirani_on_public_linear_zero_noise():
    """
    On public linear (a=178, b=1), BV circuit without noise must measure a=178 with 100% probability.
    178 in binary is 10110010.
    In Qiskit bitstring order (clbit 0 rightmost):
    bit 0 is 0, bit 1 is 1, bit 2 is 0, bit 3 is 0, bit 4 is 1, bit 5 is 1, bit 6 is 0, bit 7 is 1.
    String: '10110010'.
    """
    f = make_public_linear()
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=0.0, seed=42)

    qc = QuantumCircuit(9, 8)
    # Output qubit in |->
    qc.x(8)
    qc.h(8)
    # Input qubits in |+>
    qc.h(range(8))
    # Oracle gate
    qc.append(Gate("oracle", 9, []), range(9))
    # H on inputs
    qc.h(range(8))
    qc.measure(range(8), range(8))

    counts = sim.run(qc, shots=50)
    expected_str = f"{178:08b}"
    assert counts == {expected_str: 50}


def test_simon_circuit_on_public_simon_zero_noise():
    """
    On public simon (s=105), all measured strings y must satisfy y . s = 0 (mod 2).
    """
    f = make_public_simon()
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=0.0, seed=42)

    qc = QuantumCircuit(16, 8)
    # H on inputs
    qc.h(range(8))
    # Oracle gate: 8 inputs, 8 outputs
    qc.append(Gate("oracle", 16, []), range(16))
    # H on inputs
    qc.h(range(8))
    qc.measure(range(8), range(8))

    counts = sim.run(qc, shots=100)
    s = 105
    for bitstr, count in counts.items():
        y = int(bitstr, 2)
        # Parity of y & s
        dot = bin(y & s).count("1") & 1
        assert dot == 0, f"Measured y={y} does not satisfy y . s = 0"


def test_readout_noise_estimation():
    """Estimate readout flip rate from zero-oracle circuit."""
    f = make_public_constant()
    true_flip_rate = 0.04
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=true_flip_rate, seed=123)

    # 0-oracle circuit preparing |0...0> and measuring
    qc = QuantumCircuit(8, 8)
    qc.measure(range(8), range(8))

    shots = 500
    counts = sim.run(qc, shots=shots)
    assert sim.total_queries_used == 0

    total_bits = shots * 8
    ones_measured = sum(bitstr.count("1") * cnt for bitstr, cnt in counts.items())
    empirical_rate = ones_measured / total_bits

    assert abs(empirical_rate - true_flip_rate) < 0.015


def test_simon2_on_public_instance():
    """Verify Simon2 circuit on H={0, 15, 49, 62} produces vectors in H_perp."""
    from oracles import make_public_simon2
    f = make_public_simon2()
    sim = OracleSimulator(f, misfire_rate=0.0, readout_flip_rate=0.0, seed=42)

    qc = QuantumCircuit(16, 8)
    qc.h(range(8))
    qc.append(Gate("oracle", 16, []), range(16))
    qc.h(range(8))
    qc.measure(range(8), range(8))

    counts = sim.run(qc, shots=50)
    H = {0, 15, 49, 62}
    for bitstr in counts.keys():
        y = int(bitstr, 2)
        for h in H:
            assert (bin(y & h).count("1") & 1) == 0


def test_qubit_limit_enforced():
    """Simulator must reject circuits with more than 20 qubits."""
    f = make_public_constant()
    sim = OracleSimulator(f)
    qc = QuantumCircuit(21, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="maximum allowed qubits"):
        sim.run(qc, shots=10)

