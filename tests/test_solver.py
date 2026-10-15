"""
tests/test_solver.py - Comprehensive unit and integration tests for solver.py.
"""

import pytest
import numpy as np
from solver import solve
from simulator import OracleSimulator
from oracles import (
    make_public_linear,
    make_public_simon,
    make_public_simon2,
    make_public_perm,
    make_public_grover,
    make_public_constant,
    generate_oracle,
)

ALL_FAMILIES = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]


def test_solver_no_forbidden_imports():
    """Verify solver.py does not import simulator, oracles, or forbidden libraries."""
    import solver
    solver_code = open(solver.__file__).read()
    assert "simulator" not in solver_code
    assert "oracles" not in solver_code
    assert "matplotlib" not in solver_code
    assert "pytest" not in solver_code


def test_public_reference_linear():
    f = make_public_linear()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=10)
    family, secret = solve(8, 1, 60, ALL_FAMILIES, sim.run)
    assert family == "linear"
    assert secret == [178, 1]
    assert sim.total_queries_used <= 60


def test_public_reference_simon():
    f = make_public_simon()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=20)
    family, secret = solve(8, 8, 60, ALL_FAMILIES, sim.run)
    assert family == "simon"
    assert secret == 105
    assert sim.total_queries_used <= 60


def test_public_reference_simon2():
    f = make_public_simon2()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=30)
    family, secret = solve(8, 8, 60, ALL_FAMILIES, sim.run)
    assert family == "simon2"
    assert secret == [15, 49, 62]
    assert sim.total_queries_used <= 60


def test_public_reference_perm():
    f = make_public_perm()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=40)
    family, secret = solve(8, 8, 60, ALL_FAMILIES, sim.run)
    assert family == "perm"
    assert secret is None
    assert sim.total_queries_used <= 60


def test_public_reference_grover():
    f = make_public_grover()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=50)
    family, secret = solve(8, 1, 60, ALL_FAMILIES, sim.run)
    assert family == "grover"
    assert secret == 173
    assert sim.total_queries_used <= 60


def test_public_reference_constant():
    f = make_public_constant()
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=60)
    family, secret = solve(8, 1, 60, ALL_FAMILIES, sim.run)
    assert family == "constant"
    assert secret == 1
    assert sim.total_queries_used <= 60


@pytest.mark.parametrize("family", ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"])
def test_all_families_normal_noise(family):
    rng = np.random.default_rng(hash(family) % 100000)
    f = generate_oracle(family, rng=rng)
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=123)

    pred_family, pred_secret = solve(f.n, f.m, 60, ALL_FAMILIES, sim.run)

    assert sim.total_queries_used <= 60, f"Queries {sim.total_queries_used} exceeded 60"
    assert pred_family == f.family, f"Expected {f.family}, got {pred_family}"

    if f.secret is not None:
        assert pred_secret == f.secret, f"Family {family}: expected secret {f.secret}, got {pred_secret}"
    else:
        assert pred_secret is None


@pytest.mark.parametrize("family", ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"])
def test_all_families_stress_noise(family):
    rng = np.random.default_rng((hash(family) + 999) % 100000)
    f = generate_oracle(family, rng=rng)
    sim = OracleSimulator(f, misfire_rate=0.08, readout_flip_rate=0.04, seed=456)

    pred_family, pred_secret = solve(f.n, f.m, 60, ALL_FAMILIES, sim.run)

    assert sim.total_queries_used <= 60, f"Queries {sim.total_queries_used} exceeded 60"
    assert pred_family == f.family, f"Stress noise: expected {f.family}, got {pred_family}"

    if f.secret is not None:
        assert pred_secret == f.secret, f"Family {family}: expected secret {f.secret}, got {pred_secret}"
    else:
        assert pred_secret is None


def test_subset_families_order():
    """Verify solver handles arbitrary subsets and orderings of families."""
    subset = ["perm", "simon"]
    rng = np.random.default_rng(77)
    f = generate_oracle("simon", rng=rng)
    sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=777)

    pred_family, pred_secret = solve(f.n, f.m, 60, subset, sim.run)
    assert pred_family in subset
    assert pred_family == "simon"
    assert pred_secret == f.secret
    assert sim.total_queries_used <= 60


def test_budget_caps_never_exceeded():
    """Verify that solver strictly honors lower budgets (e.g. 20, 30, 40)."""
    for cap in [20, 30, 40, 50]:
        f = generate_oracle("linear", rng=np.random.default_rng(cap))
        sim = OracleSimulator(f, misfire_rate=0.04, readout_flip_rate=0.02, seed=cap)
        pred_family, pred_secret = solve(f.n, f.m, cap, ALL_FAMILIES, sim.run)
        assert sim.total_queries_used <= cap
