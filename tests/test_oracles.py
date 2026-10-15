"""
tests/test_oracles.py - Unit tests verifying oracles against public check values and algebraic properties.
"""

import pytest
import numpy as np
from oracles import (
    make_public_linear,
    make_public_simon,
    make_public_simon2,
    make_public_perm,
    make_public_grover,
    make_public_constant,
    generate_oracle,
    parity,
)


def test_public_linear():
    f = make_public_linear()
    assert f.family == "linear"
    assert f.secret == [178, 1]
    assert f(0) == 1
    assert f(1) == 1
    assert f(2) == 0
    assert f(255) == 1


def test_public_simon():
    f = make_public_simon()
    assert f.family == "simon"
    assert f.secret == 105
    assert f(5) == 5
    assert f(108) == 5
    assert f(105) == 0
    assert f(200) == 161


def test_public_simon2():
    f = make_public_simon2()
    assert f.family == "simon2"
    assert f.secret == [15, 49, 62]
    assert f(0) == 0
    assert f(15) == 0
    assert f(49) == 0
    assert f(62) == 0
    assert f(1) == 1
    assert f(200) == 199


def test_public_perm():
    f = make_public_perm()
    assert f.family == "perm"
    assert f.secret is None
    assert f(0) == 11
    assert f(1) == 48
    assert f(2) == 85
    assert f(255) == 230


def test_public_grover():
    f = make_public_grover()
    assert f.family == "grover"
    assert f.secret == 173
    assert f(173) == 1
    assert f(0) == 0
    assert f(255) == 0


def test_public_constant():
    f = make_public_constant()
    assert f.family == "constant"
    assert f.secret == 1
    for x in [0, 1, 42, 100, 255]:
        assert f(x) == 1


@pytest.mark.parametrize("family", ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"])
def test_randomized_generator_properties(family):
    rng = np.random.default_rng(42)
    f = generate_oracle(family, rng=rng)

    if family == "constant":
        assert f.secret in (0, 1)
        assert len(np.unique(f.table)) == 1
    elif family == "balanced":
        assert f.secret is None
        assert np.sum(f.table) == 128
    elif family == "linear":
        a, b = f.secret
        assert 1 <= a <= 255 and b in (0, 1)
        for x in range(256):
            assert f(x) == (parity(a & x) ^ b)
    elif family == "grover":
        t = f.secret
        assert 0 <= t <= 255
        assert np.sum(f.table) == 1
        assert f(t) == 1
    elif family == "randfun":
        assert f.secret is None
        assert set(np.unique(f.table)).issubset({0, 1})
    elif family == "simon":
        s = f.secret
        assert 1 <= s <= 255
        assert len(np.unique(f.table)) == 128
        for x in range(256):
            assert f(x) == f(x ^ s)
    elif family == "simon2":
        H = [0] + f.secret
        assert len(f.secret) == 3
        assert len(np.unique(f.table)) == 64
        for x in range(256):
            for h in H:
                assert f(x) == f(x ^ h)
    elif family == "perm":
        assert f.secret is None
        assert len(np.unique(f.table)) == 256
