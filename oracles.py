"""
oracles.py - Oracle implementations and generators for Quantum Oracle Forensics.

Supports all eight challenge families:
  1. constant:  m=1, f(x)=c, c in {0, 1}
  2. balanced:  m=1, exactly 128 ones, 128 zeros
  3. linear:    m=1, f(x)=a.x ^ b, a in {1..255}, b in {0, 1}
  4. grover:    m=1, f(t)=1 for exactly one t in {0..255}, else 0
  5. randfun:   m=1, f(x)=1 independently with prob 0.3
  6. simon:     m=8, f(x)=f(x ^ s), s in {1..255}, 2-to-1, random relabeling
  7. simon2:    m=8, f constant on cosets of 2D subspace H, 4-to-1, random relabeling
  8. perm:      m=8, random permutation of 0..255
"""

import numpy as np
from typing import Tuple, List, Optional, Any, Dict, Callable


class OracleFunction:
    """Represents a black-box function f: {0..255} -> {0..255} or {0, 1}."""

    def __init__(self, n: int, m: int, family: str, secret: Any, table: np.ndarray):
        self.n = n
        self.m = m
        self.family = family
        self.secret = secret
        self.table = np.asarray(table, dtype=np.uint16)
        assert len(self.table) == (1 << n), f"Table size must be 2^{n}"

    def __call__(self, x: int) -> int:
        return int(self.table[x])

    def evaluate_all(self) -> np.ndarray:
        return self.table.copy()


def parity(val: int) -> int:
    """Compute parity (popcount modulo 2) of an integer."""
    return bin(val).count("1") & 1


# ---------------------------------------------------------------------------
# Section 4 Public Reference Check Oracles
# ---------------------------------------------------------------------------

def make_public_linear() -> OracleFunction:
    """Public reference: linear a=178, b=1."""
    a, b = 178, 1
    table = np.array([parity(a & x) ^ b for x in range(256)], dtype=np.uint16)
    return OracleFunction(8, 1, "linear", [a, b], table)


def make_public_simon() -> OracleFunction:
    """Public reference: simon s=105, f(x) = min(x, x ^ s)."""
    s = 105
    table = np.array([min(x, x ^ s) for x in range(256)], dtype=np.uint16)
    return OracleFunction(8, 8, "simon", s, table)


def make_public_simon2() -> OracleFunction:
    """Public reference: simon2 H={0, 15, 49, 62}, f(x) = min over h in H of (x ^ h)."""
    H = {0, 15, 49, 62}
    table = np.array([min(x ^ h for h in H) for x in range(256)], dtype=np.uint16)
    secret = sorted([h for h in H if h != 0])
    return OracleFunction(8, 8, "simon2", secret, table)


def make_public_perm() -> OracleFunction:
    """Public reference: perm f(x) = (37 * x + 11) mod 256."""
    table = np.array([(37 * x + 11) % 256 for x in range(256)], dtype=np.uint16)
    return OracleFunction(8, 8, "perm", None, table)


def make_public_grover() -> OracleFunction:
    """Public reference: grover t=173."""
    t = 173
    table = np.zeros(256, dtype=np.uint16)
    table[t] = 1
    return OracleFunction(8, 1, "grover", t, table)


def make_public_constant() -> OracleFunction:
    """Public reference: constant c=1."""
    c = 1
    table = np.ones(256, dtype=np.uint16) * c
    return OracleFunction(8, 1, "constant", c, table)


# ---------------------------------------------------------------------------
# Randomized Oracle Generators (with random output relabeling)
# ---------------------------------------------------------------------------

def generate_constant(c: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    if rng is None:
        rng = np.random.default_rng()
    if c is None:
        c = int(rng.integers(0, 2))
    table = np.full(256, c, dtype=np.uint16)
    return OracleFunction(8, 1, "constant", c, table)


def generate_balanced(rng: Optional[np.random.Generator] = None) -> OracleFunction:
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(256)
    table = np.zeros(256, dtype=np.uint16)
    table[indices[:128]] = 1
    return OracleFunction(8, 1, "balanced", None, table)


def generate_linear(a: Optional[int] = None, b: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    if rng is None:
        rng = np.random.default_rng()
    if a is None:
        a = int(rng.integers(1, 256))
    if b is None:
        b = int(rng.integers(0, 2))
    table = np.array([parity(a & x) ^ b for x in range(256)], dtype=np.uint16)
    return OracleFunction(8, 1, "linear", [a, b], table)


def generate_grover(t: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    if rng is None:
        rng = np.random.default_rng()
    if t is None:
        t = int(rng.integers(0, 256))
    table = np.zeros(256, dtype=np.uint16)
    table[t] = 1
    return OracleFunction(8, 1, "grover", t, table)


def generate_randfun(p: float = 0.3, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    if rng is None:
        rng = np.random.default_rng()
    table = rng.binomial(1, p, size=256).astype(np.uint16)
    return OracleFunction(8, 1, "randfun", None, table)


def generate_simon(s: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    """Generate Simon oracle with random secret s and random output relabeling."""
    if rng is None:
        rng = np.random.default_rng()
    if s is None:
        s = int(rng.integers(1, 256))

    # Form 128 pairs {x, x ^ s}
    visited = set()
    pairs = []
    for x in range(256):
        if x not in visited:
            pairs.append((x, x ^ s))
            visited.add(x)
            visited.add(x ^ s)

    # Assign 128 distinct random 8-bit values
    output_values = rng.permutation(256)[:128]
    table = np.zeros(256, dtype=np.uint16)
    for (x1, x2), val in zip(pairs, output_values):
        table[x1] = val
        table[x2] = val

    return OracleFunction(8, 8, "simon", s, table)


def generate_simon2(H: Optional[List[int]] = None, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    """Generate Simon2 oracle with random 2D subspace H and random output relabeling."""
    if rng is None:
        rng = np.random.default_rng()
    if H is None:
        # Pick 2 linearly independent nonzero elements in GF(2)^8
        h1 = int(rng.integers(1, 256))
        h2 = int(rng.integers(1, 256))
        while h2 == h1 or h2 == 0:
            h2 = int(rng.integers(1, 256))
        h3 = h1 ^ h2
        H_set = {0, h1, h2, h3}
    else:
        H_set = set(H)
        assert len(H_set) == 4 and 0 in H_set, "H must be a 2D subspace containing 0 and 3 nonzero elements"

    nonzero_H = sorted([h for h in H_set if h != 0])

    # Form 64 cosets of H
    visited = set()
    cosets = []
    for x in range(256):
        if x not in visited:
            coset = [x ^ h for h in H_set]
            cosets.append(coset)
            visited.update(coset)

    # Assign 64 distinct random 8-bit values
    output_values = rng.permutation(256)[:64]
    table = np.zeros(256, dtype=np.uint16)
    for coset, val in zip(cosets, output_values):
        for x in coset:
            table[x] = val

    return OracleFunction(8, 8, "simon2", nonzero_H, table)


def generate_perm(rng: Optional[np.random.Generator] = None) -> OracleFunction:
    """Generate random permutation of 0..255."""
    if rng is None:
        rng = np.random.default_rng()
    table = rng.permutation(256).astype(np.uint16)
    return OracleFunction(8, 8, "perm", None, table)


def generate_oracle(family: str, rng: Optional[np.random.Generator] = None) -> OracleFunction:
    """Generate an oracle instance for the given family name."""
    generators = {
        "constant": generate_constant,
        "balanced": generate_balanced,
        "linear": generate_linear,
        "grover": generate_grover,
        "randfun": generate_randfun,
        "simon": generate_simon,
        "simon2": generate_simon2,
        "perm": generate_perm,
    }
    if family not in generators:
        raise ValueError(f"Unknown family: {family}")
    return generators[family](rng=rng)
