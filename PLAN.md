# Plan: Quantum Oracle Forensics Solver

## 1. System Architecture and Design Overview
The challenge requires identifying the hidden structure of an unknown black-box quantum oracle $U_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$ on $n=8$ input qubits and $m \in \{1, 8\}$ output qubits within a strict budget of at most 60 queries under misfire noise (4% normal, 8% stress) and readout bit-flip noise (2% normal, 4% stress).

The 8 families are partitioned by output dimension $m$:
- **$m=8$ subspace**: `simon`, `simon2`, `perm`
- **$m=1$ subspace**: `constant`, `balanced`, `linear`, `grover`, `randfun`

### Key Design Pillars:
1. **Zero-leakage, Zero-tolerance Budget Guard**:
   - Wrap `run` inside a strict budget controller.
   - Exact query accounting: $\text{cost} = \sum_{\text{oracle gates}} 1 \times \text{shots}$.
   - Never exceed `budget` (default 60, but adaptive to any specified integer budget).
2. **Noise-Calibrated Statistical Engine**:
   - Zero-oracle calibration circuits (0 queries) to estimate readout noise $\eta_{\text{readout}}$.
   - Bayesian / maximum likelihood estimation with mixture models accounting for misfire rate $p_{\text{misfire}}$ and readout flip $\eta$.
3. **Staged, Adaptive Sequential Decision Strategy**:
   - **Phase 0: Calibration & Free Split**: Read $m$ and measure readout noise with 0-query circuits.
   - **Phase 1 ($m=8$)**:
     - Simon circuit: $H^{\otimes n}$ on inputs, oracle with output in $|0\rangle^{\otimes m}$, $H^{\otimes n}$ on inputs, measure inputs.
     - Sequential acquisition: Run small batches of shots (e.g., 10-14 shots).
     - Noise-robust linear system solver over $\mathbb{F}_2$: Compute candidate orthogonal kernel / subspace likelihoods under readout noise and misfire.
     - Likelihood ratio test:
       - Single strong 1D orthogonal vector $s \implies$ `simon`.
       - Rank 2 subspace orthogonal to 2D subspace $H$ (yielding 3 nonzero elements in $H$) $\implies$ `simon2`.
       - Uniformly distributed samples without algebraic structure $\implies$ `perm`.
     - Targeted verification queries if margin is ambiguous.
   - **Phase 1 ($m=1$)**:
     - Deutsch-Jozsa / Bernstein-Vazirani diagnostic: Output qubit in $|-\rangle$, $H^{\otimes n}$ on inputs, 1 oracle, $H^{\otimes n}$ on inputs, measure inputs.
     - Sample 8-10 shots:
       - Sharp non-zero peak at $a \neq 0 \implies$ candidate `linear`. Verify $a$ and recover $b = f(0)$ using direct computational basis queries.
       - Dominated by $0^n$: Could be `constant(0)`, `constant(1)`, or `grover` (or misfire effect).
       - Spread over non-zero basis states: Could be `balanced` or `randfun`.
     - Targeted branching:
       - If peak at $0^n$: Run Grover search with optimal small-iteration steps ($k=8..10$ or adaptive test) + direct verification $f(t)$ to distinguish `grover` ($t$) vs `constant`.
       - Direct query at $x=0$: $f(0)=1$ immediately rules out `grover` ($t \neq 0$ or verify $t=0$) and separates $c=0$ vs $c=1$.
       - If spread over non-zero states (`balanced` vs `randfun`): Grover amplitude test or binomial counting of 1s using low-cost classical/quantum queries.
4. **Comprehensive Test & Benchmarking Harness**:
   - `simulator.py` + `oracles.py`: High-performance statevector simulator handling 8 to 20 qubits with exact misfire and readout noise modeling, verified against all public examples.
   - Unit tests suite: format validations, budget violations, noise robustness, seed reproducibility.
   - Benchmark suite: Full evaluation across 200+ instances per family for normal and stress noise, comparing against 4 mandatory baselines.
5. **Technical Deliverables**:
   - `solver.py`: Clean, standalone, zero external dependencies other than standard library, numpy, scipy, qiskit.
   - `REPORT.md` (and `REPORT.pdf`): Strict 3-page limit, mathematical decision rules, logs matching claims.
   - `AI_DISCLOSURE.md`: Full transparency on methodology and verification.
   - `README.md`: Reproduction commands, installation, test guide.
