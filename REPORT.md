# Quantum Oracle Forensics: Technical Report
**QC@IITI x QryptoNox | IBM Qiskit Fall Fest 2026**

---

## Strategy and architecture

Our solver identifies sealed quantum black boxes $U_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$ ($n=8, m \in \{1, 8\}$) within 60 queries under gate misfires (4% normal, 8% stress) and readout flips (2% normal, 4% stress) using an adaptive pipeline:

1. **Readout Calibration (0 Queries)**: A zero-oracle circuit ($|0\rangle^{\otimes 8}$ measured directly) runs for 150 shots ($k = 0$ oracle gates $\implies 0$ queries). Measuring 1200 bits yields $\eta = \text{ones} / 1200 \in [0.005, 0.15]$, calibrating downstream likelihood models.
2. **Dimension Partitioning**: Branches on output width $m$: $m=8$ (`simon`, `simon2`, `perm`) vs $m=1$ (`constant`, `balanced`, `linear`, `grover`, `randfun`).
3. **Phase 1 ($m = 8$): Subspace Orthogonality**: A Simon circuit ($H^{\otimes 8}$–oracle–$H^{\otimes 8}$–measure) runs for 48 shots. Candidates $v \in \{1..255\}$ are scored via $y \cdot v \equiv 0 \pmod 2$. `simon` satisfies $y \cdot s = 0$ for hidden period $s$; `simon2` satisfies $y \cdot h = 0$ for all three non-zero elements of 2D subspace $H$; `perm` yields uniform outcomes across $\mathbb{F}_2^8$.
4. **Phase 1 ($m = 1$): Bernstein-Vazirani (BV) Diagnostic**: An 8-shot BV circuit (ancilla in $|-\rangle$) extracts Fourier signatures:
   - **Branch A (`linear`)**: If a non-zero vector $a$ peaks with count $\ge 3$, offset $b$ is recovered by querying $f(0)$ (3 shots). A basis check $f(2^k) = 1 \oplus b$ (2 shots) confirms $[a, b]$.
   - **Branch B (`constant` / `grover`)**: If state $0^8$ dominates ($\ge 4$ counts), querying $f(0)$ (3 shots) resolves the branch: if $f(0)=1$, testing $f(1)$ (3 shots) separates `constant(1)` from `grover(0)`. If $f(0)=0$, the solver runs up to three 1-shot Grover searches ($k \in \{8, 9\}$ iterations), checking candidate $t$ and its 8 Hamming-1 neighbors via direct queries. If unconfirmed, it defaults to `constant(0)`.
   - **Branch C (`balanced` vs `randfun`)**: For diffuse BV outcomes, remaining queries (52) evaluate distinct inputs classically via Log-Likelihood Ratio (LLR) hypothesis testing.

---

## Statistics and thresholds behind each decision

All thresholds in [`solver.py`](solver.py) were tuned empirically on our simulator to optimize accuracy under 4%–8% misfire and 2%–4% readout noise, rather than derived from theory:

- **Calibration Shots (`shots = 150`)**: Yields 1200 bit observations, bounding $\text{SE}(\hat{\eta}) \le 0.004$ at $\eta=0.02$ with 0 query cost.
- **Simon Allocation (`total_shots = 48`)**: While $n-1=7$ vectors suffice noiselessly, 48 shots provide $>6\times$ redundancy against misfires while conserving 12 queries (20% saving).
- **Simon Thresholds (`s2_thresh = 36`, `s1_thresh = 38`)**: For `perm`, expected orthogonal count in 48 shots is $24 \pm 3.46$. Threshold 38 sits $4.04\sigma$ above the mean. For `simon2`, requiring three closed vectors $h_1 \oplus h_2 = h_3$ in top 12 with $\min(\text{count}) \ge \max(36, \text{top\_score} - 6)$ absorbs misfire degradation.
- **BV Diagnostic Shots (`diag_shots = 8`)**: Minimal sample budget to detect linear peaks while reserving 52 queries for Branch C or 34 queries for Grover.
- **Linear Peak Count (`top_nonzero_cnt >= 3`)**: Under uniform noise, the chance of an identical non-zero string appearing $\ge 3$ times in 8 shots is $< 0.0002$.
- **Grover Iterations (`k_iter = 8` if $\eta \ge 0.03$ else `9`)**: Noiseless optimum is 12. However, misfire probability compounds as $(1 - p_{\text{mis}})^k$. At $p_{\text{mis}}=0.08$, fidelity after 12 iterations drops to $0.92^{12} \approx 0.36$. Truncating to 8 or 9 iterations ($0.92^8 \approx 0.51$) balances rotation angle against misfire decay.
- **Direct Query Verification (`shots = 2` or `3`)**: Majority voting suppresses 2%–4% readout flips. Grover candidate verification tests the candidate and all 8 Hamming-1 neighbors to tolerate single readout flips.
- **Branch C Constant Guard**: Observing $\le 10\%$ or $\ge 90\%$ ones across $\ge 20$ samples returns `constant(0)` or `constant(1)`.
- **LLR Decision Rule (`llr >= 0.35`)**: Evaluates noise-adjusted probabilities $p_{\text{bal}} = 0.5(1-p_{\text{mis}})(1-2\eta_c) + \eta_c$ and $p_{\text{ran}} = 0.3(1-p_{\text{mis}})(1-2\eta_c) + \eta_c$ with BV prior $\text{LLR}_{\text{prior}} = -0.65 \times \text{zero\_count}$. The 0.35 threshold balances classification errors empirically.

---

## Budget split per family

Theoretical query paths match empirical averages from [`logs/benchmarks_summary.json`](logs/benchmarks_summary.json) and [`logs/benchmarks_detailed.csv`](logs/benchmarks_detailed.csv):

- **`linear`**: 8 (BV) + 3 ($f(0)$) + 2 (verify) = 13 queries. Log mean: 13.00 (normal), 13.94 (stress). Saving: 78.33%.
- **`simon`**: 48 Simon shots. Log mean: 48.00 (normal), 48.00 (stress). Saving: 20.00%.
- **`simon2`**: 48 Simon shots. Log mean: 48.00 (normal), 48.00 (stress). Saving: 20.00%.
- **`perm`**: 48 Simon shots. Log mean: 48.00 (normal), 48.00 (stress). Saving: 20.00%.
- **`constant`**: 8 (BV) + 3 ($f(0)$) + 3 ($f(1)$) = 14 ($c=1$); or up to 60 ($c=0$). Log mean: 37.78 (normal), 35.70 (stress). Saving: 37.03%.
- **`grover`**: 8 (BV) + 3 ($f(0)$) + 9 (Grover) + 3 (verify) $\approx 23$, up to 60. Log mean: 34.18 (normal), 37.10 (stress). Saving: 43.03%.
- **`balanced`**: 8 (BV) + 52 (classical sampling) = 60 queries. Log mean: 60.00 (normal), 60.00 (stress). Saving: 0.00%.
- **`randfun`**: 8 (BV) + 52 (classical sampling) = 60 queries. Log mean: 59.08 (normal), 59.08 (stress). Saving: 1.53%.
- **Overall**: Weighted mean across 8 families: 43.51 queries (normal), 43.73 queries (stress). Mean query saving: 27.93%.

---

## Baselines and diagnostics

Evaluations across 400 instances per regime (50 per family) from [`logs/benchmarks_summary.json`](logs/benchmarks_summary.json) demonstrate substantial gains over required baselines:

- **Normal Noise**: Proposed Adaptive reaches **96.50%** accuracy [95% CI: 94.21%, 97.90%], Worst: **94.00%**, Saving: **27.93%**, Avg Queries: 43.51. Outperforms Baseline 1 (Classical-Only: 32.00%, saving 23.12%), Baseline 2 (Single-Algorithm: 26.50%), Baseline 3 (Random: 8.75%), and Baseline 4 (Non-Adaptive: 38.75%).
- **Stress Noise**: Proposed Adaptive achieves **88.50%** accuracy [95% CI: 85.00%, 91.27%], Worst: **76.00%**, Saving: **28.07%**, Avg Queries: 43.73. Outperforms Baseline 1 (25.75%), Baseline 2 (25.75%), Baseline 3 (12.25%), and Baseline 4 (34.50%).

**Confusion Matrix (50 instances/family from logs)**:
- *Normal*: `constant`: 49/50 (1 `grover`); `balanced`: 47/50 (3 `randfun`); `linear`: 50/50; `grover`: 47/50 (3 `constant`); `randfun`: 47/50 (1 `constant`, 1 `balanced`, 1 `grover`); `simon`: 49/50 (1 `simon2`); `simon2`: 48/50 (2 `simon`); `perm`: 50/50.
- *Stress*: `constant`: 46/50 (2 `linear`, 2 `grover`); `balanced`: 43/50 (7 `randfun`); `linear`: 49/50 (1 `randfun`); `grover`: 41/50 (9 `constant`); `randfun`: 46/50 (1 `constant`, 3 `balanced`); `simon`: 44/50 (1 `simon2`, 5 `perm`); `simon2`: 40/50 (10 `simon`); `perm`: 50/50.

**Accuracy vs Budget Cap**: Cap 20: **56.25%** (avg 16.98 queries); Cap 30: **79.25%** (avg 26.47); Cap 40: **90.25%** (avg 33.98); Cap 50: **95.25%** (avg 40.14); Cap 60: **95.00%** (avg 43.66). *(Budget cap analysis used an independent seed stream, explaining variance with the 96.50% normal benchmark).*

---

## Where a classical strategy is enough, and where quantum helps

1. **Where Classical Strategy is Enough**:
   - **Balanced vs Randfun**: Differentiating $p=0.5$ from $p=0.3$ is well handled by classical sampling. Baseline 1 achieves 94.0% on `balanced` and 86.0% on `randfun` under normal noise without superposition.
   - **Scalar Offsets and Verification**: Recovering $b = f(0)$ and confirming candidate bits require only 1 to 3 classical queries with majority voting.
2. **Where Quantum Helps**:
   - **Grover Search ($N=256$)**: Locating marked item $t$ classically requires $\mathbb{E}[Q] \approx 128.5$ queries. Within 60 queries, Baseline 1 achieves only **6.0%** (normal) and **2.0%** (stress). Quantum Grover search reaches **94.0%** (normal) and **76.0%** (stress) in ~34 queries.
   - **Simon Subspace Finding**: Finding collisions classically via the birthday paradox requires $\approx \sqrt{\pi \times 256 / 2} \approx 20$ queries per collision. Baseline 1 scores **0.0%** on `simon2` and **0.0%** on `perm`. Quantum Simon circuits solve both in 48 queries, reaching **98.0%** (`simon`) and **96.0%** (`simon2`).
   - **Linear Mask $a$**: Probing $a \in \mathbb{F}_2^8$ classically requires 8 basis queries vulnerable to readout corruption (Baseline 1: **0.0%**). Bernstein-Vazirani identifies $a$ in 8 quantum shots (**98.0%** accuracy).

---

## Where the solver fails

Analysis across [`logs/benchmarks_summary.json`](logs/benchmarks_summary.json) (50 trials/family) and `accuracy_check.py` (from accuracy_check.py, 100 trials/family) highlights key failure modes:

1. **`grover` Confused as `constant(0)`**: In logs, 3/50 (normal) and 9/50 (stress) were confused as `constant`. In `accuracy_check.py`, 5/100 (normal) and 16/100 (stress) failed. Multi-iteration Grover circuits suffer compounded misfire decay ($(1-0.08)^9 \approx 47\%$ misfire chance under stress). When amplification fails, the solver defaults to `constant(0)`.
2. **`simon2` Confused as `simon`**: In logs, 2/50 (normal) and 10/50 (stress) were confused as `simon`. In `accuracy_check.py`, 21/100 under stress failed. When noise drops one subspace vector below $s_2=36$, the solver detects only the dominant vector and outputs `simon`.
3. **`simon` Confused as `perm` under Stress**: In logs, 5/50 stress instances were misclassified as `perm`. In `accuracy_check.py`, 11/100 stress instances failed because noise flattened counts below $s_1=38$.
4. **`balanced` Confused as `randfun`**: In logs, 3/50 (normal) and 7/50 (stress) were confused as `randfun`. In `accuracy_check.py`, 13/100 (normal) and 15/100 (stress) failed due to finite binomial sampling variance ($\sigma \approx 3.6$ over 52 queries).

---

## Honest limitations

1. **Compounding Misfires in Grover Search**: Multi-iteration Grover circuits degrade under 8% gate misfires, capping stress accuracy at 76.0% (and 79.0% in 100-trial checks).
2. **Empirical Parameter Calibration**: Thresholds ($s_1=38$, $s_2=36$, $k_{\text{iter}}=8/9$, $\text{LLR}=0.35$) were tuned against our simulator and lack formal guarantees under arbitrary hardware noise.
3. **Binomial Sample Overlap**: A 52-query classical budget leaves statistical overlap between $\text{Binomial}(52, 0.5)$ and $\text{Binomial}(52, 0.3)$, limiting balanced/randfun separation to ~86%–94%.
4. **Simon2 Subspace Rigidity**: The algebraic closure test $h_1 \oplus h_2 = h_3$ requires all three vectors to appear in the top 12 ranked candidates, which severe noise can disrupt.
