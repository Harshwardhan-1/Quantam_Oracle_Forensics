# AI-Use Disclosure

**Quantum Oracle Forensics**  
*QC@IITI x QryptoNox | IBM Qiskit Fall Fest 2026*

---

### Team Information
- **Team Name**: TODO: [Insert Team Name]
- **Team Members**: TODO: [Insert Team Member Names and Emails]

---

### AI Assistance Declaration
In accordance with Section 10 ("Mathematical integrity and AI use") and Section 12 ("Deliverables") of the challenge specification, we declare the use of AI assistance:
- **Tools Used**: Google Gemini and the Antigravity agentic coding environment.
- **Scope of Assistance**: AI assistants were utilized for brainstorming architectural pipelines, debugging noisy simulation accounting, verifying modular Qiskit 2.x gate constructs, generating baseline implementations, and drafting technical documentation.

---

### Human Independent Verification
All logic, implementations, and empirical results were independently verified on local hardware (macOS, Apple Silicon):
1. **Automated Test Suite**: Executed `pytest -v` across all 46 unit and integration tests (`tests/test_oracles.py`, `tests/test_simulator.py`, `tests/test_solver.py`), verifying 46/46 passed in 0.48s.
2. **Local Accuracy & Stress Validation**: Executed `python accuracy_check.py 100` across 800 trials under normal (4% misfire, 2% readout flip) and stress (8% misfire, 4% readout flip) noise.
3. **Reproducible Benchmarking**: Verified execution of `benchmarks/run_benchmarks.py`, validating query accounting, runtime constraints, and confidence intervals across 400 instances per noise regime.

---

### Algorithmic Integrity & Code Audit
As mandated by Section 10, all solver decisions originate strictly and exclusively from query observations returned by the judges' `run()` callback. A line-by-line inspection of [`solver.py`](solver.py) confirms:
- **No Oracle Inspection**: The solver never reads, imports, or accesses oracle tables, hidden attributes, or simulator state.
- **No Hard-Coded Answers or Seed Lookups**: Secrets are dynamically reconstructed from physical measurement counts.
- **Disclosure of Fixed PRNG Seed**: In `solve_m1`, the solver selects classical evaluation inputs using `rng = np.random.default_rng(total_diag + 42)`. The seed is derived from the BV diagnostic shot total (`total_diag = 8`) plus 42 ($\text{seed} = 50$). This selects a fixed, deterministic sequence of sample inputs to test. It does not inspect or leak oracle data, but is disclosed here in full transparency.
