import sys
import numpy as np
from solver import solve
from simulator import OracleSimulator
from oracles import generate_oracle

FAMS = ["constant", "balanced", "linear", "grover", "randfun", "simon", "simon2", "perm"]
BUDGET = 60
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

for name, mis, flip in [("normal", 0.04, 0.02), ("stress", 0.08, 0.04)]:
    print(f"\n=== {name} noise, {TRIALS} trials/family ===")
    for fam in FAMS:
        fam_ok = sec_ok = viol = 0
        wrong = {}
        for t in range(TRIALS):
            f = generate_oracle(fam, rng=np.random.default_rng(1000 + t))
            sim = OracleSimulator(f, misfire_rate=mis, readout_flip_rate=flip, seed=5000 + t)
            try:
                pf, ps = solve(f.n, f.m, BUDGET, FAMS, sim.run)
            except Exception:
                viol += 1
                continue
            if sim.total_queries_used > BUDGET:
                viol += 1
            if pf == f.family:
                fam_ok += 1
                if f.secret is None or ps == f.secret:
                    sec_ok += 1
            else:
                wrong[pf] = wrong.get(pf, 0) + 1
        print(f"{fam:9s} family {fam_ok/TRIALS*100:5.1f}% | family+secret {sec_ok/TRIALS*100:5.1f}% | violations {viol} | confused as {wrong}")
