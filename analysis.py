#!/usr/bin/env python3
"""
Reproducible analysis: entropy computing (Dirac-3) as stochastic entropic mirror descent.

DATA (not redistributed here — fetch from the authors' public release):
  git clone https://github.com/qci-github/eqc-studies
  cd eqc-studies/eqc-paradigm/eqc_vs_grad
  # unzip data/results_{0..4}.zip  -> results_500_sched_2_{0..4}.json
  # cost matrix: QPLIB_0018_OBJ.csv

Produces the three hardened results in HARDENED_RESULTS.txt.
Point DATA_DIR at the unzipped directory.
"""
import json, glob, os
import numpy as np

DATA_DIR = os.environ.get("EC_DATA_DIR", ".")  # dir with QPLIB_0018_OBJ.csv and results_500_sched_2_*.json

def load_cost():
    qp = np.loadtxt(os.path.join(DATA_DIR, "QPLIB_0018_OBJ.csv"), delimiter=",")
    c, J = qp[:, 0], qp[:, 1:]
    return c, J, (J + J.T)

def grad(c, JJ, x):     return c + JJ @ x
def md_step(c, JJ, x, lr):
    x = np.maximum(x, 1e-15); y = x * np.exp(-lr * grad(c, JJ, x)); return y / y.sum()
def sproj(x):
    n = len(x); u = np.sort(x)[::-1]; cs = np.cumsum(u)
    rho = np.nonzero(u - (cs - 1.0) / np.arange(1, n + 1) > 0)[0][-1]
    tau = (cs[rho] - 1.0) / (rho + 1); return np.maximum(x - tau, 0)
def pgd_step(c, JJ, x, lr): return sproj(x - lr * grad(c, JJ, x))

def result1_trajectory(files, c, JJ, tmax=80, nruns=40):
    """Device steps multiplicative vs Euclidean: PGD/MD one-step mismatch ratio."""
    ratios = []
    for fn in files:
        d = json.load(open(fn)); R = min(nruns, len(d["state_vector"]))
        def mism(step, lr):
            e = []
            for r in range(R):
                sv = np.array(d["state_vector"][r]); s = 0.0
                for t in range(tmax): s += np.abs(step(c, JJ, sv[t], lr) - sv[t + 1]).sum()
                e.append(s / tmax)
            return np.mean(e)
        bmd = min(mism(md_step, lr) for lr in [0.02, 0.05, 0.1, 0.2, 0.5])
        bpgd = min(mism(pgd_step, lr) for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
        ratios.append(bpgd / bmd)
    return np.mean(ratios), np.std(ratios)

def result2_noise(files, c, JJ, lr=0.05, nruns=30, t0=5, t1=150):
    """Noise scaling Var(noise_i) ~ x_i^beta, per file, with bootstrap CI."""
    out = []
    for fn in files:
        d = json.load(open(fn)); R = min(nruns, len(d["state_vector"]))
        xs, r2 = [], []
        for r in range(R):
            sv = np.array(d["state_vector"][r])
            for t in range(t0, t1):
                resid = sv[t + 1] - md_step(c, JJ, sv[t], lr)
                xs.append(sv[t]); r2.append(resid ** 2)
        xs = np.concatenate(xs); r2 = np.concatenate(r2)
        m = (xs > 1e-6) & (r2 > 0); lx, ly = np.log(xs[m]), np.log(r2[m])
        bins = np.linspace(lx.min(), lx.max(), 12); idx = np.digitize(lx, bins)
        bx, by = [], []
        for b in range(1, len(bins)):
            mm = idx == b
            if mm.sum() > 50: bx.append(lx[mm].mean()); by.append(ly[mm].mean())
        bx, by = np.array(bx), np.array(by)
        A = np.vstack([bx, np.ones_like(bx)]).T
        slope = np.linalg.lstsq(A, by, rcond=None)[0][0]
        out.append(slope)
    return np.mean(out), np.std(out), out

def make_instance(n, neg, seed):
    rng = np.random.default_rng(seed); A = rng.standard_normal((n, n)); S = (A + A.T) / 2
    w, V = np.linalg.eigh(S); w = w / np.abs(w).max(); w[:n // 2] = -neg * np.abs(w[:n // 2])
    return 0.3 * rng.standard_normal(n), V @ np.diag(w) @ V.T

def result3_advantage(n=30, T=600, nstart=20, ninst=15, levels=(1., 4., 10., 20.)):
    def en(c, J, x): return c @ x + x @ J @ x
    def run(step, c, JJ, x0, lr, T0):
        x = sproj(x0) if step is pgd_step else (np.maximum(x0, 1e-15) / np.maximum(x0, 1e-15).sum())
        for _ in range(T0): x = step(c, JJ, x, lr)
        return x
    rows = []
    for neg in levels:
        gaps = []
        for s in range(ninst):
            c, J = make_instance(n, neg, 200 + s); JJ = J + J.T
            starts = [np.random.default_rng(9000 + s * 40 + k).random(n) for k in range(nstart)]
            em = min(np.mean([en(c, J, run(md_step, c, JJ, x0, lr, T)) for x0 in starts]) for lr in [0.3,1,3,10])
            ep = min(np.mean([en(c, J, run(pgd_step, c, JJ, x0, lr, T)) for x0 in starts]) for lr in [1e-3,1e-2,3e-2,1e-1])
            gaps.append(ep - em)
        g = np.array(gaps); rows.append((neg, g.mean(), g.std() / np.sqrt(len(g))))
    return rows

if __name__ == "__main__":
    c, J, JJ = load_cost()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "results_500_sched_2_*.json")))
    print("RESULT 1 (trajectory):", result1_trajectory(files, c, JJ))
    print("RESULT 2 (noise beta):", result2_noise(files, c, JJ)[:2])
    print("RESULT 3 (advantage):")
    for row in result3_advantage(): print("   neg=%.1f gap=%.3f +/- %.3f" % row)
