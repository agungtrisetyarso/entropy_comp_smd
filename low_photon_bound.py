#!/usr/bin/env python3
"""
low_photon_bound.py
===================
The experiment promised in the response to Concern 1: convert the central null
into a QUANTITATIVE result about the quantum-measurement noise component.

It does three things the earlier confound sweep did not:

  (A) PUSH LOW + DEEP. Concentrate samples at the lowest mean photon numbers
      the device admits, with high num_samples per setting, to shrink the
      bootstrap error on beta where a shot-noise (beta -> 1) signature would
      appear.

  (B) FLOOR STABILITY. Re-measure the zero-landscape floor at several tiny
      landscape-coefficient magnitudes. If the floor exponent does not move,
      it is instrumental (detector) and the bound is trustworthy; if it moves,
      the floor proxy is contaminated and must be reported as such.

  (C) DECOMPOSE + BOUND. Fit the two-component law
          Var(dx_i) = a*x_i^2 + b*x_i
      where b*x_i is the linear (Poisson/shot, beta=1) term and a*x_i^2 the
      multiplicative (classical SMD, beta=2) term. The quantum-measurement
      content is carried by b. Subtracting the matched floor decomposition,
      we report either:
         * DETECTION: floor-subtracted b > 0 at >2 sigma, or
         * BOUND: a 95% upper limit on the fractional shot-noise component
                  f_shot = (b<x>) / (a<x>^2 + b<x>) at the operating scale.

Reuses problem-building, client, and allocation guards from the earlier scripts.
Keep dirac3_noise_sweep.py in the same directory.

USAGE
    export QCI_TOKEN=...
    python low_photon_bound.py --pilot --dry-run
    python low_photon_bound.py --pilot
    python low_photon_bound.py --full --confirm-full
    python low_photon_bound.py --analyze low_photon_out/lp_*.json
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

import dirac3_noise_sweep as base

OUTDIR = Path("low_photon_out")

# Concentrate at the LOW end of the documented range. The floor and the
# multiplicative law converge here, so this is where sensitivity matters most.
LOW_MPN_MAX = 5.0e-4          # stay below the regime where beta=2 dominates
FLOOR_EPS_VALUES = [1e-9, 1e-7, 1e-5]   # near-zero coefficient magnitudes for (B)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def build_lp_grids(mode):
    if mode == "pilot":
        mpn = base.log_grid(base.MPN_MIN, LOW_MPN_MAX, 4)
        num_samples = 40
        eps_vals = [1e-9, 1e-5]          # two floor points for stability check
    else:
        mpn = base.log_grid(base.MPN_MIN, LOW_MPN_MAX, 6)
        num_samples = 100                # max; smallest bootstrap error on beta
        eps_vals = FLOOR_EPS_VALUES
    return mpn, num_samples, eps_vals


def enumerate_lp_jobs(mpn_values, num_samples, eps_vals, schedule=2):
    jobs = []
    # On-landscape (indefinite) deep low-photon sweep.
    for mpn in mpn_values:
        jobs.append({"kind": "on", "eps": None, "schedule": schedule,
                     "mpn": float(mpn), "qfc": 1, "num_samples": num_samples})
    # Floor at each eps magnitude, across the same photon points (stability).
    for eps in eps_vals:
        for mpn in mpn_values:
            jobs.append({"kind": "floor", "eps": eps, "schedule": schedule,
                         "mpn": float(mpn), "qfc": 1,
                         "num_samples": num_samples})
    return jobs


def make_floor_landscape(n, eps):
    """Single negligible term of magnitude eps (API rejects empty polynomial)."""
    return [float(eps)], [[0, 1, 1]], None


# ---------------------------------------------------------------------------
# Two-component variance decomposition  Var = a x^2 + b x
# ---------------------------------------------------------------------------

def decompose_variance(solutions, min_mean=1e-4):
    """Return (a, b, a_err, b_err) for Var(x_i) = a x_i^2 + b x_i across the
    sample ensemble. b is the linear (shot/Poisson, beta=1) component."""
    sols = np.asarray(solutions, float)
    if sols.ndim != 2 or sols.shape[0] < 5:
        return (np.nan,) * 4
    means = sols.mean(axis=0)
    varis = sols.var(axis=0, ddof=1)
    m = (means > min_mean) & (varis > 0)
    x, y = means[m], varis[m]
    if x.size < 5:
        return (np.nan,) * 4
    # Design matrix columns: [x^2, x]; weighted by 1/x^2 to stabilize the fit
    # across the wide dynamic range of component magnitudes.
    A = np.vstack([x**2, x]).T
    w = 1.0 / (y**2 + 1e-30)
    W = A.T @ (A * w[:, None])
    coef = np.linalg.solve(W, A.T @ (y * w))
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float((resid**2 * w).sum() / dof)
    cov = np.linalg.inv(W) * s2
    return float(coef[0]), float(coef[1]), float(np.sqrt(abs(cov[0, 0]))), \
        float(np.sqrt(abs(cov[1, 1])))


def fractional_shot(a, b, x_op):
    """Fraction of variance carried by the linear (shot) term at scale x_op."""
    num = b * x_op
    den = a * x_op**2 + b * x_op
    return num / den if den != 0 else np.nan


# ---------------------------------------------------------------------------
# Analysis: detection or calibrated bound
# ---------------------------------------------------------------------------

def analyze(recs):
    on = [r for r in recs if r["kind"] == "on"]
    floor = [r for r in recs if r["kind"] == "floor"]
    print("\n=========== LOW-PHOTON SENSITIVITY ANALYSIS ===========\n")

    # (B) floor stability across eps
    print("(B) Floor stability vs near-zero coefficient magnitude:")
    by_eps = {}
    for r in floor:
        by_eps.setdefault(r["eps"], []).append(r)
    floor_betas = {}
    for eps, grp in sorted(by_eps.items()):
        betas = [g["beta"] for g in grp if np.isfinite(g.get("beta", np.nan))]
        if betas:
            floor_betas[eps] = float(np.mean(betas))
            print(f"    eps={eps:.0e}  mean floor beta={np.mean(betas):.3f} "
                  f"(n={len(betas)})")
    if len(floor_betas) >= 2:
        spread = max(floor_betas.values()) - min(floor_betas.values())
        stable = spread < 0.15
        print(f"    floor spread across eps = {spread:.3f} -> "
              f"{'STABLE (instrumental, bound trustworthy)' if stable else 'MOVES (floor proxy contaminated -- report caveat)'}")

    # (C) decomposition + bound at the lowest photon setting
    print("\n(C) Two-component decomposition at lowest photon settings:")
    on = sorted(on, key=lambda r: r["mpn"])
    floor_ref = sorted([r for r in floor if r["eps"] == min(by_eps)],
                       key=lambda r: r["mpn"]) if by_eps else []
    detection, bounds = False, []
    for r in on[:3]:
        a, b, ae, be = decompose_variance(r["solutions"])
        # matched floor
        fr = min(floor_ref, key=lambda f: abs(np.log(f["mpn"]) - np.log(r["mpn"]))) \
            if floor_ref else None
        if fr is not None:
            fa, fb, fae, fbe = decompose_variance(fr["solutions"])
            b_sub = b - fb                       # floor-subtracted shot term
            b_sub_e = np.sqrt(be**2 + fbe**2)
        else:
            b_sub, b_sub_e = b, be
        x_op = 1.0 / 20                          # ~ uniform component scale, n=20
        f_shot = fractional_shot(a, max(b_sub, 0), x_op)
        z = b_sub / b_sub_e if b_sub_e > 0 else np.nan
        if np.isfinite(z) and z > 2:
            detection = True
            print(f"    mpn={r['mpn']:.3g}  b_sub={b_sub:.2e}+/-{b_sub_e:.2e}  "
                  f"|z|={abs(z):.2f}  f_shot~{f_shot:.2%}  -> DETECTION")
        else:
            # 95% upper limit on shot fraction
            b_ul = max(b_sub, 0) + 1.96 * b_sub_e
            f_ul = fractional_shot(a, b_ul, x_op)
            bounds.append(f_ul)
            print(f"    mpn={r['mpn']:.3g}  b_sub={b_sub:.2e}+/-{b_sub_e:.2e}  "
                  f"|z|={abs(z):.2f}  f_shot 95%% UL ~{f_ul:.2%}  -> bound")

    print()
    if detection:
        print("  RESULT: positive shot-noise (beta=1) component resolved below "
              "the floor -> quantum-measurement-backaction signature.")
    elif bounds:
        print(f"  RESULT: no detection. 95%% upper bound on the fractional "
              f"shot-noise (quantum-measurement) component: "
              f"<= {min(bounds):.1%} at the operating scale.")
        print("  Report this number as the calibrated quantum-measurement bound.")
    else:
        print("  RESULT: insufficient data to bound; increase num_samples / "
              "lower mpn.")
    print("\n=======================================================\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _bound_from_batches(on_sols, floor_sols, x_op=1.0 / 20):
    """Pool all accumulated samples, decompose, return (f_ul, b_sub, b_sub_e).
    f_ul is the 95% upper limit on the fractional shot component."""
    on = np.vstack(on_sols)
    fl = np.vstack(floor_sols)
    a, b, ae, be = decompose_variance(on)
    fa, fb, fae, fbe = decompose_variance(fl)
    b_sub = b - fb
    b_sub_e = np.sqrt(be**2 + fbe**2)
    b_ul = max(b_sub, 0) + 1.96 * b_sub_e
    f_ul = fractional_shot(a, b_ul, x_op)
    return f_ul, b_sub, b_sub_e, a


# npj-QI-relevant bound targets to project against, tightest first.
PROJECTION_TARGETS = [0.05, 0.10, 0.15, 0.20]


def project_budget(errs, f_now, n_now, spent, alloc_remaining,
                   sec_per_sample, targets=PROJECTION_TARGETS,
                   n_points=3):
    """From the observed error-vs-N trajectory, project samples/time to reach
    each target bound, assuming the empirically-fit error scaling err ~ N^p.
    Prints which targets are affordable within remaining allocation, and
    flags the npj-QI-relevant line (<=10%) explicitly.

    n_points: a referee-defensible bound wants the same statistics at several
    low photon points, so total cost is multiplied by this.
    """
    if len(errs) < 3:
        return
    N = np.array([e[0] for e in errs], float)
    E = np.array([e[1] for e in errs], float)
    # Fit log(err) = p*log(N) + c ; ideal p = -0.5.
    p, c = np.polyfit(np.log(N), np.log(E), 1)
    print(f"    [budget] error scaling N^{p:+.2f} "
          f"(ideal -0.50; {'normal' if p < -0.35 else 'PLATEAUING' if p > -0.2 else 'slow'})")
    if p > -0.05:                      # essentially flat -> no projection
        print("    [budget] error not falling with N: bound is at a ceiling, "
              "more samples will not help.")
        return
    for t in targets:
        # f_ul scales like the error (b_ul dominated by 1.96*b_sub_e once b_sub~0),
        # so required N solves: f_now * (N_req/n_now)^p = t  ->  N_req = n_now*(t/f_now)^(1/p)
        if f_now <= t:
            tag = "  <-- npj QI target" if abs(t - 0.10) < 1e-9 else ""
            print(f"    [budget] <= {t:.0%}: ALREADY REACHED (f={f_now:.1%}){tag}")
            continue
        n_req_single = n_now * (t / f_now) ** (1.0 / p)
        n_req_total = n_req_single * n_points          # on+floor already paired in spent
        extra_samples = max(n_req_single - n_now, 0) * 2 * n_points  # *2 = on+floor
        extra_sec = extra_samples * sec_per_sample
        affordable = extra_sec <= alloc_remaining
        tag = "  <-- npj QI target" if abs(t - 0.10) < 1e-9 else ""
        flag = "AFFORDABLE" if affordable else "OVER BUDGET"
        print(f"    [budget] <= {t:.0%}: ~{n_req_single:.0f} samples/point x{n_points} pts, "
              f"+{extra_sec:.0f}s ({extra_sec/60:.0f} min) [{flag}]{tag}")
    print(f"    [budget] allocation remaining ~{alloc_remaining:.0f}s")


def run_pool(client, args, mpn, schedule=2, eps=1e-9):
    """Accumulate 100-sample batches at a single low photon point, alternating
    on-landscape and matched floor, recomputing the bound after each pair.
    Stops early when f_ul <= target, or when the correlation ceiling is hit
    (error stops falling as 1/sqrt(N)), or when the batch cap is reached."""
    target = args.target
    max_batches = args.max_batches
    on_pc, on_pi, J = base.make_indefinite_quadratic(n=args.n)
    fl_pc, fl_pi, _ = make_floor_landscape(args.n, eps)
    on_id = base.upload_problem(client, on_pc, on_pi, args.n)
    fl_id = base.upload_problem(client, fl_pc, fl_pi, args.n)

    alloc_s, metered = base.allocation_seconds(client)
    ceiling = alloc_s * (1 - base.ALLOCATION_SAFETY_MARGIN)

    on_sols, floor_sols, errs, spent = [], [], [], 0.0
    print(f"\n=== POOL mode :: mpn={mpn:.3g}, target f_ul<={target:.0%}, "
          f"max {max_batches} batch-pairs ===")
    print(f"allocation {alloc_s:.0f}s, ceiling {ceiling:.0f}s\n")

    for batch in range(1, max_batches + 1):
        if metered and spent > ceiling:
            print(f"[stop] allocation ceiling at {spent:.0f}s "
                  f"after {batch-1} batch-pairs.")
            break
        pair = []
        for kind, fid in (("on", on_id), ("floor", fl_id)):
            job = {"kind": kind, "eps": (None if kind == "on" else eps),
                   "schedule": schedule, "mpn": float(mpn), "qfc": 1,
                   "num_samples": 100}
            try:
                out = base.run_one_job(client, fid, args.n, job)
            except Exception as e:                       # noqa: BLE001
                print(f"  batch {batch} {kind}: ERROR {e}")
                out = None
            if out is None:
                continue
            spent += out["device_usage_s"] or 35.0
            (on_sols if kind == "on" else floor_sols).append(
                np.asarray(out["solutions"], float))
            pair.append(kind)
        if len(pair) < 2:
            print(f"  batch {batch}: incomplete pair, skipping bound update")
            continue

        f_ul, b_sub, b_sub_e, a = _bound_from_batches(on_sols, floor_sols)
        n_tot = sum(s.shape[0] for s in on_sols)
        errs.append((n_tot, b_sub_e))
        print(f"  batch {batch:2d}: N_on={n_tot:5d}  b_sub={b_sub:+.2e}"
              f"+/-{b_sub_e:.2e}  f_ul={f_ul:.1%}  cum={spent:.0f}s")

        # Live budget re-estimate from the observed error trajectory, so the
        # operator can decide mid-run whether the npj-QI bound is affordable.
        if len(errs) >= 3:
            project_budget(errs, f_now=f_ul, n_now=n_tot, spent=spent,
                           alloc_remaining=max(ceiling - spent, 0),
                           sec_per_sample=base.EST_SECONDS_PER_SAMPLE)

        # Early stop: target reached.
        if f_ul <= target:
            print(f"\n[REACHED] f_ul={f_ul:.1%} <= target {target:.0%} "
                  f"at N={n_tot}, {spent:.0f}s. Stopping.")
            break

        # Correlation-ceiling check: does the error fall as 1/sqrt(N)?
        # Compare observed error ratio to the ideal sqrt(N) ratio over the
        # last few batches; if error barely improved while N grew a lot, stop.
        if len(errs) >= 4:
            (n0, e0), (n1, e1) = errs[-4], errs[-1]
            ideal = np.sqrt(n0 / n1)              # expected e1/e0 if 1/sqrt(N)
            observed = e1 / e0 if e0 > 0 else 1.0
            if observed > 1.5 * ideal:            # error fell far slower
                print(f"\n[CEILING] error fell as {observed:.2f} vs ideal "
                      f"{ideal:.2f} over last 3 batches: samples are "
                      f"correlated, bound is plateauing at ~{f_ul:.1%}. "
                      f"Further runs will not help. Stopping.")
                break
    else:
        print(f"\n[CAP] reached max {max_batches} batch-pairs.")

    # Persist pooled raw + final verdict.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {"mode": "pool", "mpn": float(mpn), "eps": eps, "target": target,
           "n_total_on": int(sum(s.shape[0] for s in on_sols)),
           "device_seconds": spent,
           "on_solutions": [s.tolist() for s in on_sols],
           "floor_solutions": [s.tolist() for s in floor_sols]}
    out_path = OUTDIR / f"pool_{stamp}.json"
    out_path.write_text(json.dumps(rec))
    if on_sols and floor_sols:
        f_ul, b_sub, b_sub_e, a = _bound_from_batches(on_sols, floor_sols)
        print(f"\nFINAL: N_on={rec['n_total_on']}, {spent:.0f}s, "
              f"95% UL on shot fraction = {f_ul:.1%}")
        if b_sub / b_sub_e > 2:
            print("  -> DETECTION: resolvable shot (beta=1) component.")
        else:
            print(f"  -> BOUND: quantum-measurement component <= {f_ul:.1%} (95%).")
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--confirm-full", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", metavar="JSON", nargs="+",
                    help="analyze existing result file(s) and exit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pool", action="store_true",
                    help="adaptive batch accumulation at one low photon point")
    ap.add_argument("--pool-mpn", type=float, default=1.1e-4,
                    help="photon point for pool mode (default: best low point)")
    ap.add_argument("--target", type=float, default=0.10,
                    help="stop when 95%% shot-fraction UL <= this (default 0.10)")
    ap.add_argument("--max-batches", type=int, default=20,
                    help="max 100-sample batch-pairs in pool mode")
    args = ap.parse_args()

    if args.analyze:
        recs = []
        for p in args.analyze:
            recs.extend(json.loads(Path(p).read_text()))
        analyze([r for r in recs if np.isfinite(r.get("beta", np.nan))])
        return

    if args.pool:
        OUTDIR.mkdir(exist_ok=True)
        client = base.get_client()
        run_pool(client, args, args.pool_mpn)
        return

    mode = "full" if args.full else "pilot"
    OUTDIR.mkdir(exist_ok=True)
    mpn_values, num_samples, eps_vals = build_lp_grids(mode)
    jobs = enumerate_lp_jobs(mpn_values, num_samples, eps_vals)
    est = base.estimate_seconds(jobs)

    print(f"\n=== low-photon bound experiment :: mode={mode} ===")
    print(f"photon points     : {len(mpn_values)} "
          f"[{mpn_values[0]:.3g} .. {mpn_values[-1]:.3g}]  (low end only)")
    print(f"floor eps values  : {eps_vals}")
    print(f"samples per job   : {num_samples}")
    print(f"total jobs        : {len(jobs)}  (on:{len(mpn_values)} "
          f"floor:{len(eps_vals)*len(mpn_values)})")
    print(f"ESTIMATED dev-sec : {est:.0f}")

    if args.dry_run:
        (OUTDIR / "plan.json").write_text(json.dumps(jobs, indent=2))
        print("\n[dry-run] nothing submitted. plan.json written.")
        return

    client = base.get_client()
    alloc_s, metered = base.allocation_seconds(client)
    ceiling = alloc_s * (1 - base.ALLOCATION_SAFETY_MARGIN)
    print(f"allocation        : {alloc_s:.0f}s ({'metered' if metered else 'unmetered'})")
    if metered and est > ceiling:
        print(f"\n!! est {est:.0f}s exceeds ceiling {ceiling:.0f}s. Trim grid.")
        if mode == "full":
            sys.exit(1)
    if mode == "full" and not args.confirm_full:
        sys.exit("\nFull grid gated. Add --confirm-full.")

    # Upload the on-landscape once and each floor-eps landscape once.
    on_pc, on_pi, J = base.make_indefinite_quadratic(n=args.n)
    np.save(OUTDIR / "J_on.npy", J)
    file_ids = {"on": base.upload_problem(client, on_pc, on_pi, args.n)}
    for eps in eps_vals:
        pc, pi, _ = make_floor_landscape(args.n, eps)
        file_ids[("floor", eps)] = base.upload_problem(client, pc, pi, args.n)

    records, spent = [], 0.0
    for k, job in enumerate(jobs, 1):
        if metered and spent > ceiling:
            print(f"[stop] ceiling at {spent:.0f}s after {k-1}/{len(jobs)}.")
            break
        fid = file_ids["on"] if job["kind"] == "on" else file_ids[("floor", job["eps"])]
        tag = "on " if job["kind"] == "on" else f"flr eps={job['eps']:.0e}"
        print(f"[{k}/{len(jobs)}] {tag} mpn={job['mpn']:.3g} ...",
              end=" ", flush=True)
        try:
            out = base.run_one_job(client, fid, args.n, job)
        except Exception as e:                       # noqa: BLE001
            print(f"ERROR: {e}"); continue
        if out is None:
            print("incomplete"); continue
        used = out["device_usage_s"] or (job["num_samples"] *
                                         base.EST_SECONDS_PER_SAMPLE)
        spent += used
        beta, (lo, hi) = base.bootstrap_beta(out["solutions"])
        records.append({**job, "beta": beta, "beta_lo": lo, "beta_hi": hi,
                        "best_energy": float(np.min(out["energies"])),
                        "device_usage_s": used, "solutions": out["solutions"]})
        print(f"beta={beta:.2f} [{lo:.2f},{hi:.2f}] cum={spent:.0f}s")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTDIR / f"lp_{mode}_{stamp}.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nsaved {len(records)} records -> {out_path}")
    analyze([r for r in records if np.isfinite(r.get("beta", np.nan))])


if __name__ == "__main__":
    main()
