"""
bench_oracle_cost.py -- measure the *uncached* exact-simulation cost of the
known-noise-model oracle grid, so the resource comparison in the paper is a
measurement rather than an extrapolation from a cached run.

The D-VAQEM results themselves are cached on disk (``new_paper/cache``), so the
``sim_time_s`` recorded in the result files is ~0 for a re-run.  This script
times single-phase exact density-matrix simulations directly (no cache) and
reports

    t_sim(N)                one exact p_m(phi) evaluation, one core
    t_oracle(N)  = 361 t_sim / workers      the 'genie' baseline
    t_calib(N)   =  n_cal t_sim / workers   what D-VAQEM consumes

for the depolarising setting used in the scaling study, plus the number of
distribution entries each of the two needs (the data-volume comparison that
does not depend on our simulator at all).

Usage:  python bench_oracle_cost.py [--workers 6] [--n-cal 21] [--n-grid 361]
            [--Ns 4,6,8,10] [--out ../results/oracle_cost.json]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                          # noqa: E402
import json                                              # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402

import numpy as np                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vaqem_lib as vl                                   # noqa: E402

NOISE = ("depolarizing", 0.01)
MODELS = os.path.join(HERE, os.pardir, "models")
RES_LEGACY = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir,
                                         "revision_experiments", "results"))


def load_model(N, seed=0):
    for d in (MODELS, RES_LEGACY):
        p = os.path.join(d, f"vqcnni_N{N}_s{seed}.npz")
        if os.path.exists(p):
            return vl.load_vqcnni(p)
    raise FileNotFoundError(f"no checkpoint for N={N}")


def time_one(N, model, reps, fold=1):
    """Wall-clock seconds of one exact density-matrix p_m evaluation."""
    kind, p = NOISE
    phis = np.linspace(-np.pi, np.pi, reps + 1)[:reps] + 0.013
    ts = []
    for phi in phis:
        t0 = time.time()
        pr = vl.probs_dm(float(phi), N, model["theta"], model["curly"], kind, p,
                         fold=fold, readout_p=0.0)
        ts.append(time.time() - t0)
        assert np.isfinite(pr).all() and abs(pr.sum() - 1.0) < 1e-9
    return float(np.mean(ts)), float(np.min(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6,
                    help="process-pool size used by the experiment suite")
    ap.add_argument("--n-cal", type=int, default=21)
    ap.add_argument("--n-grid", type=int, default=361)
    ap.add_argument("--Ns", default="4,6,8,10")
    ap.add_argument("--reps", type=int, default=0,
                    help="simulations timed per N (default: 3, or 1 for N>=10)")
    ap.add_argument("--out", default=os.path.join(HERE, os.pardir, "results",
                                                  "oracle_cost.json"))
    a = ap.parse_args()
    Ns = [int(x) for x in a.Ns.split(",")]
    rows = []
    print(f"noise = {NOISE}, reps timed per N = "
          f"{a.reps or 'auto'}, workers = {a.workers}")
    print(f"{'N':>3}{'sector dim':>12}{'t_sim (s)':>12}{'t_oracle (s)':>14}"
          f"{'t_calib (s)':>13}{'oracle/calib':>14}{'data entries':>14}")
    for N in Ns:
        model = load_model(N)
        reps = a.reps or (1 if N >= 10 else 3)
        t, tmin, tmax = time_one(N, model, reps)
        t_or = a.n_grid * t / max(1, a.workers)
        t_cal = a.n_cal * t / max(1, a.workers)
        # data volume: distributions x outcomes.  The oracle needs the noisy
        # model on a fine grid; D-VAQEM needs n_cal calibration pairs, and a
        # full-space (2^N-outcome) map would need (2^N)^2 map parameters.
        entries_oracle = a.n_grid * (N + 1)
        entries_calib = a.n_cal * (N + 1)
        rows.append({"N": N, "sector_dim": N + 1, "full_dim": 2 ** N,
                     "process_tomography_dim": 4 ** N,
                     "linear_map_params_sector": (N + 1) ** 2,
                     "linear_map_params_full": 4 ** N,
                     "reps_timed": reps, "t_sim_s": t, "t_sim_min_s": tmin,
                     "t_sim_max_s": tmax, "workers": a.workers,
                     "n_grid": a.n_grid, "n_cal": a.n_cal,
                     "t_oracle_s": t_or, "t_calib_s": t_cal,
                     "oracle_over_calib": t_or / t_cal,
                     "oracle_data_entries": entries_oracle,
                     "calib_data_entries": entries_calib})
        print(f"{N:3d}{N + 1:12d}{t:12.3f}{t_or:14.1f}{t_cal:13.2f}"
              f"{t_or / t_cal:14.1f}{entries_oracle:14d}")
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
