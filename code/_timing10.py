"""N=10 checkpoint + per-fold simulation-cost probe.

Reports the wall-clock cost of the exact density-matrix simulations that the
scaling experiment (E4) needs at N=10, per fold factor, so the fold set can be
chosen against a known budget.  Folded circuits have no analytic shortcut, so
the cost grows ~linearly in the fold factor.  ``vm.sector_probs`` memoises its
results on disk, so the probe measures the *cold* cost on phases no experiment
has requested and reports the cache-hit cost separately.
"""
import os
import time

import numpy as np

import run_experiments as rx
import vaqem_lib as vl
import vaqem_methods as vm

FOLDS = (1, 3, 5)
N, NPHI, WORKERS = 10, 2, 1          # serial: gives the true per-phase cost

print("available_Ns:", rx.available_Ns(), flush=True)
m = rx.load_model(N)
print("N=%d theta %s curly %s" % (m["N"], m["theta"].shape,
                                   m["curly"].shape), flush=True)

# one raw density-matrix simulation, per noise kind (the cost E1 pays)
for kind, p in (("none", 0.0), ("depolarizing", 0.01), ("amplitude_damping", 0.01)):
    t0 = time.time()
    pr = vl.probs_dm(0.7, N, m["theta"], m["curly"], kind, p, fold=1,
                     readout_p=0.0)
    print("  probs_dm %-18s fold=1: %6.2f s   sum %.6f" %
          (kind, time.time() - t0, float(np.sum(pr))), flush=True)

# the folded sector simulations E4 needs.  ``vm.sector_probs`` memoises on the
# exact phase list, so the *cold* cost is measured on an off-grid phase set that
# no experiment has ever requested, and the cache-hit cost is then measured on
# the same set to show what a re-run pays.
CACHE = os.path.join(os.path.dirname(rx.OUT), "cache")


def n_cached():
    return len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0


phi = -0.77713 + np.linspace(0.0, 0.031, NPHI)    # deliberately off any grid
nz = {"kind": "depolarizing", "p": 0.01, "readout_p": 0.0}
print("cache entries before: %d" % n_cached(), flush=True)
tot = 0.0
per_fold = {}
for fold in FOLDS:
    t0 = time.time()
    _, pm = vm.sector_probs(phi, N, m["theta"], m["curly"], nz, fold, WORKERS)
    dt = time.time() - t0
    tot += dt
    per_fold[fold] = dt / NPHI
    print("  sector_probs fold=%d: %6.1fs for %d phases -> %6.2f s/phase "
          "(pm %s, rowsum %.6f)" % (fold, dt, NPHI, dt / NPHI,
                                    np.asarray(pm).shape,
                                    float(np.asarray(pm)[0].sum())), flush=True)
t0 = time.time()
for fold in FOLDS:
    vm.sector_probs(phi, N, m["theta"], m["curly"], nz, fold, WORKERS)
print("  same %d phases x folds %s again (cache hit): %.3fs total"
      % (NPHI, FOLDS, time.time() - t0), flush=True)
print("cache entries after: %d" % n_cached(), flush=True)
per_phase = tot / NPHI
print("folds %s cold total: %.1f s/phase (serial); fold=1 alone %.1f s/phase"
      % (FOLDS, per_phase, per_fold[1]), flush=True)
# E4's dataset needs folds (1,3,5) on n_cal + n_tst phases, while the oracle
# baseline needs fold=1 only, on n_oracle phases -- quote both budgets.
for label, n_phase, workers, pp in (
        ("E4 dataset  folds(1,3,5)", 26, 6, per_phase),
        ("E4 dataset  folds(1,3,5)", 50, 6, per_phase),
        ("oracle grid fold=1      ", 361, 12, per_fold[1])):
    print("  N=%d  %s  %3d phases / %2d workers: ~%5.1f min of simulation"
          % (N, label, n_phase, workers, pp * n_phase / workers / 60.0),
          flush=True)
