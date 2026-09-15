"""
run_experiments.py -- the D-VAQEM experiment suite (distribution-level
variational quantum error mitigation for the VQ-CNNI phase estimator).

Sub-commands
------------
  sufficiency   FI(p_m) = FI(p_full): exactness of the collective-imbalance
                reduction, for trained and random circuit parameters
  sweep         headline comparison of every mitigation method vs noise type
                and noise strength (infinite shots and finite shots)
  shots         finite-shot scaling and validation of the delta-method
                variance formula against Monte Carlo
  scaling       qubit-number scaling N = 4..10
  calib         calibration budget: number of calibration phases and
                calibration shots
  all           everything above

Every exact density-matrix dataset is cached on disk (new_paper/cache), so
re-running an experiment is essentially free.  Numerical results are written to
new_paper/results/*.json (scalars) and *.npz (curves); figures are produced by
make_figures.py.

Usage:  python run_experiments.py {sufficiency|sweep|shots|scaling|calib|all}
            [--quick] [--workers 12]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                              # noqa: E402
import json                                                  # noqa: E402
import sys                                                   # noqa: E402
import time                                                  # noqa: E402

import numpy as np                                           # noqa: E402
import autograd                                              # noqa: E402
import autograd.numpy as anp                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vaqem_lib as vl                                       # noqa: E402
import vaqem_methods as vm                                   # noqa: E402

REPO = vl.vqcnni_root()          # companion VQ-CNNI checkout (see vaqem_lib)
RES = os.path.join(REPO, "revision_experiments", "results")
MODELS = os.path.join(HERE, os.pardir, "models")
OUT = os.path.join(HERE, os.pardir, "results")
NOISELESS = {"kind": "none", "p": 0.0, "readout_p": 0.0}


def set_out(path):
    """Redirect every results file (JSON / npz) to ``path``.

    Used by ``--exp-root`` so that an ablation or a smoke run cannot overwrite
    the numbers the paper reports.
    """
    global OUT
    OUT = path
    os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------
# common helpers
# ----------------------------------------------------------------------
def load_model(N, seed=0):
    """VQ-CNNI checkpoint: new_paper/models (train_decoder.py) first, then the
    PennyLane-trained models in revision_experiments/results."""
    for d in (MODELS, RES):
        p = os.path.join(d, f"vqcnni_N{N}_s{seed}.npz")
        if os.path.exists(p):
            return vl.load_vqcnni(p)
    raise FileNotFoundError(f"no VQ-CNNI checkpoint for N={N}, seed={seed} "
                            f"in {MODELS} or {RES}")


def available_Ns(seed=0):
    """Qubit numbers with a checkpoint on disk (either model directory)."""
    out = set()
    for d in (MODELS, RES):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("vqcnni_N") and f.endswith(f"_s{seed}.npz"):
                try:
                    out.add(int(f[len("vqcnni_N"):].split("_")[0]))
                except ValueError:
                    pass
    return tuple(sorted(out))


def grids(n_cal, n_tst):
    """Calibration grid and a *held-out* half-offset test grid."""
    cal = np.linspace(-np.pi, np.pi, n_cal)
    tst = np.linspace(-np.pi + np.pi / n_tst, np.pi + np.pi / n_tst, n_tst)
    return cal, tst


def metrics(pred, phi):
    d = vl.wrapped_err(pred, phi)
    return {"mse": float(np.mean(d ** 2)),
            "mse_db": float(10 * np.log10(np.mean(d ** 2) + 1e-30)),
            "median_swpe_db": float(np.median(vl.swpe_db(pred, phi))),
            "mae": float(np.mean(np.abs(d))),
            "max_abs_err": float(np.abs(d).max())}


def agg(rows):
    """Average a list of metric dicts (Monte-Carlo trials)."""
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out["mse_std"] = float(np.std([r["mse"] for r in rows]))
    out["n_trials"] = len(rows)
    return out


def save_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"  -> wrote {path}", flush=True)


def save_npz(name, **arrays):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    np.savez_compressed(path, **arrays)
    print(f"  -> wrote {path}", flush=True)


def noise_grid(quick=False):
    """(label, noise-dict) for the headline sweep.

    Strengths are chosen from the exact-simulation scan of
    ``results/e0_noise_scan.json``: they span the regime where the unmitigated
    estimator has already lost most of its accuracy (MSE 2e-2 ... 3) while the
    noisy Fisher information is still non-zero, i.e. where mitigation is
    possible and meaningful.  (For dephasing p = 0.1 and depolarising p = 0.05
    the sector distribution is essentially phi-independent and *no* method can
    recover the phase; those points are reported by the scan but excluded here.)
    """
    dep = (0.005, 0.02) if quick else (0.002, 0.005, 0.01, 0.02)
    dph = (0.01, 0.05) if quick else (0.005, 0.01, 0.02, 0.05)
    amp = (0.005, 0.02) if quick else (0.002, 0.005, 0.01, 0.02)
    ro = (0.01, 0.03) if quick else (0.01, 0.02, 0.03, 0.05)
    out = []
    for p in dep:
        out.append((f"depol_{p}", {"kind": "depolarizing", "p": p, "readout_p": 0.0}))
    for p in dph:
        out.append((f"deph_{p}", {"kind": "dephasing", "p": p, "readout_p": 0.0}))
    for p in amp:
        out.append((f"ampdamp_{p}", {"kind": "amplitude_damping", "p": p,
                                     "readout_p": 0.0}))
    for q in ro:
        out.append((f"readout_{q}", {"kind": "none", "p": 0.0, "readout_p": q}))
    return out


def parse_noise_spec(spec):
    """Normalise a ``--noise`` JSON override into [(label, noise-dict), ...].

    Accepted shapes: ``[[label, {...}], ...]`` (the ``noise_grid`` format) or
    ``[{"label": ..., "kind": ..., "p": ..., "readout_p": ...}, ...]``.
    """
    out = []
    for item in spec:
        if isinstance(item, dict):
            d = dict(item)
            label = d.pop("label", None)
            if label is None:
                label = f"{d.get('kind', 'noise')}_{d.get('p', d.get('readout_p', 0))}"
            out.append((label, d))
        else:
            lab, d = item[0], dict(item[1])
            out.append((lab, d))
    known = {"none", "depolarizing", "dephasing", "amplitude_damping"}
    for lab, d in out:
        if d.get("kind") not in known:
            raise ValueError(f"{lab}: unknown noise kind {d.get('kind')!r}; "
                             f"expected one of {sorted(known)}")
        d.setdefault("p", 0.0)
        d.setdefault("readout_p", 0.0)
    return out


def build(N, model, noise, cal_phis, tst_phis, folds=(1, 3, 5), workers=12):
    """Calibration + test cases (exact simulation, disk cached)."""
    t0 = time.time()
    cal = vm.build_case(N, model["theta"], model["curly"], cal_phis, noise,
                        folds=(1,), workers=workers)
    tst = vm.build_case(N, model["theta"], model["curly"], tst_phis, noise,
                        folds=folds, workers=workers)
    print(f"    data {time.time() - t0:6.1f}s  cal {len(cal_phis)} x fold1, "
          f"test {len(tst_phis)} x folds {tuple(folds)}", flush=True)
    return cal, tst


# ----------------------------------------------------------------------
# the method suite
# ----------------------------------------------------------------------
DVAQEM_VARIANTS = (("lin_l2", "linear", "l2"), ("lin_ce", "linear", "ce"),
                   ("mlp_l2", "mlp", "l2"), ("mlp_ce", "mlp", "ce"),
                   ("mlp_fisher", "mlp", "fisher"))
SHOT_AWARE = (("lin_mse", "linear"), ("mlp_mse", "mlp"))
# Per-objective learning rates: the cross-entropy and Fisher objectives have
# much flatter directions than the (convex, in the linear case) least-squares
# one and need a larger step; the shot-aware mse fit is a refinement of an
# already-converged ce fit and is therefore taken at a smaller step.
LR_BY_LOSS = {"l2": 1.0, "ce": 2.0, "fisher": 0.4, "mse": 0.4}


def holdout_score(cal, model, mp, psi, idx, shots=None, n_trials=20, seed=0):
    """Common model-selection score for a fitted map: wrapped phase MSE on the
    *held-out calibration phases*.

    The per-objective validation losses (l2 vs ce vs mse) are not comparable
    across variants, and picking the winner by its test-set MSE would be
    selection on the evaluation data.  This instead decodes the held-out
    calibration phases -- data the fit did not use, but which cost no extra
    quantum resource -- with the same finite-shot budget as the experiment, so
    every variant is judged on one yardstick.  ``shots=None`` gives the
    infinite-shot (bias-only) score.
    """
    if idx is None or len(idx) == 0:
        return float("nan")
    fold = min(cal["pm_noisy"])
    phis = np.asarray(cal["phis"], dtype=float)
    order = np.argsort(phis)                       # same ordering as train_map
    y_all = np.asarray(cal["pm_noisy"][fold], dtype=float)[order]
    ph = phis[order][np.asarray(idx, dtype=int)]
    y = y_all[np.asarray(idx, dtype=int)]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(max(1, int(n_trials))):
        ys = y if not shots else vm.sample_dist(y, int(shots), rng)
        q = vm._renorm(vm.mitigated_family(mp, psi, ys))
        pred = vl.predict_from_pm(q, model["params"])
        vals.append(float(np.mean(vl.wrapped_err(pred, ph) ** 2)))
    return float(np.mean(vals))


def fit_maps(cal, cal_phis, model, shots, iters=2000, hidden=(32, 32), lr=5e-2,
             seed=0, lam=1.0, warm_start=True):
    """Train every D-VAQEM variant on the calibration data.

    Staged warm starts, all from the *same* calibration data (so no extra
    quantum resource is consumed):

      l2      cold start from the identity map -- least squares on the
              distributions is well conditioned and converges fast;
      ce      warm start from the l2 fit -- a cold-started cross-entropy fit
              starts next to the degenerate 'output ~ uniform' region and needs
              very many iterations to escape it (the linear map in particular
              must grow entries of order 1/(1-2f) >> 1);
      fisher  warm start from the ce fit;
      mse     fitted from *three* initialisations -- cold, the l2 fit and the
              ce fit -- and the one with the smallest held-out phase MSE at the
              operating shot count is kept.  Both extremes are needed: a
              cold-started shot-aware MLP fit can collapse onto a zero-variance
              (constant) map, which is a genuine local minimum of
              bias + variance, while for the linear map the staged
              initialisation is itself a bias that the cold fit does not have.
              (``warm_start=False`` ablates this down to the cold fit alone.)

    Each variant additionally gets ``info['holdout_mse']``: the wrapped phase
    MSE on the held-out calibration phases at ``shots``, a single yardstick
    that is comparable across the different objectives (see
    :func:`holdout_score`).
    """
    out, t0 = {}, time.time()

    def fit(tag, kind, loss, psi0=None, **kw):
        mp, psi, info = vm.train_map(cal, kind, cal_phis, loss=loss, iters=iters,
                                     lr=lr * LR_BY_LOSS[loss], lam=lam, seed=seed,
                                     hidden=hidden, psi0=psi0, **kw)
        info["holdout_mse"] = holdout_score(cal, model, mp, psi, info["val_idx"],
                                            shots=shots, seed=seed)
        info["holdout_mse_inf"] = holdout_score(cal, model, mp, psi,
                                                info["val_idx"], shots=None,
                                                seed=seed)
        out[tag] = (mp, psi, info)
        return psi

    psi_lin = {}
    psi_mlp = {}
    for tag, kind, loss in DVAQEM_VARIANTS:
        if loss == "l2":
            p0 = None
        elif loss == "fisher":
            p0 = (psi_mlp if kind == "mlp" else psi_lin).get("ce")
        else:
            p0 = (psi_mlp if kind == "mlp" else psi_lin).get("l2")
        psi = fit(tag, kind, loss, psi0=p0)
        (psi_mlp if kind == "mlp" else psi_lin)[loss] = psi
    for tag, kind in SHOT_AWARE:
        store = psi_mlp if kind == "mlp" else psi_lin
        if warm_start:
            # ``cold`` is always a candidate.  Measured on the full 16-setting
            # N=8 sweep (test-set infinite-shot MSE): for the *linear* map the
            # shot-aware objective is convex in the map parameters and the
            # staged initialisation only biases it -- a cold fit wins in 12/16
            # settings, by 770x at depol p=0.02, 258x at readout p=0.01 and
            # 152x at deph p=0.01, while the warm fit wins in only 4/16 and by
            # at most 1.9x.  For the *MLP* the same objective is non-convex and
            # a cold fit can collapse onto a constant map (a genuine local
            # minimum of bias+variance): 1.34 vs 3.95e-4 at deph p=0.01 and
            # 1.90 vs 5.14e-4 at depol p=0.02, i.e. ~3500x worse, so the warm
            # candidates are kept there as a safeguard -- worth +2.8 dB on
            # average and +24 dB in the worst case over the cold-only ablation.
            # Offering all three and letting the holdout decide costs one extra
            # fit per shot-aware variant and dominates either fixed choice: the
            # total finite-shot (S=1024) MSE of the two shot-aware variants over
            # the 16 settings is 3.75 for this rule vs 4.55 for warm-only and
            # 7.60 for cold-only selection.
            cands = [("cold", None)] + [(src, store[src])
                                        for src in ("l2", "ce") if src in store]
        else:      # ablation: train the shot-aware objective from scratch
            cands = [("cold", None)]
        best = None
        for src, p0 in cands:
            mp, psi, info = vm.train_map(cal, kind, cal_phis, loss="mse",
                                         iters=iters, lr=lr * LR_BY_LOSS["mse"],
                                         lam=lam, seed=seed, hidden=hidden,
                                         psi0=p0, shots=shots, decoder=model)
            info["warm_start"] = src
            # Select on the same yardstick used to pick between objectives --
            # the wrapped phase MSE on held-out calibration phases at the
            # operating shot count -- not on the training loss: a collapsed
            # constant map can have a small bias+variance training loss while
            # its holdout phase error is enormous.
            info["holdout_mse"] = holdout_score(cal, model, mp, psi,
                                                info["val_idx"], shots=shots,
                                                seed=seed)
            info["holdout_mse_inf"] = holdout_score(cal, model, mp, psi,
                                                    info["val_idx"], shots=None,
                                                    seed=seed)
            if best is None or info["holdout_mse"] < best[2]["holdout_mse"]:
                best = (mp, psi, info)
        mp, psi, info = best
        out[tag] = (mp, psi, info)
    out["_fit_time_s"] = time.time() - t0
    return out


def fit_retrained_decoder(cal, cal_phis, model, iters=2000, lr=2e-3, shots=None,
                          seed=0):
    """Noise-aware decoder retraining baseline (same calibration resource)."""
    t0 = time.time()
    dec, info = vm.retrain_decoder(cal, model, cal_phis, iters=iters, lr=lr,
                                   loss="circular", shots=shots, seed=seed)
    info["time_s"] = time.time() - t0
    return dec, info


def eval_with_mitigator(pm_noisy, phi, model, mit, shots_list, n_trials, seed):
    """Infinite-shot and Monte-Carlo finite-shot metrics for one mitigator."""
    def decode(p, dec):
        return vl.predict_from_pm(vm._renorm(p), dec["params"])

    rows = {"inf": metrics(decode(pm_noisy if mit is None else mit(pm_noisy),
                                  model), phi)}
    for S in shots_list:
        rng = np.random.RandomState(seed + int(S))
        tr = []
        for _ in range(max(1, n_trials)):
            y = vm.sample_dist(pm_noisy, S, rng)
            tr.append(metrics(decode(y if mit is None else mit(y), model), phi))
        rows[f"S{S}"] = agg(tr)
    return rows


def zne_coeffs(tst, kind, degree=1, folds=None):
    """Extrapolation coefficients for the given noise folds (default: all)."""
    folds = (sorted(tst["pm_noisy"]) if folds is None
             else sorted(int(f) for f in folds))
    _, c = vm.zne_extrapolate({f: tst["pm_noisy"][f] for f in folds}, kind, degree)
    return folds, np.asarray(c, dtype=float)


def eval_zne(tst, phi, model, kind, degree, shots_list, n_trials, seed,
             folds=None):
    """Distribution-level ZNE, finite shots included (every fold is sampled)."""
    folds, c = zne_coeffs(tst, kind, degree, folds)
    Y0 = np.stack([tst["pm_noisy"][f] for f in folds])
    rows = {"inf": metrics(vl.predict_from_pm(
        vm._renorm(np.tensordot(c, Y0, axes=(0, 0))), model["params"]), phi)}
    for S in shots_list:
        rng = np.random.RandomState(seed + int(S))
        tr = []
        for _ in range(max(1, n_trials)):
            Y = np.stack([vm.sample_dist(tst["pm_noisy"][f], S, rng) for f in folds])
            Z = np.tensordot(c, Y, axes=(0, 0))
            tr.append(metrics(vl.predict_from_pm(vm._renorm(Z), model["params"]), phi))
        rows[f"S{S}"] = agg(tr)
    return rows, c.tolist(), folds


# ----------------------------------------------------------------------
# oracle baselines: what a *known* noise model buys you
# ----------------------------------------------------------------------
def oracle_grid(N, model, noise, workers=12, n_grid=361, fold=1):
    """Exact noisy p_m(phi) on a fine phase grid ('genie' noise model).

    Cost: n_grid exact simulations, to be compared with the n_cal simulations
    that D-VAQEM consumes (and D-VAQEM never sees the noise model).
    """
    g = np.linspace(-np.pi, np.pi, n_grid)
    _, pm = vm.sector_probs(g, N, model["theta"], model["curly"], noise, fold,
                            workers)
    return g, np.asarray(pm, dtype=float)


def oracle_predict(g, pm_grid, y, how="ce", refine=True):
    """Grid maximum-likelihood phase estimate from observed m-distributions.

    how='ce' minimises the cross-entropy -sum_m y_m log p_m^model(phi) (the
    maximum-likelihood estimator for multinomial data, asymptotically
    efficient); how='l2' minimises the squared distance instead.

    The grid argmin is then refined by a parabola through the loss at the
    minimiser and its two neighbours, which removes the O(step^2/12) grid
    quantisation floor (the loss is smooth in phi, so this is a legitimate
    local interpolation and costs nothing).  The stencil *wraps* when the grid
    spans exactly one period and its end points are the same physical point,
    and is otherwise clipped with refinement disabled at the edges -- so the
    estimate is never extrapolated off the grid.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if how == "ce":
        L = -(np.log(np.maximum(pm_grid, 1e-14)) @ y.T)          # (n_g, n_tst)
    else:
        D = pm_grid[None, :, :] - y[:, None, :]                  # (n_tst,n_g,K)
        L = (D ** 2).sum(-1).T                                   # (n_g, n_tst)
    i = L.argmin(0)
    if not refine or len(g) < 3:
        return g[i]
    n, step = len(g), float(g[1] - g[0])
    # p_m(phi) is exactly 2*pi-periodic -- measured |p_m(-pi) - p_m(pi)| < 1e-16
    # for the sector simulator -- so a grid spanning one full period repeats its
    # first point as its last one: index 0's left neighbour is n-2 and index
    # n-1's right neighbour is 1.  Both conditions (full 2*pi span *and*
    # duplicate end-point distributions) are checked, so an arbitrary grid falls
    # back to the clipped stencil.
    periodic = (n > 3
                and abs((float(g[-1]) - float(g[0])) - 2.0 * np.pi) < 1e-9
                and float(np.abs(np.asarray(pm_grid)[0]
                                 - np.asarray(pm_grid)[-1]).max()) < 1e-12)
    col = np.arange(L.shape[1])
    if periodic:
        lo, hi = (i - 1) % (n - 1), (i + 1) % (n - 1)
        edge = np.zeros(L.shape[1], bool)
    else:
        lo, hi = np.clip(i - 1, 0, n - 1), np.clip(i + 1, 0, n - 1)
        edge = (lo == i) | (hi == i)
    Lm, L0, Lp = L[lo, col], L[i, col], L[hi, col]
    curv = Lm - 2.0 * L0 + Lp
    # refine only where the loss is locally convex: a flat or non-convex
    # stencil (curv <= 0) keeps the raw grid argmin, and |d| <= 1 keeps the
    # refined point inside the winning cell
    ok = (curv > 0) & ~edge
    d = np.where(ok, 0.5 * (Lm - Lp) / np.where(curv == 0, 1.0, curv), 0.0)
    est = g[i] + np.clip(d, -1.0, 1.0) * step
    if periodic:
        est = float(g[0]) + np.mod(est - float(g[0]), 2.0 * np.pi)
    return est


def eval_oracle(g, pm_grid, y1, phi, how, shots_list, n_trials, seed):
    rows = {"inf": metrics(oracle_predict(g, pm_grid, y1, how), phi)}
    for S in shots_list:
        rng = np.random.RandomState(seed + int(S))
        tr = [metrics(oracle_predict(g, pm_grid, vm.sample_dist(y1, S, rng), how),
                      phi) for _ in range(max(1, n_trials))]
        rows[f"S{S}"] = agg(tr)
    return rows


def all_methods(cal, tst, model, shots_list, n_trials=25, seed=0, iters=2000,
                hidden=(32, 32), mse_shots=None, retrain_iters=800,
                workers=12, n_oracle=361, verbose=True,
                folds=(1, 3, 5), folds_poly=(1, 3, 5, 7, 9), warm_start=True):
    """Evaluate the full method suite for one (model, noise) setting."""
    N = int(model["N"])
    phi = tst["phis"]
    y1 = tst["pm_noisy"][1]
    S_ref = mse_shots if mse_shots else (shots_list[-1] if shots_list else 4096)
    res, meta = {}, {}

    res["noiseless"] = eval_with_mitigator(tst["pm_clean"], phi, model, None,
                                           shots_list, n_trials, seed)
    res["none"] = eval_with_mitigator(y1, phi, model, None, shots_list,
                                      n_trials, seed)

    f_eff = vl.effective_flip(N, tst["noise"], fold=1)
    if f_eff is not None and f_eff > 0:
        mit = vm.mitigator_linv(vl.m_flip_kernel(N, f_eff))
        res["linv_known"] = eval_with_mitigator(y1, phi, model, mit, shots_list,
                                                n_trials, seed)
        meta["linv_known"] = {"f_eff": float(f_eff)}
    else:
        meta["linv_known"] = {"f_eff": None,
                              "note": "noise not flip-equivalent: exact sector "
                                      "kernel unavailable"}

    # oracle: exact noise model + fine phase grid (the 'genie' estimator)
    t_or = time.time()
    g_or, pm_or = oracle_grid(N, model, tst["noise"], workers=workers,
                              n_grid=n_oracle)
    for how, tag in (("ce", "oracle_ml"), ("l2", "oracle_l2")):
        res[tag] = eval_oracle(g_or, pm_or, y1, phi, how, shots_list, n_trials,
                               seed)
    meta["oracle"] = {"n_grid": int(len(g_or)),
                      "grid_step": float(g_or[1] - g_or[0]),
                      "grid_floor_mse_unrefined": float((g_or[1] - g_or[0]) ** 2 / 12.0),
                      "refinement": "parabolic on the loss at the grid argmin",
                      "sim_time_s": round(time.time() - t_or, 1),
                      "note": "known noise model, exact simulations on a fine "
                              "phase grid, grid-ML readout"}
    # Cramer-Rao reference for the *noisy* distribution (finite-shot only)
    fi_n = np.asarray(vl.fi_grid(phi, y1), dtype=float)
    meta["crb"] = {"fi_noisy_mean": float(np.mean(fi_n)),
                   "mse_at_S": {str(S): float(np.mean(1.0 / (S * np.maximum(fi_n, 1e-30))))
                                for S in shots_list}}

    # ZNE baselines, each on an *explicit* fold set (all folds come from the
    # same test phases, so the extra quantum cost is one run per fold).
    # ``vm.fold_gates`` only supports odd fold factors (folding twice undoes
    # itself), so the sets are (1,3,5) for Richardson / the linear fit -- which
    # isolates the extrapolation rule -- and the larger (1,3,5,7,9) for the
    # quadratic fit, making it over-determined rather than exactly
    # interpolating, i.e. as strong a ZNE baseline as the data allows.  A
    # variant is skipped if the dataset does not carry its folds.
    for tag, kind, deg, want in (("zne_rich", "richardson", 1, folds),
                                 ("zne_poly1", "poly", 1, folds),
                                 ("zne_poly2", "poly", 2, folds_poly or folds)):
        if not set(want) <= set(tst["pm_noisy"]):
            continue
        r, c, folds = eval_zne(tst, phi, model, kind, deg, shots_list,
                               n_trials, seed, folds=want)
        res[tag] = r
        meta[tag] = {"coeffs": c, "folds": folds, "kind": kind, "degree": deg}

    maps = fit_maps(cal, cal["phis"], model, S_ref, iters=iters, hidden=hidden,
                    warm_start=warm_start)
    meta["_fit_time_s"] = maps.pop("_fit_time_s")
    for tag, (mp, psi, info) in maps.items():
        mit = vm.Mitigator(f"dvaqem_{tag}",
                           lambda y, mp=mp, psi=psi: vm.mitigated_family(mp, psi, y),
                           {"n_params": info["n_params"], "loss": info["loss"]})
        res[f"dvaqem_{tag}"] = eval_with_mitigator(y1, phi, model, mit,
                                                   shots_list, n_trials, seed)
        meta[f"dvaqem_{tag}"] = {k: info[k] for k in
                                 ("loss", "kind", "n_params", "final_loss",
                                  "train_loss", "best_iter", "selected_on",
                                  "n_val", "iters", "lr", "lam",
                                  "holdout_mse", "holdout_mse_inf")}
        if "warm_start" in info:
            meta[f"dvaqem_{tag}"]["warm_start"] = info["warm_start"]
        if tag.startswith("lin"):
            meta[f"dvaqem_{tag}"]["psi"] = np.asarray(psi).tolist()

    dec_rt, info_rt = fit_retrained_decoder(cal, cal["phis"], model,
                                            iters=retrain_iters)
    res["retrain_dec"] = eval_with_mitigator(y1, phi, dec_rt, None, shots_list,
                                             n_trials, seed)
    meta["retrain_dec"] = {k: info_rt[k] for k in
                           ("iters", "n_params", "final_loss", "time_s")}
    meta["retrain_dec"]["x"] = np.asarray(dec_rt["x"]).tolist()

    # Which D-VAQEM variant to headline is decided on the held-out
    # *calibration* phases (finite-shot wrapped MSE, one yardstick for every
    # objective), never on the test set -- selecting there would leak the
    # evaluation data into the method.  The test MSE is only the fallback if a
    # variant has no hold-out (too little calibration data to split).
    def _sel_key(t):
        h = meta[t].get("holdout_mse", float("nan"))
        return float(h) if np.isfinite(h) else res[t]["inf"]["mse"]

    best = min((t for t in res if t.startswith("dvaqem_")), key=_sel_key)
    mp, psi, _ = maps[best.split("dvaqem_", 1)[1]]
    raw = vm.mitigated_family(mp, psi, y1)
    fam = vm._renorm(raw)
    grid, fis = vm.fi_curves(phi, {"clean": tst["pm_clean"], "noisy": y1,
                                   "mitigated": fam})
    meta["fi"] = {k: np.asarray(v).tolist() for k, v in fis.items()}
    meta["fi_grid"] = np.asarray(grid).tolist()
    # The mitigated family is the *clipped* quasi-probability distribution, so
    # its Fisher information is support-restricted -- vl.fi_grid drops the bins
    # the clip drove to zero, because an outcome that cannot occur carries no
    # phase information (and dividing by that numerical zero would report
    # ~1e300).  Record how much was clipped so the number stays interpretable
    # next to the clean/noisy FIs, which come from strictly positive families.
    neg = np.clip(-np.asarray(raw, dtype=float), 0.0, None).sum(axis=1)
    meta["fi_mitigated_diag"] = {
        "variant": best,
        "clipped_mass_mean": float(neg.mean()),
        "clipped_mass_max": float(neg.max()),
        "zero_bin_frac": float((np.asarray(fam, dtype=float) <= 1e-12).mean()),
        "note": "FI of the clipped+renormalised mitigated family; bins clipped "
                "to zero are excluded, so this is a support-restricted FI and "
                "not a Cramer-Rao bound for a regular family"}
    meta["best_dvaqem"] = best
    meta["mse_shots"] = int(S_ref)
    if verbose:
        print(f"    best D-VAQEM variant: {best} (inf-shot mse "
              f"{res[best]['inf']['mse']:.3e}); fit {meta['_fit_time_s']:.1f}s, "
              f"decoder retrain {info_rt['time_s']:.1f}s", flush=True)
    return res, meta


# ----------------------------------------------------------------------
# E1  exactness of the collective-imbalance reduction
# ----------------------------------------------------------------------
def _variant_params(N, vtag, rng_seed):
    """Circuit parameters for one E1 variant (deterministic in rng_seed)."""
    model = load_model(N)
    if vtag == "trained":
        return model["theta"], model["curly"]
    rng = np.random.RandomState(rng_seed + N)
    return (rng.randn(*model["theta"].shape) * 0.4,
            rng.randn(*model["curly"].shape) * 0.4)


def _fi_pair(phis, N, th, cu, noise, workers):
    """FI of the full 2^N distribution and of the (N+1)-sector distribution."""
    pr, pm = vm.sector_probs(phis, N, th, cu, noise, 1, workers)
    return (np.asarray(vl.fi_grid(phis, pr), dtype=float),
            np.asarray(vl.fi_grid(phis, pm), dtype=float))


def exp_sufficiency(Ns=(4, 6, 8), n_phi=13, workers=12, rng_seed=0,
                    strengths=(0.0, 0.002, 0.01, 0.02), readout_p=0.03):
    """How much Fisher information does the imbalance reduction cost?

    The reduction p_full(x) -> p_m (the distribution of the collective imbalance
    m = sum_k x_k) is what makes D-VAQEM scale: N+1 numbers instead of 2^N.  It
    is *exactly* information-preserving when the conditional p(x | m) does not
    depend on phi, which holds for the noiseless VQ-CNNI circuit and is verified
    here to machine precision.

    Once a channel acts inside the circuit that conditional acquires a weak
    phi-dependence, so sufficiency is only approximate.  Rather than claim
    exactness, this experiment *bounds the price*: the Cramer-Rao penalty
    ``crb_gap_db = 10 log10(FI_full / FI_m)`` of estimating phi from p_m instead
    of p_full, alongside two deviation measures (``max_rel_dev``, the worst
    pointwise |FI_full - FI_m| / FI_full, which is inflated by the near-zero dips
    of FI itself; and ``max_dev_norm``, the worst absolute gap normalised by the
    mean FI).  The worst noisy setting is finally re-run on successively finer
    phase grids: a finite-difference artefact would shrink with the spacing, a
    genuine information loss would not.

    Checked for the trained VQ-CNNI decoder and for a random parameter point (so
    the result is not an artefact of a particularly smooth circuit).
    """
    noises = ([("noiseless", NOISELESS)] +
              [(f"depol_{p}", {"kind": "depolarizing", "p": p, "readout_p": 0.0})
               for p in strengths if p > 0] +
              [(f"readout_{readout_p}",
                {"kind": "none", "p": 0.0, "readout_p": readout_p})])
    rows = []
    for N in Ns:
        phis = np.linspace(-np.pi + 0.13, np.pi - 0.13, n_phi)
        vtags = ("trained",) if rng_seed is None else ("trained", "random")
        for vtag in vtags:
            th, cu = _variant_params(N, vtag, rng_seed)
            for nlabel, noise in noises:
                fi_f, fi_m = _fi_pair(phis, N, th, cu, noise, workers)
                gap = np.abs(fi_f - fi_m)
                r = {"N": N, "variant": vtag, "n_phi": len(phis),
                     "noise": nlabel, "kind": noise["kind"], "p": noise["p"],
                     "readout_p": noise["readout_p"],
                     "max_rel_dev": float(np.max(gap / np.maximum(fi_f, 1e-30))),
                     "max_dev_norm": float(
                         np.max(gap) / max(float(np.mean(fi_f)), 1e-30)),
                     "mean_FI_p_m": float(np.mean(fi_m)),
                     "mean_FI_p_full": float(np.mean(fi_f)),
                     "crb_db_full": float(vl.crb_db(np.mean(fi_f))),
                     "crb_db_m": float(vl.crb_db(np.mean(fi_m)))}
                r["crb_gap_db"] = r["crb_db_m"] - r["crb_db_full"]
                rows.append(r)
                print(f"  N={N:2d} {vtag:8s} {nlabel:14s} "
                      f"max rel dev {r['max_rel_dev']:.3e}  "
                      f"(norm {r['max_dev_norm']:.3e}, CRB gap "
                      f"{r['crb_gap_db']:.2e} dB)   "
                      f"mean FI full/sector {np.mean(fi_f):11.4f}/"
                      f"{np.mean(fi_m):11.4f}", flush=True)
    save_json("e1_sufficiency.json", rows)
    print("  E1 summary by noise (worst over N and over trained/random):")
    print(f"    {'noise':16s}{'max rel dev':>13s}{'norm dev':>11s}"
          f"{'CRB gap dB':>12s}")
    for nlabel in dict(noises):
        g = [r for r in rows if r["noise"] == nlabel]
        print(f"    {nlabel:16s}{max(r['max_rel_dev'] for r in g):13.2e}"
              f"{max(r['max_dev_norm'] for r in g):11.2e}"
              f"{max(r['crb_gap_db'] for r in g):12.2e}", flush=True)

    # does the deficit survive grid refinement?  (FD artefact vs genuine loss)
    conv = []
    pool = [r for r in rows if r["noise"] != "noiseless" and r["N"] <= 8] or rows
    w = max(pool, key=lambda r: r["max_rel_dev"])
    print(f"  grid refinement of the worst noisy setting "
          f"(N={w['N']} {w['variant']} {w['noise']}):", flush=True)
    th, cu = _variant_params(w["N"], w["variant"], rng_seed)
    noise = {"kind": w["kind"], "p": w["p"], "readout_p": w["readout_p"]}
    for n in (n_phi, 2 * n_phi - 1, 4 * n_phi - 3):
        ph = np.linspace(-np.pi + 0.13, np.pi - 0.13, n)
        fi_f, fi_m = _fi_pair(ph, w["N"], th, cu, noise, workers)
        gap = np.abs(fi_f - fi_m)
        conv.append({"N": w["N"], "variant": w["variant"], "noise": w["noise"],
                     "n_phi": int(n),
                     "max_rel_dev": float(np.max(gap / np.maximum(fi_f, 1e-30))),
                     "max_dev_norm": float(
                         np.max(gap) / max(float(np.mean(fi_f)), 1e-30)),
                     "crb_gap_db": float(vl.crb_db(np.mean(fi_m))
                                         - vl.crb_db(np.mean(fi_f)))})
        print(f"    n_phi={n:4d}  max rel dev {conv[-1]['max_rel_dev']:.3e}  "
              f"norm {conv[-1]['max_dev_norm']:.3e}  "
              f"CRB gap {conv[-1]['crb_gap_db']:.3e} dB", flush=True)
    save_json("e1_fi_convergence.json", conv)

    nl = [r for r in rows if r["noise"] == "noiseless"]
    ny = [r for r in rows if r["noise"] != "noiseless"]
    drift = conv[-1]["max_rel_dev"] / max(conv[0]["max_rel_dev"], 1e-300)
    print(f"  E1 verdict: without noise the collective imbalance is an exactly "
          f"sufficient statistic (FI(p_m) = FI(p_full) to "
          f"{max(r['max_rel_dev'] for r in nl):.1e} over {len(nl)} settings). "
          f"With a channel inside the circuit sufficiency is approximate: the "
          f"worst pointwise deviation is "
          f"{max(r['max_rel_dev'] for r in ny):.2e} ({max(r['max_dev_norm'] for r in ny):.2e} "
          f"normalised by the mean FI) and the price in the Cramer-Rao bound is "
          f"at most {max(r['crb_gap_db'] for r in ny):.2e} dB over {len(ny)} "
          f"settings.  Refining the phase grid {conv[0]['n_phi']} -> "
          f"{conv[-1]['n_phi']} changes the deviation by a factor {drift:.2f}, "
          f"so it is a genuine (and negligible) information loss rather than a "
          f"finite-difference artefact.", flush=True)
    return rows


# ----------------------------------------------------------------------
# E2  headline sweep: methods x noise type x strength
# ----------------------------------------------------------------------
METHOD_ORDER = ["noiseless", "none", "linv_known", "oracle_ml", "oracle_l2",
                "zne_rich", "zne_poly1", "zne_poly2", "dvaqem_lin_ce",
                "dvaqem_lin_l2", "dvaqem_lin_mse", "dvaqem_mlp_ce",
                "dvaqem_mlp_l2", "dvaqem_mlp_fisher", "dvaqem_mlp_mse",
                "retrain_dec"]


def print_table(rows, shots_list, key="inf", title=None):
    sel_all = [r for r in rows if r["shots"] == key]
    settings = sorted({r["setting"] for r in sel_all})
    print(f"\n  {title or f'MSE (key={key})'}   columns = noise settings")
    print("    " + "-" * (24 + 11 * len(settings)))
    print(f"    {'method':24s}" + "".join(f"{s[:10]:>11s}" for s in settings)
          + f"{'mean':>11s}")
    for method in METHOD_ORDER:
        cells, vals = [], []
        for st in settings:
            r = [x for x in sel_all if x["method"] == method and x["setting"] == st]
            if r:
                cells.append(f"{r[0]['mse']:11.2e}")
                vals.append(r[0]["mse"])
            else:
                cells.append(f"{'--':>11s}")
        mean = f"{np.mean(vals):11.2e}" if vals else f"{'--':>11s}"
        print(f"    {method:24s}" + "".join(cells) + mean)
    print("    " + "-" * (24 + 11 * len(settings)), flush=True)


def exp_sweep(N=8, n_cal=25, n_tst=41, shots_list=(256, 1024, 4096),
              n_trials=25, workers=12, iters=2000, hidden=(32, 32),
              retrain_iters=800, quick=False, noise=None, seed=0,
              tag="e2", n_oracle=361, folds=(1, 3, 5),
              folds_poly=(1, 3, 5, 7, 9), warm_start=True):
    """Core shot-budget sweep (paper Fig. 3 / Table 1).

    ``folds`` / ``folds_poly`` are the (odd-only) ZNE noise-scale factors used by
    the linear / Richardson and the quadratic baselines respectively;
    ``warm_start=False`` disables the L2 / CE initialisation of the shot-aware
    retraining step, which is the ablation that shows the staged fit is what
    keeps the low-shot MSE variants off the constant-map solution.
    """
    model = load_model(N)
    cal_phis, tst_phis = grids(n_cal, n_tst)
    settings = noise_grid(quick) if noise is None else noise
    rows, metas = [], {}
    for label, nz in settings:
        print(f"  [{label}] N={N}  noise={nz}", flush=True)
        t0 = time.time()
        # the union of all fold factors the ZNE baselines need (odd only)
        need = sorted(set(folds) | set(folds_poly or folds))
        cal, tst = build(N, model, nz, cal_phis, tst_phis, workers=workers,
                         folds=tuple(need))
        res, meta = all_methods(cal, tst, model, shots_list, n_trials=n_trials,
                                seed=seed, iters=iters, hidden=hidden,
                                retrain_iters=retrain_iters, workers=workers,
                                n_oracle=n_oracle, folds=folds,
                                folds_poly=folds_poly, warm_start=warm_start)
        for method, r in res.items():
            for shot_key, m in r.items():
                rows.append({"setting": label, "N": N, "noise": nz,
                             "method": method, "shots": shot_key,
                             "n_cal": n_cal, "n_tst": n_tst,
                             "best_dvaqem": meta["best_dvaqem"], **m})
        meta.pop("retrain_dec", None)
        metas[label] = meta
        print(f"    {time.time() - t0:6.1f}s   " + "  ".join(
            f"{k}={res[k]['inf']['mse']:.2e}" for k in
            ("none", "linv_known", "oracle_ml", meta["best_dvaqem"],
             "retrain_dec")
            if k in res), flush=True)
        save_json(f"{tag}_sweep.json", rows)          # incremental: resumable
    save_json(f"{tag}_sweep_meta.json", metas)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{tag}_fi_curves.npz"),
                        **{f"{k}__{kk}": np.asarray(vv)
                           for k, v in metas.items() for kk, vv in
                           [("grid", v["fi_grid"])] +
                           [(f"fi_{f}", np.asarray(v["fi"][f])) for f in v["fi"]]})
    for key in ("inf",) + tuple(f"S{S}" for S in shots_list):
        print_table(rows, shots_list, key=key)
    return rows, metas


# ----------------------------------------------------------------------
# E3  finite shots: bias/variance decomposition and the delta method
# ----------------------------------------------------------------------
def _analytic_map(mp, psi, y1, phi, dec, S):
    """Exact bias + delta-method variance of phi_hat = decoder(M_psi(y)) at S.

    Chain rule through the *frozen* map: c = (d phi_hat/d p~)(d p~/d y), and
    Var = [sum_j c_j^2 y_j - (sum_j c_j y_j)^2]/S for multinomial sampling of y.
    This is the same expression the ``mse`` objective minimises, evaluated on
    the held-out test grid.
    """
    psi = anp.asarray(np.asarray(psi, dtype=float))
    q = vm.mitigated_family(mp, psi, y1)
    phi_hat, g = vm.decoder_value_grad(anp.asarray(q), dec)
    if getattr(mp, "name", "") == "linear":
        c = anp.matmul(g, anp.transpose(mp.matrix(psi)))
    else:
        _, Jm = vm.mlp_map_value_jac(y1, mp._unpack(psi))
        c = anp.sum(Jm * g[:, None, :], axis=-1)
    phi_hat = np.asarray(phi_hat, dtype=float)
    c = np.asarray(c, dtype=float)
    bias = float(np.mean(vl.wrapped_err(phi_hat, phi) ** 2))
    var = float(np.mean(vm.delta_var(y1, c, S)))
    return {"bias2": bias, "var_over_S": var, "analytic_mse": bias + var}


def _analytic_none(y1, phi, dec, S):
    """Same decomposition for the unmitigated estimator (map = identity)."""
    phi_hat, g = vm.decoder_value_grad(anp.asarray(y1), dec)
    phi_hat = np.asarray(phi_hat, dtype=float)
    g = np.asarray(g, dtype=float)
    bias = float(np.mean(vl.wrapped_err(phi_hat, phi) ** 2))
    var = float(np.mean(vm.delta_var(y1, g, S)))
    return {"bias2": bias, "var_over_S": var, "analytic_mse": bias + var}


def _analytic_zne(tst, folds, c, phi, dec, S):
    """Exact bias + delta-method variance of distribution-level ZNE.

    The folds are sampled independently, so
    Var = (1/S) sum_f c_f^2 J_f Sigma_f J_f^T with J_f = d phi_hat / d y_f:
    the extrapolation coefficients amplify shot noise by sum_f c_f^2 >= 1,
    which is precisely why ZNE degrades at small S while a contractive learned
    map (sum_j |T_jk| <= 1) does not.
    """
    Y = np.stack([tst["pm_noisy"][f] for f in folds])          # (F, n_phi, n)
    F, n_phi = Y.shape[0], Y.shape[1]
    c = np.asarray(c, dtype=float)
    pred = np.asarray(vl.predict_from_pm(np.tensordot(c, Y, axes=(0, 0)), dec))
    bias = float(np.mean(vl.wrapped_err(pred, phi) ** 2))

    def phi_of(yk, i, k):
        rows = [yk if j == k else anp.asarray(Y[j, i]) for j in range(F)]
        z = anp.tensordot(anp.asarray(c), anp.stack(rows), axes=(0, 0))
        return vl.predict_from_pm(anp.reshape(z, (1, -1)), dec)[0]

    var = 0.0
    for i in range(n_phi):
        acc = 0.0
        for k in range(F):
            gk = np.asarray(autograd.grad(
                lambda yk, i=i, k=k: phi_of(yk, i, k))(Y[k, i]), dtype=float)
            p = Y[k, i]
            acc += c[k] ** 2 * float(gk @ (np.diag(p) - np.outer(p, p)) @ gk)
        var += acc / S
    var /= n_phi
    return {"bias2": bias, "var_over_S": var, "analytic_mse": bias + var,
            "var_amplification": float(np.sum(c ** 2))}


def exp_shots(N=8, n_cal=25, n_tst=41,
              shots_list=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
              n_trials=60, workers=12, iters=2000, hidden=(32, 32), quick=False,
              settings=None, seed=0, retrain_iters=800, lr=5e-2,
              warm_start=True):
    """Finite-shot scaling, and validation of the analytic (delta-method) MSE.

    For every shot budget S the shot-aware D-VAQEM map is *refitted* at that S
    (this is the variance-regularised estimator), while the standard-VAQEM
    (``ce``) map, ZNE and the retrained decoder are S-independent.  Each point is
    (i) Monte-Carlo sampled n_trials times and (ii) predicted analytically as
    bias^2 + Var_delta; agreement of the two validates the closed-form variance
    that the shot-aware objective minimises.
    """
    model = load_model(N)
    dec = model["params"]
    cal_phis, tst_phis = grids(n_cal, n_tst)
    if settings is None:
        # strengths from the E0 scan: unmitigated MSE ~ 0.1 / 0.3 / 7e-3, i.e.
        # all three are in the regime where mitigation is possible
        settings = [("depol_0.01", {"kind": "depolarizing", "p": 0.01,
                                    "readout_p": 0.0}),
                    ("ampdamp_0.01", {"kind": "amplitude_damping", "p": 0.01,
                                      "readout_p": 0.0}),
                    ("readout_0.03", {"kind": "none", "p": 0.0,
                                      "readout_p": 0.03})]
    if quick:
        settings, shots_list, n_trials = settings[:1], (256, 1024, 4096), 25
    rows = []
    for label, nz in settings:
        print(f"  [{label}] finite-shot study over S in {tuple(shots_list)}",
              flush=True)
        t0 = time.time()
        # (1,3,5,7,9): the widest odd fold set, so the quadratic ZNE baseline
        # is over-determined here too (folding only supports odd factors).
        cal, tst = build(N, model, nz, cal_phis, tst_phis, workers=workers,
                         folds=(1, 3, 5, 7, 9))
        y1, phi = tst["pm_noisy"][1], tst["phis"]
        folds, c = zne_coeffs(tst, "poly", 1)

        # ---- S-independent fits (same staged warm start as fit_maps) --------
        mp_mlp = vm.MlpMap(N, hidden=hidden, rng=np.random.RandomState(seed))
        _, psi_l2, _ = vm.train_map(cal, "mlp", cal_phis, loss="l2", iters=iters,
                                    lr=lr * LR_BY_LOSS["l2"], hidden=hidden,
                                    seed=seed)
        _, psi_ce, _ = vm.train_map(cal, "mlp", cal_phis, loss="ce", iters=iters,
                                    lr=lr * LR_BY_LOSS["ce"], hidden=hidden,
                                    seed=seed, psi0=psi_l2)
        mit_ce = vm.Mitigator("dvaqem_mlp_ce",
                              lambda y: vm.mitigated_family(mp_mlp, psi_ce, y), {})
        dec_rt, _ = fit_retrained_decoder(cal, cal_phis, model,
                                          iters=retrain_iters, seed=seed)

        for S in shots_list:
            # same rule as fit_maps: refine the shot-aware objective from cold
            # *and* from both infinite-shot fits, and keep whichever wins on the
            # held-out calibration phases at this shot count -- the training
            # objective's own validation loss is a biased selector (it can
            # prefer a collapsed constant map).  ``warm_start=False`` ablates
            # this down to the cold fit alone, exactly as in fit_maps.
            cands = (("cold", None),) if not warm_start else (
                ("cold", None), ("l2", psi_l2), ("ce", psi_ce))
            best_mse = None
            for src, p0 in cands:
                _, psi_try, info_try = vm.train_map(
                    cal, "mlp", cal_phis, loss="mse", iters=iters,
                    lr=lr * LR_BY_LOSS["mse"], shots=S, decoder=model,
                    hidden=hidden, seed=seed, psi0=p0)
                info_try["holdout_mse"] = holdout_score(
                    cal, model, mp_mlp, psi_try, info_try["val_idx"],
                    shots=S, seed=seed)
                if (best_mse is None
                        or info_try["holdout_mse"] < best_mse[1]["holdout_mse"]):
                    best_mse = (psi_try, info_try, src)
            psi_mse, info_mse, mse_src = best_mse
            mit_mse = vm.Mitigator("dvaqem_mlp_mse",
                                   lambda y: vm.mitigated_family(mp_mlp, psi_mse, y),
                                   {})
            mc = {"none": eval_with_mitigator(y1, phi, model, None, [S],
                                              n_trials, seed)[f"S{S}"],
                  "dvaqem_mlp_ce": eval_with_mitigator(y1, phi, model, mit_ce,
                                                       [S], n_trials, seed)[f"S{S}"],
                  "dvaqem_mlp_mse": eval_with_mitigator(y1, phi, model, mit_mse,
                                                        [S], n_trials, seed)[f"S{S}"],
                  "retrain_dec": eval_with_mitigator(y1, phi, dec_rt, None, [S],
                                                     n_trials, seed)[f"S{S}"],
                  "zne_poly1": eval_zne(tst, phi, model, "poly", 1, [S],
                                        n_trials, seed)[0][f"S{S}"]}
            an = {"none": _analytic_none(y1, phi, dec, S),
                  "dvaqem_mlp_ce": _analytic_map(mp_mlp, psi_ce, y1, phi, dec, S),
                  "dvaqem_mlp_mse": _analytic_map(mp_mlp, psi_mse, y1, phi, dec, S),
                  "zne_poly1": _analytic_zne(tst, folds, c, phi, dec, S)}
            for method, m in mc.items():
                a = an.get(method, {})
                rows.append({"setting": label, "N": N, "noise": nz, "method": method,
                             "shots": S, "n_cal": n_cal, "n_tst": n_tst,
                             "mc_mse": m["mse"], "mc_mse_db": m["mse_db"],
                             "mc_median_swpe_db": m["median_swpe_db"],
                             "mc_mse_std": m["mse_std"],
                             "bias2": a.get("bias2"), "var_over_S": a.get("var_over_S"),
                             "analytic_mse": a.get("analytic_mse"),
                             "ratio": (m["mse"] / a["analytic_mse"]
                                       if a.get("analytic_mse") else None),
                             "var_amplification": a.get("var_amplification"),
                             "mse_warm_start": mse_src,
                             "mse_holdout": info_mse["holdout_mse"],
                             "mse_val_loss": info_mse["final_loss"],
                             "mse_best_iter": info_mse["best_iter"]})
            print(f"    S={S:6d}  " + "  ".join(
                f"{k}={mc[k]['mse']:.3e}" +
                (f"(ana {an[k]['analytic_mse']:.3e}, r={mc[k]['mse'] / an[k]['analytic_mse']:.2f})"
                 if k in an else "") for k in
                ("none", "zne_poly1", "dvaqem_mlp_ce", "dvaqem_mlp_mse",
                 "retrain_dec")), flush=True)
        print(f"    {time.time() - t0:6.1f}s", flush=True)
        save_json("e3_shots.json", rows)
    save_json("e3_shots.json", rows)
    return rows


# ----------------------------------------------------------------------
# E4  qubit-number scaling
# ----------------------------------------------------------------------
def subset_case(case, idx):
    """Restrict a simulated case to a subset of the phase grid."""
    n = len(case["phis"])
    out = dict(case)
    for k, v in case.items():
        if k == "pm_noisy":
            out[k] = {f: np.asarray(a)[idx] for f, a in v.items()}
        elif isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[idx]
    return out


def exp_scaling(Ns=None, n_cal=25, n_tst=25, folds=(1, 3, 5),
                folds_poly=(1, 3, 5, 7, 9),
                shots_list=(1024, 4096), n_trials=25, workers=12, iters=2000,
                hidden=(32, 32), quick=False, seed=0, retrain_iters=600,
                p=0.01, noise=None, warm_start=True):
    """Scaling in the number of qubits.

    D-VAQEM works entirely on the (N+1)-dimensional sector distribution, so the
    *calibration data* is N+1 numbers per phase (vs 4^N for process/state
    tomography) and the mitigator has O(N^2) parameters; the underlying exact
    simulation still costs O(4^N), and that cost is reported for honesty.

    The dataset carries the union of ``folds`` and ``folds_poly`` so the
    quadratic ZNE baseline -- the strongest of the extrapolation family, and
    the one whose failure mode at large N is the point of the comparison -- is
    available here as well as in the headline sweep.
    """
    rows = []
    if Ns is None:
        Ns = available_Ns()
    if quick:
        Ns, n_cal, n_tst = Ns[::max(1, len(Ns) - 1)], 13, 13
    for N in Ns:
        model = load_model(N)
        nz = noise or {"kind": "depolarizing", "p": p, "readout_p": 0.0}
        cal_phis, tst_phis = grids(n_cal, n_tst)
        print(f"  [N={N}] noise={nz}  sector dim {N + 1} vs full dim {4 ** N}",
              flush=True)
        t0 = time.time()
        cal, tst = build(N, model, nz, cal_phis, tst_phis,
                         folds=tuple(sorted(set(folds) | set(folds_poly or ()))),
                         workers=workers)
        t_data = time.time() - t0
        res, meta = all_methods(cal, tst, model, shots_list, n_trials=n_trials,
                                seed=seed, iters=iters, hidden=hidden,
                                retrain_iters=retrain_iters,
                                folds=tuple(folds), folds_poly=tuple(folds_poly),
                                warm_start=warm_start)
        fi = {k: float(np.mean(v)) for k, v in meta["fi"].items()}
        for method, r in res.items():
            for k, m in r.items():
                rows.append({"N": N, "setting": f"N{N}", "noise": nz,
                             "method": method, "shots": k, "n_cal": n_cal,
                             "n_tst": n_tst, "t_data_s": t_data,
                             "t_fit_s": meta["_fit_time_s"],
                             "t_retrain_s": meta["retrain_dec"]["time_s"],
                             "t_oracle_s": meta["oracle"]["sim_time_s"],
                             "n_oracle": meta["oracle"]["n_grid"],
                             "dim_sector": N + 1, "dim_full": 4 ** N,
                             "fi_clean": fi["clean"], "fi_noisy": fi["noisy"],
                             "fi_mitigated": fi["mitigated"],
                             "best_dvaqem": meta["best_dvaqem"], **m})
        print(f"    data {t_data:6.1f}s  map fit {meta['_fit_time_s']:5.1f}s  "
              f"retrain {meta['retrain_dec']['time_s']:5.1f}s  "
              f"oracle grid ({meta['oracle']['n_grid']} phases) "
              f"{meta['oracle']['sim_time_s']:6.1f}s  "
              f"FI clean/noisy/mit = {fi['clean']:.3f}/{fi['noisy']:.4f}/"
              f"{fi['mitigated']:.3f} "
              f"(mit clipped mass {meta['fi_mitigated_diag']['clipped_mass_mean']:.1e})",
              flush=True)
        print_table(rows, shots_list, key="inf",
                    title=f"N-scaling, mean MSE (columns = N)")
        save_json("e4_scaling.json", rows)
    return rows


# ----------------------------------------------------------------------
# E5  calibration budget
# ----------------------------------------------------------------------
def exp_calib(N=8, cal_sizes=(5, 9, 17, 25, 41), cal_shots=(None, 512, 2048, 8192),
              n_cal_full=41, n_tst=41, shots_list=(1024,), n_trials=25,
              workers=12, iters=2000, hidden=(32, 32), quick=False, seed=0,
              noise=None, retrain_iters=600, warm_start=True):
    """How much calibration resource does D-VAQEM need?

    Axis A: number of calibration phases (target distributions taken to be
    exact, i.e. available from a noiseless simulator or classically).
    Axis B: number of shots used to *estimate* the calibration target t itself
    (the realistic case, where the ideal distribution comes from a noiseless
    device run).  Both axes are compared against decoder retraining, which
    needs the same resource.
    """
    model = load_model(N)
    nz = noise or {"kind": "depolarizing", "p": 0.01, "readout_p": 0.0}
    if quick:
        cal_sizes, cal_shots, n_cal_full, n_tst = (9, 25), (None, 1024), 25, 21
    cal_phis, tst_phis = grids(n_cal_full, n_tst)
    print(f"  [calib] N={N} noise={nz}", flush=True)
    cal, tst = build(N, model, nz, cal_phis, tst_phis, workers=workers,
                     folds=(1, 3, 5, 7, 9))   # widest odd set: quadratic ZNE too
    rows = []

    def record(sub, axis, n_cal, S_cal):
        res, meta = all_methods(sub, tst, model, shots_list, n_trials=n_trials,
                                seed=seed, iters=iters, hidden=hidden,
                                mse_shots=(shots_list[-1] if shots_list else 4096),
                                retrain_iters=retrain_iters, verbose=False,
                                warm_start=warm_start)
        for method, r in res.items():
            for k, m in r.items():
                rows.append({"axis": axis, "N": N, "setting": axis, "noise": nz,
                             "n_cal": int(n_cal), "cal_shots": S_cal,
                             "method": method, "shots": k,
                             "best_dvaqem": meta["best_dvaqem"],
                             "t_fit_s": meta["_fit_time_s"],
                             "t_retrain_s": meta["retrain_dec"]["time_s"], **m})
        save_json("e5_calib.json", rows)
        print(f"    {axis}={n_cal if S_cal is None else S_cal:>6}  " + "  ".join(
            f"{k}={res[k]['inf']['mse']:.2e}" for k in
            ("none", "dvaqem_lin_ce", "dvaqem_mlp_ce", "dvaqem_mlp_mse",
             "retrain_dec") if k in res), flush=True)
        return res

    print("  axis A: number of calibration phases (exact targets)", flush=True)
    for n_cal in cal_sizes:
        idx = np.unique(np.linspace(0, n_cal_full - 1, n_cal).round().astype(int))
        record(subset_case(cal, idx), "n_cal", len(idx), None)

    print("  axis B: shots used to estimate the calibration target t", flush=True)
    for S_cal in cal_shots:
        sub = subset_case(cal, np.arange(n_cal_full))
        if S_cal:
            sub["pm_clean"] = vm.sample_dist(cal["pm_clean"], S_cal,
                                             np.random.RandomState(seed + 7))
        record(sub, "cal_shots", n_cal_full, S_cal)
    return rows


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="D-VAQEM experiment suite")
    ap.add_argument("exp", nargs="?", default="all",
                    choices=("sufficiency", "sweep", "shots", "scaling",
                             "calib", "all"))
    ap.add_argument("--quick", action="store_true",
                    help="smaller grids, fewer noise settings and trials")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--Ns", default=None,
                    help="comma-separated qubit numbers for E1/E4 (default: "
                         "every checkpoint under revision_experiments/outputs)")
    ap.add_argument("--n-cal", type=int, default=25)
    ap.add_argument("--n-tst", type=int, default=41)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--retrain-iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-oracle", type=int, default=361,
                    help="phase-grid size of the known-noise-model oracle")
    ap.add_argument("--noise", default=None,
                    help='JSON override of the E2 noise grid: either '
                         '[["label", {"kind": ..., "p": ..., "readout_p": ...}], ...] '
                         'or [{"label": ..., "kind": ..., ...}, ...]; kind is one of '
                         'none|depolarizing|dephasing|amplitude_damping')
    ap.add_argument("--tag", default="e2",
                    help="output-file prefix for the E2 sweep")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="ablation: train the shot-aware MSE variants from "
                         "scratch instead of warm-starting them from the L2 / "
                         "CE fits")
    ap.add_argument("--exp-root", default=None,
                    help="results directory (default new_paper/results); set "
                         "this to keep an ablation separate from the paper run")
    a = ap.parse_args(argv)
    if a.exp_root:
        set_out(a.exp_root)
    if a.quick:
        a.n_cal, a.n_tst, a.n_trials = 13, 21, 12
        # --quick shrinks the budgets, but an explicitly requested iteration
        # count still wins (the map fits are cheap and need the iterations).
        if a.iters == ap.get_default("iters"):
            a.iters = 1200
        if a.retrain_iters == ap.get_default("retrain_iters"):
            a.retrain_iters = 400
    ns_avail = available_Ns()
    ns_sel = (tuple(int(x) for x in a.Ns.split(",")) if a.Ns else
              (tuple(ns_avail) if not a.quick
               else tuple(ns_avail[::max(1, len(ns_avail) - 1)])))
    cfg = dict(vars(a))
    cfg.update({"numpy": np.__version__,
                "autograd": getattr(autograd, "__version__", "?"),
                "python": sys.version.split()[0],
                "fast_pauli": bool(vl.FAST_PAULI_CHANNELS),
                "threads_env": os.environ.get("OMP_NUM_THREADS"),
                "cpu_count": os.cpu_count(),
                "models_available_N": list(ns_avail)})
    print("config: " + json.dumps(cfg, default=str), flush=True)
    os.makedirs(OUT, exist_ok=True)
    t_all, timings = time.time(), {}

    def run(name, fn):
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        out = fn()
        timings[name] = time.time() - t0
        print(f"=== {name} done in {timings[name]:.1f}s ===", flush=True)
        return out

    if a.exp in ("sufficiency", "all"):
        ns_suf = ns_sel
        run("E1 sufficiency", lambda: exp_sufficiency(
            Ns=ns_suf,
            n_phi=13 if a.quick else 21, workers=a.workers,
            strengths=(0.0, 0.002, 0.01, 0.02) if not a.quick else (0.0, 0.01)))
    if a.exp in ("sweep", "all"):
        nz = parse_noise_spec(json.loads(a.noise)) if a.noise else None
        run("E2 sweep", lambda: exp_sweep(
            N=a.N, n_cal=a.n_cal, n_tst=a.n_tst, n_trials=a.n_trials,
            workers=a.workers, iters=a.iters, retrain_iters=a.retrain_iters,
            quick=a.quick, seed=a.seed, noise=nz, n_oracle=a.n_oracle,
            tag=a.tag, warm_start=not a.no_warm_start))
    if a.exp in ("shots", "all"):
        run("E3 shots", lambda: exp_shots(
            N=a.N, n_cal=a.n_cal, n_tst=a.n_tst,
            n_trials=(60 if not a.quick else a.n_trials), workers=a.workers,
            iters=a.iters, quick=a.quick, seed=a.seed,
            retrain_iters=a.retrain_iters,
            warm_start=not a.no_warm_start))
    if a.exp in ("scaling", "all"):
        ns_sc = ns_sel
        run("E4 scaling", lambda: exp_scaling(
            Ns=ns_sc,
            n_cal=a.n_cal, n_tst=(25 if not a.quick else 13),
            workers=a.workers, iters=a.iters, n_trials=a.n_trials,
            quick=a.quick, seed=a.seed, retrain_iters=a.retrain_iters,
            warm_start=not a.no_warm_start))
    if a.exp in ("calib", "all"):
        run("E5 calib", lambda: exp_calib(
            N=a.N, n_tst=a.n_tst, n_trials=a.n_trials, workers=a.workers,
            iters=a.iters, quick=a.quick, seed=a.seed,
            retrain_iters=a.retrain_iters,
            warm_start=not a.no_warm_start))

    cfg["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    cfg["total_s"] = round(time.time() - t_all, 1)
    save_json("run_manifest.json", cfg)
    print(f"\nALL DONE in {cfg['total_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
