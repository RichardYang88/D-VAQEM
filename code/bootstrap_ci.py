"""
bootstrap_ci.py -- paired bootstrap confidence intervals and exact sign tests.

Why this module exists
----------------------
Every method-to-method comparison in the paper is made at fixed (noise setting,
shot budget), and at fixed shot budget all methods see *the same* multinomial
draws, because ``run_experiments.eval_with_mitigator`` seeds the sampler with
``seed + S``.  The comparisons are therefore paired, and a paired bootstrap is
the right interval estimator -- not the per-trial standard deviation that ``agg``
reports, which describes the spread of one method rather than the uncertainty of
a *difference*.

``run_experiments.metrics(..., detail=True)`` stores exactly the arrays needed to
do this after the fact (``err2``/``err2_phase`` per test phase, ``mse_trial`` per
Monte-Carlo trial), so no experiment has to be repeated to attach an interval to
a published number.

Two resampling units, matched to the two dB conventions in the result files
--------------------------------------------------------------------------
Infinite-shot rows are one deterministic evaluation, so
``mse_db = 10log10(mean_phi err2)`` and the only random element is the finite
test grid: :func:`gain_ci_phase` resamples the test phases.

Finite-shot rows average over Monte-Carlo trials *in decibels*,
``mse_db = mean_t 10log10(mse_t)`` (Jensen: not ``10log10`` of the mean), so
:func:`gain_ci_trial` resamples trials and takes the log per trial -- which
reproduces the stored ``mse_db`` exactly and centres the interval on the number
the paper actually quotes.  The phase-level interval is reported alongside as a
robustness check.

Both are percentile intervals; the one-sided p-value is the bootstrap fraction of
replicates on the wrong side of zero with the ``+1/(B+1)`` correction, so it can
never be exactly zero.

Tests over the noise settings
-----------------------------
:func:`sign_test` is the exact two-sided binomial test and
:func:`wilcoxon_exact` the exact two-sided Wilcoxon signed-rank test (null
distribution built by a subset-sum recursion over the ranks, so ties are exact
and no asymptotic approximation is needed).  scipy is deliberately not required.

Usage:  python bootstrap_ci.py [--tag final] [--res ../results]
                               [--n-boot 20000] [--seed 0] [--conf 0.95]
Writes ``results/ci_<tag>.json`` and prints a markdown digest.
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, "results"))
SHOT_KEYS = ("inf", "S256", "S1024", "S4096")
ZNE = ("zne_rich", "zne_poly1", "zne_poly2")
EPS = 1e-30
# Presentation order of the noise settings (as in collect_numbers.setting_sort).
CHANNEL_ORDER = ("depol", "deph", "ampdamp", "readout")


# ======================================================================
# io helpers (same convention as collect_numbers.py / make_tables.py)
# ======================================================================
def load(path):
    with open(path) as fh:
        return json.load(fh)


def index(rows, *keys):
    """rows -> nested dict keyed by the given fields (last level = row list)."""
    out = {}
    for r in rows:
        d = out
        for k in keys[:-1]:
            d = d.setdefault(r[k], {})
        d.setdefault(r[keys[-1]], []).append(r)
    return out


def one(idx, *key):
    v = idx
    for k in key:
        v = v[k]
    assert len(v) == 1, (key, len(v))
    return v[0]


def get(idx, *key):
    """Like :func:`one` but returns None when the row is absent."""
    v = idx
    try:
        for k in key:
            v = v[k]
    except KeyError:
        return None
    return v[0] if len(v) == 1 else None


def db(x):
    return 10.0 * np.log10(max(float(x), EPS))


def setting_sort(settings):
    """Noise settings grouped by channel, ascending strength within a channel."""
    def key(s):
        name, val = s.rsplit("_", 1)
        return (CHANNEL_ORDER.index(name) if name in CHANNEL_ORDER else 99,
                float(val))
    return sorted(settings, key=key)


# ======================================================================
# row accessors: hide the err2 / err2_phase naming from the callers
# ======================================================================
def err2_of(row):
    """Per-phase squared errors (inf rows: ``err2``, MC rows: ``err2_phase``)."""
    if row is None:
        return None
    v = row.get("err2_phase")
    if v is None:
        v = row.get("err2")
    return None if v is None else np.asarray(v, dtype=float)


def trial_of(row):
    """Per-trial MSEs of a Monte-Carlo row (``mse_trial``)."""
    if row is None or row.get("mse_trial") is None:
        return None
    return np.asarray(row["mse_trial"], dtype=float)


def missing_detail(rows):
    """Rows that carry no per-phase/per-trial arrays (pre-``detail`` runs)."""
    return [r for r in rows
            if err2_of(r) is None and trial_of(r) is None]


# ======================================================================
# resampling primitives
# ======================================================================
def _boot_idx(n, n_boot, seed):
    """(n_boot, n) array of indices drawn with replacement from range(n)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, n, size=(n_boot, n))


def _pct_ci(reps, conf):
    lo = 50.0 * (1.0 - conf)
    return (float(np.percentile(reps, lo)),
            float(np.percentile(reps, 100.0 - lo)))


def _finish(point, reps, conf, n_boot, unit, n_unit):
    """Package a point estimate with its bootstrap replicates.

    ``p_one_sided`` is the bootstrap p-value for H0: point <= 0, i.e. the
    fraction of replicates that fail to show an improvement, with the standard
    ``+1/(B+1)`` correction so that it can never read as exactly zero.
    """
    lo, hi = _pct_ci(reps, conf)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "boot_se": float(np.std(reps, ddof=1)),
            "p_one_sided": (float(np.sum(reps <= 0.0)) + 1.0) / (n_boot + 1.0),
            "n_unit": int(n_unit), "n_boot": int(n_boot), "unit": unit,
            "conf": float(conf)}


# ======================================================================
# paired confidence intervals for a dB gain
# ======================================================================
def gain_ci_phase(err2_ref, err2_new, n_boot=20000, seed=0, conf=0.95):
    """Paired bootstrap CI of the dB gain of ``new`` over ``ref``.

    gain = 10log10 mean(err2_ref) - 10log10 mean(err2_new);  >0 means ``new`` is
    better.  Resamples test phases with replacement, keeping the pairing (both
    methods saw identical shots at this budget).
    """
    a = np.asarray(err2_ref, dtype=float)
    b = np.asarray(err2_new, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    n = a.size
    point = db(a.mean()) - db(b.mean())
    idx = _boot_idx(n, n_boot, seed)
    reps = (10.0 * np.log10(a[idx].mean(axis=1) + EPS)
            - 10.0 * np.log10(b[idx].mean(axis=1) + EPS))
    return _finish(point, reps, conf, n_boot, "phase", n)


def gain_ci_trial(mse_ref, mse_new, n_boot=20000, seed=0, conf=0.95):
    """Paired bootstrap CI of ``mean_t 10log10(mse_ref,t) - mean_t 10log10(mse_new,t)``.

    This is the convention of the stored ``mse_db`` at finite shots, so the point
    estimate equals ``mse_db[ref] - mse_db[new]`` to machine precision (asserted
    by :func:`selfcheck`).  Resamples Monte-Carlo trials.
    """
    a = np.asarray(mse_ref, dtype=float)
    b = np.asarray(mse_new, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    n = a.size
    da, dbv = 10.0 * np.log10(a + EPS), 10.0 * np.log10(b + EPS)
    point = float(da.mean() - dbv.mean())
    idx = _boot_idx(n, n_boot, seed)
    reps = da[idx].mean(axis=1) - dbv[idx].mean(axis=1)
    return _finish(point, reps, conf, n_boot, "trial", n)


def gain_ci_auto(row_ref, row_new, n_boot=20000, seed=0, conf=0.95):
    """Trial-level interval when available (finite shots), else phase-level."""
    tr_ref, tr_new = trial_of(row_ref), trial_of(row_new)
    if tr_ref is not None and tr_new is not None:
        out = gain_ci_trial(tr_ref, tr_new, n_boot, seed, conf)
        e_ref, e_new = err2_of(row_ref), err2_of(row_new)
        if e_ref is not None and e_new is not None:      # robustness check
            alt = gain_ci_phase(e_ref, e_new, n_boot, seed + 1, conf)
            out["phase_level"] = {"point": alt["point"], "ci_lo": alt["ci_lo"],
                                  "ci_hi": alt["ci_hi"]}
        return out
    e_ref, e_new = err2_of(row_ref), err2_of(row_new)
    if e_ref is not None and e_new is not None:
        return gain_ci_phase(e_ref, e_new, n_boot, seed, conf)
    return None


def mse_db_ci(err2=None, mse_trial=None, n_boot=20000, seed=0, conf=0.95):
    """CI for a single method's dB level, from whichever array is available."""
    if mse_trial is not None:
        v = 10.0 * np.log10(np.asarray(mse_trial, dtype=float) + EPS)
        point, unit = float(v.mean()), "trial"
    elif err2 is not None:
        v = np.asarray(err2, dtype=float)
        point = db(v.mean())
        unit = "phase"
    else:
        return None
    n = v.size
    if unit == "trial":
        reps = v[_boot_idx(n, n_boot, seed)].mean(axis=1)
    else:
        reps = 10.0 * np.log10(
            v[_boot_idx(n, n_boot, seed)].mean(axis=1) + EPS)
    lo, hi = _pct_ci(reps, conf)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "boot_se": float(np.std(reps, ddof=1)), "n_unit": int(n),
            "n_boot": int(n_boot), "unit": unit, "conf": float(conf)}


def gap_closure_ci(row_none, row_new, row_nl, n_boot=20000, seed=0, conf=0.95):
    """CI for the fraction of the dB gap to the noiseless device that is closed.

    closure = (db_none - db_new) / (db_none - db_noiseless) * 100.  The ratio is
    recomputed inside every replicate, so the uncertainty of the denominator is
    included rather than treated as a fixed constant.
    """
    tr = (trial_of(row_none), trial_of(row_new), trial_of(row_nl))
    if all(t is not None for t in tr):
        a, b, c = (10.0 * np.log10(np.asarray(t, float) + EPS) for t in tr)
        unit = "trial"
    else:
        e = (err2_of(row_none), err2_of(row_new), err2_of(row_nl))
        if any(x is None for x in e):
            return None
        a, b, c = (np.asarray(x, float) for x in e)
        unit = "phase"
    n = a.size
    if unit == "trial":
        A, B, C = a.mean(), b.mean(), c.mean()
    else:
        A, B, C = db(a.mean()), db(b.mean()), db(c.mean())
    if abs(A - C) < 1e-12:
        return None
    point = float((A - B) / (A - C) * 100.0)
    idx = _boot_idx(n, n_boot, seed)
    if unit == "trial":
        Ab, Bb, Cb = (v[idx].mean(axis=1) for v in (a, b, c))
    else:
        Ab, Bb, Cb = (10.0 * np.log10(v[idx].mean(axis=1) + EPS)
                      for v in (a, b, c))
    den = Ab - Cb
    reps = np.where(np.abs(den) > 1e-12, (Ab - Bb) / den * 100.0, np.nan)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return None
    lo, hi = _pct_ci(reps, conf)
    return {"point": point, "ci_lo": lo, "ci_hi": hi,
            "boot_se": float(np.std(reps, ddof=1)), "n_unit": int(n),
            "n_boot": int(n_boot), "unit": unit, "conf": float(conf)}


def mean_ci(values, n_boot=20000, seed=0, conf=0.95):
    """Bootstrap CI for the mean of per-setting values (resampling settings).

    The 16 noise settings -- not the 21 test phases -- are the population the
    headline "mean reduction over the settings" claim generalises over, so the
    setting is the resampling unit here.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = v.size
    if n == 0:
        return None
    reps = v[_boot_idx(n, n_boot, seed)].mean(axis=1)
    lo, hi = _pct_ci(reps, conf)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()),
            "ci_lo": lo, "ci_hi": hi,
            "std": float(v.std(ddof=1)) if n > 1 else 0.0,
            "n": int(n), "n_boot": int(n_boot), "conf": float(conf)}


# ======================================================================
# exact nonparametric tests over the noise settings
# ======================================================================
def sign_test(gains):
    """Exact two-sided binomial sign test; zero gains are dropped."""
    g = [float(x) for x in gains if np.isfinite(x) and float(x) != 0.0]
    n, k = len(g), sum(1 for x in g if x > 0.0)
    if n == 0:
        return {"n": 0, "n_positive": 0, "p_two_sided": 1.0}
    tail_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    tail_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
    return {"n": n, "n_positive": int(k),
            "p_two_sided": float(min(1.0, 2.0 * min(tail_le, tail_ge)))}


def wilcoxon_exact(gains):
    """Exact two-sided Wilcoxon signed-rank test over the settings.

    Ranks |gain| with average ranks for ties, takes W+ = sum of the ranks of the
    positive gains, and builds the null distribution of W+ by the subset-sum
    recursion ``atoms <- concat(atoms, atoms + r)`` over the ranks.  That is the
    exact permutation null (random sign flips), ties included; the two-sided
    p-value is ``2*min(P(W <= w), P(W >= w))`` capped at 1.

    Two conventions worth knowing when cross-checking against other software:

    * ``W_min`` is what scipy and R report as the two-sided statistic
      (``min(W+, W-)``); ``W_plus`` is reported as well.
    * scipy's ``method='exact'`` is documented to *stop being exact* once ties
      are present, because it evaluates the statistic against the tie-free
      integer-rank distribution ``_get_wilcoxon_distr(n)`` (see
      ``scipy/stats/_wilcoxon.py``, "the null distribution in dist is exact only
      if there are no ties or zeros"); scipy therefore silently switches to a
      permutation test under ``method='auto'``.  This function keeps the exact
      permutation null for tied ranks, so it can differ from a forced
      ``scipy.wilcoxon(..., method='exact')`` on tied input -- and is the more
      accurate of the two.  ``test_bootstrap_ci.py`` checks both regimes.

    n > 24 falls back to the continuity-corrected normal approximation, whose
    variance ``sum(ranks^2)/4`` is exact for tied ranks too (it reduces to
    ``n(n+1)(2n+1)/24`` when the ranks are 1..n).
    """
    g = np.asarray([float(x) for x in gains
                    if np.isfinite(x) and float(x) != 0.0], dtype=float)
    n = g.size
    if n == 0:
        return {"n": 0, "W_plus": 0.0, "W_minus": 0.0, "W_min": 0.0,
                "p_two_sided": 1.0, "method": "none"}
    abs_g = np.abs(g)
    order = np.argsort(abs_g, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:                                   # average ranks for ties
        j = i
        while j + 1 < n and abs_g[order[j + 1]] == abs_g[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    w_obs = float(ranks[g > 0].sum())
    w_min = float(min(w_obs, float(ranks.sum()) - w_obs))
    base = {"n": int(n), "W_plus": w_obs,
            "W_minus": float(ranks.sum()) - w_obs, "W_min": w_min}
    if n > 24:
        mu = float(ranks.sum()) / 2.0
        sd = math.sqrt(float((ranks ** 2).sum()) / 4.0)
        z = (w_obs - mu - (0.5 if w_obs > mu else -0.5)) / sd
        p = float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))
        return dict(base, p_two_sided=p, method="normal-approx")
    atoms = np.zeros(1, dtype=float)               # exact null distribution
    for r in ranks:
        atoms = np.concatenate([atoms, atoms + r])
    p_le = float(np.mean(atoms <= w_obs + 1e-12))
    p_ge = float(np.mean(atoms >= w_obs - 1e-12))
    return dict(base, p_two_sided=float(min(1.0, 2.0 * min(p_le, p_ge))),
                method="exact")


# ======================================================================
# self-check: the CI point estimates must reproduce the stored numbers
# ======================================================================
def selfcheck(rows, tol_db=1e-8, tol_rel=1e-12):
    """Assert that the bootstrap targets equal the stored ``mse``/``mse_db``.

    Two identities have to hold for the intervals to be interpretable as
    intervals *for the published numbers*:

    * finite shots: ``mean_t 10log10(mse_trial_t) == mse_db`` (the trial-level
      decibel convention ``agg`` uses), checked in absolute dB;
    * any row:      ``mean(mse_trial) == mse`` and ``10log10(mean(err2)) ==
      10log10(mse)``, checked as *relative* deviations -- ``agg`` averages the
      per-trial means while ``err2_phase`` averages per phase first, so the two
      agree only up to summation-order round-off (~1e-16 relative), which an
      absolute tolerance would flag spuriously.

    Returns ``(n_checked, worst_db, worst_rel)`` and raises on a violation.
    """
    n, worst_db, worst_rel = 0, 0.0, 0.0
    for r in rows:
        tr = trial_of(r)
        if tr is not None:
            worst_db = max(worst_db, abs(float((10.0 * np.log10(tr + EPS)).mean())
                                         - float(r["mse_db"])))
            scale = max(abs(float(r["mse"])), 1e-30)
            worst_rel = max(worst_rel, abs(float(tr.mean()) - float(r["mse"])) / scale)
            n += 1
        e2 = err2_of(r)
        if e2 is not None:
            scale = max(abs(float(r["mse"])), 1e-30)
            worst_rel = max(worst_rel,
                            abs(float(e2.mean()) - float(r["mse"])) / scale)
            n += 1
    if worst_db > tol_db or worst_rel > tol_rel:
        raise AssertionError(
            f"selfcheck failed: worst dB deviation {worst_db:.3e} (tol {tol_db:.0e}), "
            f"worst relative deviation {worst_rel:.3e} (tol {tol_rel:.0e})")
    return n, worst_db, worst_rel


# ======================================================================
# report builder
# ======================================================================
# Baselines D-VAQEM is compared against, in the order used by the paper.  The
# sign convention is uniform: a *positive* entry means D-VAQEM is better, so the
# two genie baselines come out negative and read as the residual gap.
COMPARISONS = (("none", "unmitigated"),
               ("retrain_dec", "noise-aware decoder retraining"),
               ("oracle_ml", "known-noise-model oracle (ML)"),
               ("linv_known", "exact sector inverse (genie noise model)"),
               ("linv_calib", "readout mitigation (calibrated assignment matrix)"))


def build(rows, n_boot=20000, seed=0, conf=0.95):
    """All confidence intervals and tests for one sweep result file.

    Mirrors the definitions in ``collect_numbers.sec_e2``/``sec_e2b`` exactly
    (the selected variant comes from the infinite-shot row's ``best_dvaqem``,
    "best ZNE" is the smallest ``mse_db`` over the three extrapolation variants)
    so that every interval is attached to a number the paper already quotes.
    """
    idx = index(rows, "setting", "method", "shots")
    settings = setting_sort({r["setting"] for r in rows})
    best = {s: one(idx, s, "none", "inf")["best_dvaqem"] for s in settings}
    out = {"meta": {"n_rows": len(rows), "n_settings": len(settings),
                    "n_boot": int(n_boot), "seed": int(seed), "conf": float(conf),
                    "selected_variant": best, "settings": settings,
                    "n_phase": int(rows[0].get("n_tst", 0)),
                    "n_trials": int(next((r["n_trials"] for r in rows
                                          if r.get("n_trials")), 0))},
           "per_setting": {}, "aggregates": {}, "tests": {}, "claims": {},
           "warnings": []}
    miss = missing_detail(rows)
    if miss:
        out["warnings"].append(
            f"{len(miss)}/{len(rows)} rows carry no per-phase or per-trial "
            "arrays; they were written before metrics(detail=True). Re-run "
            "run_experiments.py to attach intervals to them.")
    present = [k for k in SHOT_KEYS if any(r["shots"] == k for r in rows)]

    for key in present:
        gains = {name: [] for name, _ in COMPARISONS}
        gains_zne = {m: [] for m in ZNE}
        gains_best_zne, closures, per = [], [], {}
        for s in settings:
            sel = best[s]
            r_d = get(idx, s, sel, key)
            if r_d is None:
                out["warnings"].append(f"missing row ({s}, {sel}, {key})")
                continue
            entry = {"selected": sel, "mse_db": float(r_d["mse_db"]),
                     "mse_db_ci": mse_db_ci(err2=err2_of(r_d),
                                            mse_trial=trial_of(r_d),
                                            n_boot=n_boot, seed=seed,
                                            conf=conf)}
            for name, _lab in COMPARISONS:
                r_o = get(idx, s, name, key)
                ci = gain_ci_auto(r_o, r_d, n_boot, seed, conf)
                entry[name] = ci
                if ci is not None:
                    gains[name].append(ci["point"])
            # ZNE: every variant separately, plus the a-posteriori best one
            avail = [(m, get(idx, s, m, key)) for m in ZNE]
            avail = [(m, r) for m, r in avail if r is not None]
            for m, r in avail:
                ci = gain_ci_auto(r, r_d, n_boot, seed, conf)
                entry[m] = ci
                if ci is not None:
                    gains_zne[m].append(ci["point"])
            if avail:
                bm, br = min(avail, key=lambda mr: float(mr[1]["mse_db"]))
                ci = gain_ci_auto(br, r_d, n_boot, seed, conf)
                entry["best_zne"] = ci
                entry["best_zne_variant"] = bm
                # the variant is chosen on the same data the interval uses, so
                # this interval is anti-conservative; the per-variant ones above
                # are the honest numbers and the min over them is the worst case.
                if ci is not None:
                    ci["selection_biased"] = True
                    gains_best_zne.append(ci["point"])
            r_none, r_nl = get(idx, s, "none", key), get(idx, s, "noiseless", key)
            cl = gap_closure_ci(r_none, r_d, r_nl, n_boot, seed, conf)
            entry["gap_closure_pct"] = cl
            if cl is not None:
                closures.append(cl["point"])
            per[s] = entry
        out["per_setting"][key] = per

        # ---- aggregates over the 16 settings, with exact tests -------------
        agg_k = {}
        for name, lab in COMPARISONS:
            agg_k[name] = {"label": lab,
                           "mean": mean_ci(gains[name], n_boot, seed, conf),
                           "sign": sign_test(gains[name]),
                           "wilcoxon": wilcoxon_exact(gains[name])}
        for m in ZNE + ("best_zne",):
            g = gains_best_zne if m == "best_zne" else gains_zne[m]
            agg_k[m] = {"mean": mean_ci(g, n_boot, seed, conf),
                        "sign": sign_test(g),
                        "wilcoxon": wilcoxon_exact(g)}
        agg_k["gap_closure_pct"] = mean_ci(closures, n_boot, seed, conf)
        agg_k["closure_from_means"] = _closure_from_means(
            idx, settings, best, key, n_boot, seed, conf)
        out["aggregates"][key] = agg_k

    # ---- the specific sentences of the manuscript, as verdicts ------------
    cl = out["claims"]
    if "inf" in out["aggregates"]:
        a = out["aggregates"]["inf"]
        per_inf = out["per_setting"]["inf"]
        bz = [(s, e["best_zne"]) for s, e in per_inf.items()
              if e.get("best_zne")]
        cl["mean_reduction_inf_db"] = a["none"]["mean"]
        cl["beats_best_zne_settings_inf"] = sum(1 for _, c in bz if c["point"] > 0)
        cl["beats_best_zne_ci_excludes_zero_inf"] = sum(1 for _, c in bz
                                                        if c["ci_lo"] > 0)
        cl["n_settings_inf"] = len(bz)
        if bz:
            s_min, c_min = min(bz, key=lambda sc: sc[1]["point"])
            cl["min_gain_vs_best_zne_inf"] = {"setting": s_min, **c_min}
        # Selection-free evidence.  Note that "min over variants of the
        # per-setting gain" is *algebraically identical* to the gain over the
        # a-posteriori best variant, so it is NOT an independent check; the
        # honest selection-free statement is per variant, each variant being a
        # comparator fixed before the data are seen.
        per_variant = {}
        for m in ZNE:
            mu, sg, wk = a[m]["mean"], a[m]["sign"], a[m]["wilcoxon"]
            if not mu:
                continue
            per_variant[m] = {"mean": mu["mean"], "ci_lo": mu["ci_lo"],
                              "ci_hi": mu["ci_hi"], "min": mu["min"],
                              "max": mu["max"], "n": mu["n"],
                              "wins": sg["n_positive"],
                              "p_sign": sg["p_two_sided"],
                              "p_wilcoxon": wk["p_two_sided"]}
        cl["per_variant_inf"] = per_variant
        if per_variant:
            cl["weakest_variant"] = min(per_variant,
                                        key=lambda m: per_variant[m]["mean"])
            cl["weakest_variant_mean_gain_db"] = min(
                v["mean"] for v in per_variant.values())
            cl["least_favourable_gain_db"] = min(
                v["min"] for v in per_variant.values())
            cl["all_variants_win_all_settings"] = bool(
                all(v["wins"] == v["n"] for v in per_variant.values()))
            pairs = [(s, m) for s, e in per_inf.items()
                     for m in ZNE if e.get(m)]
            if pairs:
                s0, m0 = min(pairs, key=lambda sm: per_inf[sm[0]][sm[1]]["point"])
                c0 = per_inf[s0][m0]
                cl["least_favourable_pair"] = {
                    "setting": s0, "variant": m0, "point": c0["point"],
                    "ci_lo": c0["ci_lo"], "ci_hi": c0["ci_hi"],
                    "p_one_sided": c0["p_one_sided"]}
    for key in ("S256", "S1024", "S4096"):
        if key not in out["aggregates"]:
            continue
        a = out["aggregates"][key]
        cl[f"retrain_{key}"] = {"mean": a["retrain_dec"]["mean"],
                                "sign": a["retrain_dec"]["sign"],
                                "wilcoxon": a["retrain_dec"]["wilcoxon"]}
        cl[f"closure_{key}_pct"] = a["closure_from_means"]
        cl[f"mean_reduction_{key}_db"] = a["none"]["mean"]
    return out


def _closure_from_means(idx, settings, best, key, n_boot, seed, conf):
    """Gap closure computed from the *setting-averaged* dB levels.

    This is the convention of the reference table in ``collect_numbers.sec_e2``
    (``gain / distance-to-noiseless`` of the means), which is what the abstract's
    "closes 65--76% of the decibel gap" is derived from; the CI resamples
    settings, since that is the unit the mean is taken over.
    """
    rows_none, rows_d = [], []
    for s in settings:
        r_n, r_d = get(idx, s, "none", key), get(idx, s, best[s], key)
        if r_n is None or r_d is None:
            return None
        rows_none.append(float(r_n["mse_db"]))
        rows_d.append(float(r_d["mse_db"]))
    r_nl = get(idx, settings[0], "noiseless", key)
    if r_nl is None:
        return None
    nl = float(r_nl["mse_db"])
    nu, bd = float(np.mean(rows_none)), float(np.mean(rows_d))
    if abs(nu - nl) < 1e-12:
        return None
    point = (nu - bd) / (nu - nl) * 100.0
    A = np.asarray(rows_none, float)
    B = np.asarray(rows_d, float)
    n = A.size
    idx_b = _boot_idx(n, n_boot, seed)
    den = A[idx_b].mean(axis=1) - nl
    reps = np.where(np.abs(den) > 1e-12,
                    (A[idx_b].mean(axis=1) - B[idx_b].mean(axis=1)) / den * 100.0,
                    np.nan)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return None
    lo, hi = _pct_ci(reps, conf)
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "boot_se": float(np.std(reps, ddof=1)), "n_unit": int(n),
            "n_boot": int(n_boot), "unit": "setting", "conf": float(conf),
            "noiseless_db": nl, "unmitigated_mean_db": nu,
            "dvaqem_mean_db": bd}


# ======================================================================
# human-readable digest
# ======================================================================
def _row_fmt(lab, e):
    mu = e.get("mean")
    if not mu:
        return f"| {lab} | -- | -- | -- | -- |"
    return (f"| {lab} | {mu['mean']:+.2f} "
            f"| [{mu['ci_lo']:+.2f}, {mu['ci_hi']:+.2f}] "
            f"| {e['sign']['p_two_sided']:.2g} "
            f"({e['sign']['n_positive']}/{e['sign']['n']}) "
            f"| {e['wilcoxon']['p_two_sided']:.2g} |")


def digest(out):
    """Markdown digest of a :func:`build` result (printed and worth keeping)."""
    m = out["meta"]
    L = [f"# Paired bootstrap CIs and exact tests "
         f"(B={m['n_boot']}, {m['conf'] * 100:.0f}% percentile, seed={m['seed']})",
         "",
         f"Source: `{m.get('source', '?')}`  |  settings: {m['n_settings']}  |  "
         f"rows: {m['n_rows']}",
         "",
         "Positive gain = D-VAQEM is better.  Interval units: test phases at",
         "infinite shots, Monte-Carlo trials at finite shots (paired, same draws).",
         ""]
    for w in out["warnings"]:
        L.append(f"> **WARNING** {w}")
    if out["warnings"]:
        L.append("")
    for key in SHOT_KEYS:
        if key not in out["aggregates"]:
            continue
        a = out["aggregates"][key]
        L += [f"## Shot budget `{key}`", "",
              "| comparison | mean gain (dB) | 95% CI | sign p (wins) | Wilcoxon p |",
              "|---|---|---|---|---|"]
        for name, lab in COMPARISONS:
            L.append(_row_fmt(lab, a[name]))
        for zm in ZNE + ("best_zne",):
            lab = "best ZNE variant (a posteriori)" if zm == "best_zne" else zm
            L.append(_row_fmt(lab, a[zm]))
        c = a.get("closure_from_means")
        if c:
            L.append(f"| gap to noiseless closed | {c['point']:.1f}% "
                     f"| [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}] | -- | -- |")
        L.append("")
    cl = out.get("claims", {})
    if cl:
        L += ["## The manuscript claims, re-evaluated with intervals", ""]
        mr = cl.get("mean_reduction_inf_db")
        if mr:
            L.append(f"- Mean MSE reduction at infinite shots: **{mr['mean']:.2f} dB** "
                     f"[{mr['ci_lo']:.2f}, {mr['ci_hi']:.2f}] over {mr['n']} settings.")
        if cl.get("beats_best_zne_settings_inf") is not None:
            L.append(
                f"- Beats the a-posteriori best ZNE variant in "
                f"**{cl['beats_best_zne_settings_inf']}/{cl['n_settings_inf']}** "
                f"settings at infinite shots; the paired CI excludes zero in "
                f"{cl['beats_best_zne_ci_excludes_zero_inf']} of them.")
        pv = cl.get("per_variant_inf", {})
        if pv:
            L.append("- Selection-free evidence, one *fixed* ZNE comparator at a "
                     "time (each variant is chosen before the data are seen):")
            for zm in ZNE:
                v = pv.get(zm)
                if not v:
                    continue
                L.append(f"    - vs `{zm}`: mean {v['mean']:+.2f} dB "
                         f"[{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}], wins "
                         f"{v['wins']}/{v['n']} settings, worst setting "
                         f"{v['min']:+.2f} dB, sign p = {v['p_sign']:.2g}, "
                         f"Wilcoxon p = {v['p_wilcoxon']:.2g}.")
            lf = cl.get("least_favourable_pair")
            if lf:
                L.append(f"    -> least favourable (variant, setting) pair "
                         f"overall: {lf['point']:+.2f} dB [{lf['ci_lo']:+.2f}, "
                         f"{lf['ci_hi']:+.2f}] for `{lf['variant']}` at "
                         f"`{lf['setting']}`, one-sided p = "
                         f"{lf['p_one_sided']:.2g}; every variant wins every "
                         f"setting: {cl.get('all_variants_win_all_settings')}.")
        mm = cl.get("min_gain_vs_best_zne_inf")
        if mm:
            L.append(f"- Smallest per-setting gain over the a-posteriori *best* "
                     f"ZNE variant: {mm['point']:+.2f} dB [{mm['ci_lo']:+.2f}, "
                     f"{mm['ci_hi']:+.2f}] at `{mm['setting']}` (one-sided "
                     f"bootstrap p = {mm['p_one_sided']:.2g}).  This comparator "
                     f"is picked on the same data, so read it as "
                     f"anti-conservative; the per-variant rows above are the "
                     f"selection-free evidence.")
        for key in ("S256", "S1024", "S4096"):
            r = cl.get(f"retrain_{key}")
            if r and r["mean"]:
                mu, pw = r["mean"], r["wilcoxon"]["p_two_sided"]
                verdict = ("statistically indistinguishable" if pw > 0.05
                           else "significantly different")
                L.append(f"- D-VAQEM vs decoder retraining at {key}: "
                         f"{mu['mean']:+.2f} dB [{mu['ci_lo']:+.2f}, "
                         f"{mu['ci_hi']:+.2f}], sign p = {r['sign']['p_two_sided']:.2g}, "
                         f"Wilcoxon p = {pw:.2g} -> {verdict} at alpha=0.05.")
            c = cl.get(f"closure_{key}_pct")
            if c:
                L.append(f"- Gap to the noiseless device closed at {key}: "
                         f"**{c['point']:.1f}%** [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}].")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="paired bootstrap CIs and exact tests for a sweep result file")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--res", default=RES)
    ap.add_argument("--file", default=None,
                    help="explicit sweep JSON (overrides --tag)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    path = a.file or os.path.join(a.res, f"{a.tag}_sweep.json")
    if not os.path.exists(path):
        print(f"no such result file: {path}")
        return 2
    rows = load(path)
    try:
        n, worst_db, worst_rel = selfcheck(rows)
    except AssertionError as exc:
        print(f"selfcheck FAILED: {exc}")
        return 2
    print(f"selfcheck: {n} array identities reproduce the stored mse/mse_db "
          f"(worst dB deviation {worst_db:.2e}, worst relative {worst_rel:.2e})")
    if n == 0:
        print("  note: this file carries no per-phase/per-trial arrays, so no "
              "interval can be formed. Re-run run_experiments.py (metrics now "
              "stores them) or point --file at a newer result file.")

    out = build(rows, a.n_boot, a.seed, a.conf)
    out["meta"]["source"] = os.path.basename(path)
    dest = a.out or os.path.join(a.res, f"ci_{a.tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"  -> wrote {dest}")
    print()
    print(digest(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


