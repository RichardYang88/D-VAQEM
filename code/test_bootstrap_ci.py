"""
test_bootstrap_ci.py -- self-tests for the statistics in bootstrap_ci.py.

The intervals and tests added for the statistical-validation pass are pure numpy,
so they need an independent reference.  Where scipy happens to be installed it is
used as that reference (exact binomial test, exact Wilcoxon signed-rank test);
those checks report SKIP otherwise, exactly like the PennyLane cross-check in
``validate_simulator.py``, because ``bootstrap_ci`` itself must never need scipy.

Covered:
  1. the trial-level point estimate reproduces the stored ``mse_db`` convention
     (mean of per-trial decibels) and the phase-level one reproduces
     ``10log10(mean err2)``;
  2. the bootstrap standard error matches the analytic standard error of a mean;
  3. the exact null distribution built by the subset-sum recursion is a proper
     distribution with the textbook mean and variance;
  4. ``sign_test`` == the exact two-sided binomial test (and scipy);
  5. ``wilcoxon_exact`` == scipy's exact Wilcoxon, with and without ties;
  6. ``gap_closure_ci`` agrees with a direct recomputation of the ratio;
  7. ``mean_ci`` is unbiased and brackets the sample mean;
  8. a paired bootstrap CI has the nominal coverage on synthetic data with a
     known true gain (the only test that exercises coverage itself);
  9. the row accessors and the setting order used across the repo.

Usage:  python test_bootstrap_ci.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_ci as bc                                      # noqa: E402

_NPASS = 0
_NFAIL = 0
_NSKIP = 0


def check(name, cond, msg=""):
    global _NPASS, _NFAIL
    if cond:
        _NPASS += 1
        print(f"[PASS] {name}  {msg}")
    else:
        _NFAIL += 1
        print(f"[FAIL] {name}  {msg}")
    return bool(cond)


def skip(name, msg):
    global _NSKIP
    _NSKIP += 1
    print(f"[SKIP] {name} ({msg})")


# ----------------------------------------------------------------------
def test_point_estimates():
    """(1) the CI point estimates must equal the stored dB conventions."""
    rng = np.random.RandomState(0)
    n_t, n_p = 41, 21
    err2_a = rng.gamma(2.0, 1.0, size=n_p) + 1e-3
    err2_b = rng.gamma(2.0, 1.0, size=n_p) + 1e-3
    trial_a = rng.gamma(4.0, err2_a.mean(), size=n_t)
    trial_b = rng.gamma(4.0, err2_b.mean(), size=n_t)

    g = bc.gain_ci_phase(err2_a, err2_b, n_boot=2000, seed=1)
    want = bc.db(err2_a.mean()) - bc.db(err2_b.mean())
    check("phase-level point == 10log10(mean err2) difference",
          abs(g["point"] - want) < 1e-12, f"{g['point']:.6f} vs {want:.6f}")
    check("phase-level n_unit == n_test_phases", g["n_unit"] == n_p)

    g = bc.gain_ci_trial(trial_a, trial_b, n_boot=2000, seed=1)
    want = float((10 * np.log10(trial_a + bc.EPS)).mean()
                 - (10 * np.log10(trial_b + bc.EPS)).mean())
    check("trial-level point == mean of per-trial dB difference",
          abs(g["point"] - want) < 1e-12, f"{g['point']:.6f} vs {want:.6f}")
    # Jensen: the two conventions really differ, so which one is used matters
    alt = bc.db(trial_a.mean()) - bc.db(trial_b.mean())
    check("trial-level convention != 10log10 of the mean (Jensen gap visible)",
          abs(g["point"] - alt) > 1e-6, f"{g['point']:.6f} vs {alt:.6f}")

    c = bc.mse_db_ci(mse_trial=trial_a, n_boot=2000, seed=1)
    check("mse_db_ci(trial) point == mean of per-trial dB",
          abs(c["point"] - float((10 * np.log10(trial_a + bc.EPS)).mean())) < 1e-12)
    c = bc.mse_db_ci(err2=err2_a, n_boot=2000, seed=1)
    check("mse_db_ci(phase) point == 10log10(mean err2)",
          abs(c["point"] - bc.db(err2_a.mean())) < 1e-12)
    check("mse_db_ci without arrays returns None", bc.mse_db_ci(n_boot=10) is None)


def test_bootstrap_se():
    """(2) bootstrap SE must match the analytic SE of a mean."""
    rng = np.random.RandomState(7)
    v = rng.normal(0.0, 3.0, size=2000)          # large n: analytic SE is exact
    c = bc.mse_db_ci(mse_trial=np.exp(v), n_boot=4000, seed=3)
    analytic = float(np.std(10 * np.log10(np.exp(v)), ddof=1) / math.sqrt(v.size))
    rel = abs(c["boot_se"] - analytic) / analytic
    check("bootstrap SE ~= sd/sqrt(n)", rel < 0.05,
          f"boot {c['boot_se']:.5f} vs analytic {analytic:.5f} (rel {rel:.3f})")


def test_wilcoxon_null():
    """(3) the subset-sum recursion must give a proper null distribution."""
    n = 12
    ranks = np.arange(1, n + 1, dtype=float)
    atoms = np.zeros(1)
    for r in ranks:
        atoms = np.concatenate([atoms, atoms + r])
    check("null distribution has 2^n atoms", atoms.size == 2 ** n,
          f"{atoms.size} == {2 ** n}")
    mu, var = n * (n + 1) / 4.0, n * (n + 1) * (2 * n + 1) / 24.0
    check("W+ null mean == n(n+1)/4", abs(atoms.mean() - mu) < 1e-9,
          f"{atoms.mean():.4f} vs {mu:.4f}")
    check("W+ null variance == n(n+1)(2n+1)/24", abs(atoms.var() - var) < 1e-8,
          f"{atoms.var():.4f} vs {var:.4f}")
    check("W+ null is symmetric about its mean",
          abs(np.mean(atoms <= mu) - np.mean(atoms >= mu)) < 1e-12)
    w = bc.wilcoxon_exact([1.0 * (i + 1) for i in range(n)])
    check("all-positive gains -> p == 2/2^n",
          abs(w["p_two_sided"] - 2 / 2 ** n) < 1e-12,
          f"p={w['p_two_sided']:.3e}, W+={w['W_plus']}")
    w = bc.wilcoxon_exact([-1.0 * (i + 1) for i in range(n)])
    check("all-negative gains give the same two-sided p",
          abs(w["p_two_sided"] - 2 / 2 ** n) < 1e-12)
    w = bc.wilcoxon_exact([])
    check("empty input -> p = 1", w["p_two_sided"] == 1.0 and w["method"] == "none")
    w = bc.wilcoxon_exact([1.0, -1.0, 2.0, -2.0])
    check("perfectly antisymmetric gains -> p = 1",
          abs(w["p_two_sided"] - 1.0) < 1e-12, f"p={w['p_two_sided']:.4f}")


def test_sign_test():
    """(4) exact binomial two-sided p, against the closed form and scipy."""
    for n, k in ((16, 16), (16, 12), (16, 8), (5, 4), (1, 1)):
        g = [1.0] * k + [-1.0] * (n - k)
        s = bc.sign_test(g)
        tail_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
        tail_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n
        want = min(1.0, 2 * min(tail_le, tail_ge))
        check(f"sign_test n={n} k={k}", abs(s["p_two_sided"] - want) < 1e-12,
              f"p={s['p_two_sided']:.6f}")
    check("sign_test 16/16 == 2^-15",
          abs(bc.sign_test([1.0] * 16)["p_two_sided"] - 2 / 2 ** 16) < 1e-12)
    check("sign_test drops zeros", bc.sign_test([1.0, 0.0, -1.0, 1.0])["n"] == 3)
    try:
        from scipy.stats import binomtest
    except ImportError:
        skip("sign_test vs scipy.stats.binomtest", "scipy not installed")
        return
    rng = np.random.RandomState(11)
    worst = 0.0
    for _ in range(50):
        n = int(rng.randint(4, 20))
        g = rng.normal(size=n)
        k = int(np.sum(g > 0))
        worst = max(worst, abs(bc.sign_test(g)["p_two_sided"]
                               - float(binomtest(k, n, 0.5).pvalue)))
    check("sign_test == scipy binomtest (50 random cases)", worst < 1e-12,
          f"worst |diff| = {worst:.2e}")


def _brute_force_wilcoxon(g):
    """Independent reference: enumerate all 2^n sign patterns explicitly.

    Deliberately naive (no DP, no vectorisation) so that it cannot share a bug
    with the subset-sum recursion in ``bootstrap_ci.wilcoxon_exact``.
    """
    g = np.asarray([float(x) for x in g if float(x) != 0.0], dtype=float)
    n = g.size
    abs_g = np.abs(g)
    # average rank of v = (#{|g| < v} + #{|g| <= v} + 1) / 2
    ranks = np.array([0.5 * (int(np.sum(abs_g < v)) + int(np.sum(abs_g <= v)) + 1)
                      for v in abs_g])
    ws = np.array([sum(ranks[i] for i in range(n) if (m >> i) & 1)
                   for m in range(1 << n)])
    w_obs = float(ranks[g > 0].sum())
    p_le = float(np.mean(ws <= w_obs + 1e-12))
    p_ge = float(np.mean(ws >= w_obs - 1e-12))
    return w_obs, min(1.0, 2.0 * min(p_le, p_ge)), ranks


def test_wilcoxon_vs_scipy():
    """(5) exactness: tie-free against scipy, tied against brute force."""
    rng = np.random.RandomState(13)
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        skip("wilcoxon_exact vs scipy.stats.wilcoxon (tie-free)",
             "scipy not installed")
        wilcoxon = None
    if wilcoxon is not None:
        worst, ncase = 0.0, 0
        for _ in range(200):
            n = int(rng.randint(5, 18))
            g = rng.normal(loc=0.4, size=n)
            g = g[g != 0.0]
            if g.size < 3 or len(set(np.abs(g))) != g.size:
                continue                     # scipy 'exact' needs distinct |g|
            ncase += 1
            mine = bc.wilcoxon_exact(g)
            ref = wilcoxon(g, alternative="two-sided", method="exact")
            worst = max(worst, abs(mine["p_two_sided"] - float(ref.pvalue)),
                        abs(mine["W_min"] - float(ref.statistic)))
        check(f"tie-free: p and W_min == scipy exact ({ncase} random cases)",
              ncase > 0 and worst < 1e-12, f"worst |diff| = {worst:.2e}")

    # tied input: the exact permutation null is the reference, by enumeration
    worst, ncase, ntie = 0.0, 0, 0
    for _ in range(200):
        n = int(rng.randint(5, 13))
        g = rng.normal(loc=0.4, size=n)
        g[1] = abs(g[0]) * (1.0 if g[1] >= 0 else -1.0)       # force a tie
        if rng.rand() < 0.4:
            g[3] = -abs(g[2])                                 # and a second one
        g = g[g != 0.0]
        if g.size < 4 or len(set(np.abs(g))) == g.size:
            continue
        ncase += 1
        ntie += int(g.size - len(set(np.abs(g))))
        w_bf, p_bf, _ = _brute_force_wilcoxon(g)
        mine = bc.wilcoxon_exact(g)
        worst = max(worst, abs(mine["p_two_sided"] - p_bf),
                    abs(mine["W_plus"] - w_bf))
    check(f"tied: p and W+ == brute-force enumeration of 2^n sign patterns "
          f"({ncase} cases, {ntie} tied pairs)",
          ncase > 0 and worst < 1e-12, f"worst |diff| = {worst:.2e}")

    # the worked example from the debug session, and the documented scipy gap
    g = [0.9, -0.9, 0.4, 1.7, -2.3, 0.11, -3.1, 2.9]
    mine = bc.wilcoxon_exact(g)
    _, p_bf, ranks_bf = _brute_force_wilcoxon(g)
    check("tied n=8 example: subset-sum DP == brute force",
          abs(mine["p_two_sided"] - p_bf) < 1e-12,
          f"p={mine['p_two_sided']:.6f}, W+={mine['W_plus']}, "
          f"W_min={mine['W_min']}, ranks={list(ranks_bf)}")
    check("W+ + W- == sum of ranks", abs(mine["W_plus"] + mine["W_minus"]
                                         - float(np.sum(ranks_bf))) < 1e-12)
    if wilcoxon is not None:
        ref = wilcoxon(g, alternative="two-sided", method="exact")
        print(f"       info: forced scipy exact on tied data gives "
              f"p={float(ref.pvalue):.6f} vs exact permutation null "
              f"p={mine['p_two_sided']:.6f} (scipy documents this as inexact)")
    g = rng.normal(loc=0.3, size=26)
    check("exact branch used for n <= 24",
          bc.wilcoxon_exact(list(g[:24]))["method"] == "exact")
    ap = bc.wilcoxon_exact(list(g))
    check("normal-approx branch used for n > 24", ap["method"] == "normal-approx",
          f"p={ap['p_two_sided']:.4f}")


def test_gap_closure():
    """(6) the closure ratio must match a direct recomputation."""
    rng = np.random.RandomState(5)
    n = 21

    def rows(none, new, nl):
        def mk(v):
            return {"err2": [float(x) for x in v], "mse": float(np.mean(v)),
                    "mse_db": bc.db(float(np.mean(v)))}
        return mk(none), mk(new), mk(nl)

    # (a) deterministic scaling: the identity must hold exactly and the interval
    #     must collapse, because there is then no sampling variability at all
    none = rng.gamma(3.0, 1.0, size=n) + 0.5
    row_n, row_d, row_l = rows(none, none * 0.01, none * 1e-4)
    c = bc.gap_closure_ci(row_n, row_d, row_l, n_boot=2000, seed=2)
    want = ((bc.db(none.mean()) - bc.db((none * 0.01).mean()))
            / (bc.db(none.mean()) - bc.db((none * 1e-4).mean())) * 100.0)
    check("gap_closure point == direct recomputation",
          abs(c["point"] - want) < 1e-9, f"{c['point']:.6f} vs {want:.6f}")
    check("gap_closure unit == phase for infinite-shot rows", c["unit"] == "phase")
    check("deterministic scaling collapses the interval (as it must)",
          c["ci_hi"] - c["ci_lo"] < 1e-9,
          f"[{c['ci_lo']:.6f}, {c['ci_hi']:.6f}]")
    check("gap_closure of the noiseless row itself is 100%",
          abs(bc.gap_closure_ci(row_n, row_l, row_l, n_boot=500,
                                seed=2)["point"] - 100.0) < 1e-9)
    check("gap_closure returns None when the gap is degenerate",
          bc.gap_closure_ci(row_n, row_d, row_n, n_boot=100, seed=2) is None)

    # (b) three independent draws: the interval must now be informative
    a = rng.gamma(3.0, 1.0, size=n) + 0.5
    b = rng.gamma(3.0, 0.01, size=n) + 1e-4
    l = rng.gamma(3.0, 1e-4, size=n) + 1e-6
    row_n, row_d, row_l = rows(a, b, l)
    c = bc.gap_closure_ci(row_n, row_d, row_l, n_boot=4000, seed=2)
    want = ((bc.db(a.mean()) - bc.db(b.mean()))
            / (bc.db(a.mean()) - bc.db(l.mean())) * 100.0)
    check("independent draws: point == direct recomputation",
          abs(c["point"] - want) < 1e-9, f"{c['point']:.4f}% vs {want:.4f}%")
    check("independent draws: interval has positive width",
          c["ci_hi"] - c["ci_lo"] > 1e-6,
          f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] %")
    check("independent draws: interval brackets the point estimate",
          c["ci_lo"] <= c["point"] <= c["ci_hi"])

    # (c) the trial-level path must agree with the phase-level one on the
    #     point estimate when the per-trial MSEs are constant
    tr_n = np.full(5, a.mean())
    tr_d = np.full(5, b.mean())
    tr_l = np.full(5, l.mean())
    c2 = bc.gap_closure_ci({"mse_trial": list(tr_n)}, {"mse_trial": list(tr_d)},
                           {"mse_trial": list(tr_l)}, n_boot=200, seed=2)
    check("trial-level closure == phase-level closure for constant trials",
          c2 is not None and abs(c2["point"] - want) < 1e-6,
          f"{c2['point']:.4f}% vs {want:.4f}%")
    check("trial-level unit reported", c2["unit"] == "trial")


def test_mean_ci():
    """(7) the setting-level bootstrap must bracket the sample mean."""
    rng = np.random.RandomState(9)
    v = rng.normal(30.0, 8.0, size=16)
    m = bc.mean_ci(v, n_boot=5000, seed=4)
    check("mean_ci mean == sample mean", abs(m["mean"] - v.mean()) < 1e-12)
    check("mean_ci brackets the sample mean", m["ci_lo"] <= m["mean"] <= m["ci_hi"],
          f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")
    se = float(v.std(ddof=1) / math.sqrt(v.size))
    check("mean_ci half-width ~= 1.96*sd/sqrt(n)",
          abs((m["ci_hi"] - m["ci_lo"]) / 2 - 1.96 * se) / se < 0.25,
          f"half-width {(m['ci_hi'] - m['ci_lo']) / 2:.3f} vs {1.96 * se:.3f}")
    check("mean_ci of a single value has zero width",
          bc.mean_ci([2.5], n_boot=100, seed=0)["ci_hi"]
          == bc.mean_ci([2.5], n_boot=100, seed=0)["ci_lo"])
    check("mean_ci of nothing returns None", bc.mean_ci([]) is None)


def test_accessors():
    """(9) row accessors and the setting order used across the repo."""
    check("err2_of prefers err2_phase over err2",
          np.allclose(bc.err2_of({"err2": [1.0], "err2_phase": [2.0]}), [2.0]))
    check("err2_of falls back to err2",
          np.allclose(bc.err2_of({"err2": [1.0]}), [1.0]))
    check("err2_of(None) is None", bc.err2_of(None) is None)
    check("trial_of(None) is None", bc.trial_of(None) is None)
    check("missing_detail counts bare rows",
          len(bc.missing_detail([{"mse_db": 1.0}, {"err2": [1.0]}])) == 1)
    order = bc.setting_sort(["readout_0.05", "depol_0.002", "deph_0.05",
                             "ampdamp_0.01", "depol_0.02", "deph_0.005"])
    check("setting_sort groups by channel then strength",
          order == ["depol_0.002", "depol_0.02", "deph_0.005", "deph_0.05",
                    "ampdamp_0.01", "readout_0.05"], str(order))
    idx = bc.index([{"a": 1, "b": 2}, {"a": 1, "b": 3}], "a", "b")
    check("index/one round-trip", bc.one(idx, 1, 3)["b"] == 3)
    check("get returns None for an absent key", bc.get(idx, 1, 9) is None)


def test_coverage():
    """(8) actual coverage of the paired interval on synthetic data.

    The two methods must be *independent* draws whose population means differ by
    a known 6 dB; making one a deterministic rescaling of the other (as a first
    version of this test did) removes all sampling variability, collapses the
    interval to a point and makes the coverage check vacuously true.
    """
    rng = np.random.RandomState(2024)
    n, n_boot, conf, reps = 21, 1500, 0.95, 400
    true_gain = 6.0                              # dB between the two populations
    scale = 10.0 ** (-true_gain / 10.0)
    hit, widths = 0, []
    for _ in range(reps):
        a = rng.gamma(3.0, 1.0, size=n) + 1e-3   # reference method, fresh draw
        b = rng.gamma(3.0, scale, size=n) + 1e-3 * scale   # 6 dB better in mean
        ci = bc.gain_ci_phase(a, b, n_boot=n_boot,
                              seed=int(rng.randint(1 << 30)), conf=conf)
        hit += int(ci["ci_lo"] <= true_gain <= ci["ci_hi"])
        widths.append(ci["ci_hi"] - ci["ci_lo"])
    frac = hit / reps
    slack = 4 * math.sqrt(0.95 * 0.05 / reps)    # 4 sd of the coverage estimate
    check("intervals are genuinely random (not collapsed to a point)",
          float(np.mean(widths)) > 0.5,
          f"mean CI width {np.mean(widths):.3f} dB")
    check(f"paired phase bootstrap covers a true 6 dB gain in ~95% of {reps} runs",
          frac >= 0.95 - slack, f"coverage {frac * 100:.1f}% (floor "
                                f"{(0.95 - slack) * 100:.1f}%)")
    check("coverage is not grossly over-conservative either",
          frac <= 0.95 + slack + 0.02, f"coverage {frac * 100:.1f}%")


def main():
    for fn in (test_point_estimates, test_bootstrap_se, test_wilcoxon_null,
               test_sign_test, test_wilcoxon_vs_scipy, test_gap_closure,
               test_mean_ci, test_accessors, test_coverage):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{_NPASS} passed, {_NFAIL} failed, {_NSKIP} skipped")
    return 1 if _NFAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
