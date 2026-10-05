"""
test_pec.py -- self-tests for the readout PEC baseline in vaqem_methods.py.

PEC is the one baseline here whose *implementation* can be checked against an
independent exact reference rather than against itself, because the readout
channel it inverts is known in closed form.  These tests therefore pin down the
three things that would otherwise be free parameters of the design:

  1. the sampling overhead gamma is (1-2q)^-N, re-derived from the single-qubit
     quasi-probability inverse c0*I + c1*X rather than asserted;
  2. the expected sector transition matrix M of the sampled estimator equals
     K_m^-T exactly, i.e. PEC and deterministic matrix inversion are the same
     estimator in expectation and differ only in variance -- which is what makes
     the finite-shot comparison between them a comparison of variance alone;
  3. the per-shot sampler is unbiased, has variance strictly larger than the
     deterministic inverse (the sampling overhead is real, not notational), and
     leaves an unremovable bias floor when the assumed rate is wrong.

Pure numpy + math, no scipy, matching test_bootstrap_ci.py.

Usage:  python test_pec.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaqem_lib as vl                                           # noqa: E402
import vaqem_methods as vm                                       # noqa: E402

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
def test_gamma_formula():
    """(1) gamma = (1-2q)^-N, re-derived from the single-qubit inverse."""
    q = 0.037
    c0 = (1.0 - q) / (1.0 - 2.0 * q)
    c1 = -q / (1.0 - 2.0 * q)
    check("single-qubit overhead |c0|+|c1| == 1/(1-2q)",
          abs((abs(c0) + abs(c1)) - 1.0 / (1.0 - 2.0 * q)) < 1e-14,
          f"|c0|+|c1| = {abs(c0) + abs(c1):.12f}")
    check("c0 + c1 == 1 (the inverse preserves total probability)",
          abs(c0 + c1 - 1.0) < 1e-14)
    for N in (1, 2, 5, 10):
        want = (1.0 - 2.0 * q) ** (-N)
        check(f"pec_gamma(N={N}, q={q}) == (1-2q)^-N",
              abs(vm.pec_gamma(N, q) - want) < 1e-12 * max(1.0, want),
              f"got {vm.pec_gamma(N, q):.8f} want {want:.8f}")
    check("gamma == 1 at q = 0 (no noise, no overhead)",
          abs(vm.pec_gamma(7, 0.0) - 1.0) < 1e-15)
    check("gamma grows with N and with q",
          vm.pec_gamma(10, 0.05) > vm.pec_gamma(4, 0.05) > vm.pec_gamma(4, 0.01))
    bad = 0
    for q_bad in (0.5, 0.6, -0.1):
        try:
            vm.pec_gamma(4, q_bad)
        except ValueError:
            bad += 1
    check("non-invertible rates (q >= 1/2 or q < 0) are rejected", bad == 3,
          f"{bad}/3 raised ValueError")
    # the overhead quoted for the paper settings is the one practitioners quote:
    # it grows combinatorially in N, which is why PEC is not used at N = 10.
    g = vm.pec_gamma(10, 0.05)
    check("gamma at N=10, q=0.05 is 0.9^-10 (gamma^2 ~ 8.2)",
          abs(g - 0.9 ** -10) < 1e-12, f"gamma = {g:.4f}, gamma^2 = {g ** 2:.4f}")


def test_matrix_matches_exact_inverse():
    """(2) M == K_m^-T exactly: PEC and matrix inversion share an expectation."""
    worst = 0.0
    for N in (1, 2, 4, 8):
        for q in (0.01, 0.03, 0.05):
            K = vl.m_flip_kernel(N, q)
            M, gam = vm.pec_sector_matrix(N, q)
            ref = np.linalg.inv(K).T          # alpha -> 0 limit of linv_map
            dev = float(np.max(np.abs(M - ref)))
            worst = max(worst, dev)
            check(f"M == K^-T at N={N}, q={q}", dev < 1e-9,
                  f"max|M - K^-T| = {dev:.2e}, gamma = {gam:.4f}")
    check("identity holds uniformly over the grid tested", worst < 1e-9,
          f"worst deviation {worst:.2e}")
    T = vm.linv_map_from_kernel(vl.m_flip_kernel(8, 0.05), alpha=1e-6)
    M, _ = vm.pec_sector_matrix(8, 0.05)
    check("M agrees with the Tikhonov inverse the linv baselines apply",
          float(np.max(np.abs(M - T))) < 1e-5,
          f"max|M - T(alpha=1e-6)| = {np.max(np.abs(M - T)):.2e}")


def test_overhead_is_state_independent():
    """(3) sum_a' |M[a, a']| == gamma for every observed sector."""
    for N in (1, 4, 8, 10):
        for q in (0.0, 0.02, 0.05):
            M, gam = vm.pec_sector_matrix(N, q)
            row = np.sum(np.abs(M), axis=1)
            check(f"|M| row sums == gamma at N={N}, q={q}",
                  float(np.max(np.abs(row - gam))) < 1e-9,
                  f"min {row.min():.6f} max {row.max():.6f} gamma {gam:.6f}")


def test_sampler_is_unbiased():
    """(4) E[pec_sample] == y @ M, and PEC recovers the clean distribution."""
    N, q, shots, trials = 6, 0.04, 2000, 3000
    M, gam = vm.pec_sector_matrix(N, q)
    K = vl.m_flip_kernel(N, q)
    rng = np.random.RandomState(11)
    w = rng.dirichlet(np.ones(N + 1) * 3.0)      # a clean sector distribution
    y = w @ K.T                                  # pushed through the channel
    acc = np.zeros(N + 1)
    for _ in range(trials):
        acc += vm.pec_sample(y, N, q, shots, rng)
    emp = acc / trials
    want = y @ M
    # sd of the trial mean, allowing for the gamma-inflated spread
    tol = 6.0 * gam * math.sqrt(1.0 / shots) / math.sqrt(trials) + 1e-3
    dev = float(np.max(np.abs(emp - want)))
    check(f"pec_sample is unbiased for y @ M over {trials} trials", dev < tol,
          f"max dev {dev:.2e} (tol {tol:.2e})")
    dev2 = float(np.max(np.abs(emp - w)))
    check("PEC on the noisy distribution recovers the clean one in expectation",
          dev2 < tol, f"max|E[PEC] - p_clean| = {dev2:.2e} (tol {tol:.2e})")


def test_variance_inflation():
    """(5) the sampled estimator is strictly noisier than the inverse."""
    N, q, shots, trials = 6, 0.05, 512, 1200
    gam = vm.pec_gamma(N, q)
    K = vl.m_flip_kernel(N, q)
    rng = np.random.RandomState(23)
    w = rng.dirichlet(np.ones(N + 1) * 3.0)
    y = w @ K.T
    T = np.linalg.inv(K).T
    pec, lin = [], []
    for _ in range(trials):
        yh = vm.sample_dist(y[None, :], shots, rng)[0]   # multinomial shot noise
        lin.append(yh @ T)                               # deterministic inverse
        pec.append(vm.pec_sample(yh, N, q, shots, rng))  # + quasi-prob noise
    v_pec = float(np.mean(np.var(np.array(pec), axis=0)))
    v_lin = float(np.mean(np.var(np.array(lin), axis=0)))
    ratio = v_pec / max(v_lin, 1e-30)
    check("PEC variance exceeds deterministic inversion", v_pec > v_lin,
          f"var(PEC) {v_pec:.3e} vs var(inverse) {v_lin:.3e}, ratio {ratio:.2f}")
    check("the excess is bounded by the quoted overhead gamma^2",
          ratio <= gam ** 2 * 1.5 + 1.0,
          f"ratio {ratio:.2f} against gamma^2 = {gam ** 2:.2f}")
    check("both estimators are unbiased for the clean distribution",
          float(np.max(np.abs(np.mean(pec, axis=0) - w))) < 5e-3
          and float(np.max(np.abs(np.mean(lin, axis=0) - w))) < 5e-3,
          f"PEC {np.max(np.abs(np.mean(pec, axis=0) - w)):.2e}, "
          f"inv {np.max(np.abs(np.mean(lin, axis=0) - w)):.2e}")


def test_model_mismatch():
    """(6) a wrong assumed rate leaves a bias floor that shots cannot remove."""
    N, q_true = 6, 0.05
    check("zero mismatch gives exactly zero residual bias",
          vm.pec_mismatch_bias(N, q_true, q_true) < 1e-12,
          f"{vm.pec_mismatch_bias(N, q_true, q_true):.2e}")
    prev, mono = -1.0, True
    for d in (0.002, 0.005, 0.01, 0.02, 0.04):
        b = vm.pec_mismatch_bias(N, q_true, q_true + d)
        mono = mono and b > prev
        prev = b
    check("residual bias increases monotonically with the rate error", mono,
          f"bias at +0.04 = {prev:.3e}")
    prev, mono_dn = -1.0, True
    for d in (0.002, 0.005, 0.01, 0.02, 0.04):
        b = vm.pec_mismatch_bias(N, q_true, q_true - d)
        mono_dn = mono_dn and b > prev
        prev = b
    check("... and likewise when the rate is under-assumed", mono_dn,
          f"bias at -0.04 = {prev:.3e}")
    # The bias is *not* symmetric in the sign of the rate error: the inverse
    # carries 1/(1-2q), which is convex, so over- and under-correcting by the
    # same amount leave different floors.  Asserting the asymmetry (rather than
    # an assumed symmetry) keeps this a property of the model, not of the test.
    up = vm.pec_mismatch_bias(N, q_true, q_true + 0.01)
    dn = vm.pec_mismatch_bias(N, q_true, q_true - 0.01)
    check("over- and under-assuming by 0.01 give different floors (asymmetric)",
          abs(up - dn) > 1e-6, f"+0.01 -> {up:.4e}, -0.01 -> {dn:.4e}")
    b1 = vm.pec_mismatch_bias(N, q_true, q_true + 0.02)
    check("a 0.02 rate error leaves a substantial, unremovable floor",
          b1 > 1e-3, f"residual RMS sector error {b1:.3e}")


def test_mitigator_interface():
    """(7) the Mitigator wiring, metadata and per-call RNG behaviour."""
    N, q = 8, 0.05
    mit = vm.mitigator_pec(N, q, shots=None, seed=3)
    check("infinite-shot PEC is named pec_calib", mit.name == "pec_calib",
          f"name = {mit.name}")
    check("records gamma and gamma^2 for quoting",
          abs(mit.info["gamma"] - vm.pec_gamma(N, q)) < 1e-12
          and abs(mit.info["sampling_overhead_gamma_sq"]
                  - vm.pec_gamma(N, q) ** 2) < 1e-12,
          f"gamma = {mit.info['gamma']:.4f}")
    check("calibration budget is one fitted parameter, as for linv_calib",
          mit.info["n_params_fit"] == 1)
    uni = np.full((3, N + 1), 1.0 / (N + 1))
    check("infinite-shot PEC == the deterministic inverse exactly",
          float(np.max(np.abs(
              mit(uni) - vm.mitigator_linv(vl.m_flip_kernel(N, q), alpha=0.0)(
                  uni)))) < 1e-9)
    mk = vm.mitigator_pec_known(N, q, shots=None, seed=3)
    check("genie variant is named pec_known and flagged as analytic",
          mk.name == "pec_known" and "genie" in mk.info["assignment_matrix"])
    ms = vm.mitigator_pec(N, q, shots=256, seed=5)
    check("finite-shot variant switches to sampling",
          ms.info["sampled"] is True and ms.info["shots"] == 256)
    a, b = ms(uni), ms(uni)
    check("the PEC RNG advances across calls (trials stay independent)",
          not np.allclose(a, b), f"max|a-b| = {np.max(np.abs(a - b)):.2e}")
    # Total mass: E[sum_a PEC_a] = 1 exactly, but its sd is gamma-inflated --
    # sd = gamma*sqrt(1-(1-2q)^(2N))/sqrt(S), independent of the observed sector
    # -- so a *single* trial can legitimately be off by several percent.  Using
    # the analytic sd over many trials turns this into a real unbiasedness check
    # instead of a tolerance chosen to pass; a first version of this test used
    # 2/S and failed, which is the variance inflation the suite is meant to show.
    S, reps = 256, 400
    gam = vm.pec_gamma(N, q)
    rng = np.random.RandomState(7)
    y_emp = vm.sample_dist(np.full((1, N + 1), 1.0 / (N + 1)), S, rng)[0]
    check("the empirical input has integer counts summing to S",
          abs(y_emp.sum() - 1.0) < 1e-12
          and np.allclose(np.rint(y_emp * S), y_emp * S))
    mp = vm.mitigator_pec(N, q, shots=S, seed=9)
    sums = np.array([mp(y_emp[None, :]).sum(axis=1)[0] for _ in range(reps)])
    sd1 = gam * math.sqrt(1.0 - (1.0 - 2.0 * q) ** (2 * N)) / math.sqrt(S)
    tol = 4.0 * sd1 / math.sqrt(reps)
    check("mean total mass is 1 within the analytic gamma-inflated error",
          abs(float(sums.mean()) - 1.0) < tol,
          f"mean {sums.mean():.6f} +/- {sums.std() / math.sqrt(reps):.6f} "
          f"(tol {tol:.6f})")
    check("per-trial spread matches the analytic gamma-inflated sd",
          abs(float(sums.std()) - sd1) < 3.0 * sd1 / math.sqrt(2 * reps),
          f"observed {sums.std():.4f} vs analytic {sd1:.4f}")
    me = vm.mitigator_pec(N, q, shots=256, seed=5, expect=True)
    check("expect=True forces the deterministic limit even with shots set",
          me.info["sampled"] is False
          and np.allclose(me(uni), uni @ vm.pec_sector_matrix(N, q)[0]))


def main():
    for fn in (test_gamma_formula, test_matrix_matches_exact_inverse,
               test_overhead_is_state_independent, test_sampler_is_unbiased,
               test_variance_inflation, test_model_mismatch,
               test_mitigator_interface):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{_NPASS} passed, {_NFAIL} failed, {_NSKIP} skipped")
    return 1 if _NFAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
