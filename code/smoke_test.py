"""
smoke_test.py -- end-to-end check of the D-VAQEM pipeline.

Runs the whole stack on the stored N=8 VQ-CNNI decoder with exact Kraus
depolarising noise:

  1. decoder Jacobian (batched forward mode) vs autograd and finite differences;
  2. exactness of the m-sector reduction: FI(p_full) vs FI(p_m);
  3. calibration/test data from the density-matrix simulator (disk cached);
  4. every mitigation method (none, known-kernel linear inversion, ZNE,
     D-VAQEM linear/MLP with the ce, l2, fisher and shot-aware mse objectives,
     and the decoder-retraining baseline) on held-out phases;
  5. finite-shot error vs Monte Carlo, checked two ways: the delta-method
     variance in isolation, and the whole sample-then-mitigate pipeline through
     the chain rule that the shot-aware objective and E3 use;
  6. Fisher information / Cramer-Rao recovery.

Usage:  python smoke_test.py [--quick]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                              # noqa: E402
import sys                                                   # noqa: E402
import time                                                  # noqa: E402

import numpy as np                                           # noqa: E402
import autograd.numpy as anp                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vaqem_lib as vl                                       # noqa: E402
import vaqem_methods as vm                                   # noqa: E402
import run_experiments as rx                                 # noqa: E402

REPO = vl.vqcnni_root()          # companion VQ-CNNI checkout (see vaqem_lib)
DEC = os.path.join(REPO, "revision_experiments", "results", "vqcnni_N8_s0.npz")
NOISELESS = {"kind": "none", "p": 0.0, "readout_p": 0.0}

METHODS = [("lin_ce", "linear", "ce"), ("lin_l2", "linear", "l2"),
           ("lin_mse", "linear", "mse"), ("mlp_ce", "mlp", "ce"),
           ("mlp_l2", "mlp", "l2"), ("mlp_fisher", "mlp", "fisher"),
           ("mlp_mse", "mlp", "mse")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--p", type=float, default=0.02)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--mc-trials", type=int, default=40,
                    help="Monte-Carlo trials behind each analytic check in [4]")
    a = ap.parse_args()
    n_cal, n_tst = (7, 11) if a.quick else (13, 21)
    ok = True

    # ---------------- 0. frozen decoder ----------------
    model = vl.load_vqcnni(DEC)
    N = int(model["N"])
    theta, curly = model["theta"], model["curly"]
    print(f"decoder: N={N} hidden={model['hidden']} "
          f"n_params={vl.n_mlp_params(N, model['hidden'])}  "
          f"theta={np.round(theta, 4)}  curly={np.round(curly, 4)}", flush=True)

    # ---------------- 1. decoder Jacobian ----------------
    rng = np.random.RandomState(0)
    q = np.clip(rng.rand(6, N + 1), 1e-6, None)
    q /= q.sum(axis=1, keepdims=True)
    phi_g, g = vm.decoder_value_grad(anp.asarray(q), model["params"])
    phi_g, g = np.asarray(phi_g), np.asarray(g)
    phi_r = vl.predict_from_pm(q, model["params"])
    fd = np.zeros_like(g)
    for j in range(N + 1):
        e = np.zeros(N + 1)
        e[j] = 1e-6
        fd[:, j] = (vl.predict_from_pm(q + e, model["params"])
                    - vl.predict_from_pm(q - e, model["params"])) / 2e-6
    print(f"[1] phi vs predict_from_pm {np.abs(phi_g - phi_r).max():.2e}   "
          f"g vs central FD {np.abs(g - fd).max():.2e}", flush=True)
    ok &= np.abs(phi_g - phi_r).max() < 1e-12 and np.abs(g - fd).max() < 1e-5

    # ---------------- 2. exact data (cached) ----------------
    noise = {"kind": "depolarizing", "p": a.p, "readout_p": 0.0}
    cal_phis = np.linspace(-np.pi, np.pi, n_cal)
    tst_phis = np.linspace(-np.pi + np.pi / n_tst, np.pi + np.pi / n_tst, n_tst)
    t0 = time.time()
    cal = vm.build_case(N, theta, curly, cal_phis, noise, folds=(1,), workers=a.workers)
    tst = vm.build_case(N, theta, curly, tst_phis, noise, folds=(1, 3, 5),
                        workers=a.workers)
    print(f"[2] data in {time.time() - t0:.1f}s  clean sum "
          f"{cal['pm_clean'].sum(axis=1).mean():.9f}  noisy sum "
          f"{tst['pm_noisy'][1].sum(axis=1).mean():.9f}", flush=True)

    pr_tst, pm_tst = vm.sector_probs(tst_phis, N, theta, curly, NOISELESS, 1,
                                     a.workers, tst["masks"])
    fi_full = vl.fi_grid(tst_phis, pr_tst)
    fi_sec = vl.fi_grid(tst_phis, tst["pm_clean"])
    rel = np.abs(fi_full - fi_sec) / np.maximum(fi_full, 1e-30)
    print(f"[2] FI(p_full) vs FI(p_m): max rel dev {rel.max():.3e} "
          f"(mean FI {np.mean(fi_sec):.3f})", flush=True)
    ok &= rel.max() < 1e-6
    return _part2(a, model, N, theta, curly, noise, cal, tst, cal_phis, tst_phis, ok)


def _part2(a, dec, N, theta, curly, noise, cal, tst, cal_phis, tst_phis, ok):
    y_tst, t_tst = tst["pm_noisy"][1], tst["pm_clean"]
    base = {}

    def show(tag, pred, extra=""):
        d = vl.wrapped_err(pred, tst_phis)
        m2 = float(np.mean(d ** 2))
        print(f"    {tag:<24s} median SWPE {float(np.median(vl.swpe_db(pred, tst_phis))):8.3f} dB"
              f"   mean d^2 {m2:.3e}   max|d| {np.abs(d).max():.2e} {extra}", flush=True)
        base[tag] = m2
        return m2

    print("[3] infinite-shot evaluation on held-out phases", flush=True)
    show("noiseless (target)", vl.predict_from_pm(t_tst, dec["params"]))
    show("unmitigated", vl.predict_from_pm(y_tst, dec["params"]))

    f_eff = vl.effective_flip(N, noise, fold=1)
    linv = vm.mitigator_linv(vl.m_flip_kernel(N, f_eff))
    show(f"linv_known f={f_eff:.3f}",
         vl.predict_from_pm(vm._renorm(linv(y_tst)), dec["params"]))

    for kind, deg, tag in (("richardson", 1, "zne_richardson"),
                           ("poly", 1, "zne_poly1")):
        z = vm.mitigator_zne(tst["pm_noisy"], kind, deg)
        show(tag, vl.predict_from_pm(vm._renorm(z(y_tst)), dec["params"]),
             f"c={np.round(z.info['coeffs'], 3)}")

    trained = {}
    for tag, kind, loss in METHODS:
        shots = a.shots if loss == "mse" else None
        t1 = time.time()

        def _fit(psi0, kind=kind, loss=loss, shots=shots):
            return vm.train_map(cal, kind, cal_phis, loss=loss, iters=a.iters,
                                lr=5e-2, lam=1.0, shots=shots, decoder=dec,
                                seed=0, hidden=(32, 32), psi0=psi0)

        if loss == "mse":
            # mirror run_experiments.fit_maps: refine the shot-aware objective
            # from cold *and* from the l2/ce fits of the same family, keeping
            # whichever wins on the held-out calibration phases.  Neither
            # extreme alone is reliable -- a cold-started MLP fit can collapse
            # onto a constant map (a genuine local minimum of bias + variance),
            # while for the linear family the staged initialisation is itself a
            # bias that the cold fit does not have.
            pref = tag.rsplit("_", 1)[0]
            cands = [("cold", None)] + [(s, trained[f"{pref}_{s}"][1])
                                        for s in ("l2", "ce")
                                        if f"{pref}_{s}" in trained]
            best = None
            for src, p0 in cands:
                cand = _fit(p0)
                cand[2]["init"] = src
                cand[2]["holdout_mse"] = rx.holdout_score(
                    cal, dec, cand[0], cand[1], cand[2]["val_idx"],
                    shots=shots, seed=0)
                if best is None or cand[2]["holdout_mse"] < best[2]["holdout_mse"]:
                    best = cand
            mp, psi, info = best
        else:
            mp, psi, info = _fit(None)
        trained[tag] = (mp, psi, info)
        fam = vm.mitigated_family(mp, psi, y_tst)
        show(f"dvaqem_{tag}", vl.predict_from_pm(vm._renorm(fam), dec["params"]),
             f"[{time.time() - t1:.1f}s loss {info['final_loss']:.2e}"
             + (f" init {info['init']} holdout {info['holdout_mse']:.2e}]"
                if loss == "mse" else "]"))
        ok &= np.isfinite(info["final_loss"])
        # Normalisation is exact for both families and is checked strictly.
        # Non-negativity is a property of the *family*, not of the objective:
        #   mlp    -- MlpMap.apply ends in a softmax, so p~ >= 0 holds by
        #             construction under every objective (measured exactly 0.0 at
        #             both 300 and 2000 iters); checked to floating point.
        #   linear -- p~ = y @ T with the rows of T renormalised, so the sum is
        #             exact but positivity is not.  Only 'ce' carries a
        #             negativity term and it is the *soft* penalty
        #             lam*mean(max(-p~,0)^2), so a sharper fit may settle slightly
        #             outside the simplex (measured 0.0 at 300 iters, 2.6e-03 at
        #             2000).  The unpenalised objectives leave it by much more
        #             (measured l2: 1.2e-01 ... 1.6e-01 at N=8, p=0.02).
        # Either way this is legitimate quasi-probability behaviour -- the
        # pipeline clips at zero and renormalises with vm._renorm before every
        # decode -- so the mass is bounded rather than demanded to vanish, and
        # whatever is left is reported.
        neg_mass = float(np.clip(-fam, 0.0, None).sum(axis=1).max())
        if kind == "mlp":
            simplex_tol = 1e-12       # softmax output: non-negative by construction
        elif loss == "ce":
            simplex_tol = 2e-2        # soft penalty: ~8x margin over the 2.6e-03 seen
        else:
            simplex_tol = 0.25        # l2/mse/fisher: positivity is not penalised
        ok &= bool(np.allclose(fam.sum(axis=1), 1.0, atol=1e-8))
        ok &= bool(neg_mass <= simplex_tol)
        if neg_mass > 1e-12:
            print(f"      note: dvaqem_{tag} leaves the simplex by "
                  f"{neg_mass:.2e} (total negative mass, tol {simplex_tol:.0e}, "
                  f"clipped by _renorm)", flush=True)

    t1 = time.time()
    dec_rt, info_rt = vm.retrain_decoder(cal, dec, cal_phis, iters=2 * a.iters,
                                         lr=2e-3, loss="circular")
    show("retrain_decoder", vl.predict_from_pm(y_tst, dec_rt["params"]),
         f"[{time.time() - t1:.1f}s, {info_rt['n_params']} params]")

    # ---------------- 4. finite shots: analytic vs Monte Carlo --------------
    # Two identities are checked, because they fail independently.
    #
    # (a) vm.delta_var(q, g, S) is the leading-order variance of phi_hat(q_hat)
    #     for q_hat ~ Multinomial(q, S)/S.  Sampling directly from the mitigated
    #     distribution q~ isolates exactly that claim, and it has to hold for
    #     every family, including the non-linear MLP map.
    #
    # (b) vm.evaluate does *not* sample q~: it samples the noisy y and only then
    #     applies the map, phi_hat = decoder(M(y_hat)), so the shot noise is
    #     pushed through M as well.  The matching analytic variance is the chain
    #     rule c = (d phi_hat/d p~)(d p~/d y), i.e. run_experiments._analytic_map
    #     -- the same expression the shot-aware 'mse' objective minimises and the
    #     one E3 archives.  Reusing (a)'s expression here silently drops the map
    #     Jacobian and mistakes M(y) for the sampling distribution; the two agree
    #     only when M = I, which is why the unmitigated row matches while a
    #     non-linear map is off by exactly its noise amplification (measured 7.5x
    #     for the MLP map at N=8, p=0.02, S=1024).
    #
    # Tolerance: a two-sided factor 2.  Over 12 seeds x 40 trials the measured
    # MC/analytic ratio spans 0.954-1.094 for (a) and 0.905-1.117 for (b), so the
    # bound keeps ~1.8x of Monte-Carlo headroom while still rejecting any
    # factor-2 structural error -- in particular the 7.5x one above, which a
    # one-sided bound at 4x only caught for some tags.
    print(f"[4] finite-shot check at S={a.shots}, {a.mc_trials} Monte-Carlo trials",
          flush=True)
    for tag in ("none", "lin_mse", "mlp_mse"):
        if tag == "none":
            q_ex, mit, mp, psi = y_tst, None, None, None
        else:
            mp, psi, _ = trained[tag]
            q_ex = vm.mitigated_family(mp, psi, y_tst)
            mit = vm.Mitigator(tag, lambda y, mp=mp, psi=psi: vm.mitigated_family(mp, psi, y))
        qr = vm._renorm(q_ex)
        # (a) delta_var in isolation: sample the mitigated distribution itself
        _, g = vm.decoded_grad(qr, dec)
        am_iso = vm.analytic_mse(vl.predict_from_pm(qr, dec["params"]), tst_phis,
                                 qr, g, a.shots)
        mc_iso = vm.evaluate(qr, tst_phis, dec, mitigator=None, shots=a.shots,
                             n_trials=a.mc_trials, seed=1)
        # (b) whole pipeline: chain rule through the frozen map (the E3 path)
        am_ch = (rx._analytic_none(y_tst, tst_phis, dec["params"], a.shots)
                 if mp is None else
                 rx._analytic_map(mp, psi, y_tst, tst_phis, dec["params"], a.shots))
        mc = vm.evaluate(y_tst, tst_phis, dec, mitigator=mit, shots=a.shots,
                         n_trials=a.mc_trials, seed=1)
        r_iso = mc_iso["mean_swpe"] / max(am_iso["pred_mse"], 1e-30)
        r_ch = mc["mean_swpe"] / max(am_ch["analytic_mse"], 1e-30)
        print(f"    {tag:<9s} MC mse {mc['mean_swpe']:.4e}   "
              f"analytic {am_ch['analytic_mse']:.4e}"
              f"  = bias^2 {am_ch['bias2']:.3e} + var {am_ch['var_over_S']:.3e}   "
              f"ratio {r_ch:.3f}   [delta_var alone {am_iso['pred_mse']:.4e}, "
              f"ratio {r_iso:.3f}]", flush=True)
        ok &= 0.5 <= r_ch <= 2.0
        ok &= 0.5 <= r_iso <= 2.0
        base[f"finite_{tag}"] = mc["mean_swpe"]
        base[f"finite_{tag}_analytic"] = am_ch["analytic_mse"]
        base[f"finite_{tag}_ratio_chain"] = r_ch
        base[f"finite_{tag}_ratio_delta"] = r_iso

    # ---------------- 5. Fisher information / CRB ----------------
    mp, psi, _ = trained["mlp_fisher"]
    fam = vm._renorm(vm.mitigated_family(mp, psi, y_tst))
    grid, fis = vm.fi_curves(tst_phis, {"clean": t_tst, "noisy": y_tst,
                                        "mitigated": fam})
    print("[5] classical FI(phi) and Cramer-Rao bound (mean over test grid)", flush=True)
    for k, v in fis.items():
        print(f"    {k:<10s} FI {np.mean(v):10.4f}   CRB {vl.crb_db(np.mean(v)):8.3f} dB",
              flush=True)
        base[f"fi_{k}"] = float(np.mean(v))
    ok &= np.mean(fis["mitigated"]) > np.mean(fis["noisy"])

    print("\nSMOKE TEST:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

