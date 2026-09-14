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
  5. delta-method finite-shot variance vs Monte Carlo;
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

REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
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
        # Every fitted map is affine in the sector probabilities, so it preserves
        # the normalisation exactly; that is checked strictly for all objectives.
        # Non-negativity, however, is a property of the *objective*: ce, fisher
        # and the shot-aware mse are only defined on (or weighted by) the simplex
        # and stay inside it, whereas the plain l2 fit is an unconstrained affine
        # map and can leave it by a small amount (measured: total negative mass
        # 7e-2 ... 2e-1 at N=8, p=0.02).  That is legitimate quasi-probability
        # behaviour -- the pipeline clips at zero and renormalises with
        # vm._renorm before every decode -- so for l2 we bound the total
        # negative mass instead of demanding exact positivity, and report it.
        neg_mass = float(np.clip(-fam, 0.0, None).sum(axis=1).max())
        simplex_tol = 0.25 if loss == "l2" else 1e-12
        ok &= bool(np.allclose(fam.sum(axis=1), 1.0, atol=1e-8))
        ok &= bool(neg_mass <= simplex_tol)
        if neg_mass > 1e-12:
            print(f"      note: dvaqem_{tag} leaves the simplex by "
                  f"{neg_mass:.2e} (total negative mass, clipped by _renorm)",
                  flush=True)

    t1 = time.time()
    dec_rt, info_rt = vm.retrain_decoder(cal, dec, cal_phis, iters=2 * a.iters,
                                         lr=2e-3, loss="circular")
    show("retrain_decoder", vl.predict_from_pm(y_tst, dec_rt["params"]),
         f"[{time.time() - t1:.1f}s, {info_rt['n_params']} params]")

    # ---------------- 4. finite shots: delta method vs Monte Carlo ----------
    print(f"[4] finite-shot check at S={a.shots}, 40 Monte-Carlo trials", flush=True)
    for tag in ("none", "lin_mse", "mlp_mse"):
        if tag == "none":
            q_ex = y_tst
            mit = None
        else:
            mp, psi, _ = trained[tag]
            q_ex = vm.mitigated_family(mp, psi, y_tst)
            mit = vm.Mitigator(tag, lambda y, mp=mp, psi=psi: vm.mitigated_family(mp, psi, y))
        pred = vl.predict_from_pm(vm._renorm(q_ex), dec["params"])
        _, g = vm.decoded_grad(vm._renorm(q_ex), dec)
        am = vm.analytic_mse(pred, tst_phis, vm._renorm(q_ex), g, a.shots)
        mc = vm.evaluate(y_tst, tst_phis, dec, mitigator=mit, shots=a.shots,
                         n_trials=40, seed=1)
        print(f"    {tag:<9s} MC mse {mc['mean_swpe']:.4e}   analytic {am['pred_mse']:.4e}"
              f"  = bias^2 {am['bias2']:.3e} + var {am['var_delta']:.3e}   ratio "
              f"{mc['mean_swpe'] / max(am['pred_mse'], 1e-30):.2f}", flush=True)
        ok &= mc["mean_swpe"] < 4.0 * max(am["pred_mse"], 1e-12) + 1e-8
        base[f"finite_{tag}"] = mc["mean_swpe"]

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

