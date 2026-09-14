"""
validate_simulator.py -- cross-validation of the self-contained VAQEM
simulator (``vaqem_lib``) against

  (1) internal consistency of its three independent code paths
      (complex statevector, real statevector, Kraus density matrix);
  (2) the *stored* PennyLane ``default.mixed`` results of the VQ-CNNI
      revision study (``revision_experiments/results/*.npz``), i.e. the
      noiseless predictions, the QFI and the noisy SWPE/FI curves under
      gate depolarising and readout noise;
  (3) PennyLane itself, by rebuilding the same circuit with qml ops on
      ``default.mixed`` and comparing outcome probabilities (skipped
      automatically when PennyLane is not importable).

Usage
-----
    python validate_simulator.py [--quick] [--no-pl] [--N-pl 4]

Exit status is 0 when every tolerance check passes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vaqem_lib as vl  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
RES = os.path.join(REPO, "revision_experiments", "results")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def close(a, b, tol):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) <= tol


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ----------------------------------------------------------------------
# 1. internal consistency of the three simulator code paths
# ----------------------------------------------------------------------
def internal_checks(N=6, tol=1e-10):
    rng = np.random.RandomState(0)
    theta = rng.uniform(-1, 1, 3)
    curly = rng.uniform(-1, 1, 3)
    phi = 0.731

    p_c = vl.probs_pure(phi, N, theta, curly)
    p_r = vl.probs_pure_real(phi, N, theta, curly)
    p_d = vl.probs_dm(phi, N, theta, curly, "none", 0.0)
    check("pure == pure_real", close(p_c, p_r, tol),
          f"max|d|={maxdiff(p_c, p_r):.2e}")
    check("pure == dm(no noise)", close(p_c, p_d, tol),
          f"max|d|={maxdiff(p_c, p_d):.2e}")
    check("probs normalised", abs(p_c.sum() - 1) < 1e-12,
          f"sum={p_c.sum():.15f}")

    for fold in (3, 5):
        pf = vl.probs_pure(phi, N, theta, curly, fold=fold)
        check(f"fold={fold} noiseless invariance", close(p_c, pf, 1e-9),
              f"max|d|={maxdiff(p_c, pf):.2e}")

    Lq = vl.gates_per_qubit(N)
    check("channels per qubit = 4(N-1)+9", bool(np.all(Lq == 4 * (N - 1) + 9)),
          f"{Lq}")

    worst = 0.0
    for gp in vl.build_gates(N, phi, theta, curly):
        if gp[0] == 'cnot':
            continue
        M = vl.gate_matrix(gp)
        worst = max(worst, maxdiff(vl._mr_mi_block(M.real, M.imag),
                                   vl.gate_real_block(gp)))
    check("gate_real_block == real form of gate_matrix", worst < 1e-12,
          f"max|d|={worst:.2e}")

    import autograd
    w = np.arange(2 ** N, dtype=float)

    def f(th):
        return np.sum(vl.probs_pure_real(phi, N, th, curly) * w)

    g = autograd.grad(f)(theta)
    eps = 1e-6
    gn = np.array([(f(theta + eps * np.eye(3)[i])
                    - f(theta - eps * np.eye(3)[i])) / (2 * eps)
                   for i in range(3)])
    check("autograd gradient of pure_real", close(g, gn, 1e-6),
          f"max|d|={maxdiff(g, gn):.2e}")

    index_to_m, unique_m, masks = vl.m_structure(N)
    A = vl.m_aggregate_matrix(N)
    f = 0.07
    Kf = vl.flip_kernel_full(N, f)
    Km = vl.m_flip_kernel(N, f)
    p = rng.dirichlet(np.ones(2 ** N))
    check("apply_flip == flip_kernel_full", close(vl.apply_flip(p, N, f), Kf @ p, 1e-12),
          f"max|d|={maxdiff(vl.apply_flip(p, N, f), Kf @ p):.2e}")
    check("A K_full == K_m A (exact sector reduction)",
          close(A @ (Kf @ p), Km @ (A @ p), 1e-12),
          f"max|d|={maxdiff(A @ (Kf @ p), Km @ (A @ p)):.2e}")
    check("K_m column sums = 1", close(Km.sum(0), np.ones(N + 1), 1e-12), "")

    p_in = vl.probs_dm(phi, N, theta, curly, "none", 0.0, readout_p=f)
    check("probs_dm(readout_p) == apply_flip(probs_pure)",
          close(p_in, vl.apply_flip(p_c, N, f), 1e-12),
          f"max|d|={maxdiff(p_in, vl.apply_flip(p_c, N, f)):.2e}")

    for kind in ("depolarizing", "bit_flip", "dephasing", "amplitude_damping"):
        K = vl.kraus_channel(kind, 0.13)
        S = sum(k.conj().T @ k for k in K)
        check(f"{kind} Kraus trace preserving", close(S, np.eye(2), 1e-12),
              f"max|d|={maxdiff(S, np.eye(2)):.2e}")


# ----------------------------------------------------------------------
# 2. reproduce the stored PennyLane results of the VQ-CNNI revision study
# ----------------------------------------------------------------------
def _model(N=8, fname="vqcnni_N8_s0.npz"):
    path = os.path.join(RES, fname)
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None
    d = np.load(path, allow_pickle=True)
    x = np.asarray(d["x_best"], dtype=float)
    meta = json.loads(str(d["meta"]))
    N = int(meta.get("N", N))
    return {"d": d, "x": x, "N": N, "meta": meta, "theta": x[:3],
            "curly": x[3:6],
            "params": vl.unpack_mlp(x[6:], N, int(meta.get("hidden", 128)))}


def _pm_of(phis, M, noise=None, fold=1, exact=True):
    """m-sector distributions (len(phis), N+1) of model M under `noise`."""
    N, theta, curly = M["N"], M["theta"], M["curly"]
    masks = vl.m_structure(N)[2]
    noise = noise or {"kind": "none", "p": 0.0, "readout_p": 0.0}
    f_eq = vl.effective_flip(N, noise, fold)
    if exact and f_eq is not None:
        # flip-equivalent: one noiseless simulation + the exact linear kernel
        pr = np.stack([vl.probs_pure(p, N, theta, curly) for p in phis])
        if f_eq > 0:
            pr = vl.apply_flip(pr, N, f_eq)
    else:
        pr = np.stack([vl.probs_dm(p, N, theta, curly, noise["kind"],
                                   noise["p"], fold=fold,
                                   readout_p=noise.get("readout_p", 0.0))
                       for p in phis])
    return vl.probs_to_p_m(pr, masks), pr


def stored_model_checks():
    M = _model()
    if M is None:
        return
    N, theta, curly = M["N"], M["theta"], M["curly"]
    phi_test = vl.test_phases(100, 50)[1]
    pm, _ = _pm_of(phi_test, M)
    preds = vl.predict_from_pm(pm, M["params"])
    stored = np.asarray(M["d"]["phi_preds"], dtype=float)
    check("noiseless phi_preds reproduce stored PennyLane model (mod 2pi)",
          close(vl.wrapped_err(preds, stored), 0.0, 1e-8),
          f"max|wrapped d|={maxdiff(vl.wrapped_err(preds, stored), 0.0):.2e}")
    sw = vl.swpe_db(preds, phi_test)
    check("noiseless swpe_db reproduces stored PennyLane model",
          close(sw, np.asarray(M["d"]["swpe_db"]), 1e-6),
          f"median={np.median(sw):.4f} dB")

    q = vl.qfi_pure(N, theta, curly)
    check("qfi_pure == stored qfi", abs(q - float(M["d"]["qfi"])) < 1e-6,
          f"mine={q:.9f} stored={float(M['d']['qfi']):.9f}")

    phis = np.linspace(-np.pi, np.pi, 401)
    pr = np.stack([vl.probs_pure(p, N, theta, curly) for p in phis])
    masks = vl.m_structure(N)[2]
    fi_m = vl.fi_grid(phis, vl.probs_to_p_m(pr, masks))
    fi_full = vl.fi_grid(phis, pr)
    rel = maxdiff(fi_m, fi_full) / max(fi_full.max(), 1e-30)
    check("m-sector readout is information preserving: FI(m) == FI(full)",
          rel < 1e-4, f"max rel diff={rel:.2e}")
    print(f"        FI(m): min={fi_m.min():.4f} median={np.median(fi_m):.4f} "
          f"max={fi_m.max():.4f} mean={fi_m.mean():.4f} | QFI={q:.4f} | "
          f"FI(phi=0)={fi_m[200]:.4f}", flush=True)
    # random-parameter sanity check of the same statement
    rng = np.random.RandomState(11)
    rels = []
    for _ in range(3):
        th2, cu2 = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        pr2 = np.stack([vl.probs_pure(p, N, th2, cu2) for p in phis[::4]])
        a = vl.fi_grid(phis[::4], vl.probs_to_p_m(pr2, masks))
        b = vl.fi_grid(phis[::4], pr2)
        rels.append(maxdiff(a, b) / max(b.max(), 1e-30))
    check("FI(m) == FI(full) also for random circuit parameters",
          max(rels) < 1e-4, f"max rel diff={max(rels):.2e}")



def stored_noise_checks(quick=False):
    """Compare against noise_eval_{depol,readout}_N8.npz (default.mixed)."""
    M = _model()
    if M is None:
        return
    N = M["N"]
    phi_test = vl.test_phases(100, 50)[1]
    phis = phi_test[::5] if quick else phi_test
    masks = vl.m_structure(N)[2]
    pr_clean = np.stack([vl.probs_pure(p, N, M["theta"], M["curly"])
                         for p in phis])

    path = os.path.join(RES, "noise_eval_depol_N8.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        levels = np.asarray(d["levels"], dtype=float)
        sw_st = np.asarray(d["med_swpe_db"], dtype=float)
        fi_st = np.asarray(d["fi"], dtype=float)
        for k, p in enumerate(levels):
            noise = {"kind": "depolarizing", "p": float(p)}
            pm, _ = _pm_of(phis, M, noise, exact=False)     # exact Kraus DM
            preds = vl.predict_from_pm(pm, M["params"])
            sw_all = vl.swpe_db(preds, phis)
            sw = float(np.median(sw_all)) if not quick else float(np.mean(sw_all))
            sw_ref = sw_st[k] if not quick else None
            if sw_ref is not None:
                check(f"depol p={p}: median SWPE vs stored default.mixed",
                      abs(sw - sw_ref) < 0.4, f"mine={sw:.3f} stored={sw_ref:.3f} dB")
            fq = vl.qfi_mixed(N, M["theta"], M["curly"], noise)
            check(f"depol p={p}: SLD-QFI vs stored default.mixed",
                  abs(fq - fi_st[k]) < 0.02 * max(1.0, abs(fi_st[k])),
                  f"mine={fq:.5f} stored={fi_st[k]:.5f}")
            f_eq = vl.depol_equiv_flip(N, float(p))
            pm_sur = vl.probs_to_p_m(vl.apply_flip(pr_clean, N, f_eq), masks)
            print(f"        f_eq={f_eq:.6f}  max|d p_m| surrogate vs exact "
                  f"Kraus = {maxdiff(pm_sur, pm):.3e}", flush=True)
    else:
        print(f"[SKIP] {path} not found")

    path = os.path.join(RES, "noise_eval_readout_N8.npz")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return
    d = np.load(path, allow_pickle=True)
    levels = np.asarray(d["levels"], dtype=float)
    sw_st = np.asarray(d["med_swpe_db"], dtype=float)
    fi_st = np.asarray(d["fi"], dtype=float)
    for k, q in enumerate(levels):
        noise = {"kind": "none", "p": 0.0, "readout_p": float(q)}
        pm, _ = _pm_of(phis, M, noise, exact=True)
        preds = vl.predict_from_pm(pm, M["params"])
        sw_all = vl.swpe_db(preds, phis)
        if not quick:
            check(f"readout q={q}: median SWPE vs stored PennyLane kernel",
                  abs(float(np.median(sw_all)) - sw_st[k]) < 0.05,
                  f"mine={float(np.median(sw_all)):.4f} stored={sw_st[k]:.4f} dB")
        dphi = 1e-5
        pm_p, _ = _pm_of([dphi], M, noise, exact=True)
        pm_m, _ = _pm_of([-dphi], M, noise, exact=True)
        pm0, dpm = (pm_p + pm_m) / 2, (pm_p - pm_m) / (2 * dphi)
        ok = pm0 > 1e-12
        fi = float(np.sum(np.where(ok, dpm ** 2 / np.maximum(pm0, 1e-300), 0.0)))
        check(f"readout q={q}: classical FI(phi=0) vs stored",
              abs(fi - fi_st[k]) < 1e-4, f"mine={fi:.6f} stored={fi_st[k]:.6f}")


# ----------------------------------------------------------------------
# 3. direct cross-check against PennyLane default.mixed
# ----------------------------------------------------------------------
def pennylane_checks(N=4, phi=0.61, quick=False):
    try:
        import pennylane as qml
    except Exception as exc:                                  # pragma: no cover
        print(f"[SKIP] PennyLane not importable ({exc})")
        return
    print(f"PennyLane {qml.version()} cross-check at N={N}")
    rng = np.random.RandomState(3)
    theta, curly = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
    dev = qml.device("default.mixed", wires=N)

    _G = {"rx": qml.RX, "ry": qml.RY, "rz": qml.RZ}
    _C = {"depolarizing": qml.DepolarizingChannel, "bit_flip": qml.BitFlip,
          "dephasing": qml.PhaseFlip, "amplitude_damping": qml.AmplitudeDamping}

    def make(kind, p, fold):
        gates = vl.fold_gates(vl.build_gates(N, phi, theta, curly), fold)

        @qml.qnode(dev, interface="autograd")
        def q_probs():
            for gp in gates:
                if gp[0] == 'cnot':
                    qml.CNOT(wires=[gp[1], gp[2]])
                elif gp[0] == 'h':
                    qml.Hadamard(wires=gp[1])
                else:
                    _G[gp[0]](gp[2], wires=gp[1])
                if kind in _C and p > 0:
                    for q in gp[3]:
                        _C[kind](p, wires=q)
            return qml.probs(wires=list(range(N)))

        return q_probs

    cases = [("none", 0.0, 1), ("none", 0.0, 5), ("depolarizing", 0.02, 1),
             ("depolarizing", 0.01, 3), ("dephasing", 0.03, 1),
             ("amplitude_damping", 0.02, 1), ("bit_flip", 0.02, 3)]
    if quick:
        cases = cases[:4]
    for kind, p, fold in cases:
        t0 = time.time()
        p_pl = np.asarray(make(kind, p, fold)(), dtype=float)
        p_me = vl.probs_dm(phi, N, theta, curly, kind, p, fold=fold)
        d = maxdiff(p_pl, p_me)
        check(f"PennyLane default.mixed vs Kraus DM: {kind} p={p} fold={fold}",
              d < 1e-9, f"max|d|={d:.2e}  ({time.time()-t0:.1f} s PL)")

    # readout bit flips: PennyLane BitFlip before measurement == my kernel
    q = 0.05
    gates = vl.build_gates(N, phi, theta, curly)

    @qml.qnode(dev, interface="autograd")
    def q_ro():
        for gp in gates:
            if gp[0] == 'cnot':
                qml.CNOT(wires=[gp[1], gp[2]])
            elif gp[0] == 'h':
                qml.Hadamard(wires=gp[1])
            else:
                _G[gp[0]](gp[2], wires=gp[1])
        for w in range(N):
            qml.BitFlip(q, wires=w)
        return qml.probs(wires=list(range(N)))

    d = maxdiff(np.asarray(q_ro(), dtype=float),
                vl.probs_dm(phi, N, theta, curly, "none", 0.0, readout_p=q))
    check("PennyLane readout BitFlip vs probs_dm(readout_p)", d < 1e-9,
          f"max|d|={d:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-pl", action="store_true")
    ap.add_argument("--N-pl", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    internal_checks(N=6)
    stored_model_checks()
    stored_noise_checks(quick=args.quick)
    if not args.no_pl:
        pennylane_checks(N=args.N_pl, quick=args.quick)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed  ({time.time()-t0:.1f} s)")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())



