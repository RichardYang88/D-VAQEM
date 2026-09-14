"""
train_decoder.py -- train a VQ-CNNI phase decoder using *only* the new_paper
simulator (numpy + autograd; no PennyLane), so that the system-size scaling
study can be extended to qubit numbers for which no checkpoint exists.

Architecture and hyper-parameters mirror
``revision_experiments/train_vqcnni_scaling.py`` (which reproduces the original
``vqc_mlp_softsign.ipynb`` notebook exactly):

  * circuit   1 encoding + 1 decoding layer, 6 variational parameters
              x_q = [theta(3), curly(3)]  (``vl.build_gates``)
  * decoder   (N+1) -> H -> H//2 -> 2 MLP, softsign hidden layers,
              L2-normalised output, phi_hat = arctan2(v0, v1)
              (``vl.mlp_shapes`` / ``vl.net_batch``, H = 128 as in the paper)
  * data      100 uniform phases on [-pi, pi], *clean* sector distributions
  * loss      circular SWPE   2 * mean(1 - cos(phi_hat - phi))
  * optimiser Adam(lr = 0.02) on circuit + MLP parameters jointly
  * selection test loss on the 50 half-offset phases, early stopping with
              patience (min_iters guard), exactly as in the original script

Gradients.  The MLP part is exact (autograd through ``vl.net_batch``).  The
clean simulator is plain numpy, so the 6 circuit gradients are obtained by
central finite differences: 12 extra simulations per circuit step.  All
simulations of one step are dispatched to a single persistent process pool,
which keeps one joint step at ~1 s for N = 10 on 12 cores.  Because the MLP
gradient needs no simulation at all, training is staged: (1) joint circuit+MLP
steps, (2) MLP-only steps on the cached distributions (free), (3) a short joint
polish.

The checkpoint format is the one ``vl.load_vqcnni`` reads, so a model trained
here is interchangeable with the PennyLane-trained ones:

    python train_decoder.py --N 10 --out ../models/vqcnni_N10_s0.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
from multiprocessing import Pool                            # noqa: E402

import numpy as np                                          # noqa: E402
import autograd                                             # noqa: E402
import autograd.numpy as anp                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vaqem_lib as vl                                      # noqa: E402

MODELS = os.path.join(HERE, os.pardir, "models")


# ----------------------------------------------------------------------
# parallel clean simulation of the sector distribution
# ----------------------------------------------------------------------
_MASKS = {}


def _masks(N):
    if N not in _MASKS:
        _MASKS[N] = vl.m_structure(N)[2]
    return _MASKS[N]


def _sim_pm(args):
    """Worker: clean m-distribution p_m(phi) for one phase."""
    phi, N, theta, curly = args
    pr = vl.probs_pure_real(phi, N, theta, curly, 1)
    return vl.probs_to_p_m(pr, _masks(N))


def _pm_batch(phis, N, theta, curly, pool):
    args = [(float(p), N, theta, curly) for p in phis]
    if pool is None:
        return np.stack([_sim_pm(a) for a in args])
    return np.stack(pool.map(_sim_pm, args, chunksize=1))


# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------
def swpe(pred, phi):
    """Circular squared wrapped phase error, averaged (2(1-cos) form)."""
    return 2.0 * anp.mean(1.0 - anp.cos(pred - phi))


def make_costs(N, hidden, phis):
    """(autograd MLP cost, MLP gradient, plain-numpy loss) on a fixed grid."""
    def cost_mlp(p_mlp, pm):
        pred = vl.predict_from_pm(pm, vl.unpack_mlp(p_mlp, N, hidden))
        return swpe(pred, phis)

    grad_mlp = autograd.grad(lambda p, pm: cost_mlp(p, pm))

    def loss_val(p_mlp, pm):
        return float(cost_mlp(p_mlp, pm))

    return cost_mlp, grad_mlp, loss_val


# ----------------------------------------------------------------------
# the trainer
# ----------------------------------------------------------------------
def init_params(N, hidden, seed, init_seed):
    """x = [theta(3), curly(3), mlp_flat] with the notebook initialisation."""
    rng = np.random.RandomState(seed)
    xq = (rng.uniform(-0.1, 0.1, 6) if init_seed is None
          else np.random.RandomState(init_seed).uniform(-0.1, 0.1, 6))
    rs = np.random.RandomState(0)
    flat = []
    for shp in vl.mlp_shapes(N, hidden):
        lim = 1.0 / np.sqrt(shp[-1])
        flat.append(rs.uniform(-lim, lim, shp).ravel())
    return np.concatenate([xq, np.concatenate(flat)])


def train(N, hidden=128, seed=0, init_seed=42, lr=0.02, n_train=100,
          n_test=50, joint_iters=300, mlp_iters=2000, polish_iters=200,
          patience=150, min_iters=500, eval_every=25, fd_h=1e-3, workers=12,
          verbose=True, out=None):
    """Train the decoder; returns (x_best, info)."""
    phis = np.linspace(-np.pi, np.pi, n_train)
    phi_te = np.linspace(-np.pi + np.pi / n_train, np.pi + np.pi / n_train,
                         n_test)
    x = init_params(N, hidden, seed, init_seed)
    nq = 6
    n_mlp = vl.n_mlp_params(N, hidden)
    _, grad_mlp, loss_val = make_costs(N, hidden, phis)
    _, _, loss_te = make_costs(N, hidden, phi_te)
    opt = vl.Adam(lr=lr)
    pool = Pool(processes=workers) if workers > 1 else None

    def circuit_grad(p_mlp, theta, curly):
        """Central finite differences of the loss w.r.t. the 6 circuit params."""
        g = np.zeros(nq)
        for i in range(nq):
            for sgn in (+1.0, -1.0):
                t2, c2 = theta.copy(), curly.copy()
                if i < 3:
                    t2[i] += sgn * fd_h
                else:
                    c2[i - 3] += sgn * fd_h
                pm = _pm_batch(phis, N, t2, c2, pool)
                g[i] += sgn * loss_val(p_mlp, pm) / (2.0 * fd_h)
        return g

    t0 = time.time()
    best = {"loss": np.inf, "x": x.copy(), "iter": 0}
    hist, it = [], 0
    stages = (("joint", joint_iters), ("mlp", mlp_iters),
              ("joint", polish_iters))
    for stage, n_it in stages:
        if n_it <= 0:
            continue
        pm_tr = pm_te = None
        q_frozen = x[:6].copy()            # circuit block held fixed in "mlp"
        for _ in range(n_it):
            it += 1
            theta, curly = x[:3].copy(), x[3:6].copy()
            p_mlp = x[6:].copy()
            if pm_tr is None:
                pm_tr = _pm_batch(phis, N, theta, curly, pool)
            gq = (circuit_grad(p_mlp, theta, curly) if stage == "joint"
                  else np.zeros(nq))
            gm = np.asarray(grad_mlp(p_mlp, pm_tr))
            x = opt.step(x, np.clip(np.nan_to_num(np.concatenate([gq, gm])),
                                    -1e6, 1e6))
            if stage == "joint":
                pm_tr = None                  # circuit moved -> cache stale
            else:
                # Adam momentum from the previous stage would otherwise keep
                # dragging the frozen circuit parameters around.
                x[:6] = q_frozen
            if it % eval_every == 0 or it == 1:
                if pm_te is None or stage == "joint":
                    pm_te = _pm_batch(phi_te, N, x[:3], x[3:6], pool)
                l_te = loss_te(x[6:], pm_te)
                hist.append({"iter": it, "stage": stage, "test": float(l_te),
                             "train": (float(loss_val(x[6:], pm_tr))
                                       if pm_tr is not None else None),
                             "t_s": round(time.time() - t0, 1)})
                if l_te < best["loss"]:
                    best = {"loss": float(l_te), "x": x.copy(), "iter": it}
                if verbose:
                    print(f"  [{stage:5s}] iter {it:5d}  test {l_te:.6e}"
                          f"   best {best['loss']:.6e} @{best['iter']}"
                          f"   ({time.time() - t0:.0f}s)", flush=True)
                if it > min_iters and (it - best["iter"]) > patience:
                    if verbose:
                        print(f"  early stop at iter {it}", flush=True)
                    break
        pm_te = None if stage == "joint" else pm_te
    if pool is not None:
        pool.close()
        pool.join()
    return _finish(x, best, hist, it, t0, N, hidden, seed, init_seed, lr,
                   n_train, n_test, joint_iters, mlp_iters, polish_iters,
                   patience, min_iters, eval_every, fd_h, workers, phi_te,
                   nq, n_mlp, out, verbose)


def _finish(x, best, hist, it, t0, N, hidden, seed, init_seed, lr, n_train,
            n_test, joint_iters, mlp_iters, polish_iters, patience, min_iters,
            eval_every, fd_h, workers, phi_te, nq, n_mlp, out, verbose):
    """Evaluate the best iterate, fill in the metadata and save the checkpoint."""
    xb = best["x"]
    pm_te = _pm_batch(phi_te, N, xb[:3], xb[3:6], None)
    pred = np.asarray(vl.predict_from_pm(pm_te, vl.unpack_mlp(xb[6:], N, hidden)))
    d2 = vl.wrapped_err(pred, phi_te) ** 2
    info = {"N": N, "hidden": hidden, "seed": seed, "init_seed": init_seed,
            "lr": lr, "n_train": n_train, "n_test": n_test,
            "joint_iters": joint_iters, "mlp_iters": mlp_iters,
            "polish_iters": polish_iters, "patience": patience,
            "min_iters": min_iters, "eval_every": eval_every, "fd_h": fd_h,
            "workers": workers, "best_iter": best["iter"],
            "best_test_loss": float(best["loss"]), "total_iters": it,
            "time_s": round(time.time() - t0, 1), "n_quantum": nq,
            "n_classical": n_mlp, "n_params_total": nq + n_mlp,
            "model": "VQ-CNNI", "activation": "softsign",
            "trainer": "new_paper/code/train_decoder.py (numpy+autograd; "
                       "finite-difference circuit gradients)",
            "test_mse": float(np.mean(d2)),
            "test_mse_db": float(10 * np.log10(np.mean(d2))),
            "test_median_swpe_db": float(np.median(vl.swpe_db(pred, phi_te))),
            "theta": np.asarray(xb[:3]).tolist(),
            "curly": np.asarray(xb[3:6]).tolist(), "history": hist}
    if verbose:
        print(f"  best @ iter {best['iter']}: test loss {best['loss']:.4e}, "
              f"MSE {info['test_mse']:.3e} ({info['test_mse_db']:.2f} dB), "
              f"median SWPE {info['test_median_swpe_db']:.2f} dB", flush=True)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(
            out, x_best=xb, x_final=x, theta_best=xb[:3], curly_best=xb[3:6],
            mlp_flat_best=xb[6:], phi_trues=phi_te, phi_preds=pred,
            swpe_db=vl.swpe_db(pred, phi_te), meta=json.dumps(info))
        print(f"  -> wrote {out}", flush=True)
    return xb, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="train a VQ-CNNI decoder")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--joint_iters", type=int, default=300)
    ap.add_argument("--mlp_iters", type=int, default=2000)
    ap.add_argument("--polish_iters", type=int, default=200)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--min_iters", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--fd_h", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None,
                    help="default: ../models/vqcnni_N{N}_s{seed}.npz")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(MODELS, f"vqcnni_N{a.N}_s{a.seed}.npz")
    print(f"== training VQ-CNNI decoder, N={a.N}, hidden={a.hidden}, "
          f"seed={a.seed}", flush=True)
    _, info = train(N=a.N, hidden=a.hidden, seed=a.seed,
                    init_seed=a.init_seed, lr=a.lr, n_train=a.n_train,
                    n_test=a.n_test, joint_iters=a.joint_iters,
                    mlp_iters=a.mlp_iters, polish_iters=a.polish_iters,
                    patience=a.patience, min_iters=a.min_iters,
                    eval_every=a.eval_every, fd_h=a.fd_h, workers=a.workers,
                    out=out)
    print(json.dumps({k: v for k, v in info.items()
                      if k not in ("history",)}, indent=1), flush=True)
    return info


if __name__ == "__main__":
    main()
