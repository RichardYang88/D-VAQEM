"""
run_sufficiency_swpe.py -- E1: the sector reduction is free, measured in SWPE.

The claim that aggregating the 2^N computational-basis outcomes onto the N+1
collective-imbalance sectors costs the estimator nothing is stated here *on the
error scale the rest of the paper uses*, i.e. as squared wrapped phase error in
dB, and not as an information-theoretic identity.  For every system size N three
readout arms are compared on identical data:

  sector_frozen  D_w    : R^{N+1} -> phase   the certified VQ-CNNI checkpoint
  sector         D_w'   : R^{N+1} -> phase   retrained here, matched protocol
  full           D_w^F  : R^{2^N} -> phase   retrained here, matched protocol

The ``sector`` arm is the control that makes the comparison fair: the certified
checkpoint was obtained by *joint* circuit-plus-decoder training, so its residual
fitting error is not comparable to that of a freshly trained network, and a gap
between ``full`` and ``sector_frozen`` would mix the price of the reduction with
the difference in fitting quality.  ``sector`` and ``full`` are matched in every
respect that could bias the comparison:

  * the same frozen probe circuit (theta, curly) of the checkpoint, so the full
    decoder gets no chance to re-optimise the quantum part;
  * the same softsign MLP of hidden width 128 with an L2-normalised 2-vector
    output read out as atan2, i.e. the same architecture family, differing only
    in the width of the input layer;
  * the same 100 noiseless training phases, the same circular loss
    2[1-cos(phi_hat-phi)], the same Adam(lr=0.02), and the same early-stopping
    rule on the same 50 half-offset phases (``train_decoder.py``);
  * the same evaluation grid, the same noise settings, and -- at finite shots --
    the *same* multinomial draws: the full decoder reads all 2^N bins of one
    sample, the sector decoder reads the N+1 sector counts of that same sample.

Only the input representation differs, so a difference in SWPE is the price (or
the absence of a price) of the sector reduction itself.

Usage:  python run_sufficiency_swpe.py [--N 4,6,8,10] [--n_trials 41] [--quick]
Writes: results/e1_sufficiency_swpe.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                             # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402

import numpy as np                                          # noqa: E402
import autograd                                             # noqa: E402
import autograd.numpy as anp                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiments as rx                                # noqa: E402
import vaqem_lib as vl                                      # noqa: E402
import vaqem_methods as vm                                  # noqa: E402

CLEAN = {"kind": "none", "p": 0.0, "readout_p": 0.0}
# The conditions of the study: the noiseless case and pure readout noise, where
# the sector reduction is expected to be exactly free, plus one in-circuit
# channel of each family used in the headline sweep.  Depolarizing noise is
# sampled at the four strengths of the sweep so that the price of the reduction
# can be plotted against the noise rate.
SETTINGS = [("noiseless", CLEAN),
            ("readout_0.03", {"kind": "none", "p": 0.0, "readout_p": 0.03}),
            ("depol_0.002", {"kind": "depolarizing", "p": 0.002,
                             "readout_p": 0.0}),
            ("depol_0.005", {"kind": "depolarizing", "p": 0.005,
                             "readout_p": 0.0}),
            ("depol_0.01", {"kind": "depolarizing", "p": 0.01,
                            "readout_p": 0.0}),
            ("depol_0.02", {"kind": "depolarizing", "p": 0.02,
                            "readout_p": 0.0}),
            ("deph_0.02", {"kind": "dephasing", "p": 0.02, "readout_p": 0.0}),
            ("ampdamp_0.01", {"kind": "amplitude_damping", "p": 0.01,
                              "readout_p": 0.0})]
QUICK = {"noiseless", "readout_0.03", "depol_0.01"}


# ----------------------------------------------------------------------
# the full-distribution decoder: same MLP family, wider input layer
# ----------------------------------------------------------------------
def full_shapes(n_in, hidden=128):
    """Layer shapes of the n_in -> hidden -> hidden/2 -> 2 decoder."""
    return [(hidden, n_in), (hidden,), (hidden // 2, hidden), (hidden // 2,),
            (2, hidden // 2), (2,)]


def n_full_params(n_in, hidden=128):
    return int(sum(int(np.prod(s)) for s in full_shapes(n_in, hidden)))


def unpack_full(flat, n_in, hidden=128):
    """Flat vector -> [W1, b1, W2, b2, W3, b3] (autograd-safe)."""
    from autograd.tracer import Box
    flat = flat if isinstance(flat, Box) else np.asarray(flat, dtype=float)
    flat = anp.reshape(flat, -1)
    params, off = [], 0
    for shp in full_shapes(n_in, hidden):
        sz = int(np.prod(shp))
        params.append(anp.reshape(flat[off:off + sz], shp))
        off += sz
    return params


def init_full(n_in, hidden=128, seed=0):
    """The MLP initialisation scheme of ``train_decoder.init_params``."""
    rs = np.random.RandomState(seed)
    flat = []
    for shp in full_shapes(n_in, hidden):
        lim = 1.0 / np.sqrt(shp[-1])
        flat.append(rs.uniform(-lim, lim, shp).ravel())
    return np.concatenate(flat)


# ----------------------------------------------------------------------
# training: the classical half of train_decoder.train, wider input layer
# ----------------------------------------------------------------------
def train_readout(N, theta, curly, rep="full", hidden=128, n_train=100,
                  n_test=50, mlp_iters=2000, lr=0.02, patience=150,
                  min_iters=500, eval_every=25, workers=12, verbose=True):
    """Train a readout network on one of the two representations.

    ``rep="full"``  input is the 2^N computational-basis distribution p(x|phi);
    ``rep="sector"`` input is the N+1 imbalance distribution p_m(phi).

    Everything else is identical for the two arms -- same frozen circuit, same
    100 noiseless training phases, same circular loss, same Adam(lr), same
    hidden width, same initialisation scheme, same evaluation cadence and same
    early-stopping rule -- so that the *only* difference between the arms is the
    representation the readout is allowed to see.  (``full_shapes(N+1, h)`` is
    exactly ``vl.mlp_shapes(N, h)``, i.e. the sector arm reproduces the certified
    decoder architecture.)  Returns ``(flat_best, info)``.
    """
    if rep not in ("full", "sector"):
        raise ValueError(f"rep must be 'full' or 'sector', got {rep!r}")
    phi_tr, phi_te = vl.test_phases(n_train, n_test)
    pr_tr, pm_tr = vm.sector_probs(phi_tr, N, theta, curly, CLEAN, 1, workers)
    pr_te, pm_te = vm.sector_probs(phi_te, N, theta, curly, CLEAN, 1, workers)
    x_tr = np.asarray(pr_tr if rep == "full" else pm_tr, dtype=float)
    x_te = np.asarray(pr_te if rep == "full" else pm_te, dtype=float)
    n_in = x_tr.shape[1]
    want = 2 ** N if rep == "full" else N + 1
    if n_in != want:                                      # pragma: no cover
        raise ValueError(f"expected {want} inputs for rep={rep}, got {n_in}")

    def cost(p):
        """2[1-cos(phi_hat-phi)] on the training grid (VQ-CNNI objective)."""
        pred = vl.predict_from_pm(x_tr, unpack_full(p, n_in, hidden))
        return 2.0 * anp.mean(1.0 - anp.cos(pred - phi_tr))

    def test_loss(p):
        pred = np.asarray(vl.predict_from_pm(
            x_te, unpack_full(p, n_in, hidden)))
        return float(2.0 * np.mean(1.0 - np.cos(pred - phi_te)))

    grad = autograd.grad(cost)
    opt = vl.Adam(lr=lr)
    x = init_full(n_in, hidden)
    t0 = time.time()
    best = {"loss": np.inf, "x": x.copy(), "iter": 0}
    hist = []
    for it in range(1, mlp_iters + 1):
        g = np.asarray(grad(x))
        x = opt.step(x, np.clip(np.nan_to_num(g), -1e6, 1e6))
        if it % eval_every == 0 or it == 1:
            l_te = test_loss(x)
            hist.append({"iter": it, "test": l_te,
                         "t_s": round(time.time() - t0, 1)})
            if l_te < best["loss"]:
                best = {"loss": float(l_te), "x": x.copy(), "iter": it}
            if verbose:
                print(f"    [{rep:6s} N={N}] iter {it:5d}  test {l_te:.6e}"
                      f"   best {best['loss']:.6e} @{best['iter']}"
                      f"   ({time.time() - t0:.0f}s)", flush=True)
            if it > min_iters and (it - best["iter"]) > patience:
                if verbose:
                    print(f"    early stop at iter {it}", flush=True)
                break
    info = {"N": N, "decoder": rep, "hidden": hidden, "n_in": n_in,
            "n_params": n_full_params(n_in, hidden), "lr": lr,
            "n_train": n_train, "n_test": n_test, "mlp_iters": mlp_iters,
            "patience": patience, "min_iters": min_iters,
            "eval_every": eval_every, "best_iter": best["iter"],
            "best_test_loss": float(best["loss"]),
            "time_s": round(time.time() - t0, 1), "activation": "softsign",
            "circuit": "frozen at the checkpoint values", "history": hist}
    return best["x"], info



# ----------------------------------------------------------------------
# evaluation: both decoders on the same distributions and the same draws
# ----------------------------------------------------------------------
def eval_arms(N, model, flats, setting, noise, tst, masks, shots_list,
              n_trials, seed, workers, hidden_full=128):
    """Metric rows for the three readout arms, one noise setting.

    ``flats`` maps ``"sector"`` and ``"full"`` to the parameter vectors trained
    by :func:`train_readout`; the frozen certified decoder of the checkpoint is
    always evaluated as the third arm ``"sector_frozen"``, since that is the
    decoder the rest of the paper uses.
    """
    pr, pm = vm.sector_probs(tst, N, model["theta"], model["curly"], noise, 1,
                             workers, masks)
    pr = np.asarray(pr, dtype=float)
    pm = np.asarray(pm, dtype=float)
    n_in = pr.shape[1]
    dec = {"sector_frozen": (model["params"], pm),
           "sector": (unpack_full(flats["sector"], N + 1, hidden_full), pm),
           "full": (unpack_full(flats["full"], n_in, hidden_full), pr)}
    rows = []
    for shots in shots_list:
        if shots == "inf":
            per = {name: rx.metrics(
                       np.asarray(vl.predict_from_pm(x, params)),
                       tst, detail=True)
                   for name, (params, x) in dec.items()}
        else:
            S = int(shots)
            acc = {name: [] for name in dec}
            for k in range(n_trials):
                # One multinomial sample per phase.  The sector arms read the
                # N+1 sector counts of *that same sample*, so all three arms are
                # evaluated on identical shots and the comparison is paired.
                rng = np.random.default_rng(seed + S * 1000 + k)
                pr_hat = vm.sample_dist(pr, S, rng)
                pm_hat = vl.probs_to_p_m(pr_hat, masks)
                for name, (params, _) in dec.items():
                    x_hat = pr_hat if name == "full" else pm_hat
                    acc[name].append(rx.metrics(
                        np.asarray(vl.predict_from_pm(x_hat, params)),
                        tst, detail=True))
            per = {key: rx.agg(val) for key, val in acc.items()}
        for name, met in per.items():
            row = {"N": N, "setting": setting,
                   "kind": noise.get("kind", "none"),
                   "p": float(noise.get("p", 0.0)),
                   "readout_p": float(noise.get("readout_p", 0.0)),
                   "shots": shots, "decoder": name,
                   "n_decoded_numbers": int(n_in if name == "full" else N + 1)}
            row.update(met)
            rows.append(row)
    return rows


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def exp_sufficiency_swpe(Ns=(4, 6, 8, 10), n_cal=21, n_tst=21,
                         shots_list=("inf", 256, 1024, 4096), n_trials=41,
                         seed=0, workers=12, quick=False, hidden_full=128,
                         verbose=True):
    """Run the study and write ``results/e1_sufficiency_swpe.json``."""
    _, tst = rx.grids(n_cal, n_tst)
    settings = ([s for s in SETTINGS if s[0] in QUICK] if quick else SETTINGS)
    rows, trains = [], []
    for N in Ns:
        model = rx.load_model(N)
        masks = vl.m_structure(N)[2]
        hidden = int(model["hidden"])
        n_sector = int(sum(int(np.prod(s)) for s in vl.mlp_shapes(N, hidden)))
        flats = {}
        for rep in ("sector", "full"):
            print(f"== N={N}: training the {rep} readout "
                  f"(input width {2 ** N if rep == 'full' else N + 1}, "
                  f"hidden {hidden_full})", flush=True)
            flat, info = train_readout(N, model["theta"], model["curly"],
                                       rep=rep, hidden=hidden_full,
                                       workers=workers, verbose=verbose)
            info["n_params_certified_sector_decoder"] = n_sector
            info["param_ratio"] = info["n_params"] / n_sector
            trains.append(info)
            flats[rep] = flat
            print(f"   {rep:6s} readout: {info['n_params']} params "
                  f"({info['param_ratio']:.2f}x the certified decoder's "
                  f"{n_sector}), best test loss {info['best_test_loss']:.3e} "
                  f"@ iter {info['best_iter']}", flush=True)
        for setting, noise in settings:
            t0 = time.time()
            got = eval_arms(N, model, flats, setting, noise, tst, masks,
                            shots_list, n_trials, seed, workers, hidden_full)
            rows.extend(got)
            inf = {r["decoder"]: r for r in got if r["shots"] == "inf"}
            print(f"   N={N} {setting:14s} inf-shot SWPE [dB]: sector "
                  f"{inf['sector']['mse_db']:8.2f}   full "
                  f"{inf['full']['mse_db']:8.2f}   certified "
                  f"{inf['sector_frozen']['mse_db']:8.2f}   gap(full-sector) "
                  f"{inf['full']['mse_db'] - inf['sector']['mse_db']:+7.3f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
    out = {"rows": rows, "training": trains,
           "protocol": {"n_cal": n_cal, "n_tst": n_tst, "n_train": 100,
                        "n_test_early_stop": 50,
                        "shots_list": [str(s) for s in shots_list],
                        "n_trials": n_trials, "seed": seed,
                        "hidden_full": hidden_full, "quick": bool(quick),
                        "decoder_input": {
                            "sector": "N+1 collective-imbalance probabilities",
                            "full": "2^N computational-basis probabilities"},
                        "matched": ["probe circuit", "training phases",
                                    "circular loss", "Adam learning rate",
                                    "early-stopping rule", "evaluation grid",
                                    "multinomial draws"],
                        "metric": "mse_db = 10 log10 mean squared wrapped "
                                  "phase error (SWPE)"}}
    rx.save_json("e1_sufficiency_swpe.json", out)
    summarise(out)
    return out


def summarise(out):
    """Print the numbers the manuscript quotes, per shot budget."""
    for shots in ("inf", "256", "1024", "4096"):
        idx = {}
        for r in out["rows"]:
            if str(r["shots"]) != str(shots):
                continue
            idx.setdefault((r["N"], r["setting"]), {})[r["decoder"]] = r
        pairs = [p for p in idx.values() if {"sector", "full"} <= set(p)]
        if not pairs:
            continue
        # matched-protocol arms: the price of the sector reduction itself
        gap = np.asarray([p["full"]["mse_db"] - p["sector"]["mse_db"]
                          for p in pairs], dtype=float)
        # against the certified decoder actually used in the paper
        gapf = np.asarray([p["full"]["mse_db"] - p["sector_frozen"]["mse_db"]
                           for p in pairs if "sector_frozen" in p], dtype=float)
        print(f"\n-- S={shots}: {gap.size} (N, setting) pairs; "
              f"gap = SWPE(full) - SWPE(sector) [dB], >0 favours sectors --")
        print(f"   matched protocol : max {gap.max():+.3f}   min "
              f"{gap.min():+.3f}   median {np.median(gap):+.3f}   mean "
              f"{gap.mean():+.3f}")
        print(f"   full better in {int((gap < 0).sum())}/{gap.size}, "
              f"sector better by more than 0.5 dB in "
              f"{int((gap > 0.5).sum())}/{gap.size}")
        if gapf.size:
            print(f"   vs certified   : max {gapf.max():+.3f}   median "
                  f"{np.median(gapf):+.3f}   full better in "
                  f"{int((gapf < 0).sum())}/{gapf.size}")
        worst = min(pairs, key=lambda p: p["full"]["mse_db"]
                    - p["sector"]["mse_db"])
        print(f"   worst for sectors: N={worst['sector']['N']} "
              f"{worst['sector']['setting']} "
              f"({(worst['full']['mse_db'] - worst['sector']['mse_db']):+.3f}"
              f" dB)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E1: the sector reduction priced in SWPE")
    ap.add_argument("--N", default="4,6,8,10",
                    help="comma-separated qubit numbers (default 4,6,8,10)")
    ap.add_argument("--n_cal", type=int, default=21)
    ap.add_argument("--n_tst", type=int, default=21)
    ap.add_argument("--n_trials", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hidden_full", type=int, default=128)
    ap.add_argument("--quick", action="store_true",
                    help="three settings only (smoke run)")
    ap.add_argument("--exp-root", default=None,
                    help="write the JSON somewhere else than ../results")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.exp_root:
        rx.set_out(a.exp_root)
    Ns = tuple(int(x) for x in a.N.split(",") if x.strip())
    exp_sufficiency_swpe(Ns=Ns, n_cal=a.n_cal, n_tst=a.n_tst,
                         n_trials=a.n_trials, seed=a.seed, workers=a.workers,
                         quick=a.quick, hidden_full=a.hidden_full,
                         verbose=a.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


