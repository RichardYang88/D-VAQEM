"""
vaqem_methods.py -- mitigation maps, training objectives and evaluation for
distribution-level variational quantum error mitigation (D-VAQEM).

Setting
-------
A VQ-CNNI phase estimator consists of a parametrized probe circuit whose
computational-basis statistics are aggregated onto the collective imbalance
m = #0 - #1, giving a distribution p_m in the (N+1)-dimensional simplex, and
a frozen classical decoder D that maps p_m to an angle phi_hat.  ``vaqem_lib``
verifies (to machine precision) that m is an *exactly sufficient statistic*
for phi in this architecture, i.e. FI(p_m) = FI(p_full): no information about
phi is lost by the reduction, while the dimension drops from 2^N to N+1.
Distribution-level mitigation is therefore both complete and cheap.

Noise turns p_m(phi) into y(phi) = N_phi[p_m(phi)], biasing the frozen
decoder.  A mitigation map M_psi acts on the measured distribution,
p_mit = M_psi(y), and is trained on *calibration data* only: pairs
(y(phi_i), p_m(phi_i)) at known phases (the standard VAQEM resource
assumption -- a noiseless model of the same circuit, no knowledge of the
noise channel itself).

Objectives implemented
----------------------
``ce``      cross-entropy / multinomial MLE matching of the clean
            distribution  (the natural distribution-level analogue of the
            scalar VAQEM cost);
``l2``      squared-error matching of the clean distribution;
``fisher``  ce + lam * relative Fisher information distance
            D_F(t, p~) = sum_k t_k (d_phi log p~_k - d_phi log t_k)^2,
            which forces the *score* of the mitigated family to match the
            clean one and therefore preserves the Cramer-Rao sensitivity;
``mse``     the finite-shot mean squared phase error of the *decoded
            estimate*,  E[2(1-cos(phi_hat-phi))] + (1/S) Var_delta(phi_hat),
            with the variance obtained exactly (delta method) from the
            multinomial covariance of the empirical distribution and the
            gradient of the composite estimator.  S -> infinity recovers
            the bias-only (standard VAQEM) objective; finite S automatically
            shrinks the map towards the low-variance regime.
"""

import hashlib
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import autograd
import autograd.numpy as anp

import vaqem_lib as vl
from autograd.tracer import Box


def _as_anp(x):
    """autograd-safe conversion: never touch a Box (it has no VJP for asarray)."""
    return x if isinstance(x, Box) else anp.array(np.asarray(x, dtype=float))


# ======================================================================
# Exact noisy distributions
# ======================================================================
def fast_path(noise):
    """True when the noise acts only at readout (exact linear kernel)."""
    return float(noise.get("p", 0.0)) == 0.0 or noise.get("kind", "none") in ("none", None)


def _sim_one(args):
    phi, N, theta, curly, noise, fold = args
    if fast_path(noise):
        p = vl.probs_pure(phi, N, theta, curly, fold=fold)
        q = float(noise.get("readout_p", 0.0))
        if q > 0:
            p = vl.apply_flip(p, N, q)
        return p
    return vl.probs_dm(phi, N, theta, curly, noise["kind"], noise["p"],
                       fold=fold, readout_p=noise.get("readout_p", 0.0))


# ----------------------------------------------------------------------
# On-disk cache of the exact simulation datasets
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, os.pardir, "cache")

# Provenance of the exact-simulation requests served so far.  The wall-clock
# ``t_data_s`` recorded in the result files measures a *cache lookup* whenever
# the dataset is already on disk, and quoting that as the cost of the exact
# simulation would be wrong by orders of magnitude (1981 s vs 0.0 s at N=10 in
# the scaling study).  Callers therefore record ``data_from_cache`` next to it;
# see ``run_experiments.exp_scaling``.
CACHE_STATS = {"hit": 0, "miss": 0}


def cache_stats(reset=False):
    """``(hits, misses)`` of :func:`sector_probs` since the last reset."""
    if reset:
        CACHE_STATS["hit"] = CACHE_STATS["miss"] = 0
    return CACHE_STATS["hit"], CACHE_STATS["miss"]


def _skey(phis, N, theta, curly, noise, fold):
    """Stable hash of a simulation specification (cache key)."""
    h = hashlib.sha1()
    h.update(np.asarray(phis, dtype=float).tobytes())
    h.update(np.asarray([N, fold], dtype=float).tobytes())
    h.update(np.asarray(theta, dtype=float).tobytes())
    h.update(np.asarray(curly, dtype=float).tobytes())
    h.update(repr(sorted((str(k), str(v)) for k, v in noise.items())).encode())
    return h.hexdigest()[:20]


def sector_probs(phis, N, theta, curly, noise, fold=1, workers=1, masks=None,
                 use_cache=True):
    """Exact outcome probabilities (P, 2^N) and m-distributions (P, N+1).

    Results are cached on disk (``new_paper/cache``) keyed by the exact
    (phase grid, circuit parameters, noise spec, fold), because the density
    matrix simulation dominates the runtime of every experiment.
    """
    phis = np.atleast_1d(np.asarray(phis, dtype=float))
    masks = vl.m_structure(N)[2] if masks is None else masks
    path = None
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, "sp_%s.npz"
                            % _skey(phis, N, theta, curly, noise, fold))
        if os.path.exists(path):
            with np.load(path) as d:
                pr = np.array(d["probs"])
            CACHE_STATS["hit"] += 1
            return pr, vl.probs_to_p_m(pr, masks)
    CACHE_STATS["miss"] += 1
    args = [(float(p), N, theta, curly, noise, fold) for p in phis]
    if workers > 1 and len(args) > 1:
        with Pool(processes=min(workers, len(args))) as pool:
            rows = pool.map(_sim_one, args, chunksize=1)
    else:
        rows = [_sim_one(a) for a in args]
    pr = np.stack(rows)
    if path is not None:
        np.savez_compressed(path, probs=pr, phis=phis)
    return pr, vl.probs_to_p_m(pr, masks)


def build_case(N, theta, curly, phis, noise, folds=(1,), workers=8, masks=None):
    """Calibration/evaluation data for one (model, noise) setting."""
    masks = vl.m_structure(N)[2] if masks is None else masks
    t0 = time.time()
    pr_clean, pm_clean = sector_probs(phis, N, theta, curly,
                                      {"kind": "none", "p": 0.0, "readout_p": 0.0},
                                      1, workers, masks)
    pm_noisy = {}
    for fold in folds:
        _, pm = sector_probs(phis, N, theta, curly, noise, fold, workers, masks)
        pm_noisy[int(fold)] = pm
    return {"N": N, "phis": np.asarray(phis, dtype=float), "masks": masks,
            "probs_clean": pr_clean, "pm_clean": pm_clean, "pm_noisy": pm_noisy,
            "noise": noise, "folds": tuple(int(f) for f in folds),
            "time_s": time.time() - t0}


# ======================================================================
# Mitigation maps  (all act on rows of a (B, N+1) batch)
# ======================================================================
class LinearMap:
    """p~ = y @ T, with rows of T renormalised to sum to one.

    The unconstrained real matrix T (which may take negative entries) is the
    finite-dimensional generalisation of a quasi-probability / linear
    inversion map: for a known flip channel the exact inverse is recovered,
    while the data-driven fit needs no noise model at all.
    """
    name = "linear"

    def __init__(self, N, rng=None, scale=0.0):
        self.n = N + 1
        eye = np.eye(self.n)
        if scale > 0 and rng is not None:
            eye = eye + scale * rng.randn(self.n, self.n) / self.n
        self.p0 = eye.reshape(-1).copy()

    @property
    def n_params(self):
        return self.n * self.n

    def matrix(self, psi):
        T = anp.reshape(psi, (self.n, self.n))
        s = anp.sum(T, axis=1, keepdims=True)
        s = anp.where(anp.abs(s) > 1e-9, s, 1e-9 + 0.0 * s)
        return T / s

    def apply(self, y, psi):
        return anp.matmul(y, self.matrix(psi))


class MlpMap:
    """p~ = softmax(W3 softsign(W2 softsign(W1 y + b1) + b2) + b3).

    A small feed-forward net over the (N+1)-dimensional distribution: strictly
    more expressive than the linear map (non-linear shrinkage of sectors),
    always normalised and non-negative by construction.  Parameters are
    flattened into a single vector psi.
    """
    name = "mlp"

    def __init__(self, N, hidden=(32, 32), rng=None, scale=0.02):
        self.n = N + 1
        self.shapes = []
        dims = [self.n] + list(hidden) + [self.n]
        rng = np.random.RandomState(0) if rng is None else rng
        chunks = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.shapes.append((a, b))
            w = scale * rng.randn(a, b)
            if a == b and scale > 0:
                w = w + np.eye(a)          # start near the identity map
            chunks.append(w.reshape(-1))
            chunks.append(np.zeros(b))
        self.p0 = np.concatenate(chunks)

    @property
    def n_params(self):
        return self.p0.size

    def _unpack(self, psi):
        psi = anp.reshape(_as_anp(psi), -1)
        out, i = [], 0
        for a, b in self.shapes:
            out.append((anp.reshape(psi[i:i + a * b], (a, b)),
                        psi[i + a * b:i + a * b + b]))
            i += a * b + b
        return out

    def apply(self, y, psi):
        h = y
        layers = self._unpack(psi)
        for j, (w, b) in enumerate(layers):
            h = anp.matmul(h, w) + b
            if j < len(layers) - 1:
                h = vl.softsign(h)
        return _softmax(h)


MAPS = {"linear": LinearMap, "mlp": MlpMap}


def _softmax(x):
    z = x - anp.max(x, axis=-1, keepdims=True)
    e = anp.exp(z)
    return e / anp.sum(e, axis=-1, keepdims=True)



# ======================================================================
# Batched forward-mode Jacobians (needed by the shot-aware objective)
# ======================================================================
def _ss(z):
    return z / (1.0 + anp.abs(z))


def _dss(z):
    return 1.0 / (1.0 + anp.abs(z)) ** 2


def decoder_value_grad(q, params):
    """phi_hat(q) and g = d phi_hat / d q for a whole batch, in autograd form.

    ``params`` is the VQ-CNNI decoder tuple (W1,b1,W2,b2,W3,b3) with W stored
    as (out, in).  The decoder output is L2-normalised, but phi_hat =
    arctan2(v0, v1) is invariant to that positive rescaling, so the gradient
    is taken through the unnormalised pair.  A batched forward-mode sweep
    tracks J[b,j,l] = d h_l / d q_j for every sample at once, so the whole
    calibration batch costs one pass and the result stays differentiable with
    respect to upstream parameters (the mitigation map).
    """
    W1, b1, W2, b2, W3, b3 = [_as_anp(a) for a in params]
    q = _as_anp(q)
    B, Din = q.shape
    h = q[:, None, :]
    J = anp.broadcast_to(anp.eye(Din)[None, :, :], (B, Din, Din))
    for j, (w, b) in enumerate(((W1, b1), (W2, b2), (W3, b3))):
        z = anp.matmul(h, w.T) + b[None, None, :]
        J = anp.matmul(J, w.T)
        if j < 2:
            h = _ss(z)
            J = J * _dss(z)
        else:
            h = z
    o = h[:, 0, :]                                   # (B, 2) unnormalised
    o0, o1 = o[:, 0], o[:, 1]
    phi = anp.arctan2(o0, o1)
    den = o0 ** 2 + o1 ** 2 + 1e-30
    g = (J[:, :, 0] * o1[:, None] - J[:, :, 1] * o0[:, None]) / den[:, None]
    return phi, g



def mlp_map_value_jac(y, layers):
    """p~(y) and J[b,j,k] = d p~_k / d y_j for an MLP mitigation map."""
    y = _as_anp(y)
    B, Din = y.shape
    h = y[:, None, :]
    J = anp.broadcast_to(anp.eye(Din)[None, :, :], (B, Din, Din))
    nl = len(layers)
    for j in range(nl):
        w, b = layers[j]
        z = anp.matmul(h, w) + b[None, None, :]
        J = anp.matmul(J, w)
        if j < nl - 1:
            h = _ss(z)
            J = J * _dss(z)
        else:
            h = z
    m = anp.max(h, axis=-1, keepdims=True)
    e = anp.exp(h - m)
    p = e / anp.sum(e, axis=-1, keepdims=True)
    u = anp.sum(J * p, axis=-1, keepdims=True)
    return p[:, 0, :], p * (J - u)


# ======================================================================
# Training objectives
# ======================================================================
def fd_scores(q, dphi):
    """Differentiable central-difference score d/dphi log q on a uniform grid."""
    d = anp.concatenate([(q[1:2] - q[0:1]) / dphi,
                         (q[2:] - q[:-2]) / (2 * dphi),
                         (q[-1:] - q[-2:-1]) / dphi], axis=0)
    return d / anp.clip(q, 1e-12, None)


_CE_EPS = 1e-6      # smoothing scale of the cross-entropy log barrier
_LR_END = 0.05      # cosine schedule: final lr = _LR_END * lr
_SCORE_CLIP = 1e3   # guard on d log q / d phi inside the Fisher objective


def _smooth_pos(q, eps=_CE_EPS):
    """C^1 strictly-positive approximation of ``q``, stable for large |q|.

    ``0.5*(q + sqrt(q^2 + 4 eps^2))`` suffers catastrophic cancellation for
    q << -eps (it returns exactly 0, and log 0 = -inf kills the fit), so the
    negative branch is written in rationalised form 2 eps^2/(r - q).  Unlike
    ``clip``/``maximum`` it keeps a non-zero gradient everywhere, which matters
    because the *linear* map legitimately produces negative entries.

    The rationalised denominator needs the same care from the other side: once
    |q| >~ 190 eps-scale (q^2 + 4 eps^2 rounds to q^2 in float64) ``r - q``
    cancels to *exactly* zero, and since autograd's ``where`` differentiates
    both branches, that 0 in the denominator would produce inf -> NaN
    gradients (silently zeroed by train_map's nan_to_num, i.e. a stalled fit)
    for entries where the branch is not even selected.  Substituting a
    placeholder denominator there keeps the discarded branch finite without
    changing any selected value or gradient.
    """
    r = anp.sqrt(q * q + 4.0 * eps * eps)
    pos = q > 0
    den = anp.where(pos, 1.0, r - q)          # r - q >= 2 eps on the used branch
    return anp.where(pos, 0.5 * (q + r), 2.0 * eps * eps / den)


def train_map(data, map_kind, phis, loss="ce", iters=400, lr=5e-2, lam=1.0,
              shots=None, decoder=None, seed=0, hidden=(32, 32), scale=0.02,
              verbose=False, fold=None, psi0=None, val_frac=0.25):
    """Fit a mitigation map from calibration data.

    Calibration data = the noiseless model distribution p_m(phi_i) (computable
    classically, or measurable on a noiseless simulator) together with the
    noisy distributions y(phi_i) at the same known phases.  No knowledge of the
    noise channel is used.

    loss : 'ce' | 'l2' | 'fisher' | 'mse'
        'mse' additionally needs ``shots`` (S) and the frozen ``decoder``;
        it minimises the delta-method finite-shot MSE of the decoded phase,
        bias + (lam/S) Var, so it interpolates between standard VAQEM
        (S -> infinity) and variance-regularised shrinkage (small S).
    psi0 : optional warm-start parameter vector (e.g. an ``l2`` fit reused as
        the initialisation of a ``ce`` fit, or a ``ce`` fit for the shot-aware
        ``mse`` fine-tune, which avoids the zero-variance collapse of a
        cold-started shot-aware fit).
    val_frac : fraction of the calibration phases held out for model
        selection.  The maps are heavily over-parameterised relative to a
        realistic calibration budget (81 linear or ~1700 MLP parameters for 13
        phases), so the *training* objective keeps falling while the held-out
        error grows; the returned psi is the iterate with the best held-out
        loss.  The hold-out is strided in phi so that both parts cover the
        phase range uniformly (for 'fisher', whose score target is a finite
        difference along phi, it has to be a contiguous block instead).  No
        extra quantum resource is consumed -- the hold-out is a subset of the
        same calibration runs.  Set val_frac=0 to disable.
    """
    phis = np.asarray(phis, dtype=float)
    order = np.argsort(phis)
    phis = phis[order]
    fold = min(data["pm_noisy"]) if fold is None else int(fold)
    y = np.asarray(data["pm_noisy"][fold], dtype=float)[order]
    t = np.asarray(data["pm_clean"], dtype=float)[order]
    B, n = y.shape
    dphi = float(phis[1] - phis[0])
    rng = np.random.RandomState(seed)
    mp = (LinearMap(n - 1) if map_kind == "linear"
          else MlpMap(n - 1, hidden=hidden, rng=rng, scale=scale))
    psi = mp.p0.copy() if psi0 is None else np.asarray(psi0, dtype=float).copy()

    if loss == "mse" and decoder is None:
        raise ValueError("loss='mse' needs the frozen decoder")
    dec = None if decoder is None else decoder["params"]

    n_val = int(round(val_frac * B)) if val_frac > 0 else 0
    if B - n_val < 5 or n_val < 2:
        n_val = 0                                   # too little data to split
    if n_val == 0:
        idx_tr, idx_va = np.arange(B), None
    elif loss == "fisher":
        # fd_scores differences along phi and needs a *uniform* grid, so the
        # only consistent hold-out is a contiguous block.
        idx_tr = np.arange(B - n_val)
        idx_va = np.arange(B - n_val, B)
    else:
        # Strided hold-out: both parts then cover the phase range uniformly.
        # (A contiguous tail block sits at the phi ~ +-pi wrap boundary, which
        # is the hardest region, and selecting on it stops the fit far too
        # early -- measured.)
        step = max(2, B // n_val)
        idx_va = np.arange(0, B, step)[:n_val]
        idx_tr = np.setdiff1d(np.arange(B), idx_va)

    def make_f(idx):
        """Objective restricted to the calibration phases ``idx``."""
        y_, t_, ph_ = y[idx], t[idx], phis[idx]
        B_ = float(len(idx))
        dphi_ = float(ph_[1] - ph_[0]) if len(idx) > 1 else dphi
        if loss == "fisher":
            s_tgt_ = anp.clip(fd_scores(anp.asarray(t_), dphi_),
                              -_SCORE_CLIP, _SCORE_CLIP)

        def f_ce(p):
            q = mp.apply(y_, p)
            # smooth positivity + an explicit negativity penalty, so the
            # cross-entropy is finite *and* differentiable everywhere
            neg = anp.sum(anp.maximum(-q, 0.0) ** 2) / B_
            return -anp.sum(t_ * anp.log(_smooth_pos(q))) / B_ + lam * neg

        def f_l2(p):
            return anp.sum((mp.apply(y_, p) - t_) ** 2) / B_

        def f_fisher(p):
            q = _smooth_pos(mp.apply(y_, p))
            ce = -anp.sum(t_ * anp.log(q)) / B_
            sc = anp.clip(fd_scores(q, dphi_), -_SCORE_CLIP, _SCORE_CLIP)
            return ce + lam * anp.sum(t_ * (sc - s_tgt_) ** 2) / B_

        def f_mse(p):
            q = mp.apply(y_, p)
            phi_hat, g = decoder_value_grad(q, dec)
            bias = anp.mean(2.0 * (1.0 - anp.cos(phi_hat - ph_)))
            if not shots:
                return bias
            if map_kind == "linear":
                # c_j = d phi_hat / d y_j = sum_k g_k T_jk  (p~ = y @ T)
                c = anp.matmul(g, anp.transpose(mp.matrix(p)))
            else:
                _, Jm = mlp_map_value_jac(y_, mp._unpack(p))
                c = anp.sum(Jm * g[:, None, :], axis=-1)
            cy = anp.sum(c * y_, axis=1)
            var = (anp.sum(c ** 2 * y_, axis=1) - cy ** 2) / shots
            return bias + lam * anp.mean(var)

        return {"ce": f_ce, "l2": f_l2, "fisher": f_fisher, "mse": f_mse}[loss]

    f = make_f(idx_tr)
    f_sel = make_f(idx_va) if idx_va is not None else f
    grad = autograd.grad(f)
    m1, m2 = np.zeros_like(psi), np.zeros_like(psi)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    # The warm start is itself a candidate: every stage is then guaranteed
    # never to be worse (on the selection criterion) than the stage it is
    # initialised from, so the l2 -> ce -> mse pipeline is monotone.
    v0 = float(f_sel(psi))
    best = (v0 if np.isfinite(v0) else np.inf, psi.copy(), -1)
    eval_every = max(1, iters // 50)
    for it in range(iters):
        g = np.clip(np.nan_to_num(grad(psi)), -1e6, 1e6)
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        # Cosine decay: a constant step keeps bouncing around the optimum at
        # the ~lr scale, and that bounce dominates the loss of a converged fit.
        lr_t = lr * (_LR_END + (1.0 - _LR_END) * 0.5
                     * (1.0 + np.cos(np.pi * it / max(1, iters - 1))))
        psi = (psi - lr_t * (m1 / (1 - b1 ** (it + 1)))
               / (np.sqrt(m2 / (1 - b2 ** (it + 1))) + eps))
        if it % eval_every == 0 or it == iters - 1:
            v = float(f_sel(psi))
            if np.isfinite(v) and v < best[0]:
                best = (v, psi.copy(), it)
            if verbose:
                hist.append((it, v, float(f(psi))))
                print(f"    [{loss}/{map_kind}] iter {it:5d}  sel {v:.6e}"
                      f"  train {hist[-1][2]:.6e}  (lr {lr_t:.2e})")
    psi = best[1]
    info = {"loss": loss, "kind": map_kind, "iters": iters, "lr": lr, "lam": lam,
            "shots": shots, "seed": seed, "n_params": int(psi.size),
            "final_loss": float(best[0]), "best_iter": int(best[2]),
            "selected_on": "val" if idx_va is not None else "train",
            "n_val": int(n_val), "train_loss": float(f(psi)),
            "val_idx": ([] if idx_va is None else np.asarray(idx_va).tolist()),
            "history": hist, "fold": fold, "n_cal": int(B)}
    return mp, np.asarray(psi, dtype=float), info


# ======================================================================
# Non-variational baselines
# ======================================================================
def linv_map_from_kernel(K_m, alpha=1e-6):
    """Tikhonov-regularised linear inversion map for a known sector kernel.

    ``K_m[m', m] = Pr(sector m' | true sector m)`` and the noisy marginal is
    ``y = p @ K_m^T`` (rows are sectors), so the required inverse is
    ``T = (K_m^T)^{-1} = K_m^{-T}``.  Solving ``min_p ||p @ K_m - y||`` gives
    ``p~ = y @ K_m (K_m^T K_m + alpha I)^{-1}``, which tends to ``K_m^{-T}`` as
    alpha -> 0.  Transposing the kernel here instead would return ``K_m^{-1}``;
    the flip kernel is *not* symmetric for N >= 2, so that leaves a real
    residual bias (checked: 4e-2 vs 3e-7 at N=8, f=0.03).  This baseline
    *does* use the noise model, unlike the variational maps.
    """
    A = np.asarray(K_m, dtype=float)
    return A @ np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[0]))


class Mitigator:
    """Uniform interface: (B, N+1) noisy distribution -> (B, N+1) mitigated."""

    def __init__(self, name, fn, info=None):
        self.name = name
        self.fn = fn
        self.info = info or {}

    def __call__(self, y):
        return self.fn(y)


def mitigator_none():
    return Mitigator("none", lambda y: y)


def mitigator_learned(mp, psi):
    return Mitigator(f"dvaqem_{mp.name}", lambda y: mp.apply(y, psi),
                     {"n_params": int(np.asarray(psi).size)})


def mitigator_linv(K_m, alpha=1e-6):
    T = linv_map_from_kernel(K_m, alpha)
    return Mitigator("linv_known", lambda y: np.asarray(y) @ T,
                     {"alpha": alpha, "n_params": T.size})


def calib_train_indices(phis, val_frac=0.25, contiguous=False):
    """Training/hold-out split of the calibration phases, as :func:`train_map`.

    Exposed so that a non-variational baseline can be fitted on *exactly* the
    same subset of the calibration phases as the variational maps.  The indices
    refer to the phase-sorted order that ``train_map`` uses internally, and the
    branching (including the "too little data to split" fallback and the
    contiguous hold-out for the finite-difference Fisher score) mirrors it, so
    a baseline fitted through this helper consumes the same quantum resource
    and obeys the same hold-out discipline as D-VAQEM.

    Returns ``(idx_train, idx_val)``; ``idx_val`` is ``None`` when there is not
    enough calibration data to split.
    """
    phis = np.asarray(phis, dtype=float)[np.argsort(np.asarray(phis, dtype=float))]
    B = int(phis.size)
    n_val = int(round(val_frac * B)) if val_frac > 0 else 0
    if B - n_val < 5 or n_val < 2:
        n_val = 0                                   # too little data to split
    if n_val == 0:
        return np.arange(B), None
    if contiguous:
        return np.arange(B - n_val), np.arange(B - n_val, B)
    step = max(2, B // n_val)
    idx_va = np.arange(0, B, step)[:n_val]
    return np.setdiff1d(np.arange(B), idx_va), idx_va


def calib_pairs(data, idx=None, fold=None, val_frac=0.25):
    """The (target, observed) calibration pairs a mitigation fit is allowed to see.

    Returns ``(y, t)``: the noisy sector distributions and the noiseless targets
    at the phase-sorted, training-split calibration phases.  Shared by
    :func:`fit_flip_rate` and :func:`flip_rate_residual` so that a rate estimated
    from the data and a rate read off the noise model are always scored on
    exactly the same subset -- otherwise the comparison between the calibrated
    and the analytic assignment matrix would not be like for like.
    """
    phis = np.asarray(data["phis"], dtype=float)
    order = np.argsort(phis)
    fold = min(data["pm_noisy"]) if fold is None else int(fold)
    y = np.asarray(data["pm_noisy"][fold], dtype=float)[order]
    t = np.asarray(data["pm_clean"], dtype=float)[order]
    if idx is None:
        idx, _ = calib_train_indices(phis, val_frac)
    idx = np.asarray(idx, dtype=int)
    return y[idx], t[idx]


def flip_rate_residual(data, N, q, idx=None, fold=None, val_frac=0.25):
    """RMS calibration residual of the binomial sector kernel at a *given* rate.

    Used to score a rate that was not estimated from the data -- in particular
    the analytic ``effective_flip`` rate that the genie baseline inverts -- on
    the same objective and the same calibration phases as :func:`fit_flip_rate`.
    A residual larger than the fitted one means the flip-equivalent surrogate is
    not the best member of its own family for that channel.
    """
    y, t = calib_pairs(data, idx, fold, val_frac)
    r = t @ vl.m_flip_kernel(N, float(q)).T - y
    per = np.sqrt(np.sum(r ** 2, axis=1))
    return float(np.sqrt(np.mean(per ** 2)))


def fit_flip_rate(data, N, idx=None, fold=None, val_frac=0.25, fmax=0.4995,
                  n_grid=200, refine=48):
    """Effective i.i.d. bit-flip rate estimated from calibration data alone.

    Readout-error mitigation assumes the device acts as an independent
    per-qubit bit-flip channel and corrects the statistics by inverting the
    corresponding assignment matrix.  This reproduces that assumption *without*
    the noise model: of the one-parameter family of exact sector kernels
    :func:`vaqem_lib.m_flip_kernel`, it returns the member that best carries the
    noiseless calibration targets onto the observed noisy calibration
    distributions,

    .. math:: \\hat q = \\arg\\min_q \\sum_i \\| p_m(\\phi_i)\\, K(q)^T
              - y(\\phi_i) \\|_2^2 ,

    which is exactly the information budget the variational maps receive
    (noisy distributions at known reference phases plus noiseless targets from
    a classical model of the same circuit) -- no noise model, no test phases.

    For a channel that is *not* flip-equivalent (dephasing, amplitude damping)
    the family contains no correct member, and the fit returns its least-bad
    element.  That is deliberate: it is how a practitioner's readout correction
    actually behaves under model mismatch, and reporting it as "unavailable"
    (which the genie baseline :func:`mitigator_linv` has to do, since it needs
    the analytic rate) would hide the failure mode that matters.

    The objective is minimised by a global grid scan followed by ternary
    refinement inside the winning bracket, so a locally non-convex landscape
    cannot trap the estimate.

    Returns ``(q_hat, info)`` with ``info`` carrying the achieved residual, the
    grid resolution, the final bracket and the per-phase residuals.
    """
    y, t = calib_pairs(data, idx, fold, val_frac)
    idx = np.arange(len(y)) if idx is None else np.asarray(idx, dtype=int)

    def resid(q):
        return t @ vl.m_flip_kernel(N, q).T - y

    def loss(q):
        return float(np.sum(resid(q) ** 2))

    grid = np.linspace(0.0, fmax, int(n_grid))
    vals = np.array([loss(q) for q in grid])
    i = int(np.argmin(vals))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, int(n_grid) - 1)])
    for _ in range(int(refine)):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    q_hat = 0.5 * (lo + hi)
    if loss(q_hat) > vals[i]:                       # never worse than the grid
        q_hat, lo, hi = float(grid[i]), lo, hi
    per = np.sqrt(np.sum(resid(q_hat) ** 2, axis=1))
    info = {"loss": float(loss(q_hat)),
            "loss_grid_min": float(vals[i]),
            "rms_resid": float(np.sqrt(np.mean(per ** 2))),
            "max_resid": float(per.max()),
            "per_phase_resid": per.tolist(),
            "n_grid": int(n_grid), "fmax": float(fmax),
            "bracket": [float(lo), float(hi)],
            "n_fit_phases": int(idx.size),
            "family": "i.i.d. bit-flip (exact binomial sector kernel)",
            "objective": "least squares on the calibration sector distributions"}
    return float(q_hat), info


def mitigator_linv_calib(N, q_hat, alpha=1e-6):
    """Readout-error mitigation with a *calibrated* assignment matrix.

    Identical inversion machinery to :func:`mitigator_linv`, but the kernel is
    built from the rate estimated by :func:`fit_flip_rate` instead of the
    analytic one, so the baseline uses no noise model.  ``n_params_fit`` is the
    single number the estimator actually learns; ``n_params`` is the size of the
    resulting linear map, which is what the inversion costs to apply.
    """
    mit = mitigator_linv(vl.m_flip_kernel(N, q_hat), alpha)
    mit.name = "linv_calib"
    mit.info.update({"q_hat": float(q_hat), "n_params_fit": 1,
                     "family": "i.i.d. bit-flip (exact binomial sector kernel)",
                     "assignment_matrix": "calibrated, not analytic"})
    return mit


# ----------------------------------------------------------------------
# Probabilistic error cancellation (PEC) on the readout channel
# ----------------------------------------------------------------------
def pec_gamma(N, q):
    """Sampling overhead of readout PEC on ``N`` qubits at flip rate ``q``.

    The exact inverse of the single-qubit flip channel is the quasi-probability
    combination ``c0*I + c1*X`` with ``c0 = (1-q)/(1-2q)`` and
    ``c1 = -q/(1-2q)``, so the per-qubit overhead is ``|c0|+|c1| = 1/(1-2q)``
    and, the qubits being independent, the total is ``gamma = (1-2q)^-N``.  A
    PEC estimate has the *same expectation* as the deterministic inverse but a
    variance inflated by ``gamma^2`` -- that is the sampling overhead quoted for
    readout-error mitigation.  It diverges at ``q = 1/2``, where the channel
    carries no information and is not invertible at all.
    """
    N, q = int(N), float(q)
    if not 0.0 <= q < 0.5:
        raise ValueError("PEC needs an invertible flip channel: 0 <= q < 0.5")
    return (1.0 - 2.0 * q) ** (-N)


def pec_sector_matrix(N, q):
    """Expected sector transition matrix of readout PEC.

    PEC corrects *observed bitstrings*: given an observed sector holding ``w0``
    zeros and ``w1`` ones, every observed bit is flipped with the sampling
    probability ``|c1| / (|c0|+|c1|) = q`` and the outcome is reweighted by
    ``gamma * (-1)^(number of corrections)``.  Flipping an observed ``1``
    raises ``n0`` by one and flipping an observed ``0`` lowers it, so the net
    correction depends only on the observed sector: PEC is well defined on the
    ``(N+1)``-dimensional sector distribution and needs no bitstring
    bookkeeping downstream.

    Returns ``(M, gamma)`` with ``M[a_obs, a_out]`` the signed, ``gamma``-
    weighted probability that a shot observed in sector ``a_obs`` contributes to
    ``a_out``.  Two identities make this a faithful PEC implementation rather
    than an ad-hoc reweighting, and both are asserted in ``test_pec.py``:

    * ``M`` equals ``linv_map_from_kernel(m_flip_kernel(N, q))``, i.e. PEC and
      deterministic matrix inversion are the same estimator in expectation and
      differ only in variance;
    * ``sum_a_out |M[a_obs, a_out]| = gamma`` for every observed sector, i.e.
      the sampling overhead is state-independent.

    Because the sector index is ``a = n0`` (see ``vaqem_lib.m_flip_kernel``),
    correcting ``j`` of the ``w1 = N - a`` observed ones and ``i`` of the
    ``w0 = a`` observed zeros lands in sector ``a + j - i``, which always stays
    inside ``[0, N]``.
    """
    from math import comb
    N, q = int(N), float(q)
    gamma = pec_gamma(N, q)
    M = np.zeros((N + 1, N + 1))
    for a in range(N + 1):
        w0, w1 = a, N - a                    # observed zeros / ones
        for j in range(w1 + 1):              # observed ones corrected: 1 -> 0
            pj = comb(w1, j) * q ** j * (1.0 - q) ** (w1 - j)
            if pj == 0.0:
                continue
            for i in range(w0 + 1):          # observed zeros corrected: 0 -> 1
                pi = comb(w0, i) * q ** i * (1.0 - q) ** (w0 - i)
                if pi == 0.0:
                    continue
                M[a, a + j - i] += gamma * (-1.0) ** (j + i) * pj * pi
    return M, gamma


def pec_sample(y, N, q, shots, rng):
    """One faithful PEC estimate from an empirical sector distribution.

    ``y`` holds empirical frequencies from ``shots`` shots, so ``y * shots`` are
    integer counts.  Each count in observed sector ``a`` draws its corrections
    ``j ~ Binom(w1, q)`` and ``i ~ Binom(w0, q)`` independently and contributes
    ``gamma * (-1)^(j+i) / shots`` to sector ``a + j - i``.  This reproduces
    shot-by-shot PEC exactly -- the quasi-probability randomness is *additional*
    to the multinomial shot noise already present in ``y`` -- and is unbiased:
    ``E[pec_sample] = y @ M`` with ``M`` from :func:`pec_sector_matrix`.

    The reweighting by ``gamma`` is what makes the estimate unbiased and also
    what inflates its variance, so the returned distribution is signed and
    generally not normalised; callers renormalise before decoding, exactly as
    for the linear inverse.
    """
    y = np.asarray(y, dtype=float)
    two_d = (y.ndim == 2)
    Y = y if two_d else y[None, :]
    N, q, shots = int(N), float(q), int(shots)
    gamma = pec_gamma(N, q)
    out = np.zeros_like(Y)
    for b in range(Y.shape[0]):
        counts = np.rint(Y[b] * shots).astype(int)
        for a in range(N + 1):
            n = int(counts[a])
            if n <= 0:
                continue
            w0, w1 = a, N - a
            j = rng.binomial(w1, q, size=n)
            i = rng.binomial(w0, q, size=n)
            sign = np.where((j + i) % 2 == 0, gamma, -gamma)
            np.add.at(out[b], a + j - i, sign)
    out /= shots
    return out if two_d else out[0]


def mitigator_pec(N, q_hat, shots=None, seed=0, expect=False):
    """Readout PEC as a :class:`Mitigator`, with a calibrated or genie rate.

    Two regimes, selected by ``shots``:

    ``shots=None`` (or ``expect=True``)
        apply the *expectation* ``y @ M``.  This is the infinite-shot limit of
        PEC and is mathematically identical to the deterministic inverse
        :func:`mitigator_linv` -- the point of the baseline is precisely that
        PEC buys no extra accuracy in expectation, only a sampling cost.
    ``shots=S``
        draw the quasi-probability corrections shot by shot through
        :func:`pec_sample`, from a dedicated RNG that advances across calls so
        that repeated trials get independent PEC randomness.

    ``n_params_fit`` is 1, the flip rate, so the calibration budget matches
    ``linv_calib`` and D-VAQEM exactly; ``gamma`` is recorded so the sampling
    overhead can be quoted rather than inferred.
    """
    N, q_hat = int(N), float(q_hat)
    M, gamma = pec_sector_matrix(N, q_hat)
    rng = np.random.RandomState(seed)
    sample = (shots is not None) and not expect

    def fn(y):
        if sample:
            return pec_sample(y, N, q_hat, shots, rng)
        return np.asarray(y, dtype=float) @ M

    return Mitigator("pec_calib", fn,
                     {"q_hat": q_hat, "gamma": gamma,
                      "sampling_overhead_gamma_sq": gamma ** 2,
                      "n_params_fit": 1, "shots": shots,
                      "sampled": bool(sample),
                      "assignment_matrix": "calibrated, not analytic",
                      "family": "i.i.d. bit-flip (exact quasi-probability inverse)"})


def mitigator_pec_known(N, f_eff, shots=None, seed=0, expect=False):
    """:func:`mitigator_pec` at the genie analytic rate, mirroring ``linv_known``."""
    mit = mitigator_pec(N, f_eff, shots=shots, seed=seed, expect=expect)
    mit.name = "pec_known"
    mit.info["assignment_matrix"] = "analytic effective_flip rate (genie)"
    return mit


def pec_mismatch_bias(N, q_true, q_assumed):
    """Residual sector error of PEC built on a *wrong* flip rate.

    Model-based mitigation cannot average its model error away: as the shot
    count grows the estimate converges to ``p @ K(q_true)^T @ M(q_assumed)``,
    not to ``p``, so ``q_assumed != q_true`` leaves a bias floor that no amount
    of data removes.  This is the property that separates PEC from a map learnt
    from calibration data, which sees the device that is actually there.

    Returns the worst-case RMS sector distance over pure sector inputs, in the
    same units as :func:`flip_rate_residual`, so that a mismatched PEC and a
    mismatched linear inverse can be compared directly.
    """
    N = int(N)
    Kt = vl.m_flip_kernel(N, float(q_true))
    Ma, _ = pec_sector_matrix(N, float(q_assumed))
    p = np.eye(N + 1)                       # each sector as a pure input state
    err = (p @ Kt.T) @ Ma - p
    return float(np.sqrt(np.max(np.sum(err ** 2, axis=1))))


def richardson_coeffs(lams):
    """Exact-to-all-orders Richardson coefficients for extrapolation to lam=0."""
    lams = np.asarray(lams, dtype=float)
    c = []
    for k in range(len(lams)):
        num, den = 1.0, 1.0
        for j in range(len(lams)):
            if j != k:
                num *= lams[j]
                den *= (lams[j] - lams[k])
        c.append(num / den)
    return np.array(c)


def zne_extrapolate(pm_by_fold, kind="richardson", degree=1):
    """Distribution-level ZNE: extrapolate folded families to lambda = 0."""
    folds = sorted(pm_by_fold)
    Y = np.stack([np.asarray(pm_by_fold[f]) for f in folds])
    lam = np.asarray(folds, dtype=float)
    if kind == "richardson":
        c = richardson_coeffs(lam)
    else:
        # least-squares polynomial fit of every distribution entry vs lambda;
        # the extrapolated value at lambda = 0 is the intercept, i.e. row 0 of
        # the pseudo-inverse of the Vandermonde matrix applied to the data.
        V = np.vander(lam, degree + 1, increasing=True)
        c = np.linalg.pinv(V)[0]
    return np.tensordot(c / c.sum(), Y, axes=(0, 0)), (c / c.sum()).tolist()


def mitigator_zne(pm_by_fold, kind="richardson", degree=1):
    """ZNE as a Mitigator on batches matching the calibration grid."""
    Z, c = zne_extrapolate(pm_by_fold, kind, degree)

    def fn(y):
        y = np.asarray(y)
        return Z if y.shape == Z.shape else y
    return Mitigator(f"zne_{kind}", fn, {"folds": sorted(int(f) for f in pm_by_fold),
                                         "coeffs": c})


# ======================================================================
# Sampling and evaluation
# ======================================================================
def sample_dist(pm, shots, rng):
    """Multinomial sampling of empirical distributions (same shape as pm)."""
    pm = np.asarray(pm, dtype=float)
    flat = np.clip(pm.reshape(-1, pm.shape[-1]), 0, None)
    flat = flat / flat.sum(axis=1, keepdims=True)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = rng.multinomial(shots, flat[i]) / shots
    return out.reshape(pm.shape)


def _renorm(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum(axis=-1, keepdims=True)
    return np.where(s > 0, p / np.maximum(s, 1e-30), np.ones_like(p) / p.shape[-1])


def evaluate(pm_noisy, phi_true, decoder, mitigator=None, shots=None,
             n_trials=1, seed=0):
    """Decode phases from (optionally sampled, optionally mitigated) data."""
    pm_noisy = np.asarray(pm_noisy, dtype=float)
    phi_true = np.asarray(phi_true, dtype=float)
    rng = np.random.RandomState(seed)
    preds, res = [], []
    for _ in range(max(1, n_trials)):
        y = pm_noisy if not shots else sample_dist(pm_noisy, shots, rng)
        p = y if mitigator is None else mitigator(y)
        pred = vl.predict_from_pm(_renorm(p), decoder["params"])
        d = vl.wrapped_err(pred, phi_true)
        db = vl.swpe_db(pred, phi_true)
        preds.append(pred)
        res.append({"swpe_db_median": float(np.median(db)),
                    "swpe_db_mean": float(np.mean(db)),
                    "mse_db": float(10.0 * np.log10(np.mean(d ** 2) + 1e-12)),
                    "mean_swpe": float(np.mean(d ** 2)),
                    "median_swpe": float(np.median(d ** 2)),
                    "mae": float(np.mean(np.abs(d)))})
    agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    agg["swpe_db_std"] = float(np.std([r["swpe_db_median"] for r in res]))
    agg["pred"] = np.mean(np.stack(preds), axis=0)
    agg["pred_all"] = np.stack(preds)
    agg["n_trials"] = int(max(1, n_trials))
    agg["shots"] = shots
    agg["method"] = getattr(mitigator, "name", "none")
    return agg


def fi_curves(phis, pm_dict, n_interp=None):
    """Classical FI(phi) curves for several distribution families."""
    phis = np.asarray(phis, dtype=float)
    grid = phis if n_interp is None else np.linspace(phis[0], phis[-1], n_interp)
    out = {}
    for name, pm in pm_dict.items():
        pm = np.asarray(pm, dtype=float)
        if len(grid) != len(phis):
            pm = np.stack([np.interp(grid, phis, pm[:, k])
                           for k in range(pm.shape[1])], axis=1)
            pm = np.clip(pm, 1e-30, None)
            pm = pm / pm.sum(axis=1, keepdims=True)
        out[name] = vl.fi_grid(grid, pm)
    return grid, out


# ======================================================================
# Noise-aware retraining baseline (PRR-style)
# ======================================================================
def retrain_decoder(data, model, phis, iters=800, lr=2e-3, loss="circular",
                    shots=None, seed=0, fold=None, verbose=False):
    """Retrain the frozen decoder MLP on noisy data with clean phase labels.

    This is the strongest 'just recalibrate' baseline: it consumes exactly the
    same calibration resource as D-VAQEM (noisy distributions at known phases
    plus the noiseless target phases), but spends it on the decoder instead of
    on a mitigation map.  Circuit parameters are left untouched, so the noisy
    quantum data are reused and no extra quantum runtime is needed.
    """
    phis = np.asarray(phis, dtype=float)
    order = np.argsort(phis)
    phis = phis[order]
    fold = min(data["pm_noisy"]) if fold is None else int(fold)
    y = np.asarray(data["pm_noisy"][fold], dtype=float)[order]
    N, hidden = int(model["N"]), int(model["hidden"])
    p = np.asarray(model["x"], dtype=float)[6:].copy()
    rng = np.random.RandomState(seed)

    def cost(pp):
        x = y if not shots else sample_dist(y, shots, rng)
        pred = vl.predict_from_pm(x, vl.unpack_mlp(pp, N, hidden))
        d = pred - phis
        if loss == "circular":
            return anp.mean(1.0 - anp.cos(d))
        return anp.mean(anp.square(anp.arctan2(anp.sin(d), anp.cos(d))))

    grad = autograd.grad(cost)
    opt = vl.Adam(lr=lr)
    hist = []
    for it in range(iters):
        g = np.clip(np.nan_to_num(grad(p)), -1e6, 1e6)
        p = opt.step(p, g)
        if verbose and (it % max(1, iters // 4) == 0 or it == iters - 1):
            hist.append((it, float(cost(p))))
            print(f"    [retrain_dec] iter {it:5d}  loss {hist[-1][1]:.6e}")
    out = dict(model)
    out["x"] = np.concatenate([np.asarray(model["x"], dtype=float)[:6], p])
    out["params"] = vl.unpack_mlp(p, N, hidden)
    info = {"iters": iters, "lr": lr, "loss": loss, "shots": shots, "fold": fold,
            "history": hist, "final_loss": float(cost(p)),
            "n_params": int(p.size)}
    return out, info


# ======================================================================
# Delta-method (analytic) finite-shot error of the decoded phase
# ======================================================================
def delta_var(q, g, shots):
    """Analytic variance of phi_hat under multinomial sampling of q.

    Var[phi_hat] ~= (1/S) sum_j g_j^2 (q_j - mu_j^2) with mu_j = sum_k q_k g_k:
    the exact leading-order (delta method) variance of a smooth functional of a
    multinomial empirical distribution.  ``g`` is d phi_hat / d q, obtained from
    :func:`decoder_value_grad`.  This is what makes the finite-shot objective
    analytic instead of Monte-Carlo, and what the ``mse`` loss regularises.
    """
    q = np.asarray(q, dtype=float)
    g = np.asarray(g, dtype=float)
    mu = np.sum(q * g, axis=1)
    return (np.sum(g ** 2 * q, axis=1) - mu ** 2) / float(shots)


def decoded_grad(q, decoder):
    """phi_hat and d phi_hat / d q for a batch of m-distributions."""
    q = np.asarray(q, dtype=float)
    q = q[None, :] if q.ndim == 1 else q
    phi, g = decoder_value_grad(anp.asarray(q), decoder["params"])
    return np.asarray(phi), np.asarray(g)


def analytic_mse(pred, phi_true, q, g, shots=None):
    """Predicted MSE = squared bias + delta-method variance (and its dB)."""
    d = vl.wrapped_err(pred, phi_true)
    out = {"bias2": float(np.mean(d ** 2))}
    if shots:
        out["var_delta"] = float(np.mean(delta_var(q, g, shots)))
    else:
        out["var_delta"] = 0.0
    out["pred_mse"] = out["bias2"] + out["var_delta"]
    out["pred_mse_db"] = float(10.0 * np.log10(out["pred_mse"] + 1e-30))
    return out


def mitigated_family(mp, psi, pm_noisy):
    """The mitigated family p~(phi) = M_psi(y(phi)) as a plain array."""
    return np.asarray(mp.apply(anp.asarray(pm_noisy, dtype=float),
                               anp.asarray(psi, dtype=float)))





