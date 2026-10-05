"""
vaqem_lib.py -- self-contained simulator / model / metric library for the
VAQEM (variational quantum error mitigation) study.

The library reproduces, gate for gate, the VQ-CNNI probe circuit of
``revision_experiments/vqcnni_lib.py`` (enc = dec = 1) but is implemented
with plain NumPy so that

  * arbitrary single-qubit Markovian noise channels can be applied after
    every gate through an explicit Kraus density-matrix simulation,
  * the circuit can be *unitarily folded* to amplify the noise (the
    standard noise-scaling primitive of zero-noise extrapolation and of
    variational quantum error mitigation),
  * thousands of forward evaluations are fast enough for the sweeps in
    this study (a full 8-qubit density-matrix sweep costs ~1 s).

Three simulators are provided and cross-validated in
``validate_simulator.py``:

  probs_pure      -- noiseless statevector probabilities, complex dtype
                     (ground truth for the noiseless circuit);
  probs_pure_real -- the same circuit in a real (Re/Im stacked)
                     representation, so that autograd can differentiate it
                     with respect to the circuit parameters (needed by the
                     noise-aware fine-tuning baseline);
  probs_dm        -- exact Kraus density-matrix simulation with an
                     arbitrary noise channel after every gate and an
                     arbitrary folding factor.

For depolarising gate noise and for readout bit-flip noise the exact
channel can additionally be slid to the end of the circuit (depolarising
channels are unitarily covariant, readout noise acts after everything),
which yields the fast ``flip_kernel_*`` path used during training.
"""

import itertools
import json
import os

import numpy as np
import autograd.numpy as anp


def vqcnni_root():
    """Root of the companion VQ-CNNI checkout.

    That checkout stores the PennyLane-trained ``N=4,6,8`` checkpoints under
    ``revision_experiments/results``; this repository used to live inside it as
    ``new_paper/``, and is now a sibling of it, so both layouts are probed.
    ``VQCNNI_ROOT`` overrides the search.  Only needed to *re-run* experiments:
    every number the paper quotes is stored under ``results/``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("VQCNNI_ROOT"),
        os.path.join(here, os.pardir, os.pardir),             # new_paper/ layout
        os.path.join(here, os.pardir, os.pardir, "VQ-CNNI"),  # sibling checkout
        os.path.expanduser(os.path.join("~", "github", "VQ-CNNI")),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "revision_experiments",
                                            "results")):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, "VQ-CNNI"))


# ======================================================================
# m-value structure (population imbalance m = #0 - #1)
# ======================================================================
def m_structure(N):
    """Return (index_to_m, unique_m, masks) for N qubits.

    ``unique_m`` is ascending, -N, -N+2, ..., N, so the aggregated
    distribution ``p_m`` has length N+1 and index k <-> m = -N + 2k.
    Wire 0 is the most significant bit (PennyLane convention).
    """
    index_to_m = np.array(
        [format(k, f"0{N}b").count("0") - format(k, f"0{N}b").count("1")
         for k in range(2 ** N)], dtype=int)
    unique_m = np.unique(index_to_m)
    masks = [np.where(index_to_m == mm)[0] for mm in unique_m]
    return index_to_m, unique_m, masks


def probs_to_p_m(probs, masks):
    """Aggregate computational-basis probabilities onto m sectors.

    probs: (2^N,) or (B, 2^N) -> returns (N+1,) or (B, N+1).
    Uses only out-of-place NumPy ops so autograd can differentiate it.
    """
    if getattr(probs, "ndim", 1) == 1:
        return anp.stack([anp.sum(probs[idx]) for idx in masks])
    return anp.stack([anp.sum(probs[:, idx], axis=1) for idx in masks], axis=1)


# ======================================================================
# Gate list of the VQ-CNNI probe circuit
# ======================================================================
# A gate is a tuple
#   ('rx'|'ry'|'rz', qubit, angle, noise_wires)   single-qubit rotation
#   ('h',          qubit, None,   noise_wires)    Hadamard
#   ('cnot',       ctrl,  targ,   noise_wires)    CNOT
# ``noise_wires`` lists the qubits that receive the noise channel *after*
# that gate, mirroring ``apply_circuit`` in vqcnni_lib.py: a noise channel
# is applied once per qubit per layer element, i.e. after the full
# CNOT-RZ-CNOT triplet of an RZZ twist (never inside it).
def build_gates(N, phi, theta, curly):
    """Full gate list of the VQ-CNNI circuit (enc = dec = 1).

    Sequence (identical to ``vqcnni_lib.apply_circuit``):
      RY(pi/2)^N -> RZZ(t1/2) pairs -> H^N -> RZZ(t2/2) pairs -> H^N
      -> RX(t3)^N -> RZ(phi)^N                     [phase imprinting]
      -> RX(v3)^N -> H^N -> RZZ(v2/2) pairs -> H^N -> RZZ(v1/2) pairs
      -> RX(pi/2)^N
    with RZZ(chi, i, j) = CNOT(i,j) - RZ(chi,j) - CNOT(i,j).
    """
    t1, t2, t3 = theta[0], theta[1], theta[2]
    v1, v2, v3 = curly[0], curly[1], curly[2]
    pairs = list(itertools.combinations(range(N), 2))
    g = []

    def rzz(chi, i, j):
        g.append(('cnot', i, j, ()))
        g.append(('rz', j, chi, ()))
        g.append(('cnot', i, j, (i, j)))

    for q in range(N):
        g.append(('ry', q, np.pi / 2, (q,)))
    for i, j in pairs:
        rzz(t1 / 2.0, i, j)
    for q in range(N):
        g.append(('h', q, None, (q,)))
    for i, j in pairs:
        rzz(t2 / 2.0, i, j)
    for q in range(N):
        g.append(('h', q, None, (q,)))
    for q in range(N):
        g.append(('rx', q, t3, (q,)))
    for q in range(N):
        g.append(('rz', q, phi, (q,)))
    for q in range(N):
        g.append(('rx', q, v3, (q,)))
    for q in range(N):
        g.append(('h', q, None, (q,)))
    for i, j in pairs:
        rzz(v2 / 2.0, i, j)
    for q in range(N):
        g.append(('h', q, None, (q,)))
    for i, j in pairs:
        rzz(v1 / 2.0, i, j)
    for q in range(N):
        g.append(('rx', q, np.pi / 2, (q,)))
    return g


def gates_per_qubit(N):
    """Number of noise channels acting on each qubit: Lq = 4(N-1)+9."""
    g = build_gates(N, 0.0, np.zeros(3), np.zeros(3))
    cnt = np.zeros(N, dtype=int)
    for gp in g:
        for q in gp[3]:
            cnt[q] += 1
    return cnt


def adjoint_gates(gates):
    """Adjoint of a gate list: reversed order, negated rotation angles.

    The noise markers travel with their physical gate: gate ``G_i`` is
    executed as ``G_i^dag`` at the mirrored position and the channel that
    follows it in the forward circuit still follows it here, so the number
    of channel applications per qubit is preserved.
    """
    out = []
    for gp in reversed(gates):
        kind = gp[0]
        if kind in ('rx', 'ry', 'rz'):
            out.append((kind, gp[1], -gp[2], gp[3]))
        elif kind == 'h':
            out.append(('h', gp[1], None, gp[3]))
        else:
            out.append(('cnot', gp[1], gp[2], gp[3]))
    return out


def fold_gates(gates, fold):
    """Global unitary folding:  G -> G (G^dag G)^n,  noise scale 2n+1.

    Global folding leaves the *noiseless* unitary invariant while
    multiplying the number of noise-channel applications by ``fold``,
    which is the standard noise-amplification primitive of zero-noise
    extrapolation.  Every physical gate execution (forward or inverted) is
    noisy, hence the per-qubit channel count is exactly fold * Lq.
    """
    fold = int(fold)
    if fold < 1:
        raise ValueError("fold must be >= 1")
    if fold % 2 == 0:
        raise ValueError("global folding yields odd scale factors only")
    n = (fold - 1) // 2
    fwd = list(gates)
    bwd = adjoint_gates(gates)
    out = list(fwd)
    for _ in range(n):
        out += bwd + fwd
    return out


# ======================================================================
# Single-qubit matrices and Kraus channels
# ======================================================================
def _rx_mat(a):
    c, s = np.cos(a / 2.0), np.sin(a / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_mat(a):
    c, s = np.cos(a / 2.0), np.sin(a / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz_mat(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)]).astype(complex)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def gate_matrix(gp):
    """2x2 complex matrix of a single-qubit gate tuple."""
    kind = gp[0]
    if kind == 'rx':
        return _rx_mat(gp[2])
    if kind == 'ry':
        return _ry_mat(gp[2])
    if kind == 'rz':
        return _rz_mat(gp[2])
    if kind == 'h':
        return _H
    raise ValueError(kind)


def _mr_mi_block(Mr, Mi):
    """4x4 real block of a complex 2x2 gate acting on (Re0,Re1,Im0,Im1)."""
    return np.stack([np.concatenate([Mr[0], -Mi[0]]),
                     np.concatenate([Mr[1], -Mi[1]]),
                     np.concatenate([Mi[0], Mr[0]]),
                     np.concatenate([Mi[1], Mr[1]])])


def gate_real_block(gp):
    """4x4 real matrix of a single-qubit gate, autograd-differentiable.

    Built from cos/sin of the rotation angle with plain NumPy ops only (no
    complex exponentials, no np.diag of boxed values), so that
    ``autograd.grad`` can differentiate the whole statevector simulation
    with respect to the gate angles.
    """
    kind = gp[0]
    if kind == 'h':
        return _mr_mi_block(_H.real, _H.imag)
    a = gp[2]
    c, s = np.cos(a / 2.0), np.sin(a / 2.0)
    z = 0.0 * c
    if kind == 'rz':      # Mr = c I, Mi = diag(-s, s)
        Mr = np.stack([np.stack([c, z]), np.stack([z, c])])
        Mi = np.stack([np.stack([-s, z]), np.stack([z, s])])
    elif kind == 'ry':    # Mr = [[c,-s],[s,c]], Mi = 0
        Mr = np.stack([np.stack([c, -s]), np.stack([s, c])])
        Mi = np.stack([np.stack([z, z]), np.stack([z, z])])
    elif kind == 'rx':    # Mr = c I, Mi = [[0,-s],[-s,0]]
        Mr = np.stack([np.stack([c, z]), np.stack([z, c])])
        Mi = np.stack([np.stack([z, -s]), np.stack([-s, z])])
    else:
        raise ValueError(kind)
    return _mr_mi_block(Mr, Mi)



def kraus_channel(kind, p):
    """Kraus operators of a single-qubit Markovian channel of strength p."""
    if kind in ("none", None) or p == 0:
        return None
    if kind == "depolarizing":
        return [np.sqrt(1.0 - p) * _I2, np.sqrt(p / 3.0) * _X,
                np.sqrt(p / 3.0) * _Y, np.sqrt(p / 3.0) * _Z]
    if kind == "bit_flip":
        return [np.sqrt(1.0 - p) * _I2, np.sqrt(p) * _X]
    if kind == "dephasing":
        # Pauli dephasing: rho -> (1-p) rho + p Z rho Z
        return [np.sqrt(1.0 - p) * _I2, np.sqrt(p) * _Z]
    if kind == "amplitude_damping":
        return [np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]]),
                np.array([[0.0, np.sqrt(p)], [0.0, 0.0]])]
    raise ValueError(kind)


def cnot_perm(N, c, t):
    """Permutation of basis indices induced by CNOT(c, t)."""
    idx = np.arange(2 ** N)
    ctrl = ((idx >> (N - 1 - c)) & 1) == 1
    flip = 1 << (N - 1 - t)
    perm = idx.copy()
    perm[np.where(ctrl)[0]] = idx[np.where(ctrl)[0]] ^ flip
    return perm


def bit_perm(N, q):
    """Permutation exchanging |0>_q and |1>_q (the X Pauli)."""
    idx = np.arange(2 ** N)
    return idx ^ (1 << (N - 1 - q))


# ======================================================================
# Simulator 1: noiseless statevector (complex), single phase
# ======================================================================
def _apply1_c(st, U, q, N):
    st = st.reshape([2] * N)
    st = np.tensordot(U, st, axes=([1], [q]))
    return np.moveaxis(st, 0, q).reshape(-1)


def probs_pure(phi, N, theta, curly, fold=1):
    """Exact noiseless outcome probabilities (2^N,) for one phase."""
    gates = fold_gates(build_gates(N, phi, theta, curly), fold)
    st = np.zeros(2 ** N, dtype=complex)
    st[0] = 1.0
    for gp in gates:
        if gp[0] == 'cnot':
            st = st[cnot_perm(N, gp[1], gp[2])]
        else:
            st = _apply1_c(st, gate_matrix(gp), gp[1], N)
    return np.abs(st) ** 2


# ======================================================================
# Simulator 2: noiseless statevector in a real (Re/Im) representation,
# differentiable with autograd w.r.t. the gate angles.
# ======================================================================
def _real_gate_block(M):
    """4x4 real matrix of a 2x2 complex gate acting on (Re0,Re1,Im0,Im1)."""
    Mr, Mi = np.real(M), np.imag(M)
    return np.block([[Mr, -Mi], [Mi, Mr]])


def _apply1_r(R, R4, q, N):
    """Apply complex 1-qubit gate M to the real state R of shape (2, 2^N).

    R[0] = Re(psi), R[1] = Im(psi).  The gate acts on the combined
    (Re/Im, qubit) 4-dimensional space of qubit q through the real block
    [[Mr, -Mi], [Mi, Mr]], using only real NumPy ops so that autograd can
    differentiate w.r.t. the rotation angles.
    """
    pre, post = 2 ** q, 2 ** (N - q - 1)
    # (ri, before, bit, after) -> (before*after, (ri, bit))
    Rr = R.reshape(2, pre, 2, post).transpose(1, 3, 0, 2).reshape(-1, 4)
    out = Rr @ R4.T                                     # (pre*post, 4)
    out = out.reshape(pre, post, 2, 2).transpose(2, 0, 3, 1)
    return out.reshape(2, 2 ** N)



def probs_pure_real(phi, N, theta, curly, fold=1):
    """Same as :func:`probs_pure` but with only real NumPy ops, so that
    ``autograd.grad`` can differentiate it w.r.t. ``theta``/``curly``."""
    gates = fold_gates(build_gates(N, phi, theta, curly), fold)
    R = np.zeros((2, 2 ** N))
    R[0, 0] = 1.0
    for gp in gates:
        if gp[0] == 'cnot':
            R = R[:, cnot_perm(N, gp[1], gp[2])]
        else:
            R = _apply1_r(R, gate_real_block(gp), gp[1], N)
    return R[0] ** 2 + R[1] ** 2


# ======================================================================
# Simulator 3: exact Kraus density-matrix simulation with noise + folding
# ======================================================================
def _apply1_dm(rho, K, q, N):
    """rho -> K_q rho K_q^dag for a 2x2 operator K acting on qubit q.

    Cost O(4^N) instead of the O(8^N) of an explicit d x d matmul.
    """
    d = 2 ** N
    pre, post = 2 ** q, 2 ** (N - q - 1)
    r = rho.reshape(pre, 2, post, pre, 2, post)
    r = np.moveaxis(np.tensordot(K, r, axes=([1], [1])), 0, 1)
    r = np.moveaxis(np.tensordot(r, K.conj(), axes=([4], [1])), -1, 4)
    return r.reshape(d, d)


_E2 = np.eye(2)
_SGN = np.array([[1.0, -1.0], [-1.0, 1.0]])
FAST_PAULI_CHANNELS = True     # set False to force the generic Kraus loop


def _apply_pauli_dm(rho, kind, p, q, N):
    """Fast O(4^N) application of a Pauli-type channel to qubit q.

    Uses the closed forms  D_p(rho) = a rho + (1-a) I_q/2 (x) Tr_q rho  with
    a = 1-4p/3 (depolarising),  (1-p) rho + p Z rho Z  (dephasing) and
    (1-p) rho + p X rho X  (bit flip), each of which needs one broadcast
    instead of 2 x n_Kraus tensordots.  Identical to the generic Kraus loop to
    machine precision (checked in validate_simulator.py).
    """
    d = 2 ** N
    pre, post = 2 ** q, 2 ** (N - q - 1)
    r = rho.reshape(pre, 2, post, pre, 2, post)
    if kind == "depolarizing":
        a = 1.0 - 4.0 * p / 3.0
        sig = np.trace(r, axis1=1, axis2=4)                  # Tr_q rho
        emb = sig[:, None, :, :, None, :] * _E2[None, :, None, None, :, None]
        return (a * r + 0.5 * (1.0 - a) * emb).reshape(d, d)
    if kind == "dephasing":
        sgn = _SGN[None, :, None, None, :, None]
        return ((1.0 - p) * r + p * sgn * r).reshape(d, d)
    if kind == "bit_flip":
        # X_q rho X_q flips the qubit index of the row *and* of the column
        return ((1.0 - p) * r + p * r[:, ::-1, :, :, ::-1, :]).reshape(d, d)
    raise ValueError(kind)


_PAULI_KINDS = ("depolarizing", "dephasing", "bit_flip")


def _noise_dm(rho, kind, p, q, N, kraus):
    """One noise channel on qubit q (fast Pauli path when available)."""
    if FAST_PAULI_CHANNELS and kind in _PAULI_KINDS:
        return _apply_pauli_dm(rho, kind, p, q, N)
    out = np.zeros_like(rho)
    for K in kraus:
        out += _apply1_dm(rho, K, q, N)
    return out




def probs_dm(phi, N, theta, curly, noise_kind="depolarizing", noise_p=0.0,
             fold=1, readout_p=0.0):
    """Exact outcome probabilities (2^N,) of the noisy folded circuit.

    ``noise_kind``/``noise_p``: single-qubit channel applied after every
    gate (see :func:`kraus_channel`).  ``readout_p``: additional
    independent bit-flip applied to the measurement statistics.
    """
    gates = fold_gates(build_gates(N, phi, theta, curly), fold)
    kraus = kraus_channel(noise_kind, noise_p)
    d = 2 ** N
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    perms = {}
    for gp in gates:
        if gp[0] == 'cnot':
            key = (gp[1], gp[2])
            if key not in perms:
                perms[key] = cnot_perm(N, gp[1], gp[2])
            rho = rho[perms[key]][:, perms[key]]
        else:
            rho = _apply1_dm(rho, gate_matrix(gp), gp[1], N)
        if kraus is not None:
            for q in gp[3]:
                rho = _noise_dm(rho, noise_kind, noise_p, q, N, kraus)
    p = np.clip(np.real(np.diag(rho)), 0.0, None)
    if readout_p > 0:
        p = apply_flip(p, N, readout_p)
    return p / p.sum()


# ======================================================================
# Bit-flip (readout) kernels: full 2^N space and m-sector space
# ======================================================================
def apply_flip(p, N, f):
    """Apply independent per-qubit bit flips of rate f to a distribution.

    p: (2^N,) or (B, 2^N).  Linear, differentiable, and O(B 2^N) in cost
    (no 4^N kernel is ever materialised).
    """
    p = np.asarray(p)
    one_d = (p.ndim == 1)
    q = p[None, :] if one_d else p
    out = q
    for k in range(N):
        perm = bit_perm(N, k)
        out = (1.0 - f) * out + f * out[:, perm]
    return out[0] if one_d else out


def flip_kernel_full(N, f):
    """Explicit K[s', s] = P(observe s' | true s) matrix (small N only)."""
    K1 = np.array([[1 - f, f], [f, 1 - f]])
    K = K1
    for _ in range(N - 1):
        K = np.kron(K, K1)
    return K


def m_flip_kernel(N, f):
    """Exact (N+1)x(N+1) channel induced on the m-sector distribution.

    K_m[m', m] = P(observed imbalance m' | true imbalance m) for
    independent per-qubit bit flips of rate f.  With n0 = (N+m)/2 zeros
    and n1 = (N-m)/2 ones in the sector, j of the zeros and l of the ones
    flip, giving m' = m - 2j + 2l.  Because the flip channel is
    permutation-covariant, aggregating the 2^N-dimensional kernel onto
    sectors is *exact*:  A K_full = K_m A.
    """
    from math import comb
    ms = np.arange(-N, N + 1, 2)
    K = np.zeros((N + 1, N + 1))
    for a, m in enumerate(ms):
        n0, n1 = (N + m) // 2, (N - m) // 2
        for j in range(n0 + 1):
            pj = comb(n0, j) * f ** j * (1 - f) ** (n0 - j)
            for l in range(n1 + 1):
                pl = comb(n1, l) * f ** l * (1 - f) ** (n1 - l)
                b = (m - 2 * j + 2 * l + N) // 2
                K[b, a] += pj * pl
    return K


def m_aggregate_matrix(N):
    """A[k, s] = 1 if outcome s lies in m-sector k (shape (N+1, 2^N))."""
    index_to_m, unique_m, _ = m_structure(N)
    A = np.zeros((N + 1, 2 ** N))
    for k, mm in enumerate(unique_m):
        A[k, index_to_m == mm] = 1.0
    return A


# ======================================================================
# Exact noise bookkeeping for flip-equivalent channels
# ======================================================================
def depol_equiv_flip(N, p, fold=1):
    """Readout bit-flip rate *approximately* equivalent to depolarising(p).

    Depolarising channels are unitarily covariant, so a channel can be slid
    through any gate acting on the same wires, and compositions multiply the
    shrink factor: D_p(rho) = a rho + (1-a) I/2 with a = 1 - 4p/3, hence
    a_tot = a^(fold * Lq) with Lq = 4(N-1)+9 channels per qubit.  A depolarised
    qubit with shrink factor a has computational-basis statistics identical to
    a bit flip of rate f = (1-a)/2.

    The sliding is exact for single-qubit gates but *not* for entangling ones
    (a locally depolarised wire cannot be commuted through a CNOT), so this is
    a fast surrogate rather than an identity.  ``validate_simulator.py``
    quantifies the error against the exact Kraus density-matrix simulation:
    max |Delta p_m| = 7.9e-3, 1.2e-2, 1.3e-2, 9.0e-3, 6.7e-3 for
    p = 0.001, 0.002, 0.005, 0.01, 0.02 (N=8, fold=1), and exactly 0 for p=0.
    All headline results in this work use the exact Kraus path; the surrogate
    is used only for the model-based ``linv_known`` baseline and for cheap
    large-scale sweeps.
    """
    Lq = 4 * (N - 1) + 9
    a = (1.0 - 4.0 * p / 3.0) ** (fold * Lq)
    return 0.5 * (1.0 - a)


def combine_flips(f1, f2):
    """Composition of two independent bit-flip channels."""
    return f1 * (1 - f2) + f2 * (1 - f1)


def effective_flip(N, noise, fold=1):
    """Total end-of-circuit flip rate for a flip-equivalent noise spec.

    Returns None when the noise is not flip-equivalent (dephasing,
    amplitude damping), i.e. when the m-space channel is not a binomial
    flip kernel and the mitigation map must be learned from data.
    """
    kind, p, q = noise["kind"], noise["p"], noise.get("readout_p", 0.0)
    Lq = 4 * (N - 1) + 9
    if kind == "depolarizing":
        f = 0.5 * (1.0 - (1.0 - 4.0 * p / 3.0) ** (fold * Lq))
    elif kind == "bit_flip":
        # B_p(rho) = (1-2p) rho + 2p I/2, so the shrink factor composes
        f = 0.5 * (1.0 - (1.0 - 2.0 * p) ** (fold * Lq))
    elif kind in ("none", None) or p == 0:
        f = 0.0
    else:
        return None
    return combine_flips(f, q)


# ======================================================================
# Density matrix and QFI
# ======================================================================
def rho_dm(phi, N, theta, curly, noise_kind="none", noise_p=0.0, fold=1,
           readout_p=0.0):
    """Density matrix at the end of the (folded, noisy) circuit."""
    gates = fold_gates(build_gates(N, phi, theta, curly), fold)
    kraus = kraus_channel(noise_kind, noise_p)
    d = 2 ** N
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    perms = {}
    for gp in gates:
        if gp[0] == 'cnot':
            key = (gp[1], gp[2])
            if key not in perms:
                perms[key] = cnot_perm(N, gp[1], gp[2])
            rho = rho[perms[key]][:, perms[key]]
        else:
            rho = _apply1_dm(rho, gate_matrix(gp), gp[1], N)
        if kraus is not None:
            for q in gp[3]:
                rho = _noise_dm(rho, noise_kind, noise_p, q, N, kraus)
    return rho


def _state_pure(phi, N, theta, curly, fold=1):
    gates = fold_gates(build_gates(N, phi, theta, curly), fold)
    s = np.zeros(2 ** N, dtype=complex)
    s[0] = 1.0
    for gp in gates:
        if gp[0] == 'cnot':
            s = s[cnot_perm(N, gp[1], gp[2])]
        else:
            s = _apply1_c(s, gate_matrix(gp), gp[1], N)
    return s


def qfi_pure(N, theta, curly, phi0=0.0, dphi=1e-5, fold=1):
    """Pure-state QFI of the noiseless probe by central finite differences."""
    sp = _state_pure(phi0 + dphi, N, theta, curly, fold)
    sm = _state_pure(phi0 - dphi, N, theta, curly, fold)
    dpsi = (sp - sm) / (2 * dphi)
    psi = (sp + sm) / 2
    return float(4 * (np.vdot(dpsi, dpsi).real
                      - np.abs(np.vdot(psi, dpsi)) ** 2))


def qfi_mixed(N, theta, curly, noise, phi0=0.0, dphi=1e-5, fold=1):
    """SLD-QFI of the noisy (folded) probe state rho(phi)."""
    kw = dict(noise_kind=noise["kind"], noise_p=noise["p"], fold=fold,
              readout_p=noise.get("readout_p", 0.0))
    rp = rho_dm(phi0 + dphi, N, theta, curly, **kw)
    rm = rho_dm(phi0 - dphi, N, theta, curly, **kw)
    rho0, drho = (rp + rm) / 2, (rp - rm) / (2 * dphi)
    vals, vecs = np.linalg.eigh(rho0)
    vals = np.clip(vals, 0.0, None)
    D = vecs.conj().T @ drho @ vecs
    denom = vals[:, None] + vals[None, :]
    ok = denom > 1e-12
    F = np.zeros_like(denom)
    F[ok] = 2 * np.abs(D[ok]) ** 2 / denom[ok]
    return float(F.sum())


# ======================================================================
# Classical decoder (identical architecture to the trained VQ-CNNI model)
# ======================================================================
def softsign(x):
    return x / (1.0 + anp.abs(x))


def mlp_shapes(N, hidden=128):
    return [(hidden, N + 1), (hidden,), (hidden // 2, hidden), (hidden // 2,),
            (2, hidden // 2), (2,)]


def n_mlp_params(N, hidden=128):
    return int(sum(int(np.prod(s)) for s in mlp_shapes(N, hidden)))


def unpack_mlp(flat, N, hidden=128):
    """Flat vector -> [W1, b1, W2, b2, W3, b3] of the (N+1)-hidden-h/2-2 MLP.

    Works on plain arrays *and* on autograd boxes, so it can sit inside a
    differentiable cost (decoder retraining, shot-aware mitigation).
    """
    from autograd.tracer import Box
    flat = flat if isinstance(flat, Box) else np.asarray(flat, dtype=float)
    flat = anp.reshape(flat, -1)
    params, off = [], 0
    for shp in mlp_shapes(N, hidden):
        sz = int(np.prod(shp))
        params.append(anp.reshape(flat[off:off + sz], shp))
        off += sz
    return params


def net_batch(p_m, params):
    """Batched MLP forward pass: (B, N+1) -> (B, 2), L2-normalised output."""
    W1, b1, W2, b2, W3, b3 = params
    h1 = softsign(anp.matmul(p_m, anp.transpose(W1)) + b1)
    h2 = softsign(anp.matmul(h1, anp.transpose(W2)) + b2)
    out = anp.matmul(h2, anp.transpose(W3)) + b3
    norm = anp.sqrt(anp.sum(out ** 2, axis=1, keepdims=True) + 1e-12)
    return out / norm


def predict_from_pm(p_m, params):
    """phi_hat = arctan2(v0, v1) for a batch of m-distributions."""
    x = p_m if getattr(p_m, "ndim", 1) == 2 else anp.reshape(p_m, (1, -1))
    v = net_batch(x, params)
    return anp.arctan2(v[:, 0], v[:, 1])


def load_vqcnni(path):
    """Load a trained VQ-CNNI checkpoint (train_vqcnni_scaling.py format)."""
    d = np.load(path, allow_pickle=True)
    x = np.asarray(d["x_best"], dtype=float)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    N = int(meta.get("N", d["N"] if "N" in d.files else 8))
    hidden = int(meta.get("hidden", 128))
    return {"x": x, "N": N, "hidden": hidden, "meta": meta,
            "theta": x[:3].copy(), "curly": x[3:6].copy(),
            "params": unpack_mlp(x[6:], N, hidden)}


# ======================================================================
# Metrics
# ======================================================================
def wrapped_err(pred, true):
    """Wrapped phase error in (-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(pred) - np.asarray(true))))


def swpe_db(pred, true):
    """Squared wrapped phase error in dB (lower is better)."""
    return 10.0 * np.log10(wrapped_err(pred, true) ** 2 + 1e-12)


def mean_swpe_db(pred, true):
    return float(np.mean(swpe_db(pred, true)))


def circular_loss(pred, true):
    """2(1 - cos) circular loss, averaged (the VQ-CNNI training objective)."""
    return float(np.mean(1.0 - np.cos(wrapped_err(pred, true))))


def test_phases(n_phi=100, n_test=50):
    """Training grid and half-offset test grid, as in the VQ-CNNI paper."""
    phi_train = np.linspace(-np.pi, np.pi, n_phi)
    phi_test = np.linspace(-np.pi + np.pi / n_phi, np.pi + np.pi / n_phi,
                           n_test)
    return phi_train, phi_test


def fi_grid(phis, pm, floor=1e-12):
    """Classical Fisher information of a pmf family sampled on a phase grid.

    phis: (P,) ascending, pm: (P, K) pmfs.  Returns the FI at every grid
    phase, using central differences (np.gradient), i.e. the information
    actually available from the measured distribution -- the quantity that
    bounds the achievable mean squared error through the Cramer-Rao bound.

    Bins whose probability is at or below ``floor`` are dropped from the sum
    rather than divided by a numerical zero: an outcome that cannot occur
    carries no information about the phase.  This only ever matters for
    *mitigated* families, whose quasi-probability map is clipped at zero before
    decoding -- the clip leaves exact zeros, the family is not smooth there, so
    the Cramer-Rao regularity conditions fail and dividing by 1e-300 would
    report an absurd ~1e300 instead of the (finite, support-restricted)
    information the surviving outcomes carry.  Clean and noisy families are
    strictly positive and are unaffected.
    """
    pm = np.asarray(pm, dtype=float)
    dpm = np.gradient(pm, np.asarray(phis, dtype=float), axis=0)
    keep = pm > floor
    return np.sum(np.where(keep, dpm ** 2 / np.where(keep, pm, 1.0), 0.0),
                  axis=1)


def crb_db(fi):
    """Cramer-Rao bound on the mean squared phase error, in dB."""
    return 10.0 * np.log10(1.0 / np.maximum(np.asarray(fi, dtype=float), 1e-300))


class Adam:
    """Manual Adam, matching qml.AdamOptimizer(stepsize, 0.9, 0.99)."""

    def __init__(self, lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.fm = self.sm = None
        self.t = 0

    def step(self, x, g):
        x = np.asarray(x, dtype=float)
        g = np.asarray(g, dtype=float)
        if self.fm is None:
            self.fm = np.zeros_like(x)
            self.sm = np.zeros_like(x)
        self.fm = self.b1 * self.fm + (1 - self.b1) * g
        self.sm = self.b2 * self.sm + (1 - self.b2) * g ** 2
        self.t += 1
        eta = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        return x - eta * self.fm / (np.sqrt(self.sm) + self.eps)





