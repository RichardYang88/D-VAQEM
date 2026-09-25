"""
make_figures.py -- build every figure of the D-VAQEM paper from results/.

The script reads only the artefacts written by ``run_experiments.py``
(``results/*.json``, ``results/*.npz``), so figures are reproducible without
re-running any simulation:

    python make_figures.py [--tag final] [--out ../figures] [--dpi 300]

Each figure is written twice, as a vector PDF (what the manuscript includes)
and as a PNG (for quick inspection).  Panel letters, colors and method labels
are defined once in :data:`STYLE` / :data:`METHODS` and reused everywhere, so
the figures and the manuscript tables speak the same language.
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch      # noqa: E402
from matplotlib.ticker import NullFormatter                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, "results"))
FIG = os.path.abspath(os.path.join(HERE, os.pardir, "figures"))
sys.path.insert(0, HERE)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
# Target widths of the manuscript layout, in inches: one column and the full
# text width.  The Wiley USG class used for the Advanced Quantum Technologies
# submission sets a 210 mm page with 16 mm side margins and a 6 mm column
# separation, i.e. a 86 mm column and a 178 mm text width; the values below are
# within 2% of those, so a figure drawn at DOUBLE and included at
# ``width=\linewidth`` keeps its 8 pt fonts at 8 pt on the page.
SINGLE, DOUBLE = 3.4, 6.9
# The Table-of-Contents graphic the journal asks for is 55 mm x 50 mm.
TOC_W, TOC_H = 55 / 25.4, 50 / 25.4

# method -> (label, color, marker, zorder)
METHODS = {
    "none":           ("unmitigated",        "#444444", "o", 3),
    "zne_rich":       ("ZNE Richardson",     "#1b9e77", "^", 3),
    "zne_poly1":      ("ZNE linear",         "#7fc97f", "v", 3),
    "zne_poly2":      ("ZNE quadratic",      "#a6cee3", "s", 3),
    "linv_known":     ("exact sector inverse", "#e31a1c", "D", 4),
    "retrain_dec":    ("decoder retraining", "#ff7f00", "P", 4),
    "dvaqem":         ("D-VAQEM (selected)", "#1f78b4", "*", 6),
    "dvaqem_lin_mse": ("D-VAQEM linear, shot-aware", "#1f78b4", "*", 5),
    "dvaqem_mlp_mse": ("D-VAQEM MLP, shot-aware", "#6a3d9a", "X", 5),
    "dvaqem_mlp_ce":  ("D-VAQEM MLP, cross-entropy", "#b15928", "h", 4),
    "dvaqem_lin_ce":  ("D-VAQEM linear, cross-entropy", "#cab2d6", "H", 3),
    "dvaqem_lin_l2":  ("D-VAQEM linear, $\\ell_2$", "#a0a0a0", "d", 3),
    "dvaqem_mlp_l2":  ("D-VAQEM MLP, $\\ell_2$", "#d9d9d9", "p", 3),
    "dvaqem_mlp_fisher": ("D-VAQEM MLP, Fisher", "#2ca02c", "8", 4),
    "oracle_ml":      ("oracle (known noise)", "#000000", "x", 2),
    "noiseless":      ("noiseless floor", "#333333", None, 1),
}
CHANNELS = (("depol", "depolarizing", "$p$"),
            ("deph", "dephasing", "$p$"),
            ("ampdamp", "ampl. damping", "$p$"),
            ("readout", "readout", "$q$"))


def load(name, res=RES):
    with open(os.path.join(res, name)) as fh:
        return json.load(fh)


def index(rows, *keys):
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


def split_settings(settings):
    """settings -> {channel prefix: [(strength, label), ...]} ascending."""
    out = {c: [] for c, _, _ in CHANNELS}
    for s in settings:
        name, val = s.rsplit("_", 1)
        out.setdefault(name, []).append((float(val), s))
    return {k: sorted(v) for k, v in out.items() if v}


def panel(ax, letter, title=None):
    ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=9.5,
            weight="bold", va="bottom", ha="right")
    if title:
        ax.set_title(title)


def save(fig, name, out, dpi):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi)
        print(f"  -> {p}")
    plt.close(fig)


# ======================================================================
# Fig. 1  concept, and what the map does to the distribution
# ======================================================================
def _box(ax, x, y, w, h, text, fc="#eef3fb", ec="#33587a", fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8, mutation_aspect=0.35))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            linespacing=1.25)


def _arrow(ax, p0, p1, color="#333333", ls="-", lw=0.9, shrink=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=shrink, shrinkB=shrink))


def schematic(ax):
    """Panel (a): the D-VAQEM data flow.

    Boxes are matplotlib text bboxes, so they size themselves to their content
    and the text can never overflow them; the flow is a 3x3 grid read
    left-to-right, top-to-bottom.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    kw = dict(fontsize=5.4, ha="center", va="center", linespacing=1.35)

    def node(x, y, text, fc, ec):
        return ax.text(x, y, text, bbox=dict(boxstyle="round,pad=0.32",
                                             fc=fc, ec=ec, lw=0.8), **kw)

    n1 = node(0.13, 0.87, "unknown\nphase $\\phi$", "#fff6e5", "#a07020")
    n2 = node(0.48, 0.87, "probe circuit\n$U_R(\\phi;\\theta,\\vartheta)$\n"
                          "$N$ qubits", "#eef3fb", "#33587a")
    n3 = node(0.85, 0.87, "sector reduction\n$\\mathbf{p}_m\\in\\Delta^{N}$\n"
                          "$N{+}1$ numbers", "#e8f5e9", "#2e6b34")
    n4 = node(0.85, 0.53, "noisy device\n$\\mathbf{y}=\\mathcal{N}_\\phi"
                          "[\\mathbf{p}_m]$", "#fdecea", "#8c2f26")
    n5 = node(0.48, 0.53, "D-VAQEM map\n$\\tilde{\\mathbf{p}}=M_\\psi(\\mathbf{y})$\n"
                          "$(N{+}1)^2$ params", "#e7eefb", "#1f4e9c")
    n6 = node(0.13, 0.53, "frozen decoder\n$\\hat\\phi=D_{\\mathbf{w}}"
                          "(\\tilde{\\mathbf{p}})$", "#f3e8fb", "#5b2a86")
    n7 = node(0.15, 0.15, "calibration\ndata $\\mathbf{y}(\\phi_i)$\n"
                          "$i=1\\dots n_{\\rm cal}$", "#fff6e5", "#a07020")
    n8 = node(0.48, 0.15, "noiseless model\n$\\mathbf{t}(\\phi_i)$\n"
                          "(classical)", "#e8f5e9", "#2e6b34")
    n9 = node(0.85, 0.15, "fit $\\psi$ on\n$\\mathcal{L}_{\\rm CE} + "
                          "\\mathrm{Var}_S[\\hat\\phi]$\n(calibration only)",
              "#e7eefb", "#1f4e9c")
    for a, b in ((n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n7, n8),
                 (n8, n9)):
        _arrow(ax, a.get_position(), b.get_position(), color="#555555",
               shrink=15)
    _arrow(ax, n9.get_position(), n5.get_position(), color="#1f4e9c",
           ls=(0, (3, 2)), shrink=22)
    ax.text(0.02, 1.06, "inference: one forward pass, no folded circuits",
            ha="left", va="bottom", fontsize=5.8, style="italic",
            color="#333333")
    # Two wrapped lines: on a single line this note is ~4.2in wide and runs
    # straight into panel (b); wrapped it stays inside panel (a)'s footprint.
    ax.text(0.50, -0.06, "no noise-model knowledge; $N{+}1$ numbers per phase\n"
            "instead of $2^N$ outcomes or $4^N$ tomography coefficients",
            ha="center", va="top", fontsize=5.6, color="#333333",
            linespacing=1.35)



def distributions(setting, noise, meta, N=8, phi=0.6, res=RES):
    """(clean, noisy, mitigated) sector distributions at one phase.

    Uses the disk-cached exact datasets and the *stored* linear-map parameters,
    so the panel reproduces exactly the map that produced the reported numbers.
    """
    import run_experiments as rx
    import vaqem_methods as vm
    model = rx.load_model(N)
    d = vm.build_case(N, model["theta"], model["curly"], np.array([phi]), noise,
                      folds=(1,), workers=1)
    clean = np.asarray(d["pm_clean"], dtype=float)[0].ravel()
    noisy = np.asarray(d["pm_noisy"][1], dtype=float)[0].ravel()
    psi = np.asarray(meta[setting]["dvaqem_lin_mse"]["psi"], dtype=float)
    n = N + 1
    W = psi[:n * n].reshape(n, n)
    b = psi[n * n:] if psi.size > n * n else np.zeros(n)   # the linear map is
    mit = vm._renorm(np.clip(noisy @ W + b, 0.0, None))    # bias-free: 81 params
    m = np.arange(-N, N + 1, 2)
    return m, clean, noisy, np.asarray(mit, dtype=float).ravel()

def fig1(tag, res, out, dpi):
    meta = load(f"{tag}_sweep_meta.json", res)
    setting, noise = "depol_0.01", {"kind": "depolarizing", "p": 0.01,
                                    "readout_p": 0.0}
    cur = np.load(os.path.join(res, f"{tag}_fi_curves.npz"))
    grid = cur[f"{setting}__grid"]
    # Panel b needs the exact dataset (rx.load_model -> the VQ-CNNI checkpoint,
    # located by vl.vqcnni_root()) and the stored linear-map parameters.  This
    # used to degrade silently to a two-panel figure when either was missing,
    # which is how a fig1 that contradicted its own caption (panels a/b/c) and
    # the \ref{fig:concept}c reference in the text ended up archived.  Fail
    # loudly instead: an incomplete figure is worse than no figure.
    try:
        dist = distributions(setting, noise, meta, res=res)
    except Exception as exc:                                  # pragma: no cover
        raise RuntimeError(
            "fig1 panel b could not be built.  It needs rx.load_model(8), i.e. "
            "the VQ-CNNI checkpoint found by vl.vqcnni_root() ($VQCNNI_ROOT or a "
            "sibling ../VQ-CNNI/), plus dvaqem_lin_mse['psi'] in "
            f"{tag}_sweep_meta.json.  Refusing to emit a two-panel figure that "
            "would contradict the caption.  Underlying error: "
            f"{exc!r}") from exc
    ncol = 3
    fig = plt.figure(figsize=(DOUBLE if ncol == 3 else 4.7, 2.55))
    # Panel (a) is a schematic that owns the left half of the figure while the
    # (b),(c) data plots share the right half.  Nesting the (b,c) gridspec lets
    # the a|b seam carry a wider gap than the b|c seam, so the schematic never
    # crowds the plots; the explicit margins reclaim the dead space the default
    # left/right = 0.125/0.90 used to leave on both sides of the figure.
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.178,
                             left=0.01, right=0.99, top=0.86, bottom=0.16)
    inner = outer[0, 1].subgridspec(1, ncol - 1, wspace=0.30)
    ax0 = fig.add_subplot(outer[0, 0])
    schematic(ax0)
    panel(ax0, "a")
    ax1 = fig.add_subplot(inner[0, 0])
    if dist is not None:
        m, clean, noisy, mit = dist
        ax1.plot(m, clean, "-o", color="#2ca02c", ms=2.6,
                 label="noiseless $\\mathbf{p}_m$")
        ax1.plot(m, noisy, "-s", color="#d62728", ms=2.6, label="noisy $\\mathbf{y}$")
        ax1.plot(m, mit, "--*", color="#1f78b4", ms=4.4,
                 label="mitigated $M_\\psi(\\mathbf{y})$")
        ax1.set_xlabel("collective imbalance $m$")
        ax1.set_ylabel("probability")
        ax1.set_yscale("log")
        ax1.set_ylim(1e-4, 1.4)
        ax1.legend(frameon=False, loc="lower center", handlelength=1.7)
        ax1.set_title(f"$\\phi=0.6$ rad, {setting.replace('_', ' ')}", fontsize=7)
    panel(ax1, "b")
    if ncol == 3:
        ax2 = fig.add_subplot(inner[0, 1])
        for key, lab, c, ls in (("fi_clean", "clean $F(\\mathbf{p}_m)$",
                                 "#2ca02c", "-"),
                                ("fi_noisy", "noisy $F(\\mathbf{y})$",
                                 "#d62728", "-"),
                                ("fi_mitigated", "mitigated $F(M_\\psi)$",
                                 "#1f78b4", "--")):
            ax2.plot(grid, cur[f"{setting}__{key}"], ls, color=c, label=lab)
        ax2.set_yscale("log")
        ax2.set_xlabel("phase $\\phi$ [rad]")
        ax2.set_ylabel("classical Fisher information")
        ax2.legend(frameon=False, loc="upper right", handlelength=1.7)
        panel(ax2, "c")
    save(fig, "fig1_concept", out, dpi)


# ======================================================================
# Fig. 2  E1: exactness of the sector reduction
# ======================================================================
def fig2(tag, res, out, dpi):
    rows = load("e1_sufficiency.json", res)
    col = {"noiseless": "#2ca02c", "readout_0.03": "#1f78b4",
           "depol_0.002": "#fd8d3c", "depol_0.01": "#e31a1c",
           "depol_0.02": "#6a3d9a"}
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    fig.subplots_adjust(wspace=0.40)
    ax = axes[0]
    seen = set()
    for r in rows:
        lab = r["noise"]
        c = col.get(lab, "#888888")
        ax.plot(r["mean_FI_p_full"], r["mean_FI_p_m"], "o", ms=3.2, color=c,
                label=lab.replace("_", " ") if lab not in seen else None,
                markerfacecolor=("none" if r["variant"] == "random" else c),
                mew=0.9)
        seen.add(lab)
    lo = min(r["mean_FI_p_full"] for r in rows) * 0.6
    hi = max(r["mean_FI_p_full"] for r in rows) * 1.5
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("$F(\\mathbf{p}_{\\rm full})$")
    ax.set_ylabel("$F(\\mathbf{p}_m)$")
    ax.legend(frameon=False, loc="upper left", fontsize=6.0, handletextpad=0.3,
              title="noise (filled: trained, open: random)", title_fontsize=5.8)
    panel(ax, "a")

    ax = axes[1]
    dep = [r for r in rows if r["kind"] == "depolarizing"]
    for N, mk in ((4, "o"), (6, "s"), (8, "^"), (10, "D")):
        for var, ls in (("trained", "-"), ("random", ":")):
            sub = sorted((r for r in dep if r["N"] == N and r["variant"] == var),
                         key=lambda r: r["p"])
            if not sub:
                continue
            ax.plot([r["p"] for r in sub], [r["crb_gap_db"] for r in sub],
                    ls + mk, ms=3.0, color=plt.cm.viridis((N - 4) / 6.0),
                    label=f"N={N}" if var == "trained" else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("depolarizing rate $p$ per gate")
    ax.set_ylabel("CRB penalty $\\Delta_{\\rm CRB}$ [dB]")
    ax.legend(frameon=False, loc="upper left", fontsize=6.0)
    panel(ax, "b")

    ax = axes[2]
    Ns = sorted({r["N"] for r in rows})
    for lab, c, mk in (("noiseless", "#2ca02c", "o"),
                       ("readout_0.03", "#1f78b4", "s"),
                       ("depol_0.01", "#e31a1c", "^")):
        sub = [r for r in rows if r["noise"] == lab and r["variant"] == "trained"]
        sub = sorted(sub, key=lambda r: r["N"])
        ax.plot([r["N"] for r in sub], [r["mean_FI_p_m"] for r in sub], "-" + mk,
                color=c, ms=3.4, label=lab.replace("_", " "))
        ax.plot([r["N"] for r in sub], [r["mean_FI_p_full"] for r in sub],
                ":", color=c, lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("qubit number $N$")
    ax.set_ylabel("mean Fisher information")
    ax.set_xticks(Ns)
    ax.legend(frameon=False, loc="lower left", fontsize=6.0)
    panel(ax, "c")
    save(fig, "fig2_sufficiency", out, dpi)


# ======================================================================
# Fig. 3  E2: headline comparison over noise channels and shot budgets
# ======================================================================
def fig3(tag, res, out, dpi):
    rows = load(f"{tag}_sweep.json", res)
    inf = index([r for r in rows if r["shots"] == "inf"], "method", "setting")
    settings = split_settings(sorted({r["setting"] for r in rows}))
    show = ["none", "zne_rich", "zne_poly1", "zne_poly2", "linv_known",
            "retrain_dec", "dvaqem", "oracle_ml"]
    fig = plt.figure(figsize=(DOUBLE, 4.4))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.42,
                          height_ratios=[1.15, 1.0])
    nl = one(inf, "noiseless", list(inf["noiseless"])[0])["mse_db"]
    for k, (ch, cname, sym) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[0, k])
        pts = settings.get(ch, [])
        xs = [p for p, _ in pts]
        best = {s: one(inf, "none", s)["best_dvaqem"] for _, s in pts}
        for m in show:
            lab, c, mk, z = METHODS[m]
            if m == "dvaqem":
                ys = [one(inf, best[s], s)["mse_db"] for _, s in pts]
                ax.plot(xs, ys, "-" + (mk or "o"), color=c, ms=5.0, lw=1.2,
                        label=lab, zorder=z)
                continue
            ys = [one(inf, m, s)["mse_db"] if m in inf and s in inf[m] else np.nan
                  for _, s in pts]
            if not np.isfinite(ys).any():
                continue
            ax.plot(xs, ys, "-" + (mk or "o"), color=c, ms=3.2, lw=0.9,
                    label=lab, zorder=z)
        ax.axhline(nl, color="#333333", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"{cname} {sym}", labelpad=1.0)
        if k == 0:
            ax.set_ylabel("MSE [dB]")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=6.2)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(cname, fontsize=7.4, loc="left")
        panel(ax, "abcd"[k])
        ax.grid(axis="y", lw=0.3, color="#dddddd")
        if k == 0:
            hand, labl = ax.get_legend_handles_labels()
    # (e) shot-budget dependence, averaged over all settings
    ax = fig.add_subplot(gs[1, 0:2])
    keys = [k for k in ("inf", "S256", "S1024", "S4096")
            if any(r["shots"] == k for r in rows)]
    xlab = ["$\\infty$", "256", "1024", "4096"]
    allset = sorted({r["setting"] for r in rows})
    for m in ("none", "zne_rich", "retrain_dec", "dvaqem", "oracle_ml",
              "noiseless"):
        lab, c, mk, z = METHODS[m]
        ys = []
        for key in keys:
            sel = index([r for r in rows if r["shots"] == key], "method",
                        "setting")
            if m == "dvaqem":
                vals = [one(sel, one(inf, "none", s)["best_dvaqem"], s)["mse_db"]
                        for s in allset]
            else:
                vals = [one(sel, m, s)["mse_db"] for s in allset
                        if m in sel and s in sel[m]]
            ys.append(float(np.mean(vals)))
        ax.plot(range(len(keys)), ys, "-" + (mk or "o"), color=c, ms=3.4,
                label=lab, zorder=z)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(xlab)
    ax.set_xlabel("shot budget $S$")
    ax.set_ylabel("mean MSE [dB]")
    ax.set_title("mean over all 16 settings", fontsize=7.4, loc="left")
    panel(ax, "e")
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    # (f) per-setting gain waterfall
    ax = fig.add_subplot(gs[1, 2:4])
    order = sorted(allset, key=lambda s: -(one(inf, "none", s)["mse_db"] -
                                           one(inf, one(inf, "none", s)
                                               ["best_dvaqem"], s)["mse_db"]))
    g_inf = [one(inf, "none", s)["mse_db"] -
             one(inf, one(inf, "none", s)["best_dvaqem"], s)["mse_db"]
             for s in order]
    s1024 = index([r for r in rows if r["shots"] == "S1024"], "method", "setting")
    g_fin = [one(s1024, "none", s)["mse_db"] -
             one(s1024, one(inf, "none", s)["best_dvaqem"], s)["mse_db"]
             for s in order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, g_inf, width=0.4, color="#1f78b4", label="infinite shots")
    ax.bar(x + 0.2, g_fin, width=0.4, color="#a6cee3", label="$S=1024$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in order], fontsize=5.0,
                       rotation=55, ha="right", rotation_mode="anchor")
    ax.set_ylabel("MSE reduction vs\nunmitigated [dB]")
    ax.set_title("D-VAQEM gain per noise setting", fontsize=7.4, loc="left")
    panel(ax, "f")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, ncol=2)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    fig.subplots_adjust(bottom=0.20)
    fig.legend(hand, labl, loc="lower center", ncol=4, frameon=False,
               fontsize=6.2, handlelength=1.7, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig3_sweep", out, dpi)

# ======================================================================
# Fig. 4  E3: finite-shot theory -- delta method and the shot-aware map
# ======================================================================
def fig4(tag, res, out, dpi):
    rows = load("e3_shots.json", res)
    idx = index(rows, "setting", "method", "shots")
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    shots = sorted({r["shots"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.2))
    fig.subplots_adjust(hspace=0.68, wspace=0.32)

    ax, s0 = axes[0, 0], settings[0]
    for m in ("none", "zne_poly1", "dvaqem_mlp_ce", "dvaqem_mlp_mse",
              "retrain_dec"):
        lab, c, mk, z = METHODS.get(m, (m, "#888888", "o", 3))
        mc = [one(idx, s0, m, S)["mc_mse"] for S in shots]
        sd = [one(idx, s0, m, S)["mc_mse_std"] for S in shots]
        ax.errorbar(shots, mc, yerr=sd, fmt="-" + (mk or "o"), color=c, ms=3.2,
                    lw=1.0, capsize=1.4, label=lab + " (MC)", zorder=z)
        an = [one(idx, s0, m, S).get("analytic_mse") for S in shots]
        if any(a is not None for a in an):
            ax.plot(shots, [a if a is not None else np.nan for a in an], ":",
                    color=c, lw=1.1, zorder=z)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shot budget $S$")
    ax.set_ylabel("mean squared phase error")
    ax.set_xticks(shots)
    ax.set_xticklabels([str(S) for S in shots], fontsize=6.0)
    ax.legend(frameon=False, fontsize=5.7, loc="upper center", ncol=2,
              handlelength=1.8, bbox_to_anchor=(0.5, -0.22), columnspacing=1.0)
    ax.grid(lw=0.3, color="#dddddd", which="both")
    panel(ax, "a", s0.replace("_", " "))

    ax = axes[0, 1]
    colm = {"none": "#444444", "zne_poly1": "#7fc97f",
            "dvaqem_mlp_ce": "#b15928", "dvaqem_mlp_mse": "#1f78b4"}
    for m, c in colm.items():
        pts = [r for r in rows if r["method"] == m and r.get("ratio")]
        ax.plot([r["shots"] for r in pts], [r["ratio"] for r in pts], "o",
                ms=3.4, color=c, label=METHODS[m][0], mfc="none", mew=1.0)
    ax.axhline(1.0, color="k", lw=0.7)
    ax.axhspan(0.9, 1.1, color="#eeeeee", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("shot budget $S$")
    ax.set_ylabel("Monte Carlo / analytic MSE")
    ax.set_ylim(0.8, 2.2)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    ax.set_title("delta-method validation, all 3 settings", fontsize=7.4,
                 loc="left")
    panel(ax, "b")

    ax = axes[1, 0]
    for m, c in (("dvaqem_mlp_ce", "#b15928"), ("dvaqem_mlp_mse", "#1f78b4")):
        b = np.array([one(idx, s0, m, S)["bias2"] for S in shots])
        v = np.array([one(idx, s0, m, S)["var_over_S"] for S in shots])
        lab = METHODS[m][0]
        ax.plot(shots, b, "-", color=c, lw=1.1, label=f"{lab}: bias$^2$")
        ax.plot(shots, v, "--", color=c, lw=1.0, label=f"{lab}: variance")
        ax.plot(shots, b + v, ":", color=c, lw=1.3, label=f"{lab}: total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shot budget $S$")
    ax.set_ylabel("MSE contributions")
    ax.set_xticks(shots)
    ax.set_xticklabels([str(S) for S in shots], fontsize=6.0)
    ax.legend(frameon=False, fontsize=5.7, loc="upper center", ncol=2,
              handlelength=1.8, bbox_to_anchor=(0.5, -0.22), columnspacing=1.0)
    ax.grid(lw=0.3, color="#dddddd", which="both")
    panel(ax, "c", "bias/variance split (analytic)")

    ax = axes[1, 1]
    for s, c in zip(settings, ("#e31a1c", "#ff7f00", "#1f78b4")):
        d = [10 * np.log10(one(idx, s, "dvaqem_mlp_ce", S)["mc_mse"] /
                           one(idx, s, "dvaqem_mlp_mse", S)["mc_mse"])
             for S in shots]
        ax.plot(shots, d, "-o", color=c, ms=3.0, lw=1.0, label=s.replace("_", " "))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("shot budget $S$")
    ax.set_ylabel("shot-aware gain [dB]")
    ax.set_xticks(shots)
    ax.set_xticklabels([str(S) for S in shots], fontsize=6.0)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.grid(lw=0.3, color="#dddddd")
    panel(ax, "d", "shot-aware refit vs shot-independent map")
    save(fig, "fig4_finite_shots", out, dpi)




# ======================================================================
# Fig. 5  E4: scaling with qubit number, data volume and wall clock
# ======================================================================
def fig5(tag, res, out, dpi):
    rows = load("e4_scaling.json", res)
    idx = index(rows, "N", "method", "shots")
    Ns = sorted({r["N"] for r in rows})
    key = "S1024"
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.75))
    fig.subplots_adjust(wspace=0.40, bottom=0.27)

    ax = axes[0]
    for m in ("noiseless", "none", "zne_rich", "retrain_dec", "dvaqem",
              "oracle_ml"):
        lab, c, mk, z = METHODS[m]
        if m == "dvaqem":
            best = {N: one(idx, N, "none", "inf")["best_dvaqem"] for N in Ns}
            ys = [one(idx, N, best[N], key)["mse_db"] for N in Ns]
            yinf = [one(idx, N, best[N], "inf")["mse_db"] for N in Ns]
        else:
            ys = [one(idx, N, m, key)["mse_db"] for N in Ns]
            yinf = [one(idx, N, m, "inf")["mse_db"] for N in Ns]
        ax.plot(Ns, ys, "-" + (mk or "o"), color=c,
                ms=4.6 if m == "dvaqem" else 3.0,
                lw=1.2 if m == "dvaqem" else 0.9, label=lab, zorder=z)
        ax.plot(Ns, yinf, ":", color=c, lw=0.8, zorder=z)
    ax.set_xlabel("qubit number $N$")
    ax.set_ylabel("MSE [dB]")
    ax.set_xticks(Ns)
    ax.set_title("depolarizing $p=0.01$", fontsize=6.6, loc="left")
    ax.legend(frameon=False, fontsize=5.7, loc="upper center", ncol=2,
              handlelength=1.6, bbox_to_anchor=(0.5, -0.24), columnspacing=1.0)
    ax.grid(lw=0.3, color="#dddddd")
    panel(ax, "a")

    ax = axes[1]
    ax.plot(Ns, [N + 1 for N in Ns], "-o", color="#2e6b34", ms=3.2,
            label="data per phase, $N{+}1$")
    ax.plot(Ns, [(N + 1) ** 2 for N in Ns], "-s", color="#1f78b4", ms=3.2,
            label="linear map, $(N{+}1)^2$")
    ax.plot(Ns, [2 ** N for N in Ns], "--^", color="#ff7f00", ms=3.2,
            label="outcome space, $2^N$")
    ax.plot(Ns, [4 ** N for N in Ns], "--D", color="#e31a1c", ms=3.2,
            label="process tomography, $4^N$")
    ax.set_yscale("log")
    ax.set_xlabel("qubit number $N$")
    ax.set_ylabel("numbers to store / fit")
    ax.set_xticks(Ns)
    ax.legend(frameon=False, fontsize=5.9, loc="upper left", handlelength=1.6)
    ax.grid(lw=0.3, color="#dddddd", which="both")
    panel(ax, "b")

    ax = axes[2]
    fit = [one(idx, N, "none", "inf")["t_fit_s"] for N in Ns]
    data = [one(idx, N, "none", "inf")["t_data_s"] for N in Ns]
    retr = [one(idx, N, "none", "inf")["t_retrain_s"] for N in Ns]
    ax.plot(Ns, fit, "-o", color="#1f78b4", ms=3.2, label="D-VAQEM map fit")
    ax.plot(Ns, retr, "-s", color="#ff7f00", ms=3.2, label="decoder retraining")
    ax.plot(Ns, data, "--^", color="#6a3d9a", ms=3.2,
            label="exact dataset (5 folds)")
    if os.path.exists(os.path.join(res, "oracle_cost.json")):
        cost = {r["N"]: r for r in load("oracle_cost.json", res)}
        nn = [N for N in Ns if N in cost]
        ax.plot(nn, [cost[N]["t_oracle_s"] for N in nn], "--D", color="#000000",
                ms=3.2, label="oracle grid (361 phases)")
        ax.plot(nn, [cost[N]["t_calib_s"] for N in nn], ":*", color="#2e6b34",
                ms=5.0, label="calibration (21 phases)")
    ax.set_yscale("log")
    ax.set_xlabel("qubit number $N$")
    ax.set_ylabel("wall clock [s], 6 workers")
    ax.set_xticks(Ns)
    ax.legend(frameon=False, fontsize=5.7, loc="upper center", ncol=2,
              handlelength=1.6, bbox_to_anchor=(0.5, -0.24), columnspacing=1.0)
    ax.grid(lw=0.3, color="#dddddd", which="both")
    panel(ax, "c")
    save(fig, "fig5_scaling", out, dpi)




# ======================================================================
# Fig. 6  E5: calibration budget
# ======================================================================
def fig6(tag, res, out, dpi):
    rows = load("e5_calib.json", res)
    show = ["none", "zne_rich", "dvaqem_lin_mse", "dvaqem_mlp_mse",
            "dvaqem_mlp_ce", "retrain_dec", "oracle_ml", "noiseless"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.75))
    fig.subplots_adjust(wspace=0.28, bottom=0.36)
    panels = (("n_cal", "n_cal", [5, 9, 17, 25, 41], None,
               "calibration phases $n_{\\rm cal}$"),
              ("cal_shots", "cal_shots", [512, 2048, 8192, None],
               ["512", "2048", "8192", "exact"],
               "shots per calibration target $S_{\\rm cal}$"))
    for ax, (axis, xkey, xs, xlab, xname) in zip(axes, panels):
        sub = [r for r in rows if r["axis"] == axis and r[xkey] in xs]
        ix = index(sub, xkey, "method", "shots")
        skey = sorted({r["shots"] for r in sub})[0]
        for m in show:
            if xs[0] not in ix or m not in ix[xs[0]]:
                continue
            lab, c, mk, z = METHODS[m]
            ys = [one(ix, x, m, skey)["mse_db"] if x in ix and m in ix[x]
                  else np.nan for x in xs]
            ax.plot(range(len(xs)), ys, "-" + (mk or "o"), color=c, ms=3.2,
                    lw=1.0, label=lab, zorder=z)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xlab or [str(x) for x in xs])
        ax.set_xlabel(xname)
        ax.grid(lw=0.3, color="#dddddd")
        if axis == "n_cal":
            ax.set_ylabel(f"MSE [dB], evaluated at {skey[1:]} shots")
            hand, labl = ax.get_legend_handles_labels()
    panel(axes[0], "a", "how many calibration phases?")
    panel(axes[1], "b", "how precise must the targets be?")
    fig.legend(hand, labl, loc="lower center", ncol=2, frameon=False,
               fontsize=5.9, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.09))
    save(fig, "fig6_calibration", out, dpi)


# ======================================================================
# ToC graphic  -- 55 mm x 50 mm, the size Advanced Quantum Technologies
# asks for.  It is uploaded next to the Table-of-Contents text rather than
# \include'd in the manuscript body, so it carries no panel letter.
# ======================================================================
def fig_toc(tag, res, out, dpi):
    """The headline result in one small panel: mean MSE versus shot budget."""
    rows = load(f"{tag}_sweep.json", res)
    inf = index([r for r in rows if r["shots"] == "inf"], "method", "setting")
    keys = [k for k in ("inf", "S256", "S1024", "S4096")
            if any(r["shots"] == k for r in rows)]
    xlab = ["$\\infty$", "256", "1024", "4096"]
    allset = sorted({r["setting"] for r in rows})
    fig, ax = plt.subplots(figsize=(TOC_W, TOC_H))
    for m in ("none", "zne_rich", "dvaqem", "noiseless"):
        lab, c, mk, z = METHODS[m]
        ys = []
        for key in keys:
            sel = index([r for r in rows if r["shots"] == key], "method",
                        "setting")
            if m == "dvaqem":
                vals = [one(sel, one(inf, "none", s)["best_dvaqem"], s)["mse_db"]
                        for s in allset]
            else:
                vals = [one(sel, m, s)["mse_db"] for s in allset
                        if m in sel and s in sel[m]]
            ys.append(float(np.mean(vals)))
        ax.plot(range(len(keys)), ys, "-" + (mk or "o"), color=c, ms=2.6,
                lw=1.0, label=lab, zorder=z)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(xlab)
    ax.tick_params(labelsize=6.0)
    ax.set_xlabel("shot budget $S$", fontsize=7.0)
    ax.set_ylabel("mean MSE [dB]", fontsize=7.0)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.6,
              labelspacing=0.25, borderpad=0.1)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.97, bottom=0.19)
    save(fig, "fig_toc", out, dpi)


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="build the D-VAQEM paper figures")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=FIG)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset, e.g. fig1,fig3")
    a = ap.parse_args()
    want = a.only.split(",") if a.only else None
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3),
                     ("fig4", fig4), ("fig5", fig5), ("fig6", fig6),
                     ("toc", fig_toc)):
        if want and name not in want:
            continue
        print(f"[{name}]", flush=True)
        fn(a.tag, os.path.abspath(a.res), os.path.abspath(a.out), a.dpi)


if __name__ == "__main__":
    main()
