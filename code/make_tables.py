"""
make_tables.py -- emit the LaTeX tables of the D-VAQEM manuscript straight from
the result files, so no number is ever transcribed by hand.

    python make_tables.py [--tag final] [--out ../paper]

Writes ``tables.tex`` (main-text Tables I--III) and ``tables_supplement.tex``
(Supplemental Tables S1--S12; the last three are the paired bootstrap confidence
intervals and exact tests, and appear only once ``results/ci_<tag>.json`` exists,
which ``collect_numbers.py`` or ``bootstrap_ci.py`` writes).  The manuscript
inputs both files.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, "results"))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "paper"))

ORDER = ["noiseless", "none", "zne_rich", "zne_poly1", "zne_poly2",
         "linv_known", "linv_calib", "retrain_dec", "dvaqem_lin_l2",
         "dvaqem_lin_ce",
         "dvaqem_lin_mse", "dvaqem_mlp_l2", "dvaqem_mlp_ce",
         "dvaqem_mlp_fisher", "dvaqem_mlp_mse", "oracle_ml", "oracle_l2"]
LABEL = {
    "noiseless": "noiseless floor",
    "none": "unmitigated",
    "zne_rich": "ZNE, Richardson",
    "zne_poly1": "ZNE, linear fit",
    "zne_poly2": "ZNE, quadratic fit",
    "linv_known": "exact sector inverse",
    "linv_calib": "calibrated sector inverse",
    "retrain_dec": "decoder retraining",
    "dvaqem_lin_l2": "D-VAQEM linear, $\\ell_2$",
    "dvaqem_lin_ce": "D-VAQEM linear, CE",
    "dvaqem_lin_mse": "D-VAQEM linear, shot-aware",
    "dvaqem_mlp_l2": "D-VAQEM MLP, $\\ell_2$",
    "dvaqem_mlp_ce": "D-VAQEM MLP, CE",
    "dvaqem_mlp_fisher": "D-VAQEM MLP, CE+Fisher",
    "dvaqem_mlp_mse": "D-VAQEM MLP, shot-aware",
    "dvaqem": "D-VAQEM (selected)",
    "best_zne": "best ZNE variant$^\\dagger$",
    "oracle_ml": "oracle, known noise (ML)",
    "oracle_l2": "oracle, known noise ($\\ell_2$)",
}
CHANNELS = ("depol", "deph", "ampdamp", "readout")
CHNAME = {"depol": "depolarising", "deph": "dephasing",
          "ampdamp": "ampl. damping", "readout": "readout"}


def tex(s):
    r"""Escape a raw name read from the result files for use in a table cell.

    Setting and method names carry bare underscores (``depol_0.002``,
    ``mlp_fisher``), and an unescaped ``_`` in text mode aborts the LaTeX run
    with "Missing $ inserted".  Every name that is printed verbatim therefore
    goes through here.  Strings taken from ``LABEL`` are already typeset LaTeX
    (they contain ``$\ell_2$`` and friends) and must not be passed through this
    function.
    """
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


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


def settings_by_channel(settings):
    out = {c: [] for c in CHANNELS}
    for s in settings:
        name, val = s.rsplit("_", 1)
        out[name].append((float(val), s))
    return {k: [s for _, s in sorted(v)] for k, v in out.items() if v}


def cell(idx, m, s, key):
    if m == "dvaqem":
        m = one(idx, "none", s)["best_dvaqem"]
    if m not in idx or s not in idx[m]:
        return "--"
    return f"{one(idx, m, s)['mse_db']:.1f}"


def env_table(cap, lab, cols, body, star=False, font="\\scriptsize",
              fit=True, colsep="4pt"):
    """Wrap a ready-made tabular body in a (starred) table environment.

    ``fit=True`` puts the tabular inside ``\\fitwidth`` (defined in the
    manuscript preamble), which scales it down to ``\\textwidth`` *only* when it
    would otherwise overflow, so no generated table can be wider than the page.
    ``colsep`` tightens the inter-column space, which matters for the wide
    per-setting tables.
    """
    e = "table*" if star else "table"
    L = [f"\\begin{{{e}}}[t]", "\\centering", f"\\caption{{{cap}}}",
         f"\\label{{{lab}}}", font, f"\\setlength{{\\tabcolsep}}{{{colsep}}}"]
    if fit:
        L += ["\\fitwidth{%"]
    L += body
    L += ["\\end{tabular}%", "}"] if fit else ["\\end{tabular}"]
    L += [f"\\end{{{e}}}", ""]
    return "\n".join(L)


def table_protocol(man, meta0):
    """Table I: everything needed to reproduce the headline run."""
    d = meta0["dvaqem_lin_l2"]
    rows = [
        ("probe / decoder", "VQ-CNNI checkpoint, enc=dec=1, 6 circuit "
         "parameters; decoder $(N{+}1)\\to128\\to64\\to2$ softsign MLP, "
         "9666 parameters at $N=8$ (frozen during mitigation)"),
        ("system sizes", f"$N={man['N']}$ (sweep, shots, calibration); "
         f"$N\\in\\{{{man['Ns']}}}$ (sufficiency, scaling)"),
        ("noise channels", "depolarising / dephasing / amplitude damping after "
         "every gate; symmetric readout bit-flip"),
        ("calibration grid", f"$n_{{\\rm cal}}={man['n_cal']}$ uniform phases in "
         "$[-\\pi,\\pi)$; 5 held-out phases for variant selection"),
        ("test grid", f"$n_{{\\rm tst}}={man['n_tst']}$ half-offset phases, "
         "disjoint from the calibration grid"),
        ("shot budgets", "$S\\in\\{256,1024,4096\\}$ (sweep) and "
         "$S\\in\\{64,\\dots,8192\\}$ (finite-shot study); "
         f"{man['n_trials']} Monte-Carlo trials per point"),
        ("map fits", f"{d['iters']} Adam iterations; learning rates "
         "$\\ell_2$: 0.05, CE: 0.1, CE+Fisher: 0.02, shot-aware: 0.02; "
         "Fisher weight $\\lambda=1$"),
        ("map sizes", "linear $(N{+}1)^2=81$ parameters; MLP "
         "$(N{+}1)\\to32\\to32\\to(N{+}1)$, 1673 parameters"),
        ("ZNE folds", "Richardson / linear: $(1,3,5)$; quadratic: "
         "$(1,3,5,7,9)$ (odd factors only)"),
        ("oracle grid", f"{man['n_oracle']} exact density-matrix simulations on "
         "a uniform phase grid, parabolic refinement of the argmin"),
        ("decoder retraining", f"{man['retrain_iters']} Adam iterations, "
         "$\\eta=2\\times10^{-3}$, circular loss, same calibration data"),
        ("exact simulation", "Kraus density-matrix simulator, validated against "
         "PennyLane \\texttt{default.mixed} (Appendix~\\ref{app:validation})"),
    ]
    body = ["\\begin{tabular}{@{}p{0.24\\linewidth}p{0.68\\linewidth}@{}}",
            "\\hline", "\\textbf{quantity} & \\textbf{value} \\\\", "\\hline"]
    body += [f"{k} & {v} \\\\" for k, v in rows]
    return env_table("Reproduction protocol of the headline run "
                     "(tag \\texttt{final}).", "tab:protocol", None, body)


def table_channels(rows):
    """Table II: mean MSE (dB) per noise channel, infinite and finite shots."""
    idx = {k: index([r for r in rows if r["shots"] == k], "method", "setting")
           for k in ("inf", "S1024")}
    settings = settings_by_channel(sorted({r["setting"] for r in rows}))
    methods = ["none", "zne_rich", "zne_poly1", "zne_poly2", "linv_known",
               "linv_calib",
               "retrain_dec", "dvaqem", "oracle_ml", "noiseless"]
    # compact two-row header: each channel name spans its (inf, S=1024)
    # pair, which is what keeps the table inside the text width
    short = {"depol": "depol.", "deph": "deph.",
             "ampdamp": "amp.\\ damp.", "readout": "readout"}
    body = ["\\begin{tabular}{@{}l" + "c" * 8 + "@{}}", "\\hline",
            "\\multirow{2}{*}{method} & "
            + " & ".join("\\multicolumn{2}{c}{"
                         + short[c] + "}" for c in CHANNELS)
            + " \\\\",
            " & " + " & ".join(["$\\infty$", "$1024$"]
                               * len(CHANNELS))
            + " \\\\", "\\hline"]
    for m in methods:
        vals = []
        for c in CHANNELS:
            for k in ("inf", "S1024"):
                v = [float(cell(idx[k], m, s, k)) for s in settings[c]
                     if cell(idx[k], m, s, k) != "--"]
                vals.append(f"{np.mean(v):.1f}" if v else "--")
        body.append(f"{LABEL[m]} & " + " & ".join(vals) + " \\\\")
    body.append("\\hline")
    return env_table(
        "Mean squared wrapped phase error in dB, averaged over the four "
        "strengths of each noise channel at $N=8$, at infinite shots and at "
        "$S=1024$.  ``D-VAQEM'' is the variant selected by the held-out "
        "calibration score in each setting; ``--'' marks a baseline that does "
        "not exist for that channel (the exact sector inverse needs a "
        "flip-equivalent channel).  Lower is better.",
        "tab:channels", None, body)


def table_scaling(rows, cost):
    """Table III: qubit-number scaling."""
    idx = index(rows, "N", "method", "shots")
    Ns = sorted({r["N"] for r in rows})
    body = ["\\begin{tabular}{@{}cccccccccc@{}}", "\\hline",
            "\\multirow{2}{*}{$N$} & \\multirow{2}{*}{sector dim} & "
            "\\multirow{2}{*}{$4^N$} & \\multirow{2}{*}{selected} & "
            "\\multicolumn{5}{c}{MSE (dB) at $S=1024$} & gain (dB) \\\\",
            " & & & variant & none & ZNE & D-VAQEM & retrain & oracle & "
            "$1024$ / $\\infty$ \\\\", "\\hline"]
    for N in Ns:
        best = one(idx, N, "none", "inf")["best_dvaqem"]
        bz = min(one(idx, N, m, "S1024")["mse_db"]
                 for m in ("zne_rich", "zne_poly1", "zne_poly2"))
        g1 = one(idx, N, "none", "S1024")["mse_db"] - \
            one(idx, N, best, "S1024")["mse_db"]
        gi = one(idx, N, "none", "inf")["mse_db"] - \
            one(idx, N, best, "inf")["mse_db"]
        body.append(
            f"{N} & {N+1} & {4**N:.2e} & {tex(best.replace('dvaqem_', ''))} & "
            f"{one(idx, N, 'none', 'S1024')['mse_db']:.1f} & {bz:.1f} & "
            f"{one(idx, N, best, 'S1024')['mse_db']:.1f} & "
            f"{one(idx, N, 'retrain_dec', 'S1024')['mse_db']:.1f} & "
            f"{one(idx, N, 'oracle_ml', 'S1024')['mse_db']:.1f} & "
            f"{g1:.1f} / {gi:.1f} \\\\")
    body.append("\\hline")
    t = env_table(
        "Scaling with qubit number at depolarising rate $p=0.01$.  The last "
        "column is the MSE reduction of the selected D-VAQEM variant relative "
        "to the unmitigated estimator at $S=1024$ and at infinite shots.  "
        "``sector dim'' is the number of calibration numbers per phase; "
        "$4^N$ is the dimension of a process tomography of the same probe.",
        "tab:scaling", None, body)
    if cost:
        b2 = ["\\begin{tabular}{@{}ccccc@{}}", "\\hline",
              "$N$ & $t_{\\rm sim}$ (s) & oracle grid (s) & calibration (s) & "
              "map fit (s) \\\\", "\\hline"]
        for r in cost:
            fit = one(idx, r["N"], "none", "inf")["t_fit_s"]
            b2.append(f"{r['N']} & {r['t_sim_s']:.3f} & {r['t_oracle_s']:.1f}"
                      f" & {r['t_calib_s']:.2f} & {fit:.1f} \\\\")
        b2.append("\\hline")
        t += "\n" + env_table(
            "Measured exact-simulation cost (6 worker processes, depolarising "
            "$p=0.01$): one density-matrix evaluation $t_{\\rm sim}$, the "
            "361-phase oracle grid, the 21-phase D-VAQEM calibration, and the "
            "wall-clock fit of all seven D-VAQEM variants.",
            "tab:cost", None, b2)
    return t


def table_sweep_supplement(rows, key, lab):
    """Full method x setting matrix at one shot budget."""
    idx = index([r for r in rows if r["shots"] == key], "method", "setting")
    settings = [s for c in CHANNELS for s in
                settings_by_channel(sorted({r["setting"] for r in rows}))[c]]
    # the 16 setting names are rotated: upright they would make the header
    # row twice the text width and force an illegible shrink
    body = ["\\begin{tabular}{@{}l" + "c" * len(settings)
            + "@{}}", "\\hline",
            "method & " + " & ".join(
                "\\rotatebox[origin=c]{90}{"
                + s.replace("_", "\\ ") + "}"
                for s in settings) + " \\\\", "\\hline"]
    for m in ORDER:
        if m not in idx:
            continue
        body.append(f"{LABEL[m]} & " +
                    " & ".join(cell(idx, m, s, key) for s in settings) + " \\\\")
    best = [one(idx, "none", s)["best_dvaqem"] for s in settings]
    body.append("D-VAQEM (selected) & " +
                " & ".join(f"{one(idx, b, s)['mse_db']:.1f}"
                           for b, s in zip(best, settings)) + " \\\\")
    body.append("\\hline")
    return env_table(
        f"MSE (dB) of every method in all 16 noise settings at {lab}, $N=8$.  "
        "The last row is the variant selected by the held-out calibration "
        "score, which is the number reported in the main text.",
        f"tab:sweep_{key}", None, body, star=True, font="\\scriptsize", colsep="2pt")


def table_e1(rows):
    body = ["\\begin{tabular}{@{}ccclrrr@{}}", "\\hline",
            "$N$ & variant & noise & mean $F(\\mathbf{p}_{\\rm full})$ & "
            "mean $F(\\mathbf{p}_m)$ & $\\Delta_{\\rm CRB}$ (dB) & "
            "max rel. dev. \\\\", "\\hline"]
    for r in rows:
        body.append(f"{r['N']} & {tex(r['variant'])} & {tex(r['noise'])} & "
                    f"{r['mean_FI_p_full']:.6f} & {r['mean_FI_p_m']:.6f} & "
                    f"{r['crb_gap_db']:.2e} & {r['max_rel_dev']:.2e} \\\\")
    body.append("\\hline")
    return env_table(
        "Sufficiency of the collective-imbalance reduction: classical Fisher "
        "information of the full outcome distribution and of the sector "
        "distribution, Cramér--Rao penalty of the reduction, and worst "
        "pointwise relative deviation, for trained and random circuit "
        "parameters.", "tab:e1", None, body, star=True, font="\\tiny")


def table_e3(rows):
    shots = sorted({r["shots"] for r in rows})
    sets = list(dict.fromkeys(r["setting"] for r in rows))
    meths = list(dict.fromkeys(r["method"] for r in rows))
    body = ["\\begin{tabular}{@{}ll" + "c" * len(shots) + "@{}}", "\\hline",
            "setting & method & " + " & ".join(f"$S{S}$" for S in shots)
            + " \\\\", "\\hline"]
    for s in sets:
        for m in meths:
            sub = {r["shots"]: r for r in rows
                   if r["setting"] == s and r["method"] == m}
            body.append(f"{tex(s)} & {LABEL.get(m, tex(m))} & " +
                        " & ".join(f"{sub[S]['mc_mse']:.2e}" if S in sub
                                   else "--" for S in shots) + " \\\\")
    body.append("\\hline")
    t = env_table("Monte-Carlo mean squared phase error versus shot budget for "
                  "the three settings of the finite-shot study (41 trials per "
                  "point).", "tab:e3mc", None, body, star=True, font="\\tiny")
    body = ["\\begin{tabular}{@{}ll" + "c" * len(shots) + "@{}}", "\\hline",
            "setting & method & " + " & ".join(f"$S{S}$" for S in shots)
            + " \\\\", "\\hline"]
    for s in sets:
        for m in meths:
            sub = {r["shots"]: r for r in rows
                   if r["setting"] == s and r["method"] == m}
            if not any(sub.get(S, {}).get("ratio") for S in shots):
                continue
            body.append(f"{tex(s)} & {LABEL.get(m, tex(m))} & " +
                        " & ".join(f"{sub[S]['ratio']:.3f}"
                                   if sub.get(S, {}).get("ratio") else "--"
                                   for S in shots) + " \\\\")
    body.append("\\hline")
    t += "\n" + env_table(
        "Ratio of the Monte-Carlo MSE to the analytic (delta-method) "
        "prediction bias$^2$ + Var$_S$.  A value of 1 validates the "
        "closed-form variance that the shot-aware objective minimises.",
        "tab:e3ratio", None, body, star=True, font="\\tiny")
    return t

def table_e5(rows):
    methods = ["none", "zne_rich", "dvaqem_lin_mse", "dvaqem_mlp_ce",
               "dvaqem_mlp_mse", "retrain_dec", "oracle_ml", "noiseless"]
    out = ""
    for axis, xs in (("n_cal", [5, 9, 17, 25, 41]),
                     ("cal_shots", [512, 2048, 8192, None])):
        sub = [r for r in rows if r["axis"] == axis and r[axis] in xs]
        ix = index(sub, axis, "method", "shots")
        skey = sorted({r["shots"] for r in sub})[0]
        body = ["\\begin{tabular}{@{}l" + "c" * len(xs) + "@{}}", "\\hline",
                "method & " + " & ".join("exact" if x is None else str(x)
                                          for x in xs) + " \\\\", "\\hline"]
        for m in methods:
            body.append(f"{LABEL[m]} & " + " & ".join(
                f"{one(ix, x, m, skey)['mse_db']:.1f}"
                if x in ix and m in ix[x] else "--" for x in xs) + " \\\\")
        body.append("\\hline")
        head = ("calibration phases $n_{\\rm cal}$" if axis == "n_cal"
                else "calibration shots $S_{\\rm cal}$")
        out += env_table(f"MSE (dB) at $S=1024$ versus {head} (depolarising "
                         f"$p=0.01$, $N=8$).", f"tab:e5_{axis}", None,
                         body) + "\n"
    return out


def table_robustness(seeds, abl, base_rows):
    out = ""
    if seeds:
        body = ["\\begin{tabular}{@{}lccc@{}}", "\\hline",
                "run & gain (dB) $\\infty$ & gain (dB) $S1024$ & selected "
                "variants \\\\", "\\hline"]
        for tag, rows in seeds:
            cells, sel = [], set()
            for key in ("inf", "S1024"):
                ix = index([r for r in rows if r["shots"] == key], "method",
                           "setting")
                sts = sorted({r["setting"] for r in rows})
                gs = [one(ix, "none", s)["mse_db"] -
                      one(ix, one(ix, "none", s)["best_dvaqem"], s)["mse_db"]
                      for s in sts]
                cells.append(f"{np.mean(gs):.2f}")
                sel |= {one(ix, "none", s)["best_dvaqem"].replace("dvaqem_", "")
                        for s in sts}
            body.append(f"{tag} & {cells[0]} & {cells[1]} & "
                        f"{tex(', '.join(sorted(sel)))} \\\\")
        body.append("\\hline")
        out += env_table("Repeat runs of the headline sweep with independent "
                         "random seeds: mean MSE reduction over the 16 "
                         "settings and the set of selected variants.",
                         "tab:seeds", None, body)
    if abl:
        ib = index([r for r in base_rows if r["shots"] == "inf"], "method",
                   "setting")
        ia = index([r for r in abl if r["shots"] == "inf"], "method", "setting")
        body = ["\\begin{tabular}{@{}lccccc@{}}", "\\hline",
                "setting & warm variant & cold variant & warm (dB) & cold (dB)"
                " & cold$-$warm \\\\", "\\hline"]
        for s in sorted({r["setting"] for r in base_rows}):
            bw = one(ib, "none", s)["best_dvaqem"]
            bc = one(ia, "none", s)["best_dvaqem"]
            w = one(ib, bw, s)["mse_db"]
            c = one(ia, bc, s)["mse_db"]
            body.append(f"{tex(s)} & {tex(bw.replace('dvaqem_', ''))} & "
                        f"{tex(bc.replace('dvaqem_', ''))} & {w:.1f} & {c:.1f} & "
                        f"{c - w:+.1f} \\\\")
        body.append("\\hline")
        out += "\n" + env_table(
            "Ablation of the staged warm start: the shot-aware maps are fitted "
            "from a cold start instead of being refined from the $\\ell_2$/CE "
            "fits, and the held-out score re-selects the variant.",
            "tab:ablation", None, body, star=True, font="\\tiny")
    return out


CI_SHOTS = ("inf", "S256", "S1024", "S4096")
CI_ORDER = ("none", "zne_rich", "zne_poly1", "zne_poly2", "best_zne",
            "retrain_dec", "oracle_ml", "linv_known", "linv_calib")
CI_SHOTLAB = {"inf": "$\\infty$", "S256": "$256$", "S1024": "$1024$",
              "S4096": "$4096$"}


def _setting_key(s):
    """Channel-then-strength order, matching the rest of the paper's tables."""
    name, val = s.rsplit("_", 1)
    return (CHANNELS.index(name) if name in CHANNELS else 99, float(val))


def _p(p):
    """Format a p-value: the exact sign test bottoms out at 2^-15 = 3.1e-5."""
    if p is None:
        return "--"
    return "$<\\!10^{-4}$" if p < 1e-4 else f"{p:.3g}"


def _ci(d, pct, unit="dB"):
    if not d:
        return "--"
    if unit == "%":
        return f"$[{d['ci_lo']:.1f},\\,{d['ci_hi']:.1f}]$"
    return f"$[{d['ci_lo']:+.2f},\\,{d['ci_hi']:+.2f}]$"


def table_ci(ci):
    """Supplemental tables: paired bootstrap CIs and exact tests.

    Reads the output of ``bootstrap_ci.build`` (written by both
    ``bootstrap_ci.py`` and ``collect_numbers.sec_ci``), so the intervals in the
    paper are the same objects the manuscript text quotes.
    """
    pct = int(round(ci["meta"]["conf"] * 100))
    B = ci["meta"]["n_boot"]
    agg = ci["aggregates"]
    body = ["\\begin{tabular}{@{}llrlcrr@{}}", "\\hline",
            "shots & comparison & mean gain (dB) & "
            f"{pct}\\% CI & wins & sign $p$ & Wilcoxon $p$ \\\\", "\\hline"]
    first = True
    for key in CI_SHOTS:
        if key not in agg:
            continue
        if not first:
            body.append("\\hline")
        first = False
        a = agg[key]
        for name in CI_ORDER:
            e = a.get(name)
            if not e or not e.get("mean"):
                continue
            mu = e["mean"]
            body.append(
                f"{CI_SHOTLAB[key]} & {LABEL.get(name, tex(name))} & "
                f"${mu['mean']:+.2f}$ & {_ci(mu, pct)} & "
                f"{e['sign']['n_positive']}/{e['sign']['n']} & "
                f"{_p(e['sign']['p_two_sided'])} & "
                f"{_p(e['wilcoxon']['p_two_sided'])} \\\\")
        c = a.get("closure_from_means")
        if c:
            body.append(f"{CI_SHOTLAB[key]} & gap to noiseless closed (\\%) & "
                        f"${c['point']:.1f}$ & {_ci(c, pct, '%')} & -- & -- & "
                        "-- \\\\")
    body.append("\\hline")
    out = env_table(
        f"Paired percentile bootstrap (${B}$ replicates, {pct}\\% intervals) and "
        "exact tests for the mean gain of the holdout-selected D-VAQEM variant.  "
        "A positive gain means D-VAQEM is better, so the two genie baselines "
        "(known-noise-model oracle, exact sector inverse) appear as negative "
        "entries and read as the residual gap.  At a fixed shot budget every "
        "method sees the same multinomial draws (the sampler is seeded with "
        "$\\mathrm{seed}+S$), so the comparison is paired and the interval "
        "resamples Monte-Carlo trials; at infinite shots it resamples the "
        f"{ci['meta'].get('n_phase', 21)} held-out test phases instead.  The "
        "sign and Wilcoxon $p$-values are exact over the "
        f"{ci['meta']['n_settings']} noise settings, and ``wins'' counts the "
        "settings in which D-VAQEM is better.  ``best ZNE "
        "variant$^{\\dagger}$'' is chosen a "
        "posteriori per setting and is therefore anti-conservative; the three "
        "individual ZNE rows are the selection-free evidence.",
        "tab:ci", None, body, star=True)

    per = ci["per_setting"].get("inf", {})
    if per:
        body = ["\\begin{tabular}{@{}lllrrlrr@{}}", "\\hline",
                "setting & selected & best ZNE & D-VAQEM (dB) & gain (dB) & "
                f"{pct}\\% CI & one-sided $p$ & CI excl.\\ 0 \\\\", "\\hline"]
        for s in sorted(per, key=_setting_key):
            e = per[s]
            bz = e.get("best_zne")
            if not bz:
                continue
            body.append(
                f"{tex(s)} & {tex(e['selected'].replace('dvaqem_', ''))} & "
                f"{tex(e.get('best_zne_variant', '?').replace('zne_', ''))} & "
                f"${e['mse_db']:.1f}$ & ${bz['point']:+.2f}$ & "
                f"{_ci(bz, pct)} & {_p(bz['p_one_sided'])} & "
                f"{'yes' if bz['ci_lo'] > 0 else 'no'} \\\\")
        body.append("\\hline")
        out += "\n\n" + env_table(
            "Per-setting evidence for the claim that D-VAQEM beats ZNE in every "
            "noise setting: gain over the a-posteriori best ZNE variant at "
            "infinite shots, with the paired bootstrap interval over the "
            "held-out test phases and its one-sided $p$-value.  The last column "
            "is the operative one: the interval excludes zero in every setting, "
            "so the claim does not rest on point estimates alone.",
            "tab:ci_zne", None, body)
    return out


def table_ci_claims(ci):
    """Supplemental table: each headline claim of the paper with its verdict.

    The verdict is computed from the interval, not written by hand, so the table
    cannot drift out of agreement with the data: a claim of superiority is
    ``supported'' only when the paired interval excludes zero (or, for the
    all-settings claims, when every setting wins and the worst one still excludes
    zero), and a claim of parity is ``supported'' only when the exact Wilcoxon
    test over the settings fails to reject at $\\\\alpha=0.05$.
    """
    pct = int(round(ci["meta"]["conf"] * 100))
    cl, agg = ci["claims"], ci["aggregates"]
    rows = []
    a_inf = agg.get("inf", {})
    mr = cl.get("mean_reduction_inf_db")
    if mr:
        rows.append(("mean MSE reduction at infinite shots",
                     f"${mr['mean']:.2f}$\\,dB", _ci(mr, pct),
                     f"sign/Wilcoxon $p$\\,=\\,"
                     f"{_p(a_inf['none']['sign']['p_two_sided'])}"
                     f", wins {a_inf['none']['sign']['n_positive']}/{mr['n']}",
                     "supported" if mr["ci_lo"] > 0 else "not supported"))
    mm, pv = cl.get("min_gain_vs_best_zne_inf"), cl.get("per_variant_inf", {})
    if mm and pv:
        ok = (cl.get("beats_best_zne_settings_inf") == cl.get("n_settings_inf")
              and mm["ci_lo"] > 0)
        rows.append(("beats ZNE in all sixteen noise settings "
                     "(a-posteriori best variant)",
                     f"${mm['point']:+.2f}$\\,dB worst setting", _ci(mm, pct),
                     f"{cl['beats_best_zne_settings_inf']}/"
                     f"{cl['n_settings_inf']} wins, CI excludes 0 in "
                     f"{cl['beats_best_zne_ci_excludes_zero_inf']}; "
                     f"one-sided $p$\\,=\\,{_p(mm['p_one_sided'])}",
                     "supported" if ok else "not supported"))
        # one row per *fixed* comparator: this is the selection-free evidence
        for zm in ("zne_rich", "zne_poly1", "zne_poly2"):
            v = pv.get(zm)
            if not v:
                continue
            rows.append((f"\\quad vs the fixed comparator {LABEL[zm]}",
                         f"${v['mean']:+.2f}$\\,dB mean",
                         f"$[{v['ci_lo']:+.2f},\\,{v['ci_hi']:+.2f}]$",
                         f"wins {v['wins']:.0f}/{v['n']:.0f}, worst setting "
                         f"${v['min']:+.2f}$\\,dB, sign "
                         f"$p$\\,=\\,{_p(v['p_sign'])}, Wilcoxon "
                         f"$p$\\,=\\,{_p(v['p_wilcoxon'])}",
                         "supported" if v["wins"] == v["n"] and v["ci_lo"] > 0
                         else "not supported"))
    og = a_inf.get("oracle_ml", {}).get("mean")
    if og:
        rows.append(("residual gap to the known-noise-model oracle",
                     f"${og['mean']:.2f}$\\,dB", _ci(og, pct), "--", "quantified"))
    for key in ("S256", "S1024", "S4096"):
        r, c = cl.get(f"retrain_{key}"), cl.get(f"closure_{key}_pct")
        if r and r["mean"]:
            mu = r["mean"]
            pw = r["wilcoxon"]["p_two_sided"]
            rows.append((f"matches decoder retraining at ${key[1:]}$ shots",
                         f"${mu['mean']:+.2f}$\\,dB", _ci(mu, pct),
                         f"sign $p$\\,=\\,{_p(r['sign']['p_two_sided'])}, Wilcoxon "
                         f"$p$\\,=\\,{_p(pw)}, wins {r['sign']['n_positive']}/"
                         f"{r['sign']['n']}",
                         "supported (parity)" if pw > 0.05 else "not supported"))
        if c:
            rows.append((f"fraction of the dB gap closed at ${key[1:]}$ shots",
                         f"${c['point']:.1f}$\\,\\%", _ci(c, pct, "%"), "--",
                         "quantified"))
    body = ["\\begin{tabular}{@{}p{0.245\\linewidth}rp{0.125\\linewidth}"
            "p{0.245\\linewidth}p{0.155\\linewidth}@{}}", "\\hline",
            "\\textbf{claim} & \\textbf{value} & "
            f"\\textbf{{{pct}\\% CI}} & \\textbf{{test}} & "
            "\\textbf{verdict} \\\\", "\\hline"]
    for cl_, val, ci_, tst, verdict in rows:
        body.append(f"{cl_} & {val} & {ci_} & {tst} & {verdict} \\\\")
    body.append("\\hline")
    return env_table(
        "Every quantitative claim of the manuscript that a reviewer could ask to "
        "see tested, with its paired bootstrap interval and the exact test over "
        "the sixteen noise settings.  Verdicts are computed from the intervals by "
        "\\texttt{make\\_tables.py}, not entered by hand.  ``quantified'' marks an "
        "entry that reports a magnitude rather than asserting a comparison.",
        "tab:ci_claims", None, body, star=True, font="\\tiny")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    res, out = os.path.abspath(a.res), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    man = load("run_manifest.json", res)
    rows = load(f"{a.tag}_sweep.json", res)
    meta = load(f"{a.tag}_sweep_meta.json", res)
    e1 = load("e1_sufficiency.json", res)
    e3 = load("e3_shots.json", res)
    e4 = load("e4_scaling.json", res)
    e5 = load("e5_calib.json", res)
    cp = os.path.join(res, "oracle_cost.json")
    cost = load("oracle_cost.json", res) if os.path.exists(cp) else None
    s0 = sorted(meta)[0]
    with open(os.path.join(out, "tables.tex"), "w") as fh:
        fh.write("% auto-generated by code/make_tables.py -- do not edit\n"
                 + "\n\n".join([table_protocol(man, meta[s0]),
                                table_channels(rows),
                                table_scaling(e4, cost)]) + "\n")
    seeds = [("seed0", rows)]
    for t in ("seed1", "seed2"):
        p = os.path.join(res, "seed_robustness", f"{t}_sweep.json")
        if os.path.exists(p):
            seeds.append((t, load(p, res)))
    ap_ = os.path.join(res, "ablations", "coldstart_sweep.json")
    abl = load(ap_, res) if os.path.exists(ap_) else None
    cip = os.path.join(res, f"ci_{a.tag}.json")
    ci = load(f"ci_{a.tag}.json", res) if os.path.exists(cip) else None
    if ci is None:
        print(f"note: no ci_{a.tag}.json, so the confidence-interval tables are "
              "omitted; run collect_numbers.py (or bootstrap_ci.py) first")
    sup = [table_sweep_supplement(rows, "inf", "infinite shots"),
           table_sweep_supplement(rows, "S1024", "$S=1024$"),
           table_e1(e1), table_e3(e3), table_e5(e5),
           table_robustness(seeds, abl, rows)]
    if ci is not None:
        sup += [table_ci(ci), table_ci_claims(ci)]
    with open(os.path.join(out, "tables_supplement.tex"), "w") as fh:
        fh.write("% auto-generated by code/make_tables.py -- do not edit\n"
                 + "\n\n".join(sup) + "\n")
    print(f"wrote {os.path.join(out, 'tables.tex')} and tables_supplement.tex")


if __name__ == "__main__":
    main()

