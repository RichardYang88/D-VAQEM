"""
collect_numbers.py -- digest every headline number the D-VAQEM manuscript cites.

Reads the result files written by ``run_experiments.py`` (default tag
``final``, i.e. the validated headline run) and produces

    results/paper_numbers.md    human-readable digest, section by section
    results/paper_numbers.json  the same numbers as scalars, for scripts

Every figure caption and every number in the manuscript text is taken from
this digest, so the paper can be re-checked against the data with one command:

    python collect_numbers.py [--tag final] [--seeds seed1,seed2]

Usage:  python collect_numbers.py [--tag final] [--res ../results]
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT_KEYS = ("inf", "S256", "S1024", "S4096")
# Presentation order of the method suite (same grouping as the paper tables).
GROUPS = (
    ("reference", ("noiseless",)),
    ("unmitigated", ("none",)),
    ("ZNE", ("zne_rich", "zne_poly1", "zne_poly2")),
    ("D-VAQEM", ("dvaqem_lin_l2", "dvaqem_lin_ce", "dvaqem_lin_mse",
                 "dvaqem_mlp_l2", "dvaqem_mlp_ce", "dvaqem_mlp_fisher",
                 "dvaqem_mlp_mse")),
    ("retrain", ("retrain_dec",)),
    ("oracle/known", ("linv_known", "oracle_ml", "oracle_l2")),
)
ORDER = tuple(m for _, ms in GROUPS for m in ms)
NOISE_KIND = {"depol": "depolarising", "deph": "dephasing",
              "ampdamp": "amplitude damping", "readout": "readout bit-flip"}


def load(path):
    with open(path) as fh:
        return json.load(fh)


def index(rows, *keys):
    """rows -> nested dict keyed by the given fields (last level = the row)."""
    out = {}
    for r in rows:
        d = out
        for k in keys[:-1]:
            d = d.setdefault(r[k], {})
        d.setdefault(r[keys[-1]], []).append(r)
    return out


def one(idx, *key):
    """Unique row of an :func:`index` result (asserts uniqueness)."""
    v = idx
    for k in key:
        v = v[k]
    assert len(v) == 1, (key, len(v))
    return v[0]


def db(x):
    return 10.0 * np.log10(max(float(x), 1e-30))


def decoder_params(N, h1=128, h2=64, out=2):
    """Parameter count of the frozen VQ-CNNI decoder (N+1)->h1->h2->out."""
    return (N + 1) * h1 + h1 + h1 * h2 + h2 + h2 * out + out



def setting_sort(settings):
    """Noise settings grouped by channel, ascending in strength."""
    def k(s):
        name, val = s.rsplit("_", 1)
        kind = {"depol": 0, "deph": 1, "ampdamp": 2, "readout": 3}.get(name, 9)
        return (kind, float(val))
    return sorted(settings, key=k)


class Report:
    """Accumulates markdown lines and a flat dict of scalars."""

    def __init__(self):
        self.lines = []
        self.scalars = {}

    def h(self, level, text):
        self.lines.append("#" * level + " " + text)
        self.lines.append("")

    def p(self, text=""):
        self.lines.append(text)
        self.lines.append("")

    def table(self, header, rows):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def set(self, key, value):
        self.scalars[key] = value

    def text(self):
        return "\n".join(self.lines)



# ======================================================================
# E1  exactness of the collective-imbalance reduction
# ======================================================================
def sec_e1(rep, res, tag):
    rows = load(os.path.join(res, "e1_sufficiency.json"))
    rep.h(2, "E1 -- sufficiency of the m-reduction: FI(p_m) vs FI(p_full)")
    idx = index(rows, "N", "variant", "noise")
    settings = list(dict.fromkeys(r["noise"] for r in rows))
    rep.p(f"Ns = {sorted({r['N'] for r in rows})}, variants = "
          f"{sorted({r['variant'] for r in rows})}, n_phi = {rows[0]['n_phi']}, "
          f"rows = {len(rows)}")
    rep.table(["N", "variant", "noise", "mean FI(p_full)", "mean FI(p_m)",
               "CRB gap (dB)", "max rel dev", "max dev / mean FI"],
              [[r["N"], r["variant"], r["noise"], f"{r['mean_FI_p_full']:.6f}",
                f"{r['mean_FI_p_m']:.6f}", f"{r['crb_gap_db']:.3e}",
                f"{r['max_rel_dev']:.3e}", f"{r['max_dev_norm']:.3e}"]
               for r in rows])
    # headline aggregates
    noiseless = [r for r in rows if r["noise"] == "noiseless"]
    noisy = [r for r in rows if r["noise"] != "noiseless"]
    rep.p("**Headline aggregates**")
    rep.p(f"- noiseless rows ({len(noiseless)}): worst |CRB gap| = "
          f"{max(abs(r['crb_gap_db']) for r in noiseless):.3e} dB, worst "
          f"max_rel_dev = {max(r['max_rel_dev'] for r in noiseless):.3e}")
    rep.p(f"- noisy rows ({len(noisy)}): worst |CRB gap| = "
          f"{max(abs(r['crb_gap_db']) for r in noisy):.3e} dB, worst "
          f"max_rel_dev = {max(r['max_rel_dev'] for r in noisy):.3e}, worst "
          f"normalised dev = {max(r['max_dev_norm'] for r in noisy):.3e}")
    rep.set("e1_worst_crb_gap_db_noiseless",
            max(abs(r["crb_gap_db"]) for r in noiseless))
    rep.set("e1_worst_crb_gap_db_noisy",
            max(abs(r["crb_gap_db"]) for r in noisy))
    rep.set("e1_worst_rel_dev_noisy", max(r["max_rel_dev"] for r in noisy))
    rep.set("e1_worst_norm_dev_noisy", max(r["max_dev_norm"] for r in noisy))
    rep.p("worst CRB gap per noise setting (over N and trained/random):")
    rep.table(["noise", "worst |CRB gap| dB", "worst max rel dev",
               "mean FI(p_m) range"],
              [[s, f"{max(abs(r['crb_gap_db']) for r in rows if r['noise']==s):.3e}",
                f"{max(r['max_rel_dev'] for r in rows if r['noise']==s):.3e}",
                f"{min(r['mean_FI_p_m'] for r in rows if r['noise']==s):.3f}"
                f"--{max(r['mean_FI_p_m'] for r in rows if r['noise']==s):.3f}"]
               for s in settings])
    # grid-convergence control
    conv_p = os.path.join(res, "e1_fi_convergence.json")
    if os.path.exists(conv_p):
        conv = load(conv_p)
        rep.h(3, "E1b -- finite-difference grid convergence")
        if isinstance(conv, list) and conv:
            rep.table(sorted(conv[0].keys()),
                      [[f"{v:.6g}" if isinstance(v, float) else v for v in
                        [r[k] for k in sorted(conv[0].keys())]] for r in conv])
        else:
            rep.p("```json\n" + json.dumps(conv, indent=1)[:3000] + "\n```")
    return idx


# ======================================================================
# E2  headline sweep: method x noise channel x strength x shot budget
# ======================================================================
def _mse_db_table(rep, rows, key, title):
    sel = [r for r in rows if r["shots"] == key]
    idx = index(sel, "method", "setting")
    settings = setting_sort({r["setting"] for r in sel})
    rep.h(3, title)
    header = ["method (MSE dB)"] + [s.replace("_", " ") for s in settings] + ["mean"]
    body = []
    for m in ORDER:
        if m not in idx:
            continue
        vals = [(one(idx, m, s)["mse_db"] if s in idx[m] else np.nan)
                for s in settings]
        fin = [v for v in vals if np.isfinite(v)]
        body.append([m] + [f"{v:.2f}" if np.isfinite(v) else "--" for v in vals] +
                    [f"{np.mean(fin):.2f}" if fin else "--"])
    rep.table(header, body)
    return idx, settings


def sec_e2(rep, res, tag):
    rows = load(os.path.join(res, f"{tag}_sweep.json"))
    meta = load(os.path.join(res, f"{tag}_sweep_meta.json"))
    rep.h(2, f"E2 -- headline sweep ({tag}): methods x noise settings")
    rep.p(f"N = {rows[0]['N']}, n_cal = {rows[0]['n_cal']}, "
          f"n_tst = {rows[0]['n_tst']}, MC trials = {rows[0].get('n_trials','n/a')}, "
          f"rows = {len(rows)}")
    present = [k for k in SHOT_KEYS if any(r["shots"] == k for r in rows)]
    idx_inf, settings = _mse_db_table(rep, rows, "inf",
                                      "Infinite-shot MSE (dB, lower better)")
    for key in present:
        if key != "inf":
            _mse_db_table(rep, rows, key, f"Finite-shot MSE (dB) at {key[1:]}")
    # reference lines used by every figure
    rep.h(3, "Reference levels (noiseless decoder and its shot-noise floor)")
    body = []
    for key in present:
        sel = index([r for r in rows if r["shots"] == key], "method", "setting")
        nl = one(sel, "noiseless", settings[0])["mse_db"]
        nu = float(np.mean([one(sel, "none", s)["mse_db"] for s in settings]))
        bd = float(np.mean([one(sel, one(idx_inf, "none", s)["best_dvaqem"], s)
                            ["mse_db"] for s in settings]))
        body.append([key, f"{nl:.2f}", f"{nu:.2f}", f"{bd:.2f}",
                     f"{nu - bd:.2f}", f"{nl - bd:.2f}"])
    rep.table(["shot budget", "noiseless (dB)", "unmitigated mean (dB)",
               "D-VAQEM mean (dB)", "gain (dB)", "distance to noiseless (dB)"],
              body)
    for r in body:
        rep.set(f"e2_noiseless_db_{r[0]}", float(r[1]))
        rep.set(f"e2_dvaqem_mean_db_{r[0]}", float(r[3]))
        rep.set(f"e2_none_mean_db_{r[0]}", float(r[2]))
        rep.set(f"e2_dist_to_noiseless_{r[0]}_db", float(r[5]))


    rep.h(3, "Per-setting gain of the holdout-selected D-VAQEM variant")
    hdr = ["setting", "selected", "none", "best ZNE", "retrain", "oracle_ml",
           "D-VAQEM", "gain vs none", "gain vs ZNE", "gap to oracle"]
    body, gains_none, gains_zne, gap_or, wins_zne, wins_rt = [], [], [], [], 0, 0
    for s in settings:
        best = one(idx_inf, "none", s)["best_dvaqem"]

        def v(m, s=s):
            return (one(idx_inf, m, s)["mse_db"]
                    if m in idx_inf and s in idx_inf[m] else np.nan)

        bz = min(v(m) for m in ("zne_rich", "zne_poly1", "zne_poly2")
                 if np.isfinite(v(m)))
        d = v(best)
        body.append([s, best.replace("dvaqem_", ""), f"{v('none'):.2f}",
                     f"{bz:.2f}", f"{v('retrain_dec'):.2f}",
                     f"{v('oracle_ml'):.2f}", f"{d:.2f}",
                     f"{v('none') - d:.2f}", f"{bz - d:.2f}",
                     f"{d - v('oracle_ml'):.2f}"])
        gains_none.append(v("none") - d)
        gains_zne.append(bz - d)
        gap_or.append(d - v("oracle_ml"))
        wins_zne += int(bz - d > 0)
        wins_rt += int(v("retrain_dec") - d > 0)
    rep.table(hdr, body)
    rep.p("**Aggregates over the settings (infinite shots)**")
    rep.p(f"- gain vs unmitigated: mean {np.mean(gains_none):.2f} dB, median "
          f"{np.median(gains_none):.2f} dB, min {np.min(gains_none):.2f} dB, "
          f"max {np.max(gains_none):.2f} dB")
    rep.p(f"- gain vs best ZNE variant: mean {np.mean(gains_zne):.2f} dB, min "
          f"{np.min(gains_zne):.2f} dB; D-VAQEM wins in {wins_zne}/"
          f"{len(settings)} settings")
    rep.p(f"- beats noise-aware decoder retraining in {wins_rt}/"
          f"{len(settings)} settings")
    rep.p(f"- residual gap to the known-noise-model oracle: mean "
          f"{np.mean(gap_or):.2f} dB, max {np.max(gap_or):.2f} dB")
    for k, val in (("e2_gain_vs_none_mean_db", np.mean(gains_none)),
                   ("e2_gain_vs_none_median_db", np.median(gains_none)),
                   ("e2_gain_vs_none_min_db", np.min(gains_none)),
                   ("e2_gain_vs_none_max_db", np.max(gains_none)),
                   ("e2_gain_vs_zne_mean_db", np.mean(gains_zne)),
                   ("e2_wins_vs_zne", wins_zne),
                   ("e2_wins_vs_retrain", wins_rt),
                   ("e2_gap_to_oracle_mean_db", np.mean(gap_or)),
                   ("e2_gap_to_oracle_max_db", np.max(gap_or))):
        rep.set(k, float(val))
    return idx_inf, settings, meta, rows


def sec_e2b(rep, idx_inf, settings, meta, rows):
    """Finite-shot gains, variant selection, FI recovery, CRB, cost."""
    rep.h(3, "Mean gain of the selected variant vs unmitigated, by shot budget")
    body = []
    for key in SHOT_KEYS:
        sel = index([r for r in rows if r["shots"] == key], "method", "setting")
        gs = [one(sel, "none", s)["mse_db"] -
              one(sel, one(idx_inf, "none", s)["best_dvaqem"], s)["mse_db"]
              for s in settings if one(idx_inf, "none", s)["best_dvaqem"] in sel]
        body.append([key, f"{np.mean(gs):.2f}", f"{np.min(gs):.2f}",
                     f"{np.max(gs):.2f}", len(gs)])
        rep.set(f"e2_gain_mean_db_{key}", float(np.mean(gs)))
        rep.set(f"e2_gain_min_db_{key}", float(np.min(gs)))
    rep.table(["shot budget", "mean gain (dB)", "min", "max", "n"], body)

    rep.h(3, "Which variant the held-out calibration score selects")
    counts = {}
    for s in settings:
        b = one(idx_inf, "none", s)["best_dvaqem"].replace("dvaqem_", "")
        counts[b] = counts.get(b, 0) + 1
    rep.table(["variant", "times selected"],
              sorted(counts.items(), key=lambda kv: -kv[1]))
    rep.set("e2_variant_counts", counts)
    # does the holdout choice agree with the (never used) test-optimal one?
    agree, n_cmp = 0, 0
    for s in settings:
        cand = [m for m in ORDER if m.startswith("dvaqem_") and m in idx_inf]
        t_best = min(cand, key=lambda m: one(idx_inf, m, s)["mse_db"])
        n_cmp += 1
        agree += int(t_best == one(idx_inf, "none", s)["best_dvaqem"])
    rep.p(f"- holdout selection coincides with the test-set-optimal variant in "
          f"{agree}/{n_cmp} settings (selection never uses the test set)")
    rep.set("e2_holdout_test_agreement", agree / max(n_cmp, 1))

    rep.h(3, "Fisher information: clean / noisy / mitigated (mean over phases)")
    body, rec_n, rec_m, clip = [], [], [], []
    for s in settings:
        m = meta[s]
        fi = {k: float(np.mean(v)) for k, v in m["fi"].items()}
        diag = m.get("fi_mitigated_diag", {})
        body.append([s, f"{fi['clean']:.4f}", f"{fi['noisy']:.4f}",
                     f"{fi['mitigated']:.4f}", f"{fi['noisy']/fi['clean']:.4f}",
                     f"{fi['mitigated']/fi['clean']:.4f}",
                     f"{diag.get('clipped_mass_mean', float('nan')):.2e}",
                     f"{diag.get('zero_bin_frac', float('nan')):.2e}"])
        rec_n.append(fi["noisy"] / fi["clean"])
        rec_m.append(fi["mitigated"] / fi["clean"])
        clip.append(diag.get("clipped_mass_max", np.nan))
    rep.table(["setting", "FI clean", "FI noisy", "FI mitigated", "noisy/clean",
               "mitig/clean", "clipped mass", "zero bins"], body)
    rep.p(f"- noisy FI / clean FI: {np.min(rec_n):.3f}--{np.max(rec_n):.3f} "
          f"(mean {np.mean(rec_n):.3f})")
    rep.p(f"- mitigated FI / clean FI: {np.min(rec_m):.3f}--{np.max(rec_m):.3f} "
          f"(mean {np.mean(rec_m):.3f})")
    rep.p(f"- largest clipped negative mass: {np.nanmax(clip):.3e}")
    rep.set("e2_fi_noisy_over_clean_mean", float(np.mean(rec_n)))
    rep.set("e2_fi_mit_over_clean_mean", float(np.mean(rec_m)))
    rep.set("e2_fi_mit_over_clean_min", float(np.min(rec_m)))
    rep.set("e2_fi_mit_over_clean_max", float(np.max(rec_m)))
    rep.set("e2_clipped_mass_max", float(np.nanmax(clip)))

    rep.h(3, "Distance to the noisy Cramer-Rao bound at finite shots")
    body = []
    for key, S in (("S1024", 1024), ("S4096", 4096)):
        sel = index([r for r in rows if r["shots"] == key], "method", "setting")
        gaps = [one(sel, one(idx_inf, "none", s)["best_dvaqem"], s)["mse_db"] -
                db(meta[s]["crb"]["mse_at_S"][str(S)])
                for s in settings
                if one(idx_inf, "none", s)["best_dvaqem"] in sel
                and "crb" in meta[s]]
        body.append([key, f"{np.mean(gaps):.2f}", f"{np.min(gaps):.2f}",
                     f"{np.max(gaps):.2f}"])
        rep.set(f"e2_gap_to_crb_{key}_mean_db", float(np.mean(gaps)))
    rep.table(["shot budget", "mean MSE-CRB (dB)", "min", "max"], body)

    rep.h(3, "Classical cost per setting")
    fit = [m.get("_fit_time_s", np.nan) for m in meta.values()]
    orac = [m["oracle"]["sim_time_s"] for m in meta.values()]
    rep.p(f"- D-VAQEM: all 7 variants fitted in {np.nanmin(fit):.1f}--"
          f"{np.nanmax(fit):.1f} s (mean {np.nanmean(fit):.1f} s)")
    rep.p(f"- oracle: {min(orac):.0f}--{max(orac):.0f} s (mean "
          f"{np.mean(orac):.0f} s) of exact density-matrix simulation on "
          f"{meta[settings[0]]['oracle']['n_grid']} phases "
          f"(step {meta[settings[0]]['oracle']['grid_step']:.4f} rad)")
    rep.set("e2_fit_time_mean_s", float(np.nanmean(fit)))
    rep.set("e2_oracle_time_mean_s", float(np.mean(orac)))
    fe = {s: meta[s].get("linv_known", {}).get("f_eff") for s in settings}
    rep.p("- exact sector-inverse baseline exists only for flip-equivalent "
          "channels: " + ", ".join(f"{s}={v:.4f}" for s, v in fe.items()
                                   if v is not None))

    rep.h(3, "Trainable parameters and the holdout selection yardstick")
    m0 = meta[settings[0]]
    npar = {k.replace("dvaqem_", ""): v["n_params"]
            for k, v in m0.items() if k.startswith("dvaqem_")}
    rep.p("- map parameters: " + ", ".join(f"{k}={v}" for k, v in npar.items()) +
          f"; frozen decoder = {decoder_params(int(rows[0]['N']))} parameters "
          f"(N={rows[0]['N']}, (N+1)->128->64->2), left untouched by D-VAQEM")
    rep.set("n_params", npar)
    rep.set("n_params_decoder", decoder_params(int(rows[0]["N"])))
    body = []
    for s in settings:
        m = meta[s]
        row = [s, m["best_dvaqem"].replace("dvaqem_", "")]
        for v in ("lin_l2", "lin_ce", "lin_mse", "mlp_l2", "mlp_ce",
                  "mlp_fisher", "mlp_mse"):
            row.append(f"{m['dvaqem_' + v]['holdout_mse']:.2e}")
        body.append(row)
    rep.table(["setting", "selected"] + [f"holdout {v}" for v in
                                         ("lin_l2", "lin_ce", "lin_mse",
                                          "mlp_l2", "mlp_ce", "mlp_fisher",
                                          "mlp_mse")], body)
    rep.p("- calibration split: "
          f"n_fit = {m0['dvaqem_lin_l2'].get('n_fit', 'n/a')}, "
          f"n_val = {m0['dvaqem_lin_l2']['n_val']} held-out phases; "
          f"iterations = {m0['dvaqem_lin_l2']['iters']}; learning rates "
          + ", ".join(f"{v}={m0['dvaqem_' + v]['lr']:.3g}"
                      for v in ("lin_l2", "lin_ce", "lin_mse", "mlp_fisher")))
    zne = {k: v for k, v in m0.items() if k.startswith("zne")}
    for k, v in sorted(zne.items()):
        c = np.asarray(v["coeffs"], dtype=float)
        rep.p(f"- {k}: folds {v['folds']}, coefficients "
              f"{np.array2string(c, precision=3)}, sum_f c_f^2 = "
              f"{float((c ** 2).sum()):.3f} (shot-noise amplification)")
        rep.set(f"e2_{k}_var_amplification", float((c ** 2).sum()))



# ======================================================================
# E3  finite shots: bias/variance split and delta-method validation
# ======================================================================
def sec_e3(rep, res, tag):
    rows = load(os.path.join(res, "e3_shots.json"))
    rep.h(2, "E3 -- finite-shot scaling and validation of the delta method")
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    shots = sorted({r["shots"] for r in rows})
    rep.p(f"N = {rows[0]['N']}, settings = {settings}, methods = {methods}, "
          f"S = {shots}, MC trials = {rows[0].get('n_trials','n/a')}, "
          f"n_cal = {rows[0]['n_cal']}, n_tst = {rows[0]['n_tst']}")
    idx = index(rows, "setting", "method", "shots")
    for s in settings:
        rep.h(3, f"setting {s}")
        rep.table(["MC MSE"] + [f"S={S}" for S in shots],
                  [[m] + [f"{one(idx, s, m, S)['mc_mse']:.3e}" for S in shots]
                   for m in methods])
        rep.table(["bias^2 / var(S)"] + [f"S={S}" for S in shots],
                  [[m] + [(f"{one(idx, s, m, S)['bias2']:.2e} / "
                           f"{one(idx, s, m, S)['var_over_S']:.2e}"
                           if one(idx, s, m, S).get("var_over_S") is not None
                           else "--") for S in shots]
                   for m in methods if one(idx, s, m, shots[0]).get("bias2")
                   is not None])
        rep.table(["MC / analytic"] + [f"S={S}" for S in shots],
                  [[m] + [(f"{one(idx, s, m, S)['ratio']:.3f}"
                           if one(idx, s, m, S).get("ratio") else "--")
                          for S in shots]
                   for m in methods])
    ratios = [r["ratio"] for r in rows if r.get("ratio")]
    rep.p(f"**delta-method validation**: {len(ratios)} (setting, method, S) "
          f"points, MC/analytic ratio in [{min(ratios):.3f}, {max(ratios):.3f}], "
          f"mean {np.mean(ratios):.3f}, median {np.median(ratios):.3f}")
    rep.set("e3_ratio_min", float(min(ratios)))
    rep.set("e3_ratio_max", float(max(ratios)))
    rep.set("e3_ratio_mean", float(np.mean(ratios)))
    rep.set("e3_ratio_n_points", len(ratios))
    worst = max((r for r in rows if r.get("ratio")),
                key=lambda r: abs(r["ratio"] - 1))
    rep.p(f"- worst point: {worst['setting']} / {worst['method']} / "
          f"S={worst['shots']} ratio={worst['ratio']:.3f}")
    amp = [r for r in rows if r.get("var_amplification") is not None]
    if amp:
        rep.p("- ZNE shot-noise amplification sum_f c_f^2 = " +
              ", ".join(f"S{r['shots']}:{r['var_amplification']:.2f}"
                        for r in amp if r["setting"] == amp[0]["setting"]))
        rep.set("e3_zne_var_amplification_mean",
                float(np.mean([r["var_amplification"] for r in amp])))
    src = [r["mse_warm_start"] for r in rows if r.get("mse_warm_start")]
    if src:
        cnt = {k: src.count(k) for k in dict.fromkeys(src)}
        rep.p(f"- winning initialisation of the shot-aware fit: {cnt}")
        rep.set("e3_warm_start_counts", cnt)
    rep.h(3, "Shot-aware refit vs shot-independent (CE) map")
    body = []
    for s in settings:
        for S in shots:
            a = one(idx, s, "dvaqem_mlp_ce", S)["mc_mse"]
            b = one(idx, s, "dvaqem_mlp_mse", S)["mc_mse"]
            n = one(idx, s, "none", S)["mc_mse"]
            body.append([s, S, f"{n:.3e}", f"{a:.3e}", f"{b:.3e}",
                         f"{10*np.log10(a/b):+.2f}"])
    rep.table(["setting", "S", "none", "mlp_ce", "mlp_mse", "ce-mse (dB)"], body)
    return idx, settings, methods, shots


# ======================================================================
# E4  qubit-number scaling
# ======================================================================
def sec_e4(rep, res, tag):
    rows = load(os.path.join(res, "e4_scaling.json"))
    rep.h(2, "E4 -- scaling with qubit number")
    Ns = sorted({r["N"] for r in rows})
    shots = [k for k in ("inf", "S1024", "S4096")
             if any(r["shots"] == k for r in rows)]
    rep.p(f"Ns = {Ns}, noise = {rows[0]['noise']}, shots = {shots}, "
          f"n_cal = {sorted({r['n_cal'] for r in rows})}, "
          f"n_tst = {sorted({r['n_tst'] for r in rows})}, "
          f"MC trials = {rows[0].get('n_trials','n/a')}")
    idx = index(rows, "N", "method", "shots")
    for key in shots:
        rep.h(3, f"MSE (dB) at {key}")
        rep.table(["method"] + [f"N={N}" for N in Ns],
                  [[m] + [(f"{one(idx, N, m, key)['mse_db']:.2f}"
                           if m in idx[N] and key in idx[N][m] else "--")
                          for N in Ns]
                   for m in ORDER if any(m in idx[N] for N in Ns)])
    rep.h(3, "Selected variant, Fisher information and resource cost per N")
    body = []
    for N in Ns:
        r = one(idx, N, "none", "inf")
        best = r["best_dvaqem"]
        body.append([N, r["dim_sector"], f"{r['dim_full']:.3e}",
                     best.replace("dvaqem_", ""), f"{r['fi_clean']:.3f}",
                     f"{r['fi_noisy']:.3f}", f"{r['fi_mitigated']:.3f}",
                     f"{r['t_fit_s']:.1f}", f"{r['t_data_s']:.1f}",
                     f"{r['t_oracle_s']:.1f}", f"{r['t_retrain_s']:.1f}",
                     r["n_oracle"]])
    rep.table(["N", "sector dim", "tomography dim", "selected", "FI clean",
               "FI noisy", "FI mitigated", "fit s", "data s", "oracle s",
               "retrain s", "oracle grid"], body)
    for N in Ns:
        r = one(idx, N, "none", "inf")
        best = r["best_dvaqem"]
        rep.set(f"e4_N{N}_dim_sector", r["dim_sector"])
        rep.set(f"e4_N{N}_fi_noisy_over_clean", r["fi_noisy"] / r["fi_clean"])
        rep.set(f"e4_N{N}_fi_mit_over_clean", r["fi_mitigated"] / r["fi_clean"])
        rep.set(f"e4_N{N}_fit_s", r["t_fit_s"])
        rep.set(f"e4_N{N}_oracle_s", r["t_oracle_s"])
        rep.set(f"e4_N{N}_selected", best)
        for key in shots:
            if key in idx[N].get("none", {}) and key in idx[N].get(best, {}):
                b = one(idx, N, best, key)["mse_db"]
                rep.set(f"e4_N{N}_{key}_gain_db",
                        float(one(idx, N, "none", key)["mse_db"] - b))
                rep.set(f"e4_N{N}_{key}_best_mse_db", float(b))
    rep.h(3, "Gain of the selected variant vs unmitigated, by N")
    rep.table(["N"] + [f"gain {k} (dB)" for k in shots],
              [[N] + [f"{rep.scalars.get(f'e4_N{N}_{k}_gain_db', float('nan')):.2f}"
                      for k in shots] for N in Ns])
    return idx, Ns, shots


# ======================================================================
# E5  calibration budget
# ======================================================================
def sec_e5(rep, res, tag):
    rows = load(os.path.join(res, "e5_calib.json"))
    rep.h(2, "E5 -- calibration budget")
    rep.p(f"N = {rows[0]['N']}, noise = {rows[0]['noise']}, "
          f"evaluation shots = {sorted({r['shots'] for r in rows})}, "
          f"MC trials = {rows[0].get('n_trials','n/a')}, rows = {len(rows)}")
    for axis, xkey in (("n_cal", "n_cal"), ("cal_shots", "cal_shots")):
        sub = [r for r in rows if r["axis"] == axis]
        if not sub:
            continue
        xs = sorted({r[xkey] for r in sub}, key=lambda v: (v is None, v))
        idx = index(sub, xkey, "method", "shots")
        key = sorted({r["shots"] for r in sub})[0]
        rep.h(3, f"axis {axis} (MSE dB at {key})")
        methods = [m for m in ORDER if m in idx[xs[0]]]
        body = []
        for m in methods:
            body.append([m] + [(f"{one(idx, x, m, key)['mse_db']:.2f}"
                                if x in idx and m in idx[x] else "--")
                               for x in xs])
        rep.table(["method"] + [f"{axis}={x}" for x in xs], body)
        best = [one(idx, x, "none", key)["best_dvaqem"] for x in xs]
        rep.p(f"- selected variant per {axis}: " +
              ", ".join(f"{x}:{b.replace('dvaqem_', '')}"
                        for x, b in zip(xs, best)))
        gn = [one(idx, x, "none", key)["mse_db"] -
              one(idx, x, b, key)["mse_db"] for x, b in zip(xs, best)]
        rep.p(f"- gain vs unmitigated (dB): " +
              ", ".join(f"{x}:{g:+.2f}" for x, g in zip(xs, gn)))
        rep.set(f"e5_{axis}_values", [None if x is None else int(x) for x in xs])
        rep.set(f"e5_{axis}_gain_db", [float(g) for g in gn])
        ft = [one(idx, x, "none", key).get("t_fit_s") for x in xs]
        rt = [one(idx, x, "none", key).get("t_retrain_s") for x in xs]
        if any(v is not None for v in ft):
            rep.p("- fit time (s): " + ", ".join(f"{v:.1f}" for v in ft) +
                  "   decoder-retrain time (s): " +
                  ", ".join(f"{v:.1f}" for v in rt))
    return rows


# ======================================================================
# seed robustness (optional extra runs) and simulator validation
# ======================================================================
def sec_seeds(rep, res, tags):
    """Compare repeat runs of the E2 sweep with different random seeds."""
    found = []
    for t in tags:
        p = os.path.join(res, "seed_robustness", f"{t}_sweep.json")
        if os.path.exists(p):
            found.append((t, load(p)))
    if not found:
        return
    rep.h(2, "Seed robustness of the headline sweep")
    base = load(os.path.join(res, "final_sweep.json"))
    runs = [("seed0", base)] + found
    idxs = [(t, index([r for r in rows if r["shots"] == "inf"],
                      "method", "setting")) for t, rows in runs]
    settings = setting_sort({r["setting"] for r in base})
    # a seed run may still be in progress (results are written incrementally):
    # compare on the settings every run has completed.
    common = set(settings)
    for _, ix in idxs:
        common &= set(ix.get("none", {}))
    settings = [s for s in settings if s in common]
    if not settings:
        rep.p("- no completed settings in common yet (runs in progress)")
        return
    if len(settings) < len({r["setting"] for r in base}):
        rep.p(f"- note: comparing on {len(settings)} settings completed by all "
              f"runs (a seed run may still be in progress)")
    for key in ("inf", "S1024"):
        ixs = [(t, index([r for r in rows if r["shots"] == key],
                         "method", "setting")) for t, rows in runs]
        ixs = [(t, ix) for t, ix in ixs
               if all(s in ix.get("none", {}) for s in settings)]
        if len(ixs) < len(runs):
            continue
        gains, spread, sels = {}, [], {}
        for t, ix in ixs:
            gains[t] = [one(ix, "none", s)["mse_db"] -
                        one(ix, one(ix, "none", s)["best_dvaqem"], s)["mse_db"]
                        for s in settings]
            sels[t] = {one(ix, "none", s)["best_dvaqem"] for s in settings}
        for s in settings:
            vals = [one(ix, one(ix, "none", s)["best_dvaqem"], s)["mse_db"]
                    for _, ix in ixs]
            spread.append(max(vals) - min(vals))
        rep.h(3, f"repeat runs at {key}")
        rep.table(["run", "mean gain (dB)", "min", "max", "selected variants"],
                  [[t, f"{np.mean(g):.2f}", f"{np.min(g):.2f}",
                    f"{np.max(g):.2f}",
                    ",".join(sorted(x.replace("dvaqem_", "") for x in sels[t]))]
                   for (t, _), g in zip(ixs, [gains[t] for t, _ in ixs])])
        rep.p(f"- worst per-setting spread of the selected-variant MSE across "
              f"{len(ixs)} runs: {max(spread):.2f} dB "
              f"(mean {np.mean(spread):.2f} dB)")
        rep.p(f"- the selected variant is identical in all runs for "
              f"{sum(1 for s in settings if len({one(ix, 'none', s)['best_dvaqem'] for _, ix in ixs}) == 1)}"
              f"/{len(settings)} settings")
        rep.set(f"seed_max_spread_db_{key}", float(max(spread)))
        rep.set(f"seed_mean_spread_db_{key}", float(np.mean(spread)))
        rep.set(f"seed_runs_{key}", [t for t, _ in ixs])


def sec_validation(rep, res):
    p = os.path.join(res, "simulator_validation.txt")
    if not os.path.exists(p):
        return
    txt = open(p).read()
    rep.h(2, "Simulator validation (validate_simulator.py)")
    npass = len(re.findall(r"\[PASS\]", txt))
    nfail = len(re.findall(r"\[FAIL\]", txt))
    tail = [l for l in txt.splitlines() if "passed" in l]
    rep.p(f"- {npass} checks passed, {nfail} failed.  {tail[-1] if tail else ''}")
    rep.set("validation_pass", npass)
    rep.set("validation_fail", nfail)
    worst, surrogate = 0.0, 0.0
    for line in txt.splitlines():
        if line.startswith("[PASS]"):
            for m in re.finditer(r"max\|d\w*\|?=\s*([0-9.eE+-]+)", line):
                worst = max(worst, abs(float(m.group(1))))
            for m in re.finditer(r"max rel diff=\s*([0-9.eE+-]+)", line):
                worst = max(worst, abs(float(m.group(1))))
        for m in re.finditer(r"surrogate vs exact Kraus = ([0-9.eE+-]+)", line):
            surrogate = max(surrogate, abs(float(m.group(1))))
    rep.p(f"- largest deviation among the exact cross-checks (including "
          f"PennyLane ``default.mixed`` and the stored VQ-CNNI checkpoints): "
          f"{worst:.3e}")
    rep.p(f"- largest deviation of the *flip-equivalent surrogate* used only by "
          f"the fast training path / the exact-inverse baseline: {surrogate:.3e}")
    rep.set("validation_max_dev", float(worst))
    rep.set("validation_surrogate_max_dev", float(surrogate))
    rep.p("```")
    rep.p(txt.strip()[:6000])
    rep.p("```")


# ======================================================================
# resource accounting and the warm-start ablation
# ======================================================================
def sec_resources(rep, res):
    p = os.path.join(res, "oracle_cost.json")
    if not os.path.exists(p):
        return
    rows = load(p)
    rep.h(2, "Resource cost: D-VAQEM calibration vs known-noise-model oracle")
    rep.p("Measured with `code/bench_oracle_cost.py` (uncached exact "
          "density-matrix simulations, depolarising p = 0.01, "
          f"{rows[0]['workers']} worker processes).")
    rep.table(["N", "sector dim", "full dim", "process-tomography dim",
               "linear map params (sector / full)", "t_sim (s)",
               "oracle grid (s)", "calibration (s)", "oracle/calibration"],
              [[r["N"], r["sector_dim"], f"{r['full_dim']}",
                f"{r['process_tomography_dim']:.3e}",
                f"{r['linear_map_params_sector']} / "
                f"{r['linear_map_params_full']:.3e}",
                f"{r['t_sim_s']:.3f}", f"{r['t_oracle_s']:.1f}",
                f"{r['t_calib_s']:.2f}", f"{r['oracle_over_calib']:.1f}x"]
               for r in rows])
    for r in rows:
        rep.set(f"cost_N{r['N']}_t_oracle_s", r["t_oracle_s"])
        rep.set(f"cost_N{r['N']}_t_calib_s", r["t_calib_s"])
        rep.set(f"cost_N{r['N']}_t_sim_s", r["t_sim_s"])


def sec_ablation(rep, res, tag, root="ablations", name="coldstart"):
    p = os.path.join(res, root, f"{name}_sweep.json")
    if not os.path.exists(p):
        return
    base = load(os.path.join(res, f"{tag}_sweep.json"))
    abl = load(p)
    rep.h(2, f"Ablation: {name} (shot-aware maps fitted from a cold start)")
    ib = index([r for r in base if r["shots"] == "inf"], "method", "setting")
    ia = index([r for r in abl if r["shots"] == "inf"], "method", "setting")
    settings = setting_sort(set(ib["none"]) & set(ia["none"]))
    body, d_inf, d_fin = [], [], []
    for key in ("inf", "S256", "S1024"):
        b = index([r for r in base if r["shots"] == key], "method", "setting")
        a = index([r for r in abl if r["shots"] == key], "method", "setting")
        for s in settings:
            bb = one(ib, "none", s)["best_dvaqem"]
            ab = one(ia, "none", s)["best_dvaqem"]
            if key == "inf":
                body.append([s, bb.replace("dvaqem_", ""),
                             ab.replace("dvaqem_", ""),
                             f"{one(b, bb, s)['mse_db']:.2f}",
                             f"{one(a, ab, s)['mse_db']:.2f}",
                             f"{one(a, ab, s)['mse_db'] - one(b, bb, s)['mse_db']:+.2f}"])
            d = one(a, ab, s)["mse_db"] - one(b, bb, s)["mse_db"]
            (d_inf if key == "inf" else d_fin).append(d)
    rep.table(["setting", "selected (warm)", "selected (cold)",
               "MSE dB warm", "MSE dB cold", "cold - warm (dB)"], body)
    rep.p(f"- mean degradation of the cold-start pipeline: "
          f"{np.mean(d_inf):+.2f} dB (infinite shots), "
          f"{np.mean(d_fin):+.2f} dB (finite shots); worst "
          f"{np.max(d_inf + d_fin):+.2f} dB")
    rep.p(f"- the selection changes in "
          f"{sum(1 for r in body if r[1] != r[2])}/{len(body)} settings")
    rep.set("ablation_cold_mean_db_inf", float(np.mean(d_inf)))
    rep.set("ablation_cold_mean_db_finite", float(np.mean(d_fin)))
    rep.set("ablation_cold_worst_db", float(np.max(d_inf + d_fin)))


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="digest the D-VAQEM results")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--res", default=os.path.join(HERE, os.pardir, "results"))
    ap.add_argument("--seeds", default="seed1,seed2")
    a = ap.parse_args()
    res = os.path.abspath(a.res)
    man = load(os.path.join(res, "run_manifest.json"))
    rep = Report()
    rep.h(1, f"D-VAQEM paper numbers (tag = {a.tag})")
    rep.p("Auto-generated by `code/collect_numbers.py`; every number quoted in "
          "`paper/manuscript.tex` comes from this file.")
    rep.h(2, "Run manifest")
    rep.p("```json\n" + json.dumps(man, indent=1) + "\n```")
    for k in ("N", "Ns", "n_cal", "n_tst", "n_trials", "iters", "retrain_iters",
              "seed", "n_oracle", "workers", "total_s"):
        rep.set(f"cfg_{k}", man.get(k))
    sec_e1(rep, res, a.tag)
    idx_inf, settings, meta, rows = sec_e2(rep, res, a.tag)
    sec_e2b(rep, idx_inf, settings, meta, rows)
    sec_e3(rep, res, a.tag)
    sec_e4(rep, res, a.tag)
    sec_e5(rep, res, a.tag)
    sec_seeds(rep, res, [s for s in a.seeds.split(",") if s])
    sec_ablation(rep, res, a.tag)
    sec_resources(rep, res)
    sec_validation(rep, res)
    with open(os.path.join(res, "paper_numbers.md"), "w") as fh:
        fh.write(rep.text())
    with open(os.path.join(res, "paper_numbers.json"), "w") as fh:
        json.dump(rep.scalars, fh, indent=1, default=float)
    print(f"wrote {os.path.join(res, 'paper_numbers.md')} "
          f"({len(rep.lines)} lines) and paper_numbers.json "
          f"({len(rep.scalars)} scalars)")


if __name__ == "__main__":
    main()


