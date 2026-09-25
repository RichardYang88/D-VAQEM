"""check_tex.py -- integrity check for the manuscript.

Three independent checks:

1. LaTeX environment balance, figure labels and the truncated-paragraph
   heuristic (the original checks);
2. cross-references: every ``\\ref`` must have a ``\\label``, and every label
   should be referenced at least once (an unreferenced supplemental table is a
   thing reviewers notice);
3. numbers: every statistical quantity quoted in the prose is re-derived from
   ``results/paper_numbers.json`` and compared as a string, so a hand-edited
   confidence interval or p-value cannot silently disagree with the data.

Exits non-zero on an undefined reference or a numeric mismatch.

Usage:  python check_tex.py [manuscript.tex]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

tex = open(sys.argv[1] if len(sys.argv) > 1 else "manuscript.tex").read()
for f in ("tables.tex", "tables_supplement.tex"):
    try:
        tex += "\n" + open(f).read()
    except OSError:
        pass
for env in sorted(set(re.findall(r"\\begin\{(\w+\*?)\}", tex))):
    b = len(re.findall(r"\\begin\{%s\}" % env, tex))
    e = len(re.findall(r"\\end\{%s\}" % env, tex))
    if b != e:
        print("UNBALANCED", env, b, e)
print("figure labels:", re.findall(r"\\label\{(fig:\w+)\}", tex))
# heuristic: a paragraph that ends mid-sentence (truncation detector)
body = tex.split("\\appendix")[0]
# the front matter (title, addresses, keywords, abstract) is not prose, so it
# must not be scanned for truncated paragraphs
body = body.split("\\maketitle")[-1]
bad = []
for blk in body.split("\n\n"):
    lines = [l for l in blk.strip().split("\n") if l.strip()]
    if not lines:
        continue
    last = lines[-1].strip()
    if last.startswith("\\") and not last.endswith("$"):
        continue
    if last.endswith(("\\begin{equation}", "\\begin{figure*}")):
        continue
    if not re.search(r"[.:;%)$\]]$|\\\\$", last):
        bad.append(last[-70:])
print("suspect paragraph endings:")
for b in bad:
    print("   ...", b)
print("lines:", tex.count("\n"))


# ----------------------------------------------------------------------
# 2. cross-references
# ----------------------------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
undef = sorted(refs - labels)
unused = sorted(labels - refs)
print(f"\ncross-references: {len(labels)} labels, {len(refs)} referenced")
if undef:
    print("UNDEFINED \\ref (would typeset as ??):", undef)
else:
    print("no undefined \\ref")
if unused:
    print("labels never referenced (harmless, but worth a look):")
    for u in unused:
        print("   -", u)


# ----------------------------------------------------------------------
# 3. numbers quoted in the prose vs the data they come from
# ----------------------------------------------------------------------
def latex_p(p):
    """Render a p-value the way the manuscript does: 3.1e-05 -> 3.1x10^-5."""
    a, e = f"{p:.1e}".split("e")
    return f"{a}\\times10^{{{int(e)}}}"


def check_numbers(text, pn):
    """Assert each quoted statistic appears verbatim, re-derived from the data.

    A check fails when the string built from ``paper_numbers.json`` is absent
    from the manuscript, which catches both a stale manuscript and a changed
    result.  Values are rounded exactly as the prose rounds them, so the check
    never demands more precision than the paper claims.
    """
    if not pn:
        print("\nnumeric cross-check SKIPPED (no results/paper_numbers.json)")
        return 0, 0
    state = {"ok": 0, "bad": 0}
    # LaTeX treats a line break as a space, so the prose is compared with all
    # whitespace runs collapsed; otherwise a correctly quoted number fails just
    # because the source happens to wrap inside it.
    flat = re.sub(r"\s+", " ", text)

    def want(s, why):
        s = " ".join(str(s).split())
        if s and s in flat:
            state["ok"] += 1
        else:
            state["bad"] += 1
            print(f"  MISMATCH [{why}]: expected to find {s!r}")

    def g(k, dp=None):
        v = pn.get(k)
        return v if dp is None else float(f"{v:.{dp}f}")

    print("\nnumeric cross-check of the statistical claims:")
    if "ci_gain_vs_none_inf_db" not in pn:
        print("  SKIPPED: no ci_* keys (run collect_numbers.py on a sweep that "
              "carries per-phase/per-trial arrays)")
        return 0, 0
    want(f"${g('ci_gain_vs_none_inf_db', 1)}$ dB", "mean MSE reduction")
    want(f"[{g('ci_gain_vs_none_inf_lo', 1)},{g('ci_gain_vs_none_inf_hi', 1)}]",
         "CI of the mean reduction")
    want(latex_p(pn["ci_gain_vs_none_inf_sign_p"]), "exact sign-test p")
    want(f"${g('e2_gain_vs_zne_mean_db', 1)}$ dB on average", "mean gain vs ZNE")
    want(f"${g('ci_vs_zne_inf_lf_point', 1):+}$ dB "
         f"$[{g('ci_vs_zne_inf_lf_ci_lo', 1):+},"
         f"{g('ci_vs_zne_inf_lf_ci_hi', 1):+}]$",
         "least-favourable per-setting gain vs ZNE")
    want(latex_p(pn["ci_vs_zne_inf_lf_p_one_sided"]),
         "one-sided bootstrap p, least-favourable setting")
    for zm, quoted in (("zne_rich", "Richardson"), ("zne_poly1", "linear fit"),
                       ("zne_poly2", "quadratic fit")):
        want(f"${g(f'ci_vs_{zm}_inf_mean', 1):+}$ dB "
             f"$[{g(f'ci_vs_{zm}_inf_ci_lo', 1):+},"
             f"{g(f'ci_vs_{zm}_inf_ci_hi', 1):+}]$",
             f"fixed comparator {quoted}")
        want(latex_p(pn[f"ci_vs_{zm}_inf_sign_p"]), f"sign p vs {quoted}")
    for key in ("S256", "S1024", "S4096"):
        want(f"${g(f'ci_vs_retrain_{key}_db', 2):+}$ dB "
             f"$[{g(f'ci_vs_retrain_{key}_lo', 2):+},"
             f"{g(f'ci_vs_retrain_{key}_hi', 2):+}]$",
             f"parity with decoder retraining at {key}")
    wps = [pn[f"ci_vs_retrain_{k}_wilcoxon_p"] for k in ("S256", "S1024", "S4096")]
    want("($p=" + ", ".join(f"{p:.3f}$" if i == 0 else f"${p:.3f}$"
                            for i, p in enumerate(wps)) + ")",
         "Wilcoxon p-values of the parity claim")
    want(f"$p\\ge{min(wps):.2f}$", "Wilcoxon floor quoted in abstract/conclusion")
    # the prose quotes the oracle gap as a positive magnitude, while the CI
    # machinery stores it as a negative gain (D-VAQEM is the worse method)
    want(f"${g('e2_gap_to_oracle_mean_db', 1)}$ dB gap", "gap to the oracle")
    gap_lo, gap_hi = -pn["ci_oracle_gap_inf_hi"], -pn["ci_oracle_gap_inf_lo"]
    want(f"[{gap_lo:.1f},{gap_hi:.1f}]", "CI of the oracle gap")
    want(f"[{gap_lo:.0f},{gap_hi:.0f}]",
         "coarse CI of the oracle gap in the introduction")
    cl = [pn[f"ci_closure_{k}_pct"] for k in ("S256", "S1024", "S4096")]
    want(f"${round(min(cl))}$--${round(max(cl))}\\,\\%$", "gap-closure range")

    # ---- readout-error-mitigation comparator --------------------------------
    # This block exists because the infinite-shot row is a *negative* result: the
    # interval contains zero, so the prose must not claim a win there.  Pinning
    # the non-significant interval, the win counts and the two floor margins to
    # the data is what stops that sentence from drifting into an overclaim (the
    # earlier "reaches the noiseless floor, identical to the exact sector
    # inverse" was true only at the weakest readout rate and went unchecked).
    if "ci_vs_readout_inf_db" in pn:
        want(f"${g('ci_vs_readout_inf_db', 1):+}$ dB",
             "mean gain vs calibrated readout mitigation (infinite shots)")
        want(f"[{g('ci_vs_readout_inf_lo', 1):+},"
             f"{g('ci_vs_readout_inf_hi', 1):+}]",
             "CI vs calibrated readout mitigation (must contain zero)")
        want(f"$p={pn['ci_vs_readout_inf_sign_p']:.2f}$", "sign p vs readout (inf)")
        want(f"$p={pn['ci_vs_readout_inf_wilcoxon_p']:.2f}$",
             "Wilcoxon p vs readout (inf)")
        want(f"${int(pn['ci_vs_readout_inf_wins'])}$ of "
             f"${int(pn['ci_vs_readout_inf_n'])}$ settings",
             "win count vs readout mitigation (inf)")
        wins = []
        for key in ("S256", "S1024", "S4096"):
            want(f"${g(f'ci_vs_readout_{key}_db', 1):+}$ dB "
                 f"$[{g(f'ci_vs_readout_{key}_lo', 1):+},"
                 f"{g(f'ci_vs_readout_{key}_hi', 1):+}]$",
                 f"gain vs readout mitigation at {key}")
            wins.append(f"${int(pn[f'ci_vs_readout_{key}_wins'])}/"
                        f"{int(pn[f'ci_vs_readout_{key}_n'])}$")
        want(", ".join(wins[:2]) + f" and {wins[2]}",
             "per-budget win counts vs readout mitigation")
        sps = [pn[f"ci_vs_readout_{k}_sign_p"] for k in ("S256", "S1024", "S4096")]
        want(f"$p\\le{latex_p(max(sps))}$", "sign-p bound vs readout (finite)")
        want(f"${int(pn['readout_amplify_cells'])}$ of the "
             f"${int(pn['readout_amplify_total'])}$",
             "cells where the calibrated inverse amplifies noise")
        # genie inverse: undefined on half the settings, behind at every finite
        # budget, and ahead on average only in the infinite-shot limit
        want(f"${int(pn['readout_genie_defined_settings'])}$ of the "
             f"${int(pn['readout_n_settings'])}$ settings",
             "settings where the genie inverse is undefined")
        want(f"${g('ci_vs_linvknown_inf_db', 1):+}$ dB "
             f"$[{g('ci_vs_linvknown_inf_lo', 1):+},"
             f"{g('ci_vs_linvknown_inf_hi', 1):+}]$",
             "gain vs the genie sector inverse (infinite shots)")
        gd = [g(f"ci_vs_linvknown_{k}_db", 1) for k in ("S256", "S1024", "S4096")]
        want(f"${min(gd):+}$ to ${max(gd):+}$ dB",
             "finite-shot gain range vs the genie inverse")
        gw = [pn[f"ci_vs_linvknown_{k}_wilcoxon_p"]
              for k in ("S256", "S1024", "S4096")]
        want(f"$p\\le{max(gw):.3f}$", "Wilcoxon bound vs the genie inverse")
        # calibration quality and the two floor margins on pure readout noise
        want(f"${latex_p(pn['readout_qhat_max_abs_err_pure'])}$",
             "recovery of the true flip rate by calibration")
        want(f"${int(pn['readout_calib_beats_analytic_settings'])}$ of the "
             f"${int(pn['readout_calib_scored_settings'])}$ settings",
             "settings where the calibrated rate beats the analytic one")
        want(f"${latex_p(pn['readout_inf_linv_margin_max_db'])}$ dB",
             "distance of the calibrated inverse from the noiseless floor")
        want(f"${g('readout_inf_l2_margin_min_db', 2)}$ dB",
             "l2 map margin over the floor at the weakest readout rate")
        want(f"${g('readout_inf_l2_margin_max_db', 1)}$ dB",
             "l2 map margin over the floor at the strongest readout rate")
        want(f"${g('readout_inf_l2_db_weakest', 1)}$ dB against a floor of "
             f"${g('readout_inf_floor_db', 1)}$ dB",
             "learned map vs floor at the weakest readout rate")
    # ---- probabilistic error cancellation (E6) -----------------------------
    if pn.get("pec_available"):
        CHAN = {"depol": "depolarizing", "deph": "dephasing",
                "ampdamp": "amplitude damping", "readout": "readout bit-flip"}

        def gs(key, nd=1):
            """Signed decimal, the way the prose writes a dB gain: +31.1/-4.6."""
            return f"{float(pn[key]):+.{nd}f}"

        def n(key):
            """An integer count stored as a float scalar."""
            return int(pn[key])

        def sci(key, nd=1):
            """Scientific notation as the manuscript renders it."""
            a, e = f"{float(pn[key]):.{nd}e}".split("e")
            return f"{a}\\times10^{{{int(e)}}}"

        def pci(name, nd=1):
            """A PEC confidence interval written the way the prose writes it."""
            return (f"$[{float(pn['pec_ci_lo_' + name]):+.{nd}f},"
                    f"{float(pn['pec_ci_hi_' + name]):+.{nd}f}]$")

        print("\nPEC / probabilistic error cancellation (E6) claims:")
        want(f"agree to ${sci('pec_identity_max_dev_db')}$ dB across the "
             f"${n('pec_identity_n')}$ settings whose overhead satisfies "
             f"$\\gamma\\le{n('pec_gamma_wellcond')}$ (largest "
             f"$\\gamma={g('pec_identity_gamma_max', 1)}$ among them)",
             "where PEC's deterministic limit equals the calibrated inverse")
        want(f"rates, ${n('pec_n_settings')}$ in total",
             "number of noise settings in the PEC study")
        want(f"with ${n('pec_n_cal')}$ calibration phases, "
             f"${n('pec_n_phase')}$ held-out test phases and "
             f"${n('pec_n_trials')}$ Monte-Carlo trials per finite-shot cell",
             "the E6 grid: calibration phases, test phases, Monte-Carlo trials")
        want(f"differs from the headline sweep's (${n('cfg_n_cal')}$ "
             f"calibration and ${n('cfg_n_tst')}$ test phases",
             "the headline sweep's grid, quoted to contrast it with E6's")
        want(f"${n('pec_n_trials')}$ Monte-Carlo trials per finite-shot cell",
             "Monte-Carlo trials behind every PEC finite-shot number")
        want(f"between ${g('pec_gamma_min', 2)}$ and ${sci('pec_gamma_max')}$ "
             f"(median ${g('pec_gamma_median', 2)}$)",
             "range and median of the PEC sampling overhead gamma")
        # g() returns a float, which would render a 0-dp value as "215.0";
        # format this one as a string.
        want(f"up to ${float(pn['pec_gamma_sq_max_db']):.0f}$ dB of variance "
             f"amplification",
             "gamma^2 expressed as a variance amplification in dB")
        want(f"$\\hat q={g('pec_qhat_max', 3)}$ approaches",
             "calibrated flip rate closest to the q=1/2 singularity")
        want(f"On the ${n('pec_n_pure_readout')}$ pure-readout settings",
             "settings where the binomial family contains the channel exactly")
        want(f"$\\gamma$ is recovered to ${sci('pec_gamma_relerr_pure_max')}$ "
             f"relative", "accuracy of the calibrated gamma on pure readout noise")
        want(f"the same fit is up to "
             f"${round(float(pn['pec_gamma_relerr_max']) * 100)}\\%$ off",
             "worst relative error of the calibrated gamma off the flip family")
        fk = ("S256", "S1024", "S4096")
        # "fails to reject equality at all three" is a claim about the tests, so
        # verify it instead of only string-matching the sentence around it.  Note
        # the S256 interval is [+0.01,+0.06]: it excludes zero, so the honest
        # wording is a bound on the *magnitude*, not "every interval covers zero".
        rej = [k for k in fk if float(pn[f"pec_pwil_pecknown_{k}"]) < 0.05
               or float(pn[f"pec_psign_pecknown_{k}"]) < 0.05]
        if rej:
            state["bad"] += 1
            print(f"  MISMATCH [calibrated vs genie PEC]: the manuscript says the "
                  f"exact tests fail to reject equality at every finite budget, "
                  f"but they reject at {rej}")
        gmax = max(abs(float(pn[f"pec_gain_pecknown_{k}"])) for k in fk)
        pmin = min(float(pn[f"pec_pwil_pecknown_{k}"]) for k in fk)
        want(f"the mean difference is at most ${gmax:.2f}$ dB in magnitude and "
             f"the exact Wilcoxon test fails to reject equality at all three "
             f"($p\\ge{pmin:.2f}$)",
             "calibrated PEC vs genie PEC at every finite budget")
        want(f"the ${abs(float(pn['pec_gain_pecknown_inf'])):.1f}$ dB it "
             f"concedes to the genie at infinite shots {pci('pecknown_inf')}",
             "calibration penalty of PEC at infinite shots")


        # ---- the verdict: PEC against the alternatives at each budget -------
        want(f"at infinite shots (${gs('pec_gain_none_inf')}$ dB "
             f"{pci('none_inf')}, ahead in "
             f"${n('pec_wins_none_inf')}/{n('pec_n_none_inf')}$ settings, exact "
             f"sign $p\\le{latex_p(pn['pec_psign_none_inf'])}$)",
             "PEC vs no mitigation at infinite shots")
        want(f"doing nothing (${gs('pec_gain_none_S256')}$ dB "
             f"{pci('none_S256')}, sign "
             f"$p={float(pn['pec_psign_none_S256']):.2f}$)",
             "PEC vs no mitigation at S=256")
        want(f"in fact \\emph{{worse}} in ${n('pec_losses_none_S256')}$ of the "
             f"${n('pec_n_none_S256')}$ settings",
             "settings where PEC at S=256 is worse than not mitigating")
        want(f"resolves at all (${gs('pec_gain_none_S1024')}$ dB "
             f"{pci('none_S1024')}, exact Wilcoxon "
             f"$p={float(pn['pec_pwil_none_S1024']):.3f}$)",
             "PEC vs no mitigation at S=1024")
        # "it needs S>=1024 before the gain resolves" is a claim about which
        # intervals exclude zero, so verify that rather than only the numbers.
        cov256 = (float(pn["pec_ci_lo_none_S256"]) <= 0.0
                  <= float(pn["pec_ci_hi_none_S256"]))
        excl1024 = float(pn["pec_ci_lo_none_S1024"]) > 0.0
        if not (cov256 and excl1024):
            state["bad"] += 1
            print(f"  MISMATCH [PEC vs none, resolvability]: the manuscript says "
                  f"the gain is unresolved at S=256 and resolves at S=1024, but "
                  f"the S=256 interval covers zero = {cov256} and the S=1024 "
                  f"interval excludes zero = {excl1024}")
        # "loses everywhere at every finite budget" is a claim about all three
        # budgets, so assert it rather than quoting one of them.
        FIN = ("S256", "S1024", "S4096")
        lost = [k for k in FIN
                if n(f"pec_losses_linvcalib_{k}") != n(f"pec_n_linvcalib_{k}")]
        if lost:
            state["bad"] += 1
            print(f"  MISMATCH [PEC vs calibrated inverse]: the manuscript says "
                  f"PEC loses in every setting at every finite budget, but at "
                  f"{lost} it does not")
        # the quoted bound has to cover both exact tests at all three budgets
        pmax = max(max(float(pn[f"pec_psign_linvcalib_{k}"]),
                       float(pn[f"pec_pwil_linvcalib_{k}"])) for k in FIN)
        want(f"everywhere---ahead in ${n('pec_wins_linvcalib_S256')}/"
             f"{n('pec_n_linvcalib_S256')}$ settings at each of $S=256$, $1024$ "
             f"and $4096$, by ${gs('pec_gain_linvcalib_S256')}$ dB "
             f"{pci('linvcalib_S256')}, ${gs('pec_gain_linvcalib_S1024')}$ dB "
             f"{pci('linvcalib_S1024')} and ${gs('pec_gain_linvcalib_S4096')}$ dB "
             f"{pci('linvcalib_S4096')}, with exact sign and Wilcoxon "
             f"$p\\le{latex_p(pmax)}$ in all three",
             "PEC vs the deterministic calibrated inverse at every finite budget")
        want(f"the mean is ${gs('pec_gain_linvcalib_inf')}$ dB "
             f"{pci('linvcalib_inf', 2)}, the exact Wilcoxon test rejecting "
             f"equality at $p={float(pn['pec_pwil_linvcalib_inf']):.3f}$",
             "PEC vs the calibrated inverse at infinite shots")

        # ---- the two caveats: regularisation, and model error ---------------
        ill = str(pn["pec_illcond_setting"])
        chan, _, pw = ill.rpartition("_")
        want(f"across the ${n('pec_illcond_n')}$ settings with "
             f"$\\gamma>{n('pec_gamma_wellcond')}$ the Tikhonov term inside the "
             f"deterministic inverse buys up to "
             f"${g('pec_illcond_max_gain_db', 1)}$ dB "
             f"({CHAN.get(chan, chan)} $p={pw}$, "
             f"$\\gamma={sci('pec_illcond_max_gain_gamma')}$)",
             "regularization gain in the ill-conditioned settings")
        want(f"rate by ${abs(float(pn['pec_mismatch_worst_dq'])):.2f}$ costs "
             f"${g('pec_mismatch_degradation_db', 1)}$ dB "
             f"(${g('pec_mismatch_db_matched', 1)}\\to"
             f"{g('pec_mismatch_db_worst', 1)}$ dB)",
             "cost of mis-specifying the flip rate")
        want(f"stays ahead at ${n('pec_mismatch_linv_ahead')}$ of the "
             f"${n('pec_mismatch_n_dq')}$ assumed rates, by up to "
             f"${g('pec_mismatch_linv_margin_db', 1)}$ dB",
             "the deterministic inverse under model mismatch")
        want(f"mismatched map is ${g('pec_mismatch_floor_median', 2)}$",
             "residual sector distance of a mismatched PEC map")
        want(f"in ${n('pec_losses_linvcalib_S256')}/"
             f"{n('pec_n_linvcalib_S256')}$ settings at every finite budget we "
             f"tested", "Discussion: PEC is dominated at every finite budget")


    print(f"  {state['ok']} quoted numbers agree with the data, "
          f"{state['bad']} mismatch(es)")
    return state["ok"], state["bad"]


def check_costs(text, res):
    """Verify the wall-clock cost figures quoted in the scaling discussion.

    These are measurements rather than results, and they are the one class of
    number in the paper that a cached re-run silently invalidates: ``t_data_s``
    becomes a cache-lookup time (0.0 s instead of 1981 s at $N=10$).  The check
    therefore re-derives every quoted cost from ``e4_scaling.json`` and
    ``oracle_cost.json``, and additionally fails when the scaling rows report
    that their dataset came from the disk cache.
    """
    e4p, ocp = (os.path.join(res, f) for f in ("e4_scaling.json",
                                               "oracle_cost.json"))
    if not (os.path.exists(e4p) and os.path.exists(ocp)):
        print("\ncost cross-check SKIPPED (no e4_scaling.json/oracle_cost.json)")
        return 0
    e4, oc = json.load(open(e4p)), json.load(open(ocp))
    flat = re.sub(r"\s+", " ", text)
    bad = 0

    def want(s, why):
        nonlocal bad
        s = " ".join(str(s).split())
        if s in flat:
            return
        bad += 1
        print(f"  MISMATCH [{why}]: expected to find {s!r}")

    print("\ncost cross-check (wall-clock figures quoted in Sec. scaling):")
    fits = [r["t_fit_s"] for r in e4 if "t_fit_s" in r]
    if fits:
        want(f"${min(fits):.1f}$--${max(fits):.1f}$ s at every $N$",
             "map-fit time range over N")
    data = {r["N"]: r["t_data_s"] for r in e4 if "t_data_s" in r}
    if 10 in data:
        want(f"took ${data[10]:.0f}$ s at $N=10$", "N=10 exact-dataset cost")
    cached = sorted({r["N"] for r in e4 if r.get("data_from_cache")})
    if cached:
        bad += 1
        print(f"  CACHED RUN: t_data_s at N={cached} was served from the disk "
              "cache and is a lookup time, not a simulation cost; re-run with a "
              "cold cache before quoting it")
    byN = {e["N"]: e for e in oc}
    if 4 in byN and 10 in byN:
        cal = [e["t_calib_s"] for e in oc if "t_calib_s" in e]
        want(f"costs ${min(cal):.2f}$--${max(cal):.1f}$ s of exact simulation",
             "21-phase calibration cost range")
        want(f"costs ${byN[4]['t_oracle_s']:.1f}$ s at $N=4$ but "
             f"${byN[10]['t_oracle_s']:.0f}$ s at $N=10$",
             "361-phase oracle grid cost")
    print(f"  {'all quoted costs agree with the data' if not bad else str(bad) + ' cost mismatch(es)'}")
    return bad


RES = os.path.join(HERE, os.pardir, "results")
_pn_path = os.path.join(RES, "paper_numbers.json")
PN = json.load(open(_pn_path)) if os.path.exists(_pn_path) else None
_, NBAD = check_numbers(tex, PN)
CBAD = check_costs(tex, RES)
sys.exit(1 if (undef or NBAD or CBAD) else 0)

