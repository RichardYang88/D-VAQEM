# D-VAQEM — reproducibility package

Code and archived results supporting the manuscript *"Distribution-level
variational quantum error mitigation"* (D-VAQEM). This deposit contains what is
needed to re-derive every number the paper reports, to regenerate every figure
and table in it, and to re-check the simulator against an independent
implementation.

The manuscript source and bibliography are not part of this package; they are
supplied with the submission. Everything here is upstream of them: the
simulator, the mitigation suite, the experiment drivers, the statistics module,
the figure and table generators, and the archived output of the validated
headline run. Reference copies of the generated figures and LaTeX tables are
included, so regenerating them inside the deposit can be diffed against a
known-good output.

## Contents

| Path | What it is |
|---|---|
| `code/vaqem_lib.py` | exact density-matrix simulator; population-imbalance (`m`-sector) algebra; VQ-CNNI decoder |
| `code/vaqem_methods.py` | mitigation suite — linear inversion, ZNE (Richardson/polynomial), D-VAQEM variants, calibrated readout mitigation, probabilistic error cancellation |
| `code/run_experiments.py` | driver for experiments E2–E6 |
| `code/run_sufficiency_swpe.py` | driver for E1 (sector reduction priced in SWPE) |
| `code/train_decoder.py` | VQ-CNNI decoder training |
| `code/bench_oracle_cost.py` | measured cost of the known-noise-model oracle grid |
| `code/validate_simulator.py` | independent PennyLane cross-check of the simulator |
| `code/bootstrap_ci.py` | paired bootstrap intervals, exact tests, delta-method variance |
| `code/collect_numbers.py` | `results/` → `results/paper_numbers.{md,json}` |
| `code/make_tables.py` | number digest → `paper/tables.tex`, `paper/tables_supplement.tex` |
| `code/make_figures.py` | `results/` → `figures/fig1..fig6`, `figures/fig_toc` (PDF + PNG) |
| `code/smoke_test.py` | end-to-end integration check |
| `code/test_bootstrap_ci.py`, `code/test_pec.py` | self-checks for the statistics and PEC modules |
| `models/` | VQ-CNNI decoder checkpoints (`N = 4, 6, 8, 10`, seed 0) |
| `results/` | archived output of the validated headline run (`tag=final`), seed-robustness repeats, cold-start ablation, and the number digest |
| `figures/` | reference copies of the rendered publication figures |
| `paper/` | reference copies of the generated LaTeX tables (the manuscript itself is not here) |
| `requirements.txt` | exact pinned dependencies |
| `MANIFEST.sha256` | checksums for every file in this deposit |

`results/run_manifest.json` is the authoritative record of the configuration
that produced the archived numbers; `results/paper_numbers.md` is the
authoritative record of every scalar the manuscript quotes.

## Environment

Python 3 with the pinned versions in `requirements.txt`:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`autograd >= 1.9` is a hard floor, not a preference. `vaqem_lib.gate_real_block`
is built from plain-numpy `cos`/`sin` so that `autograd.grad` can differentiate
the entire statevector simulation; that only works because `ArrayBox` implements
`__array_ufunc__`, which autograd added in 1.9. On 1.8.0 the gradient
cross-check fails with `TypeError: loop of ufunc does not support argument 0 of
type ArrayBox which has no callable cos method`.

`validate_simulator.py` additionally needs PennyLane, which is deliberately not
pinned in `requirements.txt`:

```bash
pip install pennylane pennylane-lightning
```

Without PennyLane the script still runs and reports the cross-check as skipped.

## Checking the archived numbers without re-running anything

```bash
cd code
python collect_numbers.py --tag final     # regenerates results/paper_numbers.{md,json}
python bootstrap_ci.py    --tag final     # standalone interval / test digest
```

Both read the archived run in `results/` and reproduce the digest the manuscript
was written against. This is the fastest way to confirm the deposit is intact.

## Regenerating the figures and tables

```bash
cd code
python make_tables.py   --tag final    # -> ../paper/tables.tex, tables_supplement.tex
python make_figures.py  --tag final    # -> ../figures/fig1..fig6, fig_toc (.pdf/.png)
```

Both create their output directory if it is absent, and both overwrite the
reference copies shipped in `paper/` and `figures/`. To compare instead of
overwriting, point them somewhere else:

```bash
python make_tables.py  --tag final --out /tmp/check_tables
python make_figures.py --tag final --out /tmp/check_figures
diff -r /tmp/check_tables ../paper
```

The generated `.tex` tables and the rendered `.png` figures are deterministic
and diff clean against the reference copies. The `.pdf` figures embed a
`/CreationDate` (and an `/ID` derived from it), so they differ byte-wise on
every re-run even though the drawing instructions are identical: verified under
the pinned matplotlib 3.10.9, all seven PNGs match byte-for-byte and all seven
PDFs match exactly once the date and ID fields are normalised. Compare the PNGs,
or normalise the PDF metadata, rather than diffing the PDFs directly.
`make_figures.py --only fig1,fig3` renders a comma-separated subset of
`fig1`…`fig6` and `toc`.

The manuscript-side integrity checker (`paper/check_tex.py` in the full project
tree, which audits every number quoted in `manuscript.tex` against
`results/paper_numbers.md`) is not part of this deposit, since neither the
manuscript nor the bibliography is.

## Running the test suites

```bash
cd code
python test_bootstrap_ci.py     # self-checks of the statistics module
python test_pec.py              # self-checks of the quasi-probability baseline
python smoke_test.py --quick    # end-to-end: simulator, reduction, all methods
```

## Re-running the experiments

**Do not use a single `run_experiments.py all` invocation to regenerate the
archived set.** Two properties of the driver make that impossible, and both are
verifiable in the source:

1. `--tag` is a single global prefix, but the archived files need two different
   ones. E2 writes `{tag}_sweep.json` / `{tag}_sweep_meta.json` /
   `{tag}_fi_curves.npz` and E6 writes `{tag}_pec.json`, while
   `collect_numbers.py` reads E6 from a **hard-coded** `e6_pec.json`
   (`sec_e6`, line ~1114) and `bootstrap_ci.py` defaults to the same. So E2
   needs `--tag final` and E6 needs `--tag e6`; no one value of `--tag` yields
   both filenames.
2. `main()` ends with `save_json("run_manifest.json", cfg)`, where `cfg` is
   rebuilt from the CLI. Re-running into `results/` therefore **overwrites
   `run_manifest.json` and discards the two hand-written addenda in it**
   (`wp2_readout_baseline_addendum`, `e6_pec_baseline_addendum`), which carry the
   acceptance records and the corrections log. Always pass `--exp-root`.

Also note that three experiments ignore the corresponding CLI value and hard-code
their own, so passing it is harmless but meaningless: E1 `sufficiency` uses
`n_phi=21`, E3 `shots` uses `n_trials=60`, E4 `scaling` uses `n_tst=25`. The
archived files confirm this (`e4_scaling.json` rows carry `n_tst: 25`).

### Procedure

Everything below writes into a scratch directory, leaving the archived numbers
untouched. `REPRO` can be anywhere; `../repro` keeps it beside the deposit.

```bash
cd code
REPRO=../repro && mkdir -p "$REPRO"

# --- E2 sweep (headline: 16 settings x 4 shot budgets)      ~361 s
python run_experiments.py sweep   --tag final --N 8 \
       --n-cal 21 --n-tst 21 --n-trials 41 --workers 6 --exp-root "$REPRO"

# --- E3 shots (delta-method validation, shot-aware objective) ~306 s
python run_experiments.py shots   --N 8 --n-cal 21 --n-tst 21 \
       --workers 6 --exp-root "$REPRO"

# --- E4 scaling (N = 4..10); MUST be cold-cache, see caveat  ~2108 s
python run_experiments.py scaling --Ns 4,6,8,10 --n-cal 21 --n-trials 41 \
       --workers 6 --exp-root "$REPRO"

# --- E5 calib (calibration budget)                           ~214 s
python run_experiments.py calib   --N 8 --n-trials 41 \
       --workers 6 --exp-root "$REPRO"

# --- E6 pec: run with NO parameter overrides.  exp_pec()'s defaults
#     (n_cal=25, n_tst=41, n_trials=25) ARE the archived ones; the headline
#     21/21/41 do not apply to E6.  --tag e6 gives e6_pec.json.   ~550 s
python run_experiments.py pec     --tag e6 --workers 6 --exp-root "$REPRO"

# --- OPTIONAL, and not part of the paper's numbers.  `sufficiency` writes
#     e1_sufficiency.json and e1_fi_convergence.json, and no downstream
#     consumer reads either (verified: the only references in code/ are the two
#     save_json calls at run_experiments.py:762 and :794).  The paper's E1 is
#     e1_sufficiency_swpe.json from run_sufficiency_swpe.py, below.  Run this
#     only if you want the archived Fisher-information numerics in their own
#     right.
python run_experiments.py sufficiency --workers 6 --exp-root "$REPRO"

# --- E1 as the paper reports it (reduction priced in SWPE + certified
#     decoder).  Separate driver; its defaults --N 4,6,8,10 --n_cal 21
#     --n_tst 21 --n_trials 41 --seed 0 match the archived file.  THIS is the
#     one collect_numbers.sec_e1 reads.
python run_sufficiency_swpe.py --exp-root "$REPRO"

# --- measured cost of the known-noise-model oracle grid       ~857 s
#     This one has no --exp-root; it defaults to ../results/oracle_cost.json,
#     so redirect it explicitly with --out.
python bench_oracle_cost.py --workers 6 --out "$REPRO/oracle_cost.json"

# --- independent simulator cross-validation (needs PennyLane)
python validate_simulator.py > "$REPRO/simulator_validation.txt"
```

Wall-clock figures are the per-experiment `timings_s` recorded in the archived
`results/run_manifest.json` (`total_s` = 2988.8 s covers E1–E5 only; that run
predates E6, which is why the manifest has no E6 entry). E6 (~550 s) and the
oracle benchmark (~857 s) are additional. `run_sufficiency_swpe.py`'s wall clock
is not recorded in the manifest. These timings are for a 24-core machine at
`--workers 6`; scale expectations accordingly.

Then compare against the archive:

```bash
python collect_numbers.py --tag final --res "$REPRO"
diff "$REPRO/paper_numbers.md" ../results/paper_numbers.md
```

`collect_numbers.py` has no `--out`; it writes `paper_numbers.{md,json}` into
whatever `--res` points at.

**That diff will not be byte-clean, and that is expected.** The digest embeds
`run_manifest.json` verbatim — including `total_s`, the per-experiment
`timings_s`, the numpy/autograd/Python versions and `cpu_count` — and
`sec_resources` / the E4 and oracle sections record wall-clock measurements
(`cfg_total_s`, `e4_N10_fit_s`, `cost_N10_t_oracle_s`, ...). All of those are
machine- and run-specific. A fresh run also produces a `run_manifest.json`
without the two hand-written addenda, so that JSON block differs structurally.

Compare the **scientific** sections instead — E1–E6 accuracies, Fisher
information, intervals, test statistics, win counts. Those are deterministic
given the pinned seeds and should match. The two scratch subdirectories the
digest also reads, `seed_robustness/` and `ablations/`, are not regenerated by
the procedure above, so `sec_seeds` and `sec_ablation` will report UNAVAILABLE
in `$REPRO`; they are archived-only sections.

Experiment map: `sufficiency` = E1 (archived Fisher-information numerics of the
sector reduction), `run_sufficiency_swpe.py` = E1 as the paper reports it,
`sweep` = E2, `shots` = E3, `scaling` = E4, `calib` = E5, `pec` = E6
(probabilistic error cancellation on its own 20-setting grid).

Two caveats that matter for reproducing the archived numbers exactly:

- **Keep `--workers 6` and watch free memory.** The default is
  `cpu_count - 2`; with only a few GB free and no swap, a run that large has
  been seen to die with an intermittent SIGSEGV at a varying setting. It is not
  reproducible at `--workers 6` with the paper parameters. The sweep writes its
  JSON after every setting, so a crash costs only the in-flight setting.
- **E4 (`scaling`) must be run with a cold cache.** Its `t_data_s` is a
  simulation-cost measurement. Rows therefore carry `data_from_cache`,
  `cache_hits` and `cache_simulated`, and a warm re-run records ~0 s where the
  `N = 10` dataset really costs 1982 s. A warm re-run reproduces every
  *scientific* scalar of the headline run bit-for-bit (verified), but it must
  not overwrite the archived `e4_scaling.json`, whose timings are the cold
  measurement the manuscript quotes. For the same reason the archived E4 has no
  `linv_calib` rows; the readout baseline is compared in E2/E3/E5, where cache
  provenance is uniform. See `wp2_readout_baseline_addendum` in
  `results/run_manifest.json`.

Pass `--exp-root <dir>` to keep repeats and ablations out of `results/`, so the
archived numbers cannot be overwritten by accident.

The exact-simulation cache in `cache/` is keyed by (N, parameters, noise,
phases, folds) and is not part of this deposit; it is rebuilt on demand, and
deleting it only costs re-simulation time.

## Decoder checkpoints

`models/` holds all four VQ-CNNI checkpoints the study uses
(`vqcnni_N{4,6,8,10}_s0.npz`), so this package is self-contained. The `N = 10`
checkpoint is produced by the included `code/train_decoder.py`:

```bash
cd code
python train_decoder.py --N 10 --out ../models/vqcnni_N10_s0.npz
```

The `N = 4, 6, 8` checkpoints were trained in PennyLane as part of the companion
VQ-CNNI study; they are redistributed here so that E4 can be re-run without a
second checkout. `run_experiments.load_model()` resolves `../models/` first, so
no environment variable or sibling directory is required.

## Known limitations

Discussed in the manuscript, Sec. V: Markovian stationary noise only, a single
probe family, clipped quasi-probability diagnostics, and an information-theoretic
breakdown at very strong dephasing.
