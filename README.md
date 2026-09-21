# D-VAQEM: distribution-level variational quantum error mitigation

Everything needed to reproduce the numbers, figures and tables of the
manuscript in `paper/`:

* `code/` — simulator, mitigation suite, experiment drivers, generators, the
  statistics module (`bootstrap_ci.py`) and the test suites (`smoke_test.py`,
  `validate_simulator.py`, `test_bootstrap_ci.py`)
* `models/` — VQ-CNNI checkpoints used by the scaling study (`N=10`; the
  `N=4,6,8` checkpoints come from the companion VQ-CNNI checkout, which
  `vaqem_lib.vqcnni_root()` locates automatically: `$VQCNNI_ROOT`, a sibling
  `../VQ-CNNI/`, or the old `new_paper/` parent directory)
* `results/` — every result file of the validated headline run (`tag=final`),
  the machine-readable number digest (`paper_numbers.{md,json}`), the paired
  bootstrap intervals and exact tests (`ci_final.json`), seed-robustness and
  ablation runs
* `figures/` — generated publication figures (PDF + PNG)
* `paper/` — the LaTeX draft, generated tables, bibliography
* `cache/` — disk cache of exact density-matrix datasets (makes re-runs cheap)

## Environments

* main venv (e.g. `../VQ-CNNI/.venv`, or any environment installed from
  `requirements.txt` with `install.sh`): `numpy`, `scipy`, `matplotlib`,
  `autograd` — everything except the PennyLane cross-check
* `./.venv-pl`: adds `pennylane` + `pennylane-lightning` — only
  `code/validate_simulator.py` needs it.  With `autograd>=1.9` installed it is a
  *complete* environment: it runs the whole chain (drivers, statistics, tables,
  figures, tests) and all 53 validation checks, so a single venv is enough.

`autograd>=1.9` is a hard floor, not a preference.  `vaqem_lib.gate_real_block`
is deliberately built from plain-numpy `cos`/`sin` so that `autograd.grad` can
differentiate the entire statevector simulation; that only works because
`ArrayBox` implements `__array_ufunc__`, which autograd added in 1.9.  On 1.8.0
the gradient cross-check dies with `TypeError: loop of ufunc does not support
argument 0 of type ArrayBox which has no callable cos method`.

## Reproducing the experiments

```bash
cd code
# headline chain (~50 min cold, minutes warm thanks to cache/):
python run_experiments.py all --tag final --workers 6 \
       --N 8 --Ns 4,6,8,10 --n-cal 21 --n-tst 21 --n-trials 41
# simulator cross-validation (needs .venv-pl):
../.venv-pl/bin/python validate_simulator.py > ../results/simulator_validation.txt
# seed robustness and the cold-start ablation (kept out of results/ on purpose):
python run_experiments.py sweep --seed 1 --tag seed1 --exp-root ../results/seed_robustness ...
python run_experiments.py sweep --no-warm-start --tag coldstart --exp-root ../results/ablations ...
# measured cost of the known-noise-model oracle grid:
python bench_oracle_cost.py --workers 6
```

Two caveats that matter for the archived numbers:

* **Keep `--workers 6` and watch free memory.**  The default is
  `cpu_count - 2` (22 here); with only a few GB free and no swap, a run that
  large has been seen to die with an intermittent SIGSEGV at a varying setting.
  It is not reproducible at `--workers 6` with the paper parameters.  The sweep
  writes its JSON after every setting, so a crash costs only the in-flight
  setting.
* **E4 (`scaling`) must be run with a cold `cache/`.**  Its `t_data_s` is a
  simulation cost, and `paper/check_tex.py` deliberately fails when the scaling
  rows report `data_from_cache`.  For that reason `results/e4_scaling.json` is
  *not* regenerated when a new method is added: splicing freshly timed rows into
  the archived file would mix two cache provenances in one file.  The archived
  E4 therefore has no `linv_calib` rows, and the readout baseline is compared in
  E2/E3/E5, where provenance is uniform.  See
  `wp2_readout_baseline_addendum` in `results/run_manifest.json`.

Experiment map: `sufficiency` = E1 (Theorem 1 numerics), `sweep` = E2
(headline 16 settings × 4 shot budgets), `shots` = E3 (delta-method validation
and the shot-aware objective), `scaling` = E4 (N = 4…10), `calib` = E5
(calibration budget), `pec` = E6 (probabilistic error cancellation: the
quasi-probability baseline on its own 20-setting grid).  `--exp-root` keeps
ablations and repeats out of `results/` so the paper numbers can never be
overwritten by accident.

## Regenerating the paper artefacts

```bash
cd code
python collect_numbers.py --tag final   # paper_numbers.{md,json}
                                        #   + ci_final.json, ci_pec_final.json
python make_tables.py     --tag final   # paper/tables.tex, tables_supplement.tex (S1–S12)
python make_figures.py    --tag final   # figures/fig1..fig6 (.pdf/.png)
python bootstrap_ci.py    --tag final   # standalone interval/test digest (optional)
python test_bootstrap_ci.py             # 78 self-checks of the statistics
cd ../paper && python check_tex.py      # integrity + audit of every quoted number
```

`results/run_manifest.json` (tag `final`) is the single source of truth for the
configuration; `results/paper_numbers.md` is the single source of truth for
every number quoted in `paper/manuscript.tex`.

## Notes

* `results/full3_*`, `results/pre_*` are earlier/ablation runs kept for
  provenance; they are **not** used by any figure or table.
* The exact-simulation cache in `cache/` is keyed by (N, parameters, noise,
  phases, folds); deleting it only costs re-simulation time.
* Known limitations of the study are discussed in `paper/manuscript.tex`,
  Sec. V (Markovian stationary noise, single probe family, clipped
  quasi-probability diagnostics, information-theoretic breakdown at very strong
  dephasing).
* **Cost figures and the disk cache.** `t_data_s` in `results/e4_scaling.json`
  is a wall-clock measurement, and it is only a *cost* figure when the run was
  cold: the cache in `cache/` serves the exact density-matrix datasets
  otherwise, so a warm re-run records ~0 s where the $N=10$ dataset really
  costs 1982 s.  Rows therefore carry `data_from_cache`, `cache_hits` and
  `cache_simulated`, and `paper/check_tex.py` fails if a quoted cost comes from
  a cached run.  A warm re-run reproduces every *scientific* scalar of the
  headline run bit-for-bit (verified), but it must not overwrite the archived
  `e4_scaling.json`, whose timings are the cold measurement the manuscript
  quotes.
