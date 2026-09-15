# D-VAQEM: distribution-level variational quantum error mitigation

Everything needed to reproduce the numbers, figures and tables of the
manuscript in `paper/`:

* `code/` — simulator, mitigation suite, experiment drivers, generators
* `models/` — VQ-CNNI checkpoints used by the scaling study (`N=10`; the
  `N=4,6,8` checkpoints come from the companion VQ-CNNI checkout, which
  `vaqem_lib.vqcnni_root()` locates automatically: `$VQCNNI_ROOT`, a sibling
  `../VQ-CNNI/`, or the old `new_paper/` parent directory)
* `results/` — every result file of the validated headline run (`tag=final`),
  the machine-readable number digest, seed-robustness and ablation runs
* `figures/` — generated publication figures (PDF + PNG)
* `paper/` — the LaTeX draft, generated tables, bibliography
* `cache/` — disk cache of exact density-matrix datasets (makes re-runs cheap)

## Environments

* main venv (e.g. `../VQ-CNNI/.venv`, or any environment installed from
  `requirments.txt` with `install.sh`): `numpy`, `scipy`, `matplotlib`,
  `autograd` — everything except the PennyLane cross-check
* `./.venv-pl`: adds `pennylane` + `pennylane-lightning` — only
  `code/validate_simulator.py` needs it

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

Experiment map: `sufficiency` = E1 (Theorem 1 numerics), `sweep` = E2
(headline 16 settings × 4 shot budgets), `shots` = E3 (delta-method validation
and the shot-aware objective), `scaling` = E4 (N = 4…10), `calib` = E5
(calibration budget).  `--exp-root` keeps ablations and repeats out of
`results/` so the paper numbers can never be overwritten by accident.

## Regenerating the paper artefacts

```bash
cd code
python collect_numbers.py --tag final   # results/paper_numbers.{md,json}
python make_tables.py     --tag final   # paper/tables.tex, tables_supplement.tex
python make_figures.py    --tag final   # figures/fig1..fig6 (.pdf/.png)
cd ../paper && python check_tex.py      # manuscript integrity
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
