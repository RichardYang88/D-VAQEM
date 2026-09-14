"""Smoke test for the revised E1/E2 code paths (writes to /tmp/smoke_results)."""
import os
import sys

sys.path.insert(0, "/home/yqc/github/VQ-CNNI/new_paper/code")
import run_experiments as rx                                    # noqa: E402

rx.OUT = "/tmp/smoke_results"
os.makedirs(rx.OUT, exist_ok=True)

print("== E1 (N=4, n_phi=9, two strengths) ==", flush=True)
rows = rx.exp_sufficiency(Ns=(4,), n_phi=9, workers=4, rng_seed=0,
                          strengths=(0.0, 0.01))
print("E1 rows:", len(rows), "keys:", sorted(rows[0]), flush=True)

print("\n== E2 quick (N=6, 5 folds, n_oracle=121) ==", flush=True)
rows2, metas = rx.exp_sweep(N=6, n_cal=13, n_tst=13, shots_list=(1024,),
                            n_trials=9, workers=4, iters=600, retrain_iters=200,
                            quick=True, tag="smoke", n_oracle=121)
methods = sorted({r["method"] for r in rows2})
print("methods:", methods, flush=True)
m = metas[rows2[0]["setting"]]
for k in ("zne_rich", "zne_poly1", "zne_poly2"):
    print(f"  {k}: {m.get(k, 'MISSING')}", flush=True)
print("  best_dvaqem:", m["best_dvaqem"], flush=True)
for k, v in m.items():
    if k.startswith("dvaqem_"):
        print(f"  {k}: holdout {v['holdout_mse']:.3e} (inf {v['holdout_mse_inf']:.3e})"
              f" best_iter {v['best_iter']} warm {v.get('warm_start', '-')}",
              flush=True)
print("SMOKE OK", flush=True)
