# this project: reproduction runbook (Colab order)

Paths below use the layout from earlier sessions; change `D` if yours differs.
```
D=/content/drive/MyDrive/ttaguard_data ; P=/content/drive/MyDrive/ttaguard_out
```
**Everything is resumable** (existing outputs are skipped), so a Colab disconnect only loses the job in flight.
Every script prints `SYNTHETIC` if it could not find real data/checkpoints; never report those numbers.

## 0. Setup
```
!pip install wfdb
%cd /content/ttaguard_repo        # the uploaded folder
```

## 1. Data
```
!python dataset.py --download --data_dir $D/data/mitdb --cache_dir $P/cache
!python download_data.py --root $D/data --which nstdb,incartdb,svdb
```

## 2. Source models
**2a. Independent checkpoints** (for checkpoint-level statistics; use the SAME flags as your main model,
and copy your existing checkpoint to `ckpt_seed0.pt` only if it was trained with `--seed 0` and these flags):
```
!python train_multi_ckpt.py --seeds 0,1,2,3,4,5,6,7 --out_dir $P/ckpts --data_dir $D/data/mitdb --cache_dir $P/cache -- --narrow --epochs 40 <your other flags>
```
**2b. Leave-record-group-out folds** (for hyper-parameter selection that does not depend on 4 records):
```
!python cv_folds.py --data_dir $D/data/mitdb --cache_dir $P/cache --k 5 --ckpt_dir $P/cv --out $P/folds.json --train_args "--narrow --epochs 40 <your other flags>"
# run the 5 printed train_stage1.py commands
```

## 2c. FIRST PRIORITY: why does the source model collapse under shift, and can training fix it?
```
!pip install wfdb
!python download_data.py --root $D/data --which nstdb
# (1) which corruption breaks the current model? (no adaptation; --stride 3 = every 3rd beat, all records kept)
!python diag_corruption.py --checkpoint $P/checkpoint_stage1_narrow.pt --data_dir $D/data/mitdb --cache_dir $P/cache --nstdb_dir $D/data/nstdb --stride 3 --out_dir $P/results/diag_corruption
# (2) train a corruption-aware source model (same flags as before + these; real noise uses the FIRST half of NSTDB only)
!python train_stage1.py --data_dir $D/data/mitdb --cache_dir $P/cache --narrow <your usual flags> --corrupt_aug 0.6 --corrupt_max_severity 0.6 --nstdb_dir $D/data/nstdb --real_noise_prob 0.4 --checkpoint_path $P/checkpoint_robust.pt
# (3) same diagnostic on the new model: compare accuracy under each corruption and clean accuracy (must not drop)
!python diag_corruption.py --checkpoint $P/checkpoint_robust.pt --data_dir $D/data/mitdb --cache_dir $P/cache --nstdb_dir $D/data/nstdb --stride 3 --out_dir $P/results/diag_corruption_robust
```
Evaluation uses the SECOND half of NSTDB, so real-noise results for the robust model are held-out. Parametric-simulator
results for a model trained with the simulator are in-distribution and must be labelled so in the paper.

## 3. Source-model report (needed before any adaptation claim)
```
!python eval_source_model.py --checkpoint $P/ckpts/ckpt_seed0.pt,$P/ckpts/ckpt_seed1.pt,... --data_dir $D/data/mitdb --cache_dir $P/cache --out_dir $P/results/source_model
```

## 4. Hyper-parameter selection (DS1 folds only; DS2 untouched)
```
!python tune_cv.py --folds_json $P/folds.json --data_dir $D/data/mitdb --cache_dir $P/cache --out_dir $P/results/tune_cv \
  --grid '{"pg":{"tau":[0.05,0.1],"gamma":[0.15,0.3]},"tent":{"lr":[1e-4,1e-3]},"rotta":{"lr":[3e-4,1e-3]},"eata":{"d_margin":[0.05,0.4]},"sar":{"lr":[3e-4,1e-3]},"cotta":{"lr":[3e-4,1e-3]}}'
```
If a tuned setting beats the default by a clear margin, pass it with `--hp '{"rotta":{"lr":0.0003}}'` below and report both.

## 5. Benchmark (the paper's numbers). Full DS2 by default (`--n_beats 0`).
```
!python run_benchmark.py --checkpoint $P/ckpts/ckpt_seed0.pt,...,$P/ckpts/ckpt_seed7.pt \
  --data_dir $D/data/mitdb --cache_dir $P/cache --nstdb_dir $D/data/nstdb --incart_dir $D/data/incartdb --svdb_dir $D/data/svdb \
  --scenarios clean,leo,noise_mixed_snr12,noise_mixed_snr6,noise_mixed_snr0,incart,svdb \
  --methods main,pg_gate --seeds 0,1,2 --out_dir $P/bench
!python run_benchmark.py ... --scenarios leo,noise_mixed_snr6 --methods ablation --seeds 0,1,2 --out_dir $P/bench   # ablations / controls
```
It prints `stream <scenario>: N beats, X of Y records` first: for DS2 scenarios Y must be 22 and N about 49.7k.
Slowest methods: beatrhythm (5 steps/beat), cotta (teacher + augmentations), rotta. Shard across notebooks with
disjoint `--methods` / `--checkpoint` lists writing to the same `--out_dir`.

## 6. Statistics and tables
```
!python analyze_results.py --bench_dir $P/bench --ref pg --out_dir $P/analysis --latex_scenarios leo,noise_mixed_snr6,incart,svdb
```
Writes one `.md` per scenario (means, 95% record-bootstrap CIs, paired deltas vs `pg`, Holm-adjusted p) and `main_table.tex`.

## 7. Edge cost and drift bound
```
!python benchmark_edge.py --checkpoint $P/ckpts/ckpt_seed0.pt --data_dir $D/data/mitdb --cache_dir $P/cache --methods main,pg_gate --threads 1 --out_dir $P/edge
!python drift_bound.py --checkpoint $P/ckpts/ckpt_seed0.pt --data_dir $D/data/mitdb --cache_dir $P/cache
```
For real latency copy the folder to the target device (e.g. Raspberry Pi) and run `benchmark_edge.py` there.

## 8. Paper
Fill `paper_related_work_and_framing.tex`: replace the PILOT table with `\input{results/analysis/main_table.tex}`,
paste the edge table and the drift-bound ratio, and re-derive every sentence in "What We Claim" from the new tables.
