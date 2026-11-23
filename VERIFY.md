# Verification guide

Each row links a number in the paper to the script that produces it and the result files that hold it. All result files are in `results/`; file names there are the original folder name and file name joined by `__` (for example `fixed_weights_seed0__fixed_weights_summary.json`). Model checkpoints are in `checkpoints/`.

Streaming-fault run files are named `run_<method>_r<rate>_s<fault seed>_all.json`: `r0`, `r2`, `r5` are 0, 2 and 5 flips per 1,000 beats, `s0` to `s2` are the three fault seeds, and `all` means a random bit of all 32.

## A. Fault study

| Paper claim | Value | Script | Result files |
|---|---|---|---|
| Catastrophic flips per model (seeds 0 to 4) | 6.7, 6.5, 6.4, 6.7, 6.9 % unguarded; 0.00 % guarded | `bitflip_map.py` | `bitflip_final_tmplaug_seed0..4__bitflip_rows.csv`, `..._bitflip_summary.md` |
| Bit 30 share | 77 to 82 % catastrophic (80.2, 77.3, 76.8, 80.2, 81.5) | `bitflip_map.py` | same |
| Map bit set | 12 positions: 31, 30 to 23, 22, 18, 12 (4,608 flips per model) | `bitflip_map.py` | constant `DEFAULT_BITS` |
| Table I, macro-F1 pooled over 5 models | 0 / 2 / 5 flips: frozen .830 / .667 / .534; pg .745 / .622 / .513; guarded .852 / .853 / .853 | `run_fault_experiment.py` | `fault__final_tmplaug_seed0__run_*` (seed 0) and `fault_seeds1to4__final_tmplaug_seed1..4__run_*`; summary in `fault__fault_report.md` |
| Beats on a corrupted state at 5/1,000 | 90.8 % frozen, 0.0 % guarded | `run_fault_experiment.py` | same run files, key `corrupt_state_frac` |
| Accuracy at 5/1,000 | 96.1 % guarded, 92.2 % frozen; 96.0 % frozen without faults | `run_fault_experiment.py` | same run files, key `accuracy` |
| Streaming bit set | random bit of 32 | `fault_injection.py` | `BIT_SETS["all"]`; the `_all` suffix of every run file |
| Blind scrubber | .866 at 2 and .865 at 5 flips per 1,000 (model 0 only) | `run_fault_experiment.py` | `fault__final_tmplaug_seed0__run_scrub_100_*` |
| Snapshot corruption and three-copy vote (exploratory, one model) | see reports | `run_fault_experiment.py` | `fault_snapshot__*`, `fault_snapshot_tmr__*`, `fault_snapshot_tmr_p75__*` |
| Fixed state vs adapted set | 2.8 to 5.7 % (conv/head), 6.7 to 7.5 % (other BN affine), 21.3 to 23.3 % (BN running statistics) | fixed-weights script (not archived) | `fixed_weights_seed0..4__fixed_weights_summary.json`, `.md`, `_rows.csv` |

## B. Accuracy

| Paper claim | Value | Script | Result files |
|---|---|---|---|
| DS1 out-of-fold macro-F1 | 70.1 [60.3, 79.5] | noise-compare script (not archived) | `noise_compare_ds1__noise_compare_summary.json` |
| DS2 per-model and five-seed mean | 80.1 (82.2, 72.1, 84.5, 78.3, 83.3) | `eval_source_model.py`, `protocol_eval.py`, `ci_eval.py` | `final_protocols_tmplaug__protocols.json`, `ci_tmplaug__ci.json` |
| DS2 seed 0: confusion, SVEB and VEB metrics | macro-F1 0.822; SVEB Se 59.8, +P 52.5; VEB Se 96.5, +P 90.4 | `eval_source_model.py` | `see the confusion matrix at the end of this section` |
| DS2 without record 232 | 73.2 | recomputed from the per-record confusion matrices | not stored as a file |
| Decision tree | 59.5 [52.0, 67.9] | baseline-tree script (not archived) | `baseline_tree_ds1__baseline_summary.json` |
| Forest clean / noise-trained | 72.4 / 70.2; CNN minus forest -0.1 [-3.9, 3.5] | noise-compare script (not archived) | `noise_compare_ds1__noise_compare_summary.json`, `forest_size_sweep__forest_size_sweep.json` |
| Noise fingerprint of the final models | similar to noise-trained fold models | noise-fingerprint script (not archived) | `noise_fingerprint__noise_fingerprint.json` |

## C. Causal pipeline

| Scenario | DS2 N/S/V F1 | Difference from A |
|---|---|---|
| A annotated peaks, offline | 80.0 | |
| B XQRS peaks | 78.3 | -1.7 |
| C causal filter | 76.4 | -3.6 |
| D XQRS + causal | 75.4 | -4.6 [-7.0, -0.5] |
| E causal Pan-Tompkins | 68.9 | -11.1 [-15.4, -5.3] |
| E tuned on DS1 (threshold 0.5, shift 4) | 70.8 | -9.2 [-16.5, -3.3] |

Files: `realistic_pipeline__realistic_pipeline.json`, `realistic_pipeline__improve_detector.json`. The pipeline script is not archived; `pan_tompkins.py` and `pi_pipeline_bench.py` are. The first 10 s of each record are excluded (49,440 of 49,691 beats scored); this is why A is 80.0 and not 80.1.

## D. Edge

| Paper claim | Value | Script | Result files |
|---|---|---|---|
| Model size | ONNX 70 KB float32 / 56 KB INT8; estimated stored size 23.6 KB | `export_onnx.py`, `quant_eval.py` | `quant_seed0__quant_eval.json`, `edge_final__edge_costs.json` |
| Compute | 739k MACs frozen or guarded, 3,599k adapting | `edge_budget.py` | `edge_final__edge_costs.json` |
| Effect of INT8 weights on F1 | at most 0.3 points | `quant_eval.py` | `quant_seed0__quant_eval.json`, `int8_check__int8_check.json` |
| Colab host timings | 5.30, 7.61, 13.74 and 14.57 ms | `edge_profiler.py` | `edge_final__edge_costs.json`, `edge_final__host.json` |
| Raspberry Pi per-beat latency, memory, balanced recall; pipeline per 30 s segment | 0.40 to 0.41 ms INT8, 0.49 to 0.52 ms float32; 64 to 69 ms per segment | `bench_pi.py`, `pi_pipeline_bench.py`, `make_segments.py`, `pan_tompkins.py` | the Pi output files were on the device and are not in this repository |

The Pi scripts were hashed on the machine they were copied from to the device: `bench_pi.py` 60f3efe4..., `pi_pipeline_bench.py` 5dcdc8eb..., `make_segments.py` bed19f9a..., `pan_tompkins.py` 18abc095....

## E. Reproduce one number end to end (Colab)

### DS2 seed 0 confusion matrix (rows: true class, columns: predicted; order N, S, V, F, Q)

|   | N | S | V | F | Q |
|---|---|---|---|---|---|
| N | 42789 | 970 | 190 | 237 | 53 |
| S | 635 | 1099 | 97 | 6 | 0 |
| V | 24 | 24 | 3108 | 61 | 3 |
| F | 292 | 0 | 39 | 54 | 3 |
| Q | 4 | 0 | 3 | 0 | 0 |

49,691 beats, 22 records. N/S/V macro-F1 0.822. SVEB: sensitivity 59.8, PPV 52.5. VEB: sensitivity 96.5, PPV 90.4.

```
!pip install -r requirements.txt wfdb
!python dataset.py --download --data_dir $D/data/mitdb --cache_dir $P/cache
!python eval_source_model.py --checkpoint checkpoints/final_tmplaug_seed0.pt --data_dir $D/data/mitdb --cache_dir $P/cache --out_dir out/source_model
```

The printed DS2 macro-F1 (N/S/V) for seed 0 should be 0.822.
