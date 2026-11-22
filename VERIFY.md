# Verification guide

Each row links a number in the paper to the script that produces it and the result file that holds it.
Status: **in repo** means the file is committed under `results/`; **Drive** means it is a large output listed in `results/README.md` with its sha256.

## A. Fault study

| Paper claim | Value | Script | Result file | How to check |
|---|---|---|---|---|
| Catastrophic flips per model (seeds 0 to 4) | 6.7, 6.5, 6.4, 6.7, 6.9 % unguarded; 0.00 % guarded | `bitflip_map.py` | `results/bitflip_map_seed*.json` | count flips with balanced-recall drop > 20 % over 4,608 |
| Bit 30 share | 77 to 82 % catastrophic (80.2, 77.3, 76.8, 80.2, 81.5) | `bitflip_map.py` | same | rate for bit 30 per seed |
| Table I, macro-F1 pooled | 0 / 2 / 5 flips: frozen .830/.667/.534; pg .745/.622/.513; guarded .852/.853/.853 | `run_fault_experiment.py` | `results/fault_stream_*.json` | pool confusion over 5 models x 3 fault seeds, 13,200 beats each |
| Beats on corrupted state, 5/1,000 | 90.8 % frozen, 0.0 % guarded | `run_fault_experiment.py` | same, key `corrupt_state_frac` | mean over runs |
| Accuracy at 5/1,000 | 96.1 % guarded, 92.2 % frozen, 96.0 % frozen fault-free | same | same, key `accuracy` | mean over runs |
| Streaming bit set | random bit of 32 | `fault_injection.py` | `BIT_SETS["all"] = range(32)` | read the config in each record (`bits="all"`) |
| Map bit set | 12 positions: 31, 30..23, 22, 18, 12 | `bitflip_map.py` | `DEFAULT_BITS` | read the constant |
| Fixed state vs adapted set | 2.8 to 5.7 % (conv/head), 6.7 to 7.5 % (other BN affine), 21.3 to 23.3 % (BN running stats) | `fixed_weights` script | `results/fixed_weights*.json` | per-group catastrophic rate |

## B. Accuracy

| Paper claim | Value | Script | Result file |
|---|---|---|---|
| DS1 out-of-fold macro-F1 | 70.1 [60.3, 79.5] | `eval_source_model.py`, `ci_eval.py` | `noise_compare_summary.json` |
| DS2 five-seed mean | 80.1 (82.2, 72.1, 84.5, 78.3, 83.3) | `eval_source_model.py` | `source_model.json` |
| DS2 without record 232 | 73.2 | `eval_source_model.py` with record 232 removed | recomputed from per-record confusion |
| Decision tree | 59.5 [52.0, 67.9] | baseline-tree script | `noise_compare_summary.json` |
| Forest clean / noise-trained | 72.4 / 70.2; CNN minus forest -0.1 [-3.9, 3.5] | noise-compare script | same |

## C. Causal pipeline (`results/improve_detector.json` is in this repo)

| Scenario | DS2 N/S/V F1 | Difference from A |
|---|---|---|
| A annotated peaks, offline | 80.0 | |
| B XQRS peaks | 78.3 | -1.7 |
| C causal filter | 76.4 | -3.6 |
| D XQRS + causal | 75.4 | -4.6 [-7.0, -0.5] |
| E causal Pan-Tompkins | 68.9 | -11.1 [-15.4, -5.3] |
| E tuned on DS1 (threshold 0.5, shift 4) | 70.8 | -9.2 [-16.5, -3.3] |

The first 10 s of each record are excluded (49,440 of 49,691 beats scored); this is why A is 80.0 and not 80.1.

## D. Edge (Raspberry Pi 4 Model B, ONNX Runtime, one thread, batch 1)

| Paper claim | Value | Script | Result file |
|---|---|---|---|
| Latency per beat, float32 | p50 0.493 / 0.523 ms, p95 up to 0.713, max 0.921 | `bench_pi.py` | `results/pi_bench_*.json` |
| Latency per beat, INT8 | p50 0.396 / 0.406 ms | `bench_pi.py` | same |
| Added memory | 8.6 MB float32, 1.8 MB INT8 | `bench_pi.py` | same |
| Balanced recall | 85.3 % float32, 82.0 % INT8 | `bench_pi.py` | same |
| Pipeline per 30 s segment | median 64 to 69 ms, p95 85 to 94 ms | `pi_pipeline_bench.py` | `results/pi_pipeline_*.json` |
| Model size | ONNX 70 KB float32 / 56 KB INT8; estimated stored size 23.6 KB (15,504 int8 weights + 4 x (869 + 453 + 832) bytes) | `export_onnx.py`, `quant_eval.py` | `results/export_*.json` |
| Compute | 739k MACs frozen or guarded, 3,599k adapting | `edge_budget.py` | `results/edge_budget.json` |

Notes: the second Pi run had load average about 1.8 and the CPU at 1800 MHz (first run 1400 MHz). "Two runs" means two invocations of three passes each.

## E. Reproduce one number end to end (Colab)

```python
!pip install -r requirements.txt wfdb
!python dataset.py --download --data_dir $D/data/mitdb --cache_dir $P/cache
!python eval_source_model.py --checkpoint $P/ckpts/final_tmplaug_seed0.pt --data_dir $D/data/mitdb --cache_dir $P/cache --out_dir $P/results/source_model
```
The printed DS2 macro-F1 for seed 0 should be 82.2.
