# Guarding test-time adaptation against bit flips: ECG beat classifier

Code and result files for the paper *Edge-Based Arrhythmia Detection for Deep-Space Missions: Guarding Test-Time Adaptation Against Bit Flips*
(Dr. Gokul Kannan Sadasivam, Ria S Nair, R V Sukeerthi, Bhumi Sinha, Satwik Bankapur; Dept. of CSE, PES University, Bangalore, India).

## What this is

A small CNN classifies ECG beats as N, S or V (MIT-BIH, inter-patient split). Only 384 depthwise batch-norm values (12,288 bits) adapt at test time, without labels.
Bit flips in those writable values can destroy the classifier. A **guard** clamps each adapted value to source value +/- k * max(std of its source tensor, small floor), with k = 2,
and resets non-finite values to the source. The study measures the damage, the guard, its limits, and the edge cost on a Raspberry Pi 4.

```
ECG -> filter -> R peak -> beat window (256) + RR features -> GUARD (clamp to source copy) -> CNN predict -> test-time adaptation of 384 values
```

## Scope and honesty notes

- Fault rates are **accelerated stress levels; no radiation testing was done.**
- Primary accuracy is DS1 out-of-fold. DS2 was seen during design and is secondary.
- Pi timing covers the frozen classifier and a forward-only pipeline on pre-recorded segments. The guard and adaptation were not timed on the device. No energy was measured.
- Fixed weights are assumed to be in protected memory; the guard covers only the adapted values.

## Headline results (every number is traced in [VERIFY.md](VERIFY.md))

| Quantity | Value |
|---|---|
| Catastrophic single-bit flips, unguarded / guarded | 6.4 to 6.9 % / 0.00 % (4,608 flips per model, 5 models) |
| Macro-F1 at 5 flips per 1,000 beats, frozen / guarded | .534 / .853 (pooled over 5 models) |
| Beats on a corrupted state at 5/1,000, unguarded / guarded | 90.8 % / 0.0 % |
| Accuracy at 5/1,000, guarded / frozen (fault-free frozen) | 96.1 % / 92.2 % (96.0 %) |
| Macro-F1 N/S/V, DS1 out-of-fold | 70.1 [60.3, 79.5] |
| Macro-F1 N/S/V, DS2 five-seed mean | 80.1 (73.2 without record 232) |
| Raspberry Pi 4, per beat, INT8 / float32 | 0.40 to 0.41 ms / 0.49 to 0.52 ms |
| Model file (ONNX), INT8 / float32 | 56 KB / 70 KB |
| Full pipeline per 30 s segment | 64 to 69 ms |

## Repository layout

| Path | Purpose |
|---|---|
| `dataset.py`, `download_data.py` | MIT-BIH and NSTDB download, beat cache |
| `base_cnn.py`, `train_stage1.py`, `train_multi_ckpt.py`, `cv_folds.py` | model and training |
| `eval_source_model.py`, `ci_eval.py`, `protocol_eval.py` | accuracy and record-clustered intervals |
| `bitflip_map.py` | exhaustive single-bit-flip map (12 selected bit positions) |
| `fault_injection.py`, `run_fault_experiment.py`, `safety_controller.py` | streaming faults and the guard |
| `export_onnx.py`, `quant_eval.py` | ONNX export, INT8 |
| `bench_pi.py`, `pi_pipeline_bench.py`, `make_segments.py`, `pan_tompkins.py` | Raspberry Pi timing and pipeline |
| `results/` | result JSON files that the paper numbers come from |
| `MANIFEST.sha256` | file hashes (`sha256sum -c MANIFEST.sha256`) |
| `RUNBOOK.md` | order of the Colab runs |
| `VERIFY.md` | **claim by claim check of every number in the paper** |

## Quick start

```bash
git clone https://github.com/<user>/ecg-tta-bitflip-guard && cd ecg-tta-bitflip-guard
pip install -r requirements.txt
sha256sum -c MANIFEST.sha256          # file integrity
python safety_controller.py           # self-test of the guard logic; prints SYNTHETIC if real data is absent
```

Data (MIT-BIH Arrhythmia Database, MIT-BIH Noise Stress Test Database) comes from PhysioNet and is not redistributed. Follow `RUNBOOK.md` for the full runs.
Every script prints `SYNTHETIC` when it cannot find real data or checkpoints; do not report such numbers.

## Environment

Python 3.10. Colab for training and fault studies. Raspberry Pi 4 Model B Rev 1.5 (8 GB), Ubuntu 22.04 64-bit, Python 3.10.12, ONNX Runtime 1.23.2, NumPy 2.2.6, one thread.

## License

MIT (see `LICENSE`).
