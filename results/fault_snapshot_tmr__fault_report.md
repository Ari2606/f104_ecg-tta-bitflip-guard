
### flips/1000 beats = 2.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| scrub_100 | 3 | 26 | 96.3±0.0 | 0.869±0.001 | 60.3±0.2 | 96.3±0.5 | 1.1±0.4 | 90.5±0.0 |
| pull_guard | 3 | 26 | 96.3±0.0 | 0.869±0.000 | 60.4±0.1 | 96.6±0.0 | 0.0±0.0 | 90.5±0.0 |
| pg_cbkl_guard_k2 | 3 | 26 | 96.3±0.0 | 0.887±0.000 | 65.9±0.0 | 96.9±0.0 | 0.0±0.0 | 90.3±0.0 |

mean±sd over fault seeds (x checkpoints). 'corrupted-state' = max |adapted - source| > 0.5 (or non-finite) at the end of a beat. Same fault seed => identical flip schedule for every method. Accelerated fault rates are a stress test, not an orbit prediction.
