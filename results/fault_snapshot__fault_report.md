
### flips/1000 beats = 2.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| scrub_100 | 3 | 26 | 94.4±2.2 | 0.692±0.163 | 26.1±25.5 | 72.8±32.2 | 2.1±0.9 | 93.9±3.1 |
| pull_guard | 3 | 26 | 94.4±2.2 | 0.692±0.164 | 26.1±25.5 | 73.1±32.6 | 0.0±0.0 | 93.9±3.1 |
| pg_cbkl_guard_k2 | 3 | 26 | 94.3±2.2 | 0.698±0.170 | 28.2±28.3 | 73.3±32.8 | 0.0±0.0 | 93.8±3.3 |

mean±sd over fault seeds (x checkpoints). 'corrupted-state' = max |adapted - source| > 0.5 (or non-finite) at the end of a beat. Same fault seed => identical flip schedule for every method. Accelerated fault rates are a stress test, not an orbit prediction.
