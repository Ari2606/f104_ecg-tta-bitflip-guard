
### flips/1000 beats = 0.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 1 | 0 | 96.3 | 0.870±0.000 | 60.4 | 96.6 | 0.0 | 90.5 |
| scrub_100 | 1 | 0 | 96.3 | 0.870±0.000 | 60.4 | 96.6 | 0.0 | 90.5 |
| pull_guard | 1 | 0 | 96.3 | 0.870±0.000 | 60.4 | 96.6 | 0.0 | 90.5 |
| pg | 1 | 0 | 95.8 | 0.765±0.000 | 22.9 | 92.1 | 0.0 | 93.4 |
| pg_cbkl_guard_k2 | 1 | 0 | 96.3 | 0.887±0.000 | 65.9 | 96.9 | 0.0 | 90.3 |

### flips/1000 beats = 2.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 3 | 26 | 93.7±2.0 | 0.681±0.154 | 26.2±19.7 | 66.4±29.0 | 38.0±33.8 | 94.1±2.9 |
| scrub_100 | 3 | 26 | 96.3±0.1 | 0.866±0.005 | 59.0±2.0 | 96.5±0.1 | 0.7±0.7 | 90.5±0.1 |
| pull_guard | 3 | 26 | 96.3±0.0 | 0.870±0.000 | 60.4±0.0 | 96.6±0.0 | 0.0±0.0 | 90.5±0.0 |
| pg | 3 | 26 | 93.8±1.9 | 0.637±0.125 | 12.9±8.4 | 63.2±28.1 | 21.1±22.0 | 95.5±2.1 |
| pg_cbkl_guard_k2 | 3 | 26 | 96.3±0.0 | 0.887±0.000 | 65.9±0.0 | 96.9±0.0 | 0.0±0.0 | 90.3±0.0 |

### flips/1000 beats = 5.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 3 | 69 | 92.3±2.5 | 0.532±0.184 | 10.4±11.6 | 41.5±39.5 | 90.8±2.3 | 96.8±3.0 |
| scrub_100 | 3 | 69 | 96.2±0.1 | 0.865±0.006 | 58.5±2.3 | 96.4±0.4 | 3.9±1.2 | 90.6±0.1 |
| pull_guard | 3 | 69 | 96.3±0.0 | 0.870±0.000 | 60.5±0.1 | 96.7±0.1 | 0.0±0.0 | 90.5±0.0 |
| pg | 3 | 69 | 92.3±2.5 | 0.523±0.177 | 9.1±9.7 | 39.5±38.3 | 42.1±30.7 | 97.1±2.7 |
| pg_cbkl_guard_k2 | 3 | 69 | 96.3±0.0 | 0.887±0.000 | 65.8±0.1 | 96.9±0.0 | 0.0±0.0 | 90.3±0.0 |

mean±sd over fault seeds (x checkpoints). 'corrupted-state' = max |adapted - source| > 0.5 (or non-finite) at the end of a beat. Same fault seed => identical flip schedule for every method. Accelerated fault rates are a stress test, not an orbit prediction.
