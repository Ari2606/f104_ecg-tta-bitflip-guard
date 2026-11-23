
### flips/1000 beats = 0.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 4 | 0 | 95.9±0.5 | 0.820±0.061 | 45.5±19.7 | 95.1±0.4 | 0.0±0.0 | 91.2±1.0 |
| pull_guard | 4 | 0 | 95.9±0.5 | 0.820±0.061 | 45.5±19.7 | 95.1±0.4 | 0.0±0.0 | 91.2±1.0 |
| pg | 4 | 0 | 95.6±0.8 | 0.740±0.086 | 22.0±21.8 | 90.6±0.8 | 0.0±0.0 | 93.3±0.8 |
| pg_cbkl_guard_k2 | 4 | 0 | 96.0±0.7 | 0.843±0.078 | 57.7±27.8 | 94.8±0.3 | 0.0±0.0 | 90.7±1.4 |

### flips/1000 beats = 2.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 12 | 26 | 93.5±2.0 | 0.663±0.149 | 23.3±20.6 | 65.5±28.8 | 39.9±32.2 | 94.0±2.9 |
| pull_guard | 12 | 26 | 95.9±0.5 | 0.820±0.061 | 45.5±19.7 | 95.1±0.4 | 0.0±0.0 | 91.2±1.0 |
| pg | 12 | 26 | 93.6±1.9 | 0.618±0.129 | 11.4±15.6 | 62.0±27.9 | 21.1±22.0 | 95.5±2.1 |
| pg_cbkl_guard_k2 | 12 | 26 | 96.0±0.7 | 0.844±0.078 | 57.7±27.8 | 94.8±0.3 | 0.0±0.0 | 90.7±1.4 |

### flips/1000 beats = 5.0  (bits: all)

| method | n | flips/run | accuracy % | macro-F1 (N/S/V) | S recall % | V recall % | corrupted-state beats % | pred=N % |
|---|---|---|---|---|---|---|---|---|
| frozen | 12 | 69 | 92.2±2.6 | 0.535±0.195 | 14.0±20.1 | 40.7±39.0 | 90.8±2.3 | 96.4±3.5 |
| pull_guard | 12 | 69 | 95.9±0.5 | 0.820±0.061 | 45.4±19.7 | 95.0±0.4 | 0.0±0.0 | 91.2±1.0 |
| pg | 12 | 69 | 92.2±2.5 | 0.510±0.177 | 8.5±15.7 | 38.4±38.1 | 42.1±30.7 | 97.2±2.8 |
| pg_cbkl_guard_k2 | 12 | 69 | 96.0±0.7 | 0.844±0.078 | 57.7±27.8 | 94.8±0.3 | 0.0±0.0 | 90.7±1.4 |

mean±sd over fault seeds (x checkpoints). 'corrupted-state' = max |adapted - source| > 0.5 (or non-finite) at the end of a beat. Same fault seed => identical flip schedule for every method. Accelerated fault rates are a stress test, not an orbit prediction.
