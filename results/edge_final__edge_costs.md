Adapted parameters (all methods, sparse dw_bn scope): 384 of 16373 (2.3%). MCU latency = total MACs / 150 MMAC/s (an assumption, not a measurement); host latency = this machine.

| method | MACs/beat (k) | no-grad fwd | grad fwd | bwd (est) | MCU ms (est) | host p50 / p95 ms | persistent state KB | anchor pass | adapt rate |
|---|---|---|---|---|---|---|---|---|---|
| source_only | 739 | 739 | 0 | 0 | 4.92 | 5.30 / 6.80 | 0.0 | no | 0.00 |
| pull_guard | 739 | 739 | 0 | 0 | 4.92 | 7.61 / 9.95 | 0.0 | no | 0.00 |
| pg | 3599 | 1572 | 644 | 644 | 23.99 | 13.74 / 18.58 | 1.5 | yes | 1.00 |
| pg_cbkl_guard_k2 | 3599 | 1572 | 644 | 644 | 23.99 | 14.57 / 18.56 | 1.5 | yes | 1.00 |
