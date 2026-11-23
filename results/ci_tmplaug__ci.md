| records | class | Se | PPV | F1 |
|---|---|---|---|---|
| all DS2 | S | 52.3 [31.5, 63.5] | 50.7 [11.3, 80.6] | 50.8 [17.1, 61.6] |
| all DS2 | V | 95.2 [91.8, 97.6] | 89.3 [70.7, 97.4] | 92.1 [80.6, 96.7] |
| 232 removed | S | 50.0 [21.3, 69.0] | 20.9 [7.4, 33.0] | 29.4 [10.6, 40.7] |
| 232 removed | V | 95.2 [92.0, 97.4] | 89.5 [69.3, 97.6] | 92.2 [80.2, 96.7] |

Mean over seeds with 95% CI from a record-level bootstrap of DS2 records (22 records, so CIs are wide by construction). Strict 5-class scoring.

| first checkpoint, all DS2 beats | acc | S Se/PPV/F1 | V Se/PPV/F1 |
|---|---|---|---|
| fp32 | 94.7 | 59.8/52.5/55.9 | 96.5/90.4/93.4 |
| int8 (ONNX QDQ) | 95.1 | 51.5/55.8/53.6 | 94.0/92.3/93.1 |

int8 vs fp32 argmax agreement on all DS2 beats: 98.27%
