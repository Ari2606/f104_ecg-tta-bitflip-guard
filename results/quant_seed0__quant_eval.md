Model: 18842 parameters; fp32 73.6 KB; int8 weights + fp32 scales/biases/BN 23.0 KB (BN not folded).
Prediction agreement fp32 vs int8-sim: 98.18% of 49691 beats.

| | accuracy | macro-F1 (N/S/V) | N Se/PPV | S Se/PPV | V Se/PPV |
|---|---|---|---|---|---|
| fp32 | 94.69% | 0.822 | 96.7 / 97.8 | 59.8 / 52.5 | 96.5 / 90.4 |
| int8-sim | 95.15% | 0.818 | 97.7 / 97.3 | 53.7 / 54.7 | 94.9 / 92.3 |

SIMULATED PTQ (fake-quant at every module output, conservative); not measured on int8 kernels.
