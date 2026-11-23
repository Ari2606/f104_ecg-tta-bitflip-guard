Fixed-state single-bit exposure (UNGUARDED; guard does not cover these values). 3600 sampled flips, 600 class-balanced beats, unflipped balanced recall 71.5%.

| group | values in group | flips sampled | catastrophic % [95% Wilson] | mean drop | max drop | catastrophic % at bit 30 |
|---|---|---|---|---|---|---|
| depthwise_stem_conv | 1552 | 600 | 5.7 [4.1, 7.8] | 0.0217 | 0.482 | 66.7 |
| bn_affine_other | 448 | 600 | 6.7 [4.9, 9.0] | 0.0262 | 0.382 | 74.1 |
| pointwise_conv | 5376 | 600 | 4.5 [3.1, 6.5] | 0.0187 | 0.715 | 54.2 |
| projection_conv | 6144 | 600 | 4.5 [3.1, 6.5] | 0.0176 | 0.498 | 52.9 |
| classifier_head | 2469 | 600 | 4.2 [2.8, 6.1] | 0.0210 | 0.715 | 54.3 |
| bn_running_stats | 832 | 600 | 21.3 [18.2, 24.8] | 0.0839 | 0.712 | 38.5 |

All groups pooled (unweighted by group size): 7.8% catastrophic. Fixed-state values: 16821; adapted values: 384.
