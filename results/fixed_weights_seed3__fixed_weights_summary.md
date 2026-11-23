Fixed-state single-bit exposure (UNGUARDED; guard does not cover these values). 3600 sampled flips, 600 class-balanced beats, unflipped balanced recall 78.5%.

| group | values in group | flips sampled | catastrophic % [95% Wilson] | mean drop | max drop | catastrophic % at bit 30 |
|---|---|---|---|---|---|---|
| depthwise_stem_conv | 1552 | 600 | 4.3 [3.0, 6.3] | 0.0222 | 0.782 | 52.1 |
| bn_affine_other | 448 | 600 | 7.0 [5.2, 9.3] | 0.0348 | 0.785 | 77.8 |
| pointwise_conv | 5376 | 600 | 2.8 [1.8, 4.5] | 0.0152 | 0.723 | 33.3 |
| projection_conv | 6144 | 600 | 4.2 [2.8, 6.1] | 0.0217 | 0.772 | 49.0 |
| classifier_head | 2469 | 600 | 4.3 [3.0, 6.3] | 0.0224 | 0.785 | 56.5 |
| bn_running_stats | 832 | 600 | 22.7 [19.5, 26.2] | 0.1122 | 0.785 | 28.8 |

All groups pooled (unweighted by group size): 7.6% catastrophic. Fixed-state values: 16821; adapted values: 384.
