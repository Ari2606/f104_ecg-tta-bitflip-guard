Fixed-state single-bit exposure (UNGUARDED; guard does not cover these values). 3600 sampled flips, 600 class-balanced beats, unflipped balanced recall 85.3%.

| group | values in group | flips sampled | catastrophic % [95% Wilson] | mean drop | max drop | catastrophic % at bit 30 |
|---|---|---|---|---|---|---|
| depthwise_stem_conv | 1552 | 600 | 5.5 [3.9, 7.6] | 0.0330 | 0.745 | 62.5 |
| bn_affine_other | 448 | 600 | 7.0 [5.2, 9.3] | 0.0398 | 0.853 | 77.8 |
| pointwise_conv | 5376 | 600 | 4.2 [2.8, 6.1] | 0.0245 | 0.762 | 52.1 |
| projection_conv | 6144 | 600 | 4.7 [3.2, 6.7] | 0.0266 | 0.853 | 54.9 |
| classifier_head | 2469 | 600 | 4.8 [3.4, 6.9] | 0.0290 | 0.853 | 63.0 |
| bn_running_stats | 832 | 600 | 21.8 [18.7, 25.3] | 0.1177 | 0.853 | 23.1 |

All groups pooled (unweighted by group size): 8.0% catastrophic. Fixed-state values: 16821; adapted values: 384.
