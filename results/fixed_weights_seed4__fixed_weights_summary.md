Fixed-state single-bit exposure (UNGUARDED; guard does not cover these values). 3600 sampled flips, 600 class-balanced beats, unflipped balanced recall 86.3%.

| group | values in group | flips sampled | catastrophic % [95% Wilson] | mean drop | max drop | catastrophic % at bit 30 |
|---|---|---|---|---|---|---|
| depthwise_stem_conv | 1552 | 600 | 5.0 [3.5, 7.0] | 0.0272 | 0.753 | 56.2 |
| bn_affine_other | 448 | 600 | 7.5 [5.7, 9.9] | 0.0429 | 0.863 | 83.3 |
| pointwise_conv | 5376 | 600 | 4.7 [3.2, 6.7] | 0.0243 | 0.862 | 54.2 |
| projection_conv | 6144 | 600 | 4.2 [2.8, 6.1] | 0.0231 | 0.707 | 49.0 |
| classifier_head | 2469 | 600 | 5.7 [4.1, 7.8] | 0.0352 | 0.863 | 73.9 |
| bn_running_stats | 832 | 600 | 23.0 [19.8, 26.5] | 0.1189 | 0.863 | 28.8 |

All groups pooled (unweighted by group size): 8.3% catastrophic. Fixed-state values: 16821; adapted values: 384.
