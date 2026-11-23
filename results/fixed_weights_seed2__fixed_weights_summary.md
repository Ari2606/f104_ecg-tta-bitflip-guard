Fixed-state single-bit exposure (UNGUARDED; guard does not cover these values). 3600 sampled flips, 600 class-balanced beats, unflipped balanced recall 88.7%.

| group | values in group | flips sampled | catastrophic % [95% Wilson] | mean drop | max drop | catastrophic % at bit 30 |
|---|---|---|---|---|---|---|
| depthwise_stem_conv | 1552 | 600 | 4.5 [3.1, 6.5] | 0.0258 | 0.582 | 52.1 |
| bn_affine_other | 448 | 600 | 7.5 [5.7, 9.9] | 0.0462 | 0.887 | 81.5 |
| pointwise_conv | 5376 | 600 | 3.5 [2.3, 5.3] | 0.0212 | 0.873 | 41.7 |
| projection_conv | 6144 | 600 | 4.3 [3.0, 6.3] | 0.0254 | 0.867 | 51.0 |
| classifier_head | 2469 | 600 | 4.7 [3.2, 6.7] | 0.0326 | 0.887 | 60.9 |
| bn_running_stats | 832 | 600 | 23.3 [20.1, 26.9] | 0.1311 | 0.887 | 30.8 |

All groups pooled (unweighted by group size): 8.0% catastrophic. Fixed-state values: 16821; adapted values: 384.
