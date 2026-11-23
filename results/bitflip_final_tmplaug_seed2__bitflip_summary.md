Exhaustive single-bit-flip map: 4608 flips, evaluated on 600 class-balanced beats (unflipped balanced recall 88.7%).

Overall: catastrophic (drop>20%) unguarded 6.4% of flips, guarded 0.00%. Worst single flip: unguarded drop 0.887, guarded 0.017. Mean prediction change: unguarded 5.33%, guarded 0.26%.

| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |
|---|---|---|---|---|---|---|---|---|
| 31 | sign | 384 | 0.0034 | 0.155 | 0.0 | 0.0006 | 0.010 | 0.00 |
| 30 | exponent | 384 | 0.4440 | 0.887 | 76.8 | 0.0006 | 0.017 | 0.00 |
| 29 | exponent | 384 | 0.0038 | 0.093 | 0.0 | 0.0006 | 0.010 | 0.00 |
| 28 | exponent | 384 | 0.0038 | 0.093 | 0.0 | 0.0006 | 0.010 | 0.00 |
| 27 | exponent | 384 | 0.0038 | 0.093 | 0.0 | 0.0006 | 0.010 | 0.00 |
| 26 | exponent | 384 | 0.0052 | 0.217 | 0.3 | 0.0006 | 0.012 | 0.00 |
| 25 | exponent | 384 | 0.0048 | 0.240 | 0.3 | 0.0008 | 0.017 | 0.00 |
| 24 | exponent | 384 | 0.0026 | 0.068 | 0.0 | 0.0007 | 0.012 | 0.00 |
| 23 | exponent | 384 | 0.0024 | 0.143 | 0.0 | 0.0007 | 0.010 | 0.00 |
| 22 | mantissa | 384 | 0.0001 | 0.037 | 0.0 | 0.0002 | 0.010 | 0.00 |
| 18 | mantissa | 384 | 0.0002 | 0.005 | 0.0 | 0.0002 | 0.005 | 0.00 |
| 12 | mantissa | 384 | 0.0000 | 0.000 | 0.0 | 0.0000 | 0.000 | 0.00 |

By parameter type:

| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |
|---|---|---|---|---|
| dw_bn.weight | 2304 | 8.3 | 0.553 | 0.010 |
| dw_bn.bias | 2304 | 4.6 | 0.887 | 0.017 |

Most damaging unguarded flips:

| param | element | bit | old -> new | drop | guarded drop |
|---|---|---|---|---|---|
| backbone.block1.branches.0.dw_bn.bias | 3 | 30 | 0.0204 -> 6.96e+36 | 0.887 | 0.017 |
| backbone.block1.branches.0.dw_bn.bias | 6 | 30 | 0.0069 -> 2.35e+36 | 0.887 | 0.003 |
| backbone.block1.branches.0.dw_bn.bias | 7 | 30 | 0.0341 -> 1.16e+37 | 0.887 | -0.003 |
| backbone.block1.branches.1.dw_bn.bias | 14 | 30 | 0.0204 -> 6.94e+36 | 0.887 | 0.012 |
| backbone.block1.branches.2.dw_bn.bias | 3 | 30 | 0.00286 -> 9.74e+35 | 0.887 | -0.005 |
| backbone.block2.branches.0.dw_bn.bias | 14 | 30 | 0.0295 -> 1.01e+37 | 0.887 | 0.008 |
| backbone.block2.branches.0.dw_bn.bias | 17 | 30 | 0.0227 -> 7.72e+36 | 0.887 | 0.003 |
| backbone.block2.branches.0.dw_bn.bias | 20 | 30 | 0.0222 -> 7.56e+36 | 0.887 | 0.007 |
