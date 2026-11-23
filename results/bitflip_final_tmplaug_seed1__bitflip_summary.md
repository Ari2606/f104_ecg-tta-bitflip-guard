Exhaustive single-bit-flip map: 4608 flips, evaluated on 600 class-balanced beats (unflipped balanced recall 71.5%).

Overall: catastrophic (drop>20%) unguarded 6.5% of flips, guarded 0.00%. Worst single flip: unguarded drop 0.715, guarded 0.013. Mean prediction change: unguarded 4.43%, guarded 0.17%.

| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |
|---|---|---|---|---|---|---|---|---|
| 31 | sign | 384 | 0.0020 | 0.073 | 0.0 | -0.0001 | 0.013 | 0.00 |
| 30 | exponent | 384 | 0.3036 | 0.715 | 77.3 | -0.0008 | 0.013 | 0.00 |
| 29 | exponent | 384 | 0.0006 | 0.077 | 0.0 | -0.0001 | 0.013 | 0.00 |
| 28 | exponent | 384 | 0.0006 | 0.077 | 0.0 | -0.0001 | 0.013 | 0.00 |
| 27 | exponent | 384 | 0.0005 | 0.077 | 0.0 | -0.0001 | 0.013 | 0.00 |
| 26 | exponent | 384 | -0.0001 | 0.217 | 0.3 | -0.0004 | 0.013 | 0.00 |
| 25 | exponent | 384 | -0.0000 | 0.178 | 0.0 | -0.0006 | 0.013 | 0.00 |
| 24 | exponent | 384 | 0.0006 | 0.063 | 0.0 | -0.0001 | 0.010 | 0.00 |
| 23 | exponent | 384 | 0.0004 | 0.047 | 0.0 | -0.0002 | 0.010 | 0.00 |
| 22 | mantissa | 384 | -0.0006 | 0.020 | 0.0 | -0.0005 | 0.008 | 0.00 |
| 18 | mantissa | 384 | -0.0001 | 0.003 | 0.0 | -0.0001 | 0.003 | 0.00 |
| 12 | mantissa | 384 | 0.0000 | 0.000 | 0.0 | 0.0000 | 0.000 | 0.00 |

By parameter type:

| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |
|---|---|---|---|---|
| dw_bn.weight | 2304 | 8.3 | 0.382 | 0.008 |
| dw_bn.bias | 2304 | 4.6 | 0.715 | 0.013 |

Most damaging unguarded flips:

| param | element | bit | old -> new | drop | guarded drop |
|---|---|---|---|---|---|
| backbone.block2.branches.0.dw_bn.bias | 7 | 30 | 0.0102 -> 3.47e+36 | 0.715 | 0.005 |
| backbone.block2.branches.1.dw_bn.bias | 22 | 30 | 0.0268 -> 9.1e+36 | 0.715 | 0.003 |
| backbone.block2.branches.1.dw_bn.bias | 29 | 30 | 0.0167 -> 5.7e+36 | 0.715 | -0.003 |
| backbone.block2.branches.1.dw_bn.bias | 30 | 30 | 0.0176 -> 5.99e+36 | 0.715 | -0.002 |
| backbone.block2.branches.1.dw_bn.bias | 31 | 30 | 0.0346 -> 1.18e+37 | 0.715 | -0.007 |
| backbone.block2.branches.1.dw_bn.bias | 42 | 30 | 0.00768 -> 2.61e+36 | 0.715 | -0.003 |
| backbone.block2.branches.2.dw_bn.bias | 1 | 30 | 0.0236 -> 8.02e+36 | 0.715 | 0.000 |
| backbone.block2.branches.2.dw_bn.bias | 29 | 30 | 0.0329 -> 1.12e+37 | 0.715 | 0.002 |
