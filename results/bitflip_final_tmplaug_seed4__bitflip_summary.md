Exhaustive single-bit-flip map: 4608 flips, evaluated on 600 class-balanced beats (unflipped balanced recall 86.3%).

Overall: catastrophic (drop>20%) unguarded 6.9% of flips, guarded 0.00%. Worst single flip: unguarded drop 0.863, guarded 0.038. Mean prediction change: unguarded 5.89%, guarded 0.30%.

| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |
|---|---|---|---|---|---|---|---|---|
| 31 | sign | 384 | 0.0031 | 0.165 | 0.0 | -0.0006 | 0.017 | 0.00 |
| 30 | exponent | 384 | 0.4497 | 0.863 | 81.5 | -0.0010 | 0.038 | 0.00 |
| 29 | exponent | 384 | 0.0036 | 0.163 | 0.0 | -0.0005 | 0.008 | 0.00 |
| 28 | exponent | 384 | 0.0036 | 0.163 | 0.0 | -0.0005 | 0.008 | 0.00 |
| 27 | exponent | 384 | 0.0036 | 0.163 | 0.0 | -0.0005 | 0.008 | 0.00 |
| 26 | exponent | 384 | 0.0037 | 0.163 | 0.0 | -0.0006 | 0.008 | 0.00 |
| 25 | exponent | 384 | 0.0055 | 0.512 | 1.3 | -0.0009 | 0.038 | 0.00 |
| 24 | exponent | 384 | 0.0018 | 0.153 | 0.0 | -0.0006 | 0.008 | 0.00 |
| 23 | exponent | 384 | -0.0014 | 0.113 | 0.0 | -0.0006 | 0.012 | 0.00 |
| 22 | mantissa | 384 | -0.0009 | 0.093 | 0.0 | -0.0002 | 0.012 | 0.00 |
| 18 | mantissa | 384 | -0.0000 | 0.002 | 0.0 | -0.0000 | 0.002 | 0.00 |
| 12 | mantissa | 384 | 0.0000 | 0.000 | 0.0 | 0.0000 | 0.000 | 0.00 |

By parameter type:

| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |
|---|---|---|---|---|
| dw_bn.weight | 2304 | 8.3 | 0.530 | 0.012 |
| dw_bn.bias | 2304 | 5.5 | 0.863 | 0.038 |

Most damaging unguarded flips:

| param | element | bit | old -> new | drop | guarded drop |
|---|---|---|---|---|---|
| backbone.block1.branches.0.dw_bn.bias | 8 | 30 | 0.0189 -> 6.41e+36 | 0.863 | -0.003 |
| backbone.block1.branches.0.dw_bn.bias | 11 | 30 | 0.00801 -> 2.73e+36 | 0.863 | 0.002 |
| backbone.block1.branches.0.dw_bn.bias | 15 | 30 | 0.0119 -> 4.07e+36 | 0.863 | -0.017 |
| backbone.block1.branches.1.dw_bn.bias | 7 | 30 | 0.00958 -> 3.26e+36 | 0.863 | 0.027 |
| backbone.block1.branches.1.dw_bn.bias | 9 | 30 | 0.00664 -> 2.26e+36 | 0.863 | 0.002 |
| backbone.block2.branches.0.dw_bn.bias | 10 | 30 | 0.0345 -> 1.17e+37 | 0.863 | 0.000 |
| backbone.block2.branches.0.dw_bn.bias | 12 | 30 | 0.00665 -> 2.26e+36 | 0.863 | -0.010 |
| backbone.block2.branches.0.dw_bn.bias | 26 | 30 | 0.00795 -> 2.71e+36 | 0.863 | 0.000 |
