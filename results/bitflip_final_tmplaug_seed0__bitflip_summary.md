Exhaustive single-bit-flip map: 4608 flips, evaluated on 600 class-balanced beats (unflipped balanced recall 85.3%).

Overall: catastrophic (drop>20%) unguarded 6.7% of flips, guarded 0.00%. Worst single flip: unguarded drop 0.853, guarded 0.027. Mean prediction change: unguarded 5.30%, guarded 0.17%.

| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |
|---|---|---|---|---|---|---|---|---|
| 31 | sign | 384 | 0.0062 | 0.117 | 0.0 | 0.0012 | 0.027 | 0.00 |
| 30 | exponent | 384 | 0.4293 | 0.853 | 80.2 | 0.0012 | 0.027 | 0.00 |
| 29 | exponent | 384 | 0.0067 | 0.157 | 0.0 | 0.0009 | 0.012 | 0.00 |
| 28 | exponent | 384 | 0.0067 | 0.157 | 0.0 | 0.0009 | 0.012 | 0.00 |
| 27 | exponent | 384 | 0.0067 | 0.157 | 0.0 | 0.0009 | 0.012 | 0.00 |
| 26 | exponent | 384 | 0.0077 | 0.208 | 0.3 | 0.0010 | 0.012 | 0.00 |
| 25 | exponent | 384 | 0.0101 | 0.392 | 0.5 | 0.0014 | 0.027 | 0.00 |
| 24 | exponent | 384 | 0.0052 | 0.100 | 0.0 | 0.0011 | 0.012 | 0.00 |
| 23 | exponent | 384 | 0.0031 | 0.102 | 0.0 | 0.0008 | 0.010 | 0.00 |
| 22 | mantissa | 384 | 0.0016 | 0.065 | 0.0 | 0.0008 | 0.010 | 0.00 |
| 18 | mantissa | 384 | 0.0001 | 0.002 | 0.0 | 0.0001 | 0.002 | 0.00 |
| 12 | mantissa | 384 | 0.0000 | 0.000 | 0.0 | 0.0000 | 0.000 | 0.00 |

By parameter type:

| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |
|---|---|---|---|---|
| dw_bn.weight | 2304 | 8.3 | 0.520 | 0.010 |
| dw_bn.bias | 2304 | 5.2 | 0.853 | 0.027 |

Most damaging unguarded flips:

| param | element | bit | old -> new | drop | guarded drop |
|---|---|---|---|---|---|
| backbone.block1.branches.0.dw_bn.bias | 5 | 30 | 0.0192 -> 6.54e+36 | 0.853 | 0.000 |
| backbone.block1.branches.1.dw_bn.bias | 15 | 30 | 0.00638 -> 2.17e+36 | 0.853 | 0.005 |
| backbone.block2.branches.0.dw_bn.bias | 37 | 30 | 0.00795 -> 2.71e+36 | 0.853 | 0.002 |
| backbone.block2.branches.1.dw_bn.bias | 10 | 30 | 0.0192 -> 6.54e+36 | 0.853 | 0.002 |
| backbone.block2.branches.2.dw_bn.bias | 6 | 30 | 0.00465 -> 1.58e+36 | 0.853 | 0.003 |
| backbone.block2.branches.2.dw_bn.bias | 8 | 30 | 0.0262 -> 8.9e+36 | 0.853 | 0.005 |
| backbone.block2.branches.2.dw_bn.bias | 9 | 30 | 0.0177 -> 6.03e+36 | 0.853 | 0.002 |
| backbone.block2.branches.2.dw_bn.bias | 15 | 30 | 0.00031 -> 1.06e+35 | 0.853 | 0.002 |
