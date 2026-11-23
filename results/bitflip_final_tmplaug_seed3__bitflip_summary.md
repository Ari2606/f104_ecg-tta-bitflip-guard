Exhaustive single-bit-flip map: 4608 flips, evaluated on 600 class-balanced beats (unflipped balanced recall 78.5%).

Overall: catastrophic (drop>20%) unguarded 6.7% of flips, guarded 0.00%. Worst single flip: unguarded drop 0.785, guarded 0.027. Mean prediction change: unguarded 5.61%, guarded 0.30%.

| bit | field | flips | mean drop | max drop | catastrophic % | guarded: mean drop | guarded: max drop | guarded: catastrophic % |
|---|---|---|---|---|---|---|---|---|
| 31 | sign | 384 | -0.0002 | 0.162 | 0.0 | 0.0011 | 0.015 | 0.00 |
| 30 | exponent | 384 | 0.3898 | 0.785 | 80.2 | 0.0009 | 0.027 | 0.00 |
| 29 | exponent | 384 | 0.0003 | 0.073 | 0.0 | 0.0010 | 0.013 | 0.00 |
| 28 | exponent | 384 | 0.0003 | 0.073 | 0.0 | 0.0010 | 0.013 | 0.00 |
| 27 | exponent | 384 | 0.0003 | 0.073 | 0.0 | 0.0010 | 0.013 | 0.00 |
| 26 | exponent | 384 | 0.0021 | 0.435 | 0.3 | 0.0011 | 0.013 | 0.00 |
| 25 | exponent | 384 | 0.0007 | 0.137 | 0.0 | 0.0013 | 0.027 | 0.00 |
| 24 | exponent | 384 | 0.0007 | 0.062 | 0.0 | 0.0011 | 0.022 | 0.00 |
| 23 | exponent | 384 | 0.0006 | 0.227 | 0.3 | 0.0009 | 0.013 | 0.00 |
| 22 | mantissa | 384 | 0.0012 | 0.047 | 0.0 | 0.0010 | 0.013 | 0.00 |
| 18 | mantissa | 384 | 0.0003 | 0.007 | 0.0 | 0.0003 | 0.007 | 0.00 |
| 12 | mantissa | 384 | 0.0000 | 0.000 | 0.0 | 0.0000 | 0.000 | 0.00 |

By parameter type:

| type | flips | catastrophic % (unguarded) | max drop (unguarded) | max drop (guarded) |
|---|---|---|---|---|
| dw_bn.weight | 2304 | 8.4 | 0.467 | 0.013 |
| dw_bn.bias | 2304 | 5.1 | 0.785 | 0.027 |

Most damaging unguarded flips:

| param | element | bit | old -> new | drop | guarded drop |
|---|---|---|---|---|---|
| backbone.block1.branches.0.dw_bn.bias | 0 | 30 | 0.00656 -> 2.23e+36 | 0.785 | -0.002 |
| backbone.block1.branches.0.dw_bn.bias | 2 | 30 | 0.00446 -> 1.52e+36 | 0.785 | 0.012 |
| backbone.block1.branches.0.dw_bn.bias | 8 | 30 | 0.0311 -> 1.06e+37 | 0.785 | 0.020 |
| backbone.block1.branches.0.dw_bn.bias | 9 | 30 | 0.0492 -> 1.67e+37 | 0.785 | 0.002 |
| backbone.block1.branches.0.dw_bn.bias | 12 | 30 | 0.0159 -> 5.41e+36 | 0.785 | -0.002 |
| backbone.block1.branches.1.dw_bn.bias | 10 | 30 | 0.0612 -> 2.08e+37 | 0.785 | 0.003 |
| backbone.block1.branches.2.dw_bn.bias | 1 | 30 | 0.00615 -> 2.09e+36 | 0.785 | -0.010 |
| backbone.block1.branches.2.dw_bn.bias | 3 | 30 | 0.0219 -> 7.44e+36 | 0.785 | -0.005 |
