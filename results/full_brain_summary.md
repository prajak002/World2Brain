# Whole-brain students, in-house tasks (batch A)

## Arena (NAV + ARRANGE)

400 test episodes; blind window 8:16; MSE in whitened PCA-256 space; lower is better.

| wiring | params | seeds | blind MSE (mean over seeds) | sd over seeds | 95 % CI over episodes | diff vs connectome [95 % CI] | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| connectome | 41,253,927 | 3 | 0.6236 | 0.0046 | [0.5935, 0.6551] | ref | | |
| shuffled | 41,253,927 | 3 | 0.7028 | 0.0048 | [0.6701, 0.7370] | +0.0792 [+0.0704, +0.0881] | 0.0005 | worse than connectome |
| block_shuffled | 41,253,927 | 3 | 0.6928 | 0.0041 | [0.6604, 0.7269] | +0.0692 [+0.0614, +0.0771] | 0.0005 | worse than connectome |
| random | 41,253,927 | 3 | 0.6800 | 0.0019 | [0.6476, 0.7143] | +0.0564 [+0.0492, +0.0636] | 0.0005 | worse than connectome |
| gru | 41,242,236 | 3 | 0.4164 | 0.0044 | [0.3920, 0.4422] | -0.2072 [-0.2260, -0.1886] | 0.0005 | better than connectome |
| transformer | 40,558,848 | 3 | 0.6708 | 0.0399 | [0.6379, 0.7048] | +0.0472 [+0.0371, +0.0576] | 0.0005 | worse than connectome |

## Arena, continuous turn angles

400 test episodes; blind window 8:16; MSE in whitened PCA-256 space; lower is better.

| wiring | params | seeds | blind MSE (mean over seeds) | sd over seeds | 95 % CI over episodes | diff vs connectome [95 % CI] | p (Holm) | verdict |
|---|---|---|---|---|---|---|---|---|
| connectome | 41,253,927 | 3 | 0.6179 | 0.0027 | [0.5921, 0.6444] | ref | | |
| shuffled | 41,253,927 | 3 | 0.6972 | 0.0042 | [0.6690, 0.7265] | +0.0793 [+0.0695, +0.0894] | 0.0005 | worse than connectome |
| block_shuffled | 41,253,927 | 3 | 0.6838 | 0.0009 | [0.6558, 0.7127] | +0.0658 [+0.0574, +0.0745] | 0.0005 | worse than connectome |
| random | 41,253,927 | 3 | 0.6742 | 0.0022 | [0.6458, 0.7040] | +0.0563 [+0.0478, +0.0652] | 0.0005 | worse than connectome |
| gru | 41,242,236 | 3 | 0.4149 | 0.0070 | [0.3952, 0.4353] | -0.2030 [-0.2210, -0.1860] | 0.0005 | better than connectome |
| transformer | 40,558,848 | 3 | 0.6580 | 0.0505 | [0.6297, 0.6870] | +0.0401 [+0.0305, +0.0497] | 0.0005 | worse than connectome |

