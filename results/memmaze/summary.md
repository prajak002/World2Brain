# Memory Maze 9x9 offline probing (PREREG sections 2-5)

| condition | seeds | 9x9 Walls acc % | 9x9 Objects MSE |
|---|---|---|---|
| C | 5 | 80.7 +- 0.1 | 20.13 +- 0.30 |
| S | 3 | 80.7 +- 0.1 | 22.35 +- 0.02 |
| B | 3 | 80.7 +- 0.1 | 22.49 +- 0.02 |
| R | 3 | 80.7 +- 0.1 | 22.33 +- 0.09 |
| G | 5 | 80.6 +- 0.2 | 12.66 +- 0.11 |
| T | 3 | 80.9 +- 0.1 | 12.55 +- 0.09 |
| N | 3 | 80.8 +- 0.1 | 12.74 +- 0.07 |
| K | 1 | 80.8 +- 0.0 | 24.01 +- 0.00 |

Published (Pasukonis et al. 2023, Table 4, 3 seeds):

| model | 9x9 Walls % | 9x9 Objects MSE |
|---|---|---|
| Constant | 80.8 | 23.9 |
| VAE (no memory) | 84.1 | 17.2 |
| RSSM | 89.0 | 8.3 |
| VAE+GRU (TBTT) | 90.0 | 7.8 |
| RSSM (TBTT) | 94.0 | 6.2 |
| Supervised oracle | 99.7 | 0.3 |

| test | metric | C vs | diff (>0 = C better) | 95 % CI | p | p Holm | verdict |
|---|---|---|---|---|---|---|---|
| H1 | objects | G | -7.470 | [-7.790, -7.167] | 1.0000 | 1.0000 | FAIL |
| H2a | objects | S | 2.221 | [1.941, 2.492] | 0.0000 | 0.0000 | PASS |
| H2b | objects | B | 2.366 | [2.084, 2.637] | 0.0000 | 0.0000 | PASS |
| H3 | walls | G | 0.107 | [-0.044, 0.284] | 0.0960 | 0.3840 | INCONCLUSIVE |
| H4a | objects | N | -7.389 | [-7.697, -7.095] | 1.0000 | 1.0000 | FAIL |
| H4b | walls | N | -0.088 | [-0.217, 0.065] | 0.8838 | 1.0000 | INCONCLUSIVE |

Hypotheses: H1 FAIL; H2 PASS; H3 INCONCLUSIVE; H4 FAIL; H5 descriptive: C objects MSE 20.13 vs published RSSM (TBTT) 6.2 -> not below
