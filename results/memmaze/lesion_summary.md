# Memory Maze in-silico lesions (PREREG Amendment 2)

Values = fraction of the above-constant performance lost (> 0 = worse), mean [95 % CI].

| condition | seeds | quantity | lesion | 9x9 Walls | 9x9 Objects |
|---|---|---|---|---|---|
| C | 5 | deficit | CX | 0.005 [-0.005, 0.024] | 0.596 [0.526, 0.666] |
| C | 5 | deficit | randCX | 0.017 [-0.005, 0.076] | 0.141 [0.088, 0.196] |
| C | 5 | specific effect | CX | -0.013 [-0.067, 0.014] | 0.455 [0.365, 0.531] |
| C | 5 | deficit | MB | 0.003 [-0.008, 0.020] | 0.007 [-0.008, 0.023] |
| C | 5 | deficit | randMB | 0.015 [0.000, 0.059] | 0.044 [0.025, 0.061] |
| C | 5 | specific effect | MB | -0.012 [-0.055, 0.006] | -0.037 [-0.063, -0.013] |
| C | 5 | deficit | CXdeaf | 0.002 [-0.012, 0.019] | 0.734 [0.530, 0.990] |
| C | 5 | deficit | OUT | 0.004 [-0.034, 0.055] | 0.192 [0.125, 0.259] |
| S | 3 | deficit | CX | 0.000 [0.000, 0.000] | 0.916 [0.851, 0.972] |
| S | 3 | deficit | randCX | 0.000 [0.000, 0.000] | 0.017 [-0.000, 0.035] |
| S | 3 | specific effect | CX | 0.000 [0.000, 0.000] | 0.899 [0.825, 0.958] |
| S | 3 | deficit | MB | 0.000 [0.000, 0.000] | 0.011 [-0.009, 0.030] |
| S | 3 | deficit | randMB | 0.000 [0.000, 0.000] | 0.008 [-0.008, 0.024] |
| S | 3 | specific effect | MB | 0.000 [0.000, 0.000] | 0.002 [-0.013, 0.018] |
| S | 3 | deficit | CXdeaf | 0.000 [0.000, 0.000] | 0.848 [0.817, 0.880] |
| S | 3 | deficit | OUT | 0.000 [0.000, 0.000] | 0.161 [0.104, 0.209] |
| B | 3 | deficit | CX | 0.000 [0.000, 0.000] | 0.680 [0.630, 0.740] |
| B | 3 | deficit | randCX | 0.000 [0.000, 0.000] | 0.003 [-0.023, 0.028] |
| B | 3 | specific effect | CX | 0.000 [0.000, 0.000] | 0.678 [0.644, 0.719] |
| B | 3 | deficit | MB | 0.000 [0.000, 0.000] | -0.019 [-0.053, 0.016] |
| B | 3 | deficit | randMB | 0.000 [0.000, 0.000] | -0.009 [-0.033, 0.015] |
| B | 3 | specific effect | MB | 0.000 [0.000, 0.000] | -0.011 [-0.048, 0.024] |
| B | 3 | deficit | CXdeaf | 0.000 [0.000, 0.000] | 0.770 [0.712, 0.836] |
| B | 3 | deficit | OUT | 0.000 [0.000, 0.000] | 0.186 [0.156, 0.218] |

| test | estimate | 95 % CI | p | p Holm | verdict |
|---|---|---|---|---|---|
| H7a | 0.455 | [0.365, 0.531] | 0.0000 | 0.0000 | PASS |
| H7b | 0.491 | [0.393, 0.564] | 0.0000 | 0.0000 | PASS |
| H7c | -0.444 | [-0.552, -0.338] | 1.0000 | 1.0000 | FAIL |

Group sizes: CX 2875 neurons / 75544 outgoing edges; MB 5604 neurons / 45899 outgoing edges; OUT 1303 neurons / 30982 outgoing edges
