# Whole-brain students, linear-readout homing (batch A)

## Arena

12 visible + 24 dark steps, 50 episodes per run, success radius 1.0, mean distance from the start when the dark begins 3.89. Lower distance and drift are better.

| wiring | seeds | success (per seed) | final distance (sd) | diff vs connectome [95 % CI], p (Holm), verdict | drift (sd) | diff vs connectome [95 % CI], p (Holm), verdict |
|---|---|---|---|---|---|---|
| connectome | 3 | 0.073 (0.06, 0.12, 0.04) | 2.270 (0.058) | ref | 2.755 (0.163) | ref |
| shuffled | 3 | 0.113 (0.12, 0.14, 0.08) | 2.244 (0.070) | -0.026 [-0.251, +0.193], 1.0000, no difference | 4.427 (0.124) | +1.673 [+1.253, +2.113], 0.0005, worse than connectome |
| block_shuffled | 3 | 0.187 (0.16, 0.24, 0.16) | 2.086 (0.041) | -0.183 [-0.391, +0.027], 0.2607, no difference | 3.946 (0.314) | +1.191 [+0.792, +1.595], 0.0005, worse than connectome |
| random | 3 | 0.173 (0.20, 0.16, 0.16) | 2.256 (0.117) | -0.014 [-0.271, +0.245], 1.0000, no difference | 3.119 (0.136) | +0.365 [+0.026, +0.710], 0.0345, worse than connectome |
| gru | 3 | 0.053 (0.06, 0.04, 0.06) | 3.469 (0.251) | +1.199 [+0.845, +1.570], 0.0005, worse than connectome | 3.577 (0.200) | +0.823 [+0.446, +1.194], 0.0005, worse than connectome |
| transformer | 3 | 0.007 (0.00, 0.00, 0.02) | 4.475 (0.161) | +2.206 [+1.760, +2.677], 0.0005, worse than connectome | 4.314 (0.092) | +1.559 [+1.084, +2.065], 0.0005, worse than connectome |

## Arena, continuous turn angles

12 visible + 24 dark steps, 50 episodes per run, success radius 1.0, mean distance from the start when the dark begins 3.92. Lower distance and drift are better.

| wiring | seeds | success (per seed) | final distance (sd) | diff vs connectome [95 % CI], p (Holm), verdict | drift (sd) | diff vs connectome [95 % CI], p (Holm), verdict |
|---|---|---|---|---|---|---|
| connectome | 3 | 0.160 (0.16, 0.22, 0.10) | 2.283 (0.027) | ref | 2.569 (0.101) | ref |
| shuffled | 3 | 0.113 (0.06, 0.16, 0.12) | 2.263 (0.181) | -0.020 [-0.343, +0.275], 1.0000, no difference | 4.666 (0.439) | +2.097 [+1.655, +2.549], 0.0005, worse than connectome |
| block_shuffled | 3 | 0.127 (0.10, 0.10, 0.18) | 2.324 (0.071) | +0.041 [-0.230, +0.306], 1.0000, no difference | 4.543 (0.525) | +1.974 [+1.565, +2.376], 0.0005, worse than connectome |
| random | 3 | 0.147 (0.12, 0.18, 0.14) | 2.043 (0.151) | -0.240 [-0.510, +0.009], 0.1839, no difference | 3.398 (0.359) | +0.830 [+0.536, +1.129], 0.0005, worse than connectome |
| gru | 3 | 0.027 (0.04, 0.02, 0.02) | 3.694 (0.180) | +1.411 [+0.991, +1.848], 0.0005, worse than connectome | 3.705 (0.080) | +1.136 [+0.746, +1.543], 0.0005, worse than connectome |
| transformer | 3 | 0.013 (0.02, 0.00, 0.02) | 4.300 (0.527) | +2.017 [+1.618, +2.436], 0.0005, worse than connectome | 4.258 (0.341) | +1.689 [+1.262, +2.117], 0.0005, worse than connectome |

