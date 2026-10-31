# wm2cx results summary

## E1 Blind-horizon latent prediction, arena (NAV + ARRANGE) (test; 8 visible, then 1..8 blind steps)

MSE in whitened PCA-256 space (students are trained in it) and in the original teacher space (3,840-d ViT-B mean + 2x2 pooled tokens). 95% CI: bootstrap over test episodes of the horizon-mean MSE. Paired: mean difference vs `connectome_K1_s0` over the same episodes with bootstrap 95% CI and two-sided p. Trivial baselines are in the same space and windows (#6).

| run | neurons | params | MSE h1 | MSE h4 | MSE h8 | MSE mean [95% CI] | teacher-space MSE mean | cos h1 | cos h8 | paired Δ vs connectome [95% CI], p |
|---|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | 6935 | 7.25M | 0.4759 | 0.5808 | 0.7169 | 0.6292 | - | 0.708 | 0.585 | ref |
| connectome_K1_s1 | 6935 | 7.25M | 0.4747 | 0.5683 | 0.7150 | 0.6209 | - | 0.711 | 0.586 | - |
| connectome_K2_s0 | 13870 | 14.50M | 0.4758 | 0.5662 | 0.7150 | 0.6207 | - | 0.710 | 0.586 | - |
| connectome_nosign_K1_s0 | 6935 | 7.25M | 0.4760 | 0.5589 | 0.7050 | 0.6120 | - | 0.709 | 0.595 | - |
| dense_K1_s0 | 6935 | 55.25M | 0.6290 | 0.6578 | 0.8227 | 0.7193 | - | 0.575 | 0.484 | - |
| gru_K1_s0 | 1400 | 9.50M | 0.4833 | 0.4876 | 0.6545 | 0.5444 | - | 0.696 | 0.615 | - |
| random_K1_s0 | 6935 | 7.25M | 0.5568 | 0.5995 | 0.7601 | 0.6511 | - | 0.639 | 0.534 | - |
| random_K1_s1 | 6935 | 7.25M | 0.5567 | 0.5954 | 0.7419 | 0.6442 | - | 0.641 | 0.548 | - |
| shuffled_K1_s0 | 6935 | 7.25M | 0.5377 | 0.6361 | 0.7808 | 0.6800 | - | 0.666 | 0.521 | - |
| shuffled_K1_s1 | 6935 | 7.25M | 0.5389 | 0.6355 | 0.7647 | 0.6757 | - | 0.666 | 0.537 | - |

### Wiring comparison, K=1, arena (NAV + ARRANGE): mean over horizons, mean ± sd over seeds (95% t-CI when n ≥ 3)

| wiring | seeds | MSE | cos |
|---|---|---|---|
| connectome | 2 | 0.6251 ± 0.0058 | 0.605 ± 0.005 |
| connectome_nosign | 1 | 0.6120 ± 0.0000 | 0.619 ± 0.000 |
| dense | 1 | 0.7193 ± 0.0000 | 0.509 ± 0.000 |
| gru | 1 | 0.5444 ± 0.0000 | 0.663 ± 0.000 |
| random | 2 | 0.6476 ± 0.0049 | 0.579 ± 0.005 |
| shuffled | 2 | 0.6778 ± 0.0030 | 0.562 ± 0.002 |

## E1 Blind-horizon latent prediction, NYC (test; 8 visible, then 1..8 blind steps)

MSE in whitened PCA-256 space (students are trained in it) and in the original teacher space (3,840-d ViT-B mean + 2x2 pooled tokens). 95% CI: bootstrap over test episodes of the horizon-mean MSE. Paired: mean difference vs `connectome_K1_s0` over the same episodes with bootstrap 95% CI and two-sided p. Trivial baselines are in the same space and windows (#6).

| run | neurons | params | MSE h1 | MSE h4 | MSE h8 | MSE mean [95% CI] | teacher-space MSE mean | cos h1 | cos h8 | paired Δ vs connectome [95% CI], p |
|---|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0_nyc | 6935 | 7.25M | 0.8554 | 0.8625 | 0.8815 | 0.8846 | - | 0.461 | 0.349 | ref |
| gru_K1_s0_nyc | 1400 | 9.50M | 1.0244 | 0.8341 | 0.9457 | 0.9071 | - | 0.259 | 0.245 | - |
| random_K1_s0_nyc | 6935 | 7.25M | 0.9746 | 0.8885 | 0.9319 | 0.9171 | - | 0.320 | 0.274 | - |
| shuffled_K1_s0_nyc | 6935 | 7.25M | 0.9330 | 0.8932 | 0.9220 | 0.9055 | - | 0.378 | 0.278 | - |

### Wiring comparison, K=1, NYC: mean over horizons, mean ± sd over seeds (95% t-CI when n ≥ 3)

| wiring | seeds | MSE | cos |
|---|---|---|---|
| connectome | 1 | 0.8846 ± 0.0000 | 0.309 ± 0.000 |
| gru | 1 | 0.9071 ± 0.0000 | 0.280 ± 0.000 |
| random | 1 | 0.9171 ± 0.0000 | 0.262 ± 0.000 |
| shuffled | 1 | 0.9055 ± 0.0000 | 0.271 ± 0.000 |

## E4 Multi-brain scaling (connectome, seed 0, arena)

| K | neurons | params | MSE mean | cos mean |
|---|---|---|---|---|
| 1 | 6935 | 7.25M | 0.6292 | 0.602 |
| 2 | 13870 | 14.50M | 0.6207 | 0.610 |

## E3 Head-direction / position linear probes (median heading error, deg; R² in brackets)

`actions only`: ridge on cumulative one-hot actions, no neural state (a leaky integrator of discrete turns exposes relative heading linearly, so this is the floor any recurrent state must beat, #7). `all@1400`: random 1,400-neuron subset, feature count matched to the GRU.

| run | actions only (blind) | vis all | blind all | blind all@1400 | vis EPG | blind EPG | vis CX | blind CX | blind pos R² (all) |
|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | - | 16.7 (0.50) | 9.5 (0.82) | - | 50.3 (0.04) | 47.0 (0.16) | 19.7 (0.36) | 14.9 (0.69) | 0.49 |
| connectome_K1_s0_nyc | - | 19.7 (0.38) | 30.0 (0.42) | - | 44.1 (0.15) | 51.9 (0.12) | 23.5 (0.36) | 35.7 (0.34) | 0.31 |
| connectome_K1_s1 | - | 16.4 (0.50) | 9.4 (0.81) | - | 55.5 (0.00) | 55.7 (0.06) | 20.2 (0.33) | 17.4 (0.61) | 0.51 |
| connectome_K2_s0 | - | 15.9 (0.52) | 9.6 (0.83) | - | 40.4 (0.06) | 41.9 (0.19) | 17.0 (0.41) | 11.3 (0.75) | 0.53 |
| connectome_nosign_K1_s0 | - | 22.4 (0.36) | 19.6 (0.57) | - | 32.0 (0.08) | 36.5 (0.21) | 21.2 (0.34) | 19.9 (0.52) | 0.38 |
| dense_K1_s0 | - | 32.8 (0.17) | 36.1 (0.35) | - | 58.2 (-0.03) | 50.3 (0.06) | 35.1 (0.13) | 39.4 (0.27) | -0.14 |
| gru_K1_s0 | - | 30.1 (0.12) | 37.2 (0.29) | - | - | - | - | - | -0.05 |
| gru_K1_s0_nyc | - | 48.6 (-0.13) | 45.5 (0.11) | - | - | - | - | - | -0.10 |
| random_K1_s0 | - | 16.5 (0.40) | 12.1 (0.68) | - | 33.1 (0.13) | 34.9 (0.29) | 17.7 (0.40) | 14.3 (0.66) | 0.49 |
| random_K1_s0_nyc | - | 27.6 (0.07) | 32.7 (0.33) | - | 49.9 (0.09) | 50.8 (0.13) | 27.4 (0.03) | 35.0 (0.30) | 0.47 |
| random_K1_s1 | - | 16.7 (0.37) | 13.4 (0.65) | - | 33.9 (0.13) | 35.4 (0.27) | 17.7 (0.37) | 15.1 (0.63) | 0.47 |
| shuffled_K1_s0 | - | 16.8 (0.45) | 11.0 (0.75) | - | 46.1 (0.03) | 37.4 (0.18) | 16.9 (0.44) | 11.3 (0.73) | 0.48 |
| shuffled_K1_s0_nyc | - | 24.6 (0.27) | 27.3 (0.44) | - | 42.9 (0.13) | 46.0 (0.22) | 22.7 (0.29) | 29.5 (0.42) | 0.41 |
| shuffled_K1_s1 | - | 17.3 (0.43) | 10.4 (0.76) | - | 40.4 (0.09) | 41.7 (0.20) | 16.9 (0.42) | 11.6 (0.72) | 0.50 |

## E2 Teacher-free planning (success rate, mean goal distance start -> end)

| run | env | task | policy | dynamics | blackout | success | SPL | coll/100 m | dist start | dist end | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.11 | 30 |
| connectome_K1_s0 | arena | arrange | random | student | 0 | 0.00 | - | - | 4.09 | 4.14 | 30 |
| connectome_K1_s0 | arena | arrange | scripted | student | 0 | 0.93 | - | - | 4.09 | 0.98 | 30 |
| connectome_K1_s0 | arena | nav | mpc | student | 0 | 0.03 | - | - | 5.45 | 6.05 | 30 |
| connectome_K1_s0 | nyc twin | nav | mpc | oracle | 1 | 1.00 | 0.98 | 0.0 | 76.41 | 10.20 | 10 |
| connectome_K1_s0 | nyc twin | nav | mpc | oracle | 3 | 1.00 | 0.98 | 0.0 | 76.41 | 10.22 | 10 |
| connectome_K1_s0 | nyc twin | nav | mpc | oracle | 5 | 0.80 | 0.78 | 0.0 | 76.41 | 11.17 | 10 |
| connectome_K1_s0 | nyc twin | nav | mpc | oracle | 0 | 1.00 | 0.99 | 0.0 | 76.41 | 10.61 | 10 |
| connectome_K1_s0 | arena | nav | random | student | 0 | 0.07 | - | - | 5.45 | 5.70 | 30 |
| connectome_K1_s0 | arena | nav | scripted | student | 0 | 1.00 | - | - | 5.45 | 0.92 | 30 |
| connectome_K1_s0 | nyc twin | nav | scripted | - | 3 | 0.90 | 0.89 | 3.3 | 76.41 | 17.37 | 10 |
| connectome_K1_s0 | nyc twin | nav | scripted | - | 0 | 0.90 | 0.89 | 0.0 | 76.41 | 17.35 | 10 |
| connectome_K2_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.06 | 30 |
| connectome_K2_s0 | arena | nav | mpc | student | 0 | 0.07 | - | - | 5.45 | 6.00 | 30 |
| dense_K1_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.09 | 30 |
| dense_K1_s0 | arena | nav | mpc | student | 0 | 0.03 | - | - | 5.45 | 6.11 | 30 |
| gru_K1_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.09 | 30 |
| gru_K1_s0 | arena | nav | mpc | student | 0 | 0.03 | - | - | 5.45 | 5.63 | 30 |
| random_K1_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.05 | 30 |
| random_K1_s0 | arena | nav | mpc | student | 0 | 0.03 | - | - | 5.45 | 5.35 | 30 |
| shuffled_K1_s0 | arena | arrange | mpc | student | 0 | 0.00 | - | - | 4.09 | 4.17 | 30 |
| shuffled_K1_s0 | arena | nav | mpc | student | 0 | 0.03 | - | - | 5.45 | 6.04 | 30 |
