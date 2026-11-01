# wm2cx results summary

## E1 Blind-horizon latent prediction (test, 8 visible then 1..8 blind steps, PCA-256 whitened space)

| run | neurons | params (rec / in / out) | MSE mean [95% CI] | MSE h1 | MSE h8 | cos h1 | cos h8 | raw-space MSE | const-mean | copy-last | alt window MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6292 | 0.4759 | 0.7169 | 0.708 | 0.585 | nan | nan | nan | nan (t0=-) |
| connectome_K1_s0_nyc | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.8846 | 0.8554 | 0.8815 | 0.461 | 0.349 | nan | nan | nan | nan (t0=-) |
| connectome_K1_s1 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6209 | 0.4747 | 0.7150 | 0.711 | 0.586 | nan | nan | nan | nan (t0=-) |
| connectome_K2_s0 | 13870 | 14.50M (legacy, incl. dead W_in rows) | 0.6207 | 0.4758 | 0.7150 | 0.710 | 0.586 | nan | nan | nan | nan (t0=-) |
| connectome_nosign_K1_s0 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6120 | 0.4760 | 0.7050 | 0.709 | 0.595 | nan | nan | nan | nan (t0=-) |
| dense_K1_s0 | 6935 | 55.25M (legacy, incl. dead W_in rows) | 0.7193 | 0.6290 | 0.8227 | 0.575 | 0.484 | nan | nan | nan | nan (t0=-) |
| gru_K1_s0 | 1400 | 9.50M (legacy, incl. dead W_in rows) | 0.5444 | 0.4833 | 0.6545 | 0.696 | 0.615 | nan | nan | nan | nan (t0=-) |
| gru_K1_s0_nyc | 1400 | 9.50M (legacy, incl. dead W_in rows) | 0.9071 | 1.0244 | 0.9457 | 0.259 | 0.245 | nan | nan | nan | nan (t0=-) |
| random_K1_s0 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6511 | 0.5568 | 0.7601 | 0.639 | 0.534 | nan | nan | nan | nan (t0=-) |
| random_K1_s0_nyc | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.9171 | 0.9746 | 0.9319 | 0.320 | 0.274 | nan | nan | nan | nan (t0=-) |
| random_K1_s1 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6442 | 0.5567 | 0.7419 | 0.641 | 0.548 | nan | nan | nan | nan (t0=-) |
| shuffled_K1_s0 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6800 | 0.5377 | 0.7808 | 0.666 | 0.521 | nan | nan | nan | nan (t0=-) |
| shuffled_K1_s0_nyc | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.9055 | 0.9330 | 0.9220 | 0.378 | 0.278 | nan | nan | nan | nan (t0=-) |
| shuffled_K1_s1 | 6935 | 7.25M (legacy, incl. dead W_in rows) | 0.6757 | 0.5389 | 0.7647 | 0.666 | 0.537 | nan | nan | nan | nan (t0=-) |

Trivial baselines are computed in the same whitened PCA space on the same windows; raw-space MSE is the MSE of the PCA-reconstructed
3,840-d teacher latent (residual variance outside the 256 components is identical for every model and excluded).

### Wiring comparison, K=1, mean over horizons (mean ± sd over seeds; paired bootstrap 95% CI of the difference vs `connectome`, seed-matched)

| data | wiring | seeds | MSE | cos | Δ vs connectome [95% CI] |
|---|---|---|---|---|---|
| arena | connectome | 2 | 0.6251 ± 0.0041 | 0.605 ± 0.004 |  |
| arena | connectome_nosign | 1 | 0.6120 ± 0.0000 | 0.619 ± 0.000 |  |
| arena | dense | 1 | 0.7193 ± 0.0000 | 0.509 ± 0.000 |  |
| arena | gru | 1 | 0.5444 ± 0.0000 | 0.663 ± 0.000 |  |
| arena | random | 2 | 0.6476 ± 0.0035 | 0.579 ± 0.003 |  |
| arena | shuffled | 2 | 0.6778 ± 0.0022 | 0.562 ± 0.002 |  |
| nyc | connectome | 1 | 0.8846 ± 0.0000 | 0.309 ± 0.000 |  |
| nyc | gru | 1 | 0.9071 ± 0.0000 | 0.280 ± 0.000 |  |
| nyc | random | 1 | 0.9171 ± 0.0000 | 0.262 ± 0.000 |  |
| nyc | shuffled | 1 | 0.9055 ± 0.0000 | 0.271 ± 0.000 |  |

## E4 Multi-brain scaling (connectome, seed 0)

| K | neurons | params | MSE mean | cos mean |
|---|---|---|---|---|
| 1 | 6935 | 7.25M | 0.6292 | 0.602 |
| 2 | 13870 | 14.50M | 0.6207 | 0.610 |

## E3 Head-direction / position linear probes (median heading error, deg; R²)

| run | actions-only (linear) | actions integrated (ceiling) | vis all | blind all | blind matched-762 | vis EPG | blind EPG | vis CX | blind CX | blind pos R² (all) |
|---|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | 39.3 (0.13) | - | 16.7 (0.50) | 9.5 (0.82) | 17.8 (0.63) | 50.3 (0.04) | 47.0 (0.16) | 19.7 (0.36) | 14.9 (0.69) | 0.49 |
| connectome_K1_s0_nyc | - | - | 19.7 (0.38) | 30.0 (0.42) | - | 44.1 (0.15) | 51.9 (0.12) | 23.5 (0.36) | 35.7 (0.34) | 0.31 |
| connectome_K1_s1 | - | - | 16.4 (0.50) | 9.4 (0.81) | - | 55.5 (0.00) | 55.7 (0.06) | 20.2 (0.33) | 17.4 (0.61) | 0.51 |
| connectome_K2_s0 | - | - | 15.9 (0.52) | 9.6 (0.83) | - | 40.4 (0.06) | 41.9 (0.19) | 17.0 (0.41) | 11.3 (0.75) | 0.53 |
| connectome_nosign_K1_s0 | - | - | 22.4 (0.36) | 19.6 (0.57) | - | 32.0 (0.08) | 36.5 (0.21) | 21.2 (0.34) | 19.9 (0.52) | 0.38 |
| dense_K1_s0 | - | - | 32.8 (0.17) | 36.1 (0.35) | - | 58.2 (-0.03) | 50.3 (0.06) | 35.1 (0.13) | 39.4 (0.27) | -0.14 |
| gru_K1_s0 | - | - | 30.1 (0.12) | 37.2 (0.29) | - | - | - | - | - | -0.05 |
| gru_K1_s0_nyc | - | - | 48.6 (-0.13) | 45.5 (0.11) | - | - | - | - | - | -0.10 |
| random_K1_s0 | - | - | 16.5 (0.40) | 12.1 (0.68) | - | 33.1 (0.13) | 34.9 (0.29) | 17.7 (0.40) | 14.3 (0.66) | 0.49 |
| random_K1_s0_nyc | - | - | 27.6 (0.07) | 32.7 (0.33) | - | 49.9 (0.09) | 50.8 (0.13) | 27.4 (0.03) | 35.0 (0.30) | 0.47 |
| random_K1_s1 | - | - | 16.7 (0.37) | 13.4 (0.65) | - | 33.9 (0.13) | 35.4 (0.27) | 17.7 (0.37) | 15.1 (0.63) | 0.47 |
| shuffled_K1_s0 | - | - | 16.8 (0.45) | 11.0 (0.75) | - | 46.1 (0.03) | 37.4 (0.18) | 16.9 (0.44) | 11.3 (0.73) | 0.48 |
| shuffled_K1_s0_nyc | - | - | 24.6 (0.27) | 27.3 (0.44) | - | 42.9 (0.13) | 46.0 (0.22) | 22.7 (0.29) | 29.5 (0.42) | 0.41 |
| shuffled_K1_s1 | - | - | 17.3 (0.43) | 10.4 (0.76) | - | 40.4 (0.09) | 41.7 (0.20) | 16.9 (0.42) | 11.6 (0.72) | 0.50 |

Heading is a deterministic function of the discrete turn actions (the integrated-actions ceiling), so a probe measures whether
the network *represents* the integral linearly, not whether the information exists. `matched-762` uses random 762-neuron subsets (the GRU width).

## E2 Teacher-free planning (success rate, SPL, mean goal distance start -> end)

| run | task | policy | dynamics | success | SPL | dist start | dist end | steps | episodes |
|---|---|---|---|---|---|---|---|---|---|
| connectome_K1_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.11 | nan | 30 |
| connectome_K1_s0 | arrange | random | student | 0.00 | nan | 4.09 | 4.14 | nan | 30 |
| connectome_K1_s0 | arrange | scripted | student | 0.93 | nan | 4.09 | 0.98 | nan | 30 |
| connectome_K1_s0 | nav | mpc | student | 0.03 | nan | 5.45 | 6.05 | nan | 30 |
| connectome_K1_s0 | nav | mpc | oracle_render | 0.00 | 0.00 | 4.41 | 4.07 | 30.0 | 3 |
| connectome_K1_s0 | nav | mpc | oracle_state | 0.90 | 0.77 | 5.62 | 1.02 | 16.6 | 10 |
| connectome_K1_s0 | nav | random | student | 0.07 | nan | 5.45 | 5.70 | nan | 30 |
| connectome_K1_s0 | nav | scripted | student | 1.00 | nan | 5.45 | 0.92 | nan | 30 |
| connectome_K2_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.06 | nan | 30 |
| connectome_K2_s0 | nav | mpc | student | 0.07 | nan | 5.45 | 6.00 | nan | 30 |
| dense_K1_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.09 | nan | 30 |
| dense_K1_s0 | nav | mpc | student | 0.03 | nan | 5.45 | 6.11 | nan | 30 |
| gru_K1_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.09 | nan | 30 |
| gru_K1_s0 | nav | mpc | student | 0.03 | nan | 5.45 | 5.63 | nan | 30 |
| random_K1_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.05 | nan | 30 |
| random_K1_s0 | nav | mpc | student | 0.03 | nan | 5.45 | 5.35 | nan | 30 |
| shuffled_K1_s0 | arrange | mpc | student | 0.00 | nan | 4.09 | 4.17 | nan | 30 |
| shuffled_K1_s0 | nav | mpc | student | 0.03 | nan | 5.45 | 6.04 | nan | 30 |
