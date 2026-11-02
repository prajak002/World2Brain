# Table 3: compute cost (device: cuda, NVIDIA RTX A5000; connectome: data/connectome.npz; torch 2.11.0+cu128)

Action latency = one recurrent step at B=1 (p50 / p95 over 200 steps). Train step = forward+backward, B=32, T=24. MFLOP = analytic multiply-adds per sample per step. Peak memory = CUDA max allocated (GPU) or RSS growth (CPU).

| model | neurons | params | synapses | MFLOP/step | action p50 ms | p95 ms | train step s | peak MB | ckpt MB |
|---|---|---|---|---|---|---|---|---|---|
| connectome:1 | 6935 | 3.71M | 91,663 | 7.4 | 0.28 | 0.29 | 0.14 | 790 | 16 |
| connectome:2 | 13870 | 7.43M | 183,326 | 14.8 | 0.28 | 0.29 | 0.26 | 1486 | 32 |
| connectome:3 | 20805 | 11.14M | 274,989 | 22.1 | 0.29 | 0.30 | 0.45 | 2240 | 48 |
| shuffled:1 | 6935 | 3.71M | 91,663 | 7.4 | 0.28 | 0.29 | 0.14 | 784 | 16 |
| random:1 | 6935 | 3.71M | 91,663 | 7.4 | 0.28 | 0.29 | 0.15 | 789 | 16 |
| gru | 762 | 3.71M | 580,644 | 7.4 | 0.28 | 0.28 | 0.01 | 124 | 14 |
| dense | 6935 | 51.72M | 48,094,225 | 103.4 | 0.65 | 0.67 | 0.13 | 1285 | 197 |
