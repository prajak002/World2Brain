# Table 3: compute cost (device: cpu, cpu x16; connectome: synthetic (matched sizes); torch 2.13.0+cpu)

Action latency = one recurrent step at B=1 (p50 / p95 over 50 steps). Train step = forward+backward, B=32, T=24. MFLOP = analytic multiply-adds per sample per step. Peak memory = CUDA max allocated (GPU) or RSS growth (CPU).

| model | neurons | params | synapses | MFLOP/step | action p50 ms | p95 ms | train step s | peak MB | ckpt MB |
|---|---|---|---|---|---|---|---|---|---|
| connectome:1 | 6935 | 7.25M | 91,570 | 14.5 | 6.63 | 16.73 | 6.83 | n/a | 29 |
| connectome:2 | 13870 | 14.50M | 183,140 | 28.9 | 5.30 | 7.39 | 4.05 | n/a | 59 |
| connectome:3 | 20805 | 21.75M | 274,710 | 43.4 | 7.14 | 8.29 | 9.84 | n/a | 88 |
| shuffled:1 | 6935 | 7.25M | 91,583 | 14.5 | 11.81 | 46.59 | 2.90 | n/a | 29 |
| gru | 1400 | 9.50M | 5,880,000 | 19.0 | 2.53 | 3.31 | 1.56 | n/a | 36 |
| dense | 6935 | 55.25M | 48,094,225 | 110.5 | 14.15 | 16.82 | 8.75 | n/a | 211 |
