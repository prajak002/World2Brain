---
title: "[P0] Add trivial baselines (constant-mean, copy-last) to every prediction table"
labels: P0, evaluation
---
## Problem
No table reports a trivial predictor. Recomputed in the same PCA-256 space (blind h1–h8 MSE):

| | const-mean | copy-last-visible | best student |
|---|---|---|---|
| arena | 1.043 | 0.97–1.19 | 0.544 GRU / 0.625 connectome |
| NYC | 0.977 | 1.46–1.64 | 0.885 connectome |

On NYC the connectome is only 9% better than predicting the mean, one seed, 30 episodes.

## Fix
- Add `const_mean` and `copy_last` rows to `scripts/summarize.py` and README tables.
- Also report MSE in the original 3,840-d teacher space, not only whitened PCA space.
- Reframe NYC as a pilot until scaled.
