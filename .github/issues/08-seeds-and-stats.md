---
title: "[P1] Seeds and confidence intervals"
labels: P1, statistics
---
## Problem
2 seeds for connectome/shuffled/random; 1 seed for GRU, dense, nosign, K=2; 1 seed for every NYC run. No CIs, no paired tests over the shared test episodes.

## Fix
- 5 seeds for every condition on arena, ≥3 on NYC.
- Paired bootstrap 95% CI per horizon over test episodes; add to `summarize.py`.
- Report at least one alternative blind window besides steps 8–15.
