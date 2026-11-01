---
title: "[P0] Degree-shuffled control destroys region block structure; add block-preserving rewire"
labels: P0, controls, experiment
---
## Problem
`build_mask` in `wm2cx/student.py` permutes post-synaptic endpoints globally. This changes the macro-architecture, not just synapse identity:

| block | real edges | shuffled |
|---|---|---|
| CX→CX | 71,415 | 56,737 |
| CX→OUT | 237 | 9,741 |
| VIS→CX | 619 | 5,771 |

So "wiring identity beats degree-matched control" may only mean "region-level block structure matters".

## Fix
- Add `wiring=block_shuffled`: degree-preserving rewire **within each (pre-region, post-region) block** (stochastic-block-model control).
- Add a Maslov–Sneppen switch rewire so the edge count is exact (shuffle currently yields 90,388 unique edges + 32 self-loops vs 91,663 real; ER yields 91,595).
- Run 5 seeds on arena + NYC.

## Acceptance
Real wiring beats block-shuffled with non-overlapping 95% CIs, or the claim is rescoped to macro-architecture.
