---
title: "[P1] Show the effect is not one graph"
labels: P1, experiment
---
**Owner:** Atharva

## Problem
One connectome subgraph, one species.

## Fix
1. `scripts/ablate_connectome.sh` on the new controls (min_syn 1 / 3 / 5, region sets), 3 seeds.
2. K = 1, 2, 3 with bootstrap CIs (exists, needs run 2 controls).
3. A second connectome: larval Drosophila CNS (Winding 2023) or hemibrain CX-only; same pipeline via `wm2cx/connectome.py`.
