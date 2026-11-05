---
title: "[P1] Show the effect is not one graph"
labels: P1, experiment
---
**Owner:** Atharva  ·  **Priority:** P1  ·  **Related:** #13

## Why this matters
Every result is on one subgraph of one fly. "One graph, one species" is the first attack a reviewer will make.

## What to do
1. `scripts/ablate_connectome.sh` on the run-2 controls: `min_syn` 1 / 3 / 5 and region subsets, 3 seeds.
2. Multi-brain K = 1, 2, 3 (already in `scripts/train.py --n_brains`) with bootstrap CIs on the new controls.
3. A second connectome through the same pipeline (`wm2cx/connectome.py`): larval Drosophila CNS (Winding et al. 2023) or hemibrain central complex only.

## Done when
A figure showing the probe / homing effect across subgraphs and across K, and one row for a second connectome.
