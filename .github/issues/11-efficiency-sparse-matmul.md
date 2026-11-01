---
title: "[P1] Sparse student is not sparse in compute: dense N×N matrix materialised per step; K=3 OOM"
labels: P1, performance
---
## Problem
`FlyBrain.W()` builds a dense 6,935² matrix each forward pass; K=3 (20,805 neurons) swaps 16 GB. Any efficiency claim (latency, FLOPs, memory) is false as implemented, and checkpoints are 420 MB.

## Fix
- `torch.sparse` or index-add recurrence over the edge list.
- Measure p50/p95 action latency, FLOPs, peak memory for every model on identical hardware; add Table 3.
- Rerun K=3 and a fixed-total-parameter single-brain control for the multi-brain claim.
