---
title: "[P1] GRU and dense baselines are not parameter-, input- or tuning-matched"
labels: P1, baselines
---
## Problem
- GRU 9.50M params vs 7.25M connectome.
- GRU receives an explicit `visible` flag and observations at every unit; FlyBrain gets obs only at VIS neurons and no flag.
- Dense trained with lr 1e-3, others 2e-3. No per-architecture hyperparameter search.

## Fix
- GRU hidden size chosen to match total params after #3; give FlyBrain the same `visible` flag (or remove it from GRU).
- Small lr sweep {5e-4, 1e-3, 2e-3, 4e-3} per architecture, pick on a validation split.
- 3–5 seeds each.
