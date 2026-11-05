---
title: "[P2] Larger teacher and real odometry for the real-video stage"
labels: P2, data
---
**Owner:** Priyanshi  ·  **Priority:** P2

## Why this matters
Two caveats a reviewer will raise: (1) only one teacher, ViT-B at 256 px; (2) on the real phone clips the actions are estimated from optical flow, so the heading probe on real video has no ground truth.

## What to do
1. Teacher: V-JEPA 2.1 ViT-L at 384 px, re-collect targets with `scripts/collect.py --causal 1` (one GPU day), retrain the connectome and GRU, 3 seeds.
2. Odometry: add one public egocentric dataset with real odometry (RECON or SCAND) to `scripts/ingest_video.py`, so flow-estimated actions can be validated against ground truth.

## Done when
One row with the ViT-L teacher in the prediction table, and a validation plot of flow-estimated vs true actions.
