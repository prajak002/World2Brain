---
title: "[P1] A distillation objective that uses the circuit structure"
labels: P1, method
---
**Owner:** Prajak  ·  **Priority:** P1

## Why this matters
The training loss (next-latent regression) only rewards carrying the V-JEPA latent, which free weights do better. Any heading or position in the connectome state is an accidental by-product. If we *ask* for a path integrator, the circuit might deliver one, and the paper's claim becomes a design rather than an observation.

## What to do (two one-day ablations)
1. **Auxiliary dead-reckoning loss** on CX neurons only, during blind steps: a small linear head predicts the egocentric displacement since blackout start.
2. **Across-blackout contrastive term**: the state at the end of the blackout must match the post-blackout teacher latent better than other episodes' latents.
Keep the sign/mask constraints untouched. Report both against the plain loss on all nine cores (probes, homing, prediction).

## Done when
A table with plain / aux / contrastive x cores, and a paragraph in the method section presenting the objective as a design choice.
