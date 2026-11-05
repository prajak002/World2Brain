---
title: "[P1] A distillation objective that uses the circuit structure"
labels: P1, method
---
**Owner:** Prajak

## Problem
Latent regression rewards whatever carries the V-JEPA latent, which free weights do better. Heading in the connectome state is an unsupervised by-product.

## Fix
Two one-day ablations: (a) auxiliary dead-reckoning loss on CX neurons only during blind steps; (b) contrastive term matching the state across the blackout to the post-blackout teacher latent. Report as a design choice with the plain loss as the ablation. Keep the sign/mask constraints.
