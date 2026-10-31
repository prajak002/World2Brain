---
title: "[P0] Teacher latents are non-causal: blind-window targets encode future frames"
labels: P0, data, teacher
---
## Problem
`scripts/collect.py` computes V-JEPA latents per 8-step clip (frames duplicated, 1 tubelet = 1 step) with bidirectional attention. The latent at step t contains information from frames up to the end of its clip. The eval blind window (steps 8–15) is exactly clip 2, so the student is asked to predict a target that already encodes what it is blind to.

## Fix
- Causal targets: sliding window ending at t (clip = steps t-7..t), or a causal attention mask.
- Rerun all E1/E3 runs. Report whether the leak favoured any wiring.
