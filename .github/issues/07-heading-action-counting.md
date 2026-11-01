---
title: "[P1] Heading probe may measure action counting; add action-only baseline and drop EPG ring claim"
labels: P1, evaluation, neuroscience
---
## Problem
Turn actions are discrete, fixed-angle, and fed one-hot to CX/OUT neurons. Relative heading = cumulative turn count, which any leaky integrator exposes linearly. The probe also uses 6,935 features for FlyBrain vs 1,400 for GRU. EPG alone: R² ≤ 0.2 (51 neurons), so there is no ring attractor.

## Fix
- Ridge baseline on cumulative one-hot actions alone; report alongside.
- Probe with matched feature count (random subset of 1,400 neurons).
- Optional: continuous or noisy turn angles so integration is non-trivial.
- Remove "emergent EPG head-direction code" from title/claims; keep population-level claim with caveat.
