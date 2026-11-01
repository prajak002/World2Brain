---
title: "[P1] Sign handling is inconsistent across controls and parametrisations"
labels: P1, controls
---
## Problem
- `shuffled` keeps presynaptic signs (Dale consistent); `random` permutes signs per edge (Dale broken). Shuffled-vs-random mixes topology with Dale's law.
- `connectome` uses `softplus(W_raw)*sign` with `|w|` init; `connectome_nosign` uses raw `W_raw*0.8`. Different effective lr and initial spectral radius, so "signs do not help MSE" is uninterpretable.

## Fix
- ER control: assign signs per presynaptic neuron (Dale consistent).
- Unsigned control: same softplus parametrisation with Dale-consistent **random** signs, or raw weights for both.
- Report both.
