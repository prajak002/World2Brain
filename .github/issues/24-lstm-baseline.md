---
title: "[P2] Unidirectional LSTM at the same parameter budget"
labels: P2, baseline
---
**Owner:** Priyanshi  ·  **Priority:** P2

## Why this matters
The GRU is our only gated recurrent baseline. Reviewers will ask "why not an LSTM".

## What to do
1. Add `--wiring lstm` to `scripts/train.py`, next to the GRU path. Hidden size chosen so that `4 h^2 + 4 h d` plus input/readout equals the 3.71 M budget (same logic as `gru_hidden_for` in `wm2cx/student.py`).
2. Train 3 seeds on the arena and on the `_tn` data, 20 epochs, lr from a small validation sweep.
3. Run the probes (`scripts/eval_hd.py`) and homing (`scripts/eval_homing.py`) so it appears in every table.

## Important
Only a **unidirectional** LSTM. A bidirectional LSTM reads frames inside the blind window, which is exactly what the task forbids; it may only appear as a clearly labelled non-causal upper bound.

## Done when
An LSTM row in the prediction, probe and homing tables. Expected: within noise of the GRU.
