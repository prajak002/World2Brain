---
title: "[P2] Unidirectional LSTM at the same parameter budget"
labels: P2, baseline
---
**Owner:** Priyanshi

## Problem
GRU is the only gated recurrent baseline. Reviewers ask "why not LSTM".

## Fix
`--wiring lstm` in `scripts/train.py`, hidden size chosen by `gru_hidden_for` logic (4 h^2 + 4 h d <= budget), 3 seeds, arena + tn. Expected within noise of the GRU. A *bidirectional* LSTM is non-causal (sees frames inside the blind window) and must not be used as a baseline; at most as a labelled non-causal upper bound.
