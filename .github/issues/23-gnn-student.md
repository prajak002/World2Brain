---
title: "[P1] Graph neural network student on the connectome (--wiring gnn)"
labels: P1, method, experiment
---
**Owner:** Atharva  ·  **Priority:** P1  ·  **Related:** #18

## Why this matters
The connectome student has exactly one trainable number per synapse (91,663). The matched GRU has 0.7 M recurrent weights. The GRU wins on prediction; we cannot tell whether that is because it has more capacity or because the fly's wiring is wrong for the task. A graph neural network gives the **real graph** vector capacity without changing who talks to whom.

## What the student already is
One round of message passing per time step with a scalar message `w_ij * h_j`, sum aggregation, tanh update (`FlyBrain.step` in `wm2cx/student.py`, edge list + `index_add`).

## What to build: `--wiring gnn` in `scripts/train.py`, class `FlyGNN` in `wm2cx/student.py`
1. Node state `h_i` of dimension d (try 4, 8, 16) per neuron instead of a scalar.
2. Message on each real edge `m_ij = s_ij * softplus(w_ij) * phi(h_j)`, with `phi` a small MLP shared per (pre cell type, post cell type). Sign `s_ij` stays fixed, so Dale's law survives. Sum with `index_add`, cost O(E d).
3. Node update: a GRU cell shared per cell type, input = summed messages + visual input (VIS only) + action input (CX, OUT).
4. Readout: per-type linear pool, then the existing 256-dim linear readout.
5. Hold the total at 3.71 M parameters (`param_breakdown`), shrink d or share more until it matches.

## The control that makes it a result
Train the same GNN on `connectome`, `block_shuffled` and `random`, 3 seeds. Beats the GRU only on the real graph -> wiring matters. Beats it on all three -> the gain is the message function. Also run d = 1 with learned `phi`.

## Order of work (about one week)
d = 1 reproducing `FlyBrain` when `phi` is identity -> d = 8 on the arena -> three wirings -> probes and homing through the existing `eval_hd.py` / `eval_homing.py` -> `_tn` data. Tests in `tests/test_student.py`. Full write-up in README "Graph neural network student".
