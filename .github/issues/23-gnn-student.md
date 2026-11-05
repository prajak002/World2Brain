---
title: "[P1] Graph neural network student on the connectome (--wiring gnn)"
labels: P1, method, experiment
---
**Owner:** Atharva

## Problem
The connectome student has one scalar magnitude per synapse (91,663 free recurrent parameters). The matched GRU has 0.7 M. The GRU wins on prediction; the paper cannot tell whether that is capacity or wiring.

## Build
`FlyGNN` in `wm2cx/student.py`, selected by `--wiring gnn` in `scripts/train.py`:
1. d-dim node state per neuron (d in {1, 4, 8, 16}).
2. Message on each real edge: `s_ij * softplus(w_ij) * phi_type(h_j)`, `phi` a small MLP shared per (pre type, post type); sign fixed (Dale). Sum via `index_add` on the edge list, O(E d).
3. Node update: GRUCell shared per cell type, inputs = aggregated message + `W_in o_t` (VIS only) + `W_act a_t` (CX, OUT).
4. Readout: per-type linear pool then the existing 256-dim linear readout.
5. Hold total parameters at 3.71 M (`param_breakdown`), shrinking d or sharing more until it matches.

## Controls
Same GNN on `connectome`, `block_shuffled`, `random`, 3 seeds, arena + `_tn`; plus d = 1 with learned phi. Only a real-graph-specific gain is a wiring result. Probes and homing through the existing `eval_hd.py` / `eval_homing.py`.

## Tests
`tests/test_student.py`: with phi = identity and d = 1 the GNN step equals `FlyBrain.step` to 1e-5; parameter count equals the budget; Dale signs unchanged after training steps.
