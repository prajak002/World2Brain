---
title: "[P0] Latent MPC fails for every student; no closed-loop navigation claim is possible"
labels: P0, planning, experiment
---
## Problem
`scripts/eval_plan.py`: nav success 3–7% (random 7%), arrange 0%, for all wirings. Goal encoding is a nearest-latent over 8 rendered headings; CEM over 6 steps, 64 candidates, 2 iterations. Not run on NYC. Manuscript contribution C2 and Table 1 have no evidence.

## Fix, in order
1. Validate the planner with the **teacher as dynamics model** and with ground-truth state (oracle upper bounds). If those fail, the planner is the problem.
2. Longer horizon, more CEM iterations, goal latent from a frame at the goal pose.
3. Then student MPC on NYC with SPL, collision rate per 100 m, timeout, stop-at-goal.
Until this works, the paper is a representation-distillation pilot.
