---
title: "[P0] Closed-loop navigation on 3D Gaussian splat twins of the phone clips"
labels: P0, planning, experiment
---
**Owner:** Priyanshi  ·  **Priority:** P0  ·  **Depends on:** #9 (goal cost / controller fix)

## Why this matters
Everything in the paper is open-loop or on a box-world twin. A CVPR audience expects an agent that drives itself through a **real** place. We cannot walk a robot around a crowded street at will, so we rebuild the street from the phone clips and let the agent move inside the reconstruction.

## What to build
1. **Reconstruct each clip**: COLMAP for camera poses and a sparse cloud, then a 3D Gaussian splat with Nerfstudio Splatfacto or gsplat (~30 min per clip on the L4). Check PSNR on held-out frames.
2. **Environment wrapper** `wm2cx/env_splat.py` with the same interface as the arena (`reset(pose)`, `step(action) -> frame, done`), the agent kept in a tube around the capture path where the splat is valid.
3. **Protocol**: start/goal poses along the path, blackout stretches of 1 / 3 / 5 steps, nine cores, oracle-dynamics ceiling. Metrics: success, SPL, collisions.
4. Use the goal-vector controller from `scripts/eval_goalnav.py` (not raw latent MPC, which fails, see #9).

## Data
`data/realvid_all.npz` (9 phone clips, 82 episodes); raw clips on the laptop / Kaggle mirror `popsensharma/world2brain-realvid`.

## Done when
At least three reconstructed clips, a closed-loop table across the nine cores with blackouts, and a figure of one episode. Not this submission; next cycle.
