---
title: "[P0] Closed-loop navigation on 3D Gaussian splat twins of the phone clips"
labels: P0, planning, experiment
---
**Owner:** Priyanshi

## Problem
No closed-loop result on real imagery; the NYC twin is a raycast box world. This is the computer-vision contribution a CVPR audience expects.

## Plan
1. Per clip: COLMAP poses + sparse cloud, then Splatfacto (nerfstudio) or gsplat, ~30 min per clip on the L4. Validate PSNR on held-out frames.
2. `wm2cx/env_splat.py`: same interface as the arena env (reset(pose), step(action) -> frame), agent constrained to a tube around the capture path.
3. Fix the goal-latent cost first (#9: 0 % success even with oracle dynamics); goal as pose, then as goal image.
4. Protocol: start/goal along the path, blackout stretches 1/3/5 steps, metrics success, SPL, collisions, plus oracle-dynamics ceiling. Nine cores.
Depends on #9. Not this submission.
