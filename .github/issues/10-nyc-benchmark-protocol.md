---
title: "[P1] NYC benchmark: geographic split, scale, blackout-in-the-loop, appearance shifts"
labels: P1, data, benchmark
---
## Problem
NYC data is one location (Times Square), random episode split, 120 train / 30 test, medium-quality WebGL renders, no weather/density/appearance variation. Manuscript promises a 70/15/15 geographically disjoint split, blackout stress tests of 1/3/5 s during obstacle encounters, and shift robustness.

## Fix
- Define train/val/test blocks before training; ≥3 locations.
- Scale collection (80 s/episode on M1; needs a GPU box or the Windows/UE5 client).
- Closed-loop blackout and appearance-shift evals after #9.
