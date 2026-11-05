---
title: "[P0] Find a task the connectome wins against the matched GRU"
labels: P0, experiment
---
**Owner:** Atharva

## Problem
Everything so far is "the connectome encodes heading where the GRU does not" while the GRU predicts better. A spotlight needs an outright win on a task.

## Candidates (all cheap on the L4)
1. Blind homing on continuous-turn data (`_tn`, run 2 stage `homing`): success within radius, not R².
2. Longer blackouts: probes at 8:24 and 4:24 (run 2 `_b8_24`), homing with `--t_blind` larger than training windows.
3. Heading under action noise at test time (train clean, test with turn noise): does the GRU's dead reckoning collapse while the connectome's does not?
4. Low-data regime: data-efficiency axis (`run_seeds.sh` n-subsets) where the fixed graph may act as a prior.
Report whichever wins honestly, with all seeds; do not cherry-pick the blackout length.
