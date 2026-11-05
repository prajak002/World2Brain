---
title: "[P0] Find a task the connectome wins against the matched GRU"
labels: P0, experiment
---
**Owner:** Atharva  ·  **Priority:** P0

## Why this matters
Right now the paper can only say "the connectome *encodes* heading where the GRU does not", while the GRU still *predicts* better. For a spotlight we need one task where the connectome student beats the parameter-matched GRU outright, on a metric people care about.

## Where to look (all cheap on the L4, run-2 checkpoints already exist)
1. **Blind homing on continuous-turn data** (`_tn` runs, `scripts/eval_homing.py --turn_noise 0.3`): report success within a radius, not just R².
2. **Longer blackouts**: probes at `--blind 8:24` and `4:24` (`scripts/eval_hd.py`), homing with `--t_blind` beyond the training window.
3. **Action noise at test time**: train clean, test with turn noise. Does the GRU's dead reckoning collapse while the connectome's holds?
4. **Low data**: the data-efficiency subsets in `scripts/run_seeds.sh`; a fixed graph may act as a prior.

## Rules
Run all three seeds for every condition, report whichever wins honestly, and never pick the blackout length after seeing the result.

## Done when
One table where connectome > GRU with non-overlapping CIs, plus the conditions where it does not, both in `paper_v2/sec/4_experiments.tex`.
