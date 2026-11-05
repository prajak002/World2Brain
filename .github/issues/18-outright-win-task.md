---
title: "[P0] Blind return after a long blackout: the task the connectome wins outright"
labels: P0, experiment
---
**Owner:** Atharva  ·  **Priority:** P0

## The task
Blind return after a long blackout (dead-reckoning homing). The agent sees the world for 12 steps while it is driven
away from a start point, then the screen goes black for 24 to 40 steps and it must walk back to the start using only its
own actions and its internal state. Success = ending within a radius of the start (12 m on the NYC twin, 1 unit on the
arena). The controller is fixed and shared; only the state readout differs between wirings, so the task isolates the one
thing the connectome state does better than everything else: holding heading and position through darkness.

## Why this is the right shape
- Prediction lets a GRU win by modelling appearance. In the dark there is no appearance, only integration.
- The oracle pose scores 1.00, so the ceiling is high and the task is solvable.
- Dark length is a dial: heading error after 8 dark steps is 9 deg (connectome) vs 31 deg (GRU), and the gap widens with
  length. Success vs dark length is the figure: the GRU collapses early, the connectome holds.

## What run 2 already shows (NYC twin, 30 dark steps, 3 seeds x 50 episodes, `results/homing/*_nyc_nyc_state.json`)
| wiring | success | 95% CI |
|---|---|---|
| connectome | 0.47 | +-0.08 |
| block-shuffled | 0.51 | +-0.08 |
| GRU, parameter-matched | 0.12 | +-0.05 |
| oracle pose | 1.00 | |

Connectome vs GRU is already non-overlapping. The steps below turn it into a clean claim.

## What to do (all on the L4 with the run-2 checkpoints; `scripts/eval_homing.py`)
1. Missing rows: transformer and random wiring on NYC homing (`--env nyc --t_blind 30 --radius 12`), the
   budget-matched GRU, and the LSTM when #24 lands. The win must hold against every baseline.
2. 5 seeds x 100 episodes per cell; bootstrap the CI over episodes and seeds. Connectome seed 1 scored 0.28 vs 0.68 for
   seed 0, so the spread is real and must be shown.
3. Sweep dark length `--t_blind 8 16 24 30 40` for every wiring. Plot success vs dark length; that is the paper figure.
4. Continuous turn angles (`--turn_noise 0.3` on the `_tn` checkpoints) so action counting cannot solve it. Report the
   action-only dead-reckoning floor (`--readout actions`) on the same plot. On the arena the connectome is at 15 deg
   heading error vs a 40 deg action-only floor, so the margin survives.
5. Readout as an MLP on >=300 rollouts (`--readout mlp --fit_rollouts 300`). On NYC goal-nav the MLP readout gave
   0.27 vs 0.18 ridge, so the linear readout understates every wiring.

## The honest caveat that goes into the claim
Block-shuffled ties or beats the real wiring. So the outright win is "connectome-scale sparse signed recurrence beats
dense recurrent and attention baselines at blind return", not "the real synapse list beats shuffled wiring". To test
the real wiring itself, push dark length to 40+ steps with continuous turns, where the 2 to 6 deg heading margin over
shuffled compounds into position error. If that curve separates, it is a second, stronger claim; if not, say so.

## Rules
Run every seed for every condition, report whichever wins honestly, and fix the dark lengths before looking at results.

## Done when
`tab:homing` in `paper_v2/main.tex` has every baseline with bootstrap CIs, a success-vs-dark-length figure, and one
sentence in the abstract that states the win and its caveat.
