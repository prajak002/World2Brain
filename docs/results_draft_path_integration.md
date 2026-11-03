# Results draft: the connectome as a prior for path integration

Draft of the results section for the reframed paper. Numbers are from the 2026-11-02 GPU run (`results/summary.md`,
`results/hd`, `results/homing`, `results/goalnav`, `results/bench`), complete as of 2026-11-03 (overnight probe sweep,
capacity-matched GRU, subgraph ablation, goal-vector navigation).

## Claim

Distilling a video world model (V-JEPA 2.1, ViT-B) into a recurrent student whose only recurrent wiring is the FlyWire
connectome (6,935 neurons, 91,663 signed synapses) produces an internal state that **path-integrates**: heading and
position relative to the start are linearly decodable through 8 to 20 steps of darkness, and a linear readout of that
state drives blind homing. A parameter-matched GRU trained on the same objective predicts the teacher *better* and does
**not** form this code, even under a non-linear probe, and neither does a GRU matched to the connectome's recurrent
weight budget. The code is a property of the sparse, connectome-sized recurrence rather than of the exact synapse list:
block-shuffled and random wirings of the same size form it too, slightly worse. The behavioural readout is the weak
link: homing from the state is well above the random floor but far below the dead-reckoning ceiling, and goal-vector
navigation from the state does not beat random actions (reported as negative results).

## Setup (one paragraph for the paper)

Students are leaky tanh rate networks, h <- (1-a) h + a tanh(W h + W_in o * visible + W_act a + b), trained to predict the
teacher's PCA-256 latent at every step of 24-step episodes with random blind windows (vision cut, actions still given).
Controls fix neuron count, edge count, input routing and parameter count: block-shuffled (within-region degree-preserving),
degree-shuffled (global), Erdos-Renyi random, unsigned, random-sign, dense, and a GRU with the same total parameter count
(762 hidden units). Five seeds on the arena, three on the NYC raycast twin; teacher targets are causal; learning rates were
chosen per architecture on a validation split. All probes are ridge regressions fit on training episodes and evaluated on
held-out test episodes; the blind window is steps 8 to 16 unless stated.

## R1. Prediction accuracy is not what the wiring buys

Blind-horizon latent MSE on the arena (mean over horizons 1 to 8, mean ± sd over 5 seeds):

| wiring | MSE | cos |
|---|---|---|
| GRU, parameter-matched | 0.434 ± 0.004 | 0.722 |
| dense | 0.562 ± 0.004 | 0.632 |
| connectome | 0.579 ± 0.003 | 0.618 |
| random | 0.583 ± 0.002 | 0.613 |
| block-shuffled | 0.585 ± 0.005 | 0.612 |
| degree-shuffled | 0.610 ± 0.002 | 0.589 |
| copy-last-visible (trivial) | 0.811 | - |
| constant mean (trivial) | 0.992 | - |

Every student beats the trivial predictors. The real connectome beats its block-shuffled and random controls by about
1 percent, consistently across seeds (paired p < 0.01) but small. The GRU is 25 percent better than any sparse student,
and the ordering is the same on the NYC twin (GRU 0.609, connectome 0.701, random 0.707, block-shuffled 0.726) and at
every training-set size from 50 to 500 episodes. We therefore make no claim that the connectome is a better predictor.
A second GRU matched to the connectome's *recurrent* budget instead (174 hidden units, 90,828 recurrent weights, 0.54 M
parameters in total) reaches MSE 0.501 ± 0.001 (cos 0.671, 3 seeds): still 13 percent better than the connectome with a
seventh of its parameters. Neither GRU is the point of comparison the wiring wins on.

## R2. The connectome state carries a linearly readable heading and position code in the dark

Median heading error (degrees) and position R² of a ridge probe on the full state during the blind window
(mean ± sd over seeds; NYC twin in the right columns):

| wiring | heading err, arena | pos R², arena | heading err, NYC | pos R², NYC |
|---|---|---|---|---|
| connectome | 9.1 ± 1.2 | 0.94 ± 0.01 | 9.4 ± 0.7 | 0.94 ± 0.01 |
| connectome, random signs | 9.2 ± 1.3 | 0.94 ± 0.01 | - | - |
| block-shuffled | 10.8 ± 0.8 | 0.93 ± 0.01 | 10.1 ± 1.1 | 0.94 ± 0.02 |
| degree-shuffled | 12.7 ± 0.6 | 0.89 ± 0.02 | 14.3 ± 2.0 | 0.84 ± 0.01 |
| random | 12.9 ± 0.7 | 0.88 ± 0.01 | 15.2 ± 0.7 | 0.89 ± 0.00 |
| connectome, unsigned | 13.8 ± 0.8 | 0.88 ± 0.01 | - | - |
| dense | 21.3 ± 0.7 | 0.68 ± 0.01 | - | - |
| GRU, parameter-matched | 30.6 ± 0.6 | 0.41 ± 0.01 | 45.5 ± 2.6 | 0.25 ± 0.01 |
| GRU, recurrent-budget-matched | 35.5 ± 0.7 | 0.30 ± 0.01 | - | - |

Three controls make the claim honest:

* **Information ceiling.** Turns are fixed-angle, so a closed-form integrator of the action record recovers heading exactly
  (0 degrees) in the dark. The probes therefore do not ask whether heading *can* be known; they ask whether the state
  *keeps* it in a population code a linear readout can use. An MLP on the six cumulative action counts alone reaches
  1.5 degrees; the earlier "action-only ridge floor" of 40 degrees was a linearity artefact and is reported as such.
* **The GRU does not hide the code non-linearly.** An MLP probe on the GRU state gives 27.4 ± 0.3 degrees (3 seeds)
  against 30.6 for ridge; on the sparse students the same MLP probe *improves* heading to 4.7 ± 0.1 (connectome),
  5.9 ± 0.3 (block-shuffled), 6.4 ± 0.4 (random) and 6.9 ± 0.7 (degree-shuffled) degrees. The gap between the
  connectome and the GRU widens, not narrows, under the non-linear probe. On the NYC twin the MLP probe overfits
  (17.7 degrees against 9.4 for ridge, 3 seeds) and the ridge numbers are the ones we report.
* **Feature count.** A random 762-neuron subset of the connectome (the GRU's width) decodes heading at 19 degrees,
  still better than the full GRU state.

The sign structure matters, but not which signs: the unsigned student loses the code (13.8 degrees, R² 0.88) while
random Dale-consistent signs match the real ones. The exact synapse list matters a little: the real wiring beats
block-shuffled by about 2 degrees and random by about 4, on both environments.

**Not region-specific.** The 51 EPG neurons carry almost no heading (50 degrees), and the central complex (12.6 degrees)
decodes no better than a random subset of the same size (10.7 degrees, seed 0). We make no ring-attractor claim.

**Dark duration.** The code degrades gracefully with the length of the blind window (ridge probe, 3 seeds; median heading
error in degrees, position R²):

| wiring | 8 dark steps (8:16) | 16 dark steps (8:24) | 20 dark steps (4:24) | NYC, 16 dark | NYC, 20 dark |
|---|---|---|---|---|---|
| connectome | 9.1, 0.94 | 17.7 ± 2.1, 0.82 | 12.6 ± 1.2, 0.82 | 12.2 ± 2.2, 0.88 | 9.5 ± 1.3, 0.88 |
| connectome, random signs | 9.2, 0.94 | 18.6 ± 1.1, 0.82 | 13.6 ± 1.4, 0.82 | - | - |
| block-shuffled | 10.8, 0.93 | 23.3 ± 0.4, 0.77 | 15.9 ± 0.6, 0.77 | 16.4 ± 1.5, 0.86 | 12.2 ± 2.0, 0.84 |
| degree-shuffled | 12.7, 0.89 | 24.0 ± 1.0, 0.71 | 17.2 ± 0.6, 0.76 | 25.2 ± 1.8, 0.69 | 15.9 ± 0.7, 0.73 |
| random | 12.9, 0.88 | 24.9 ± 0.6, 0.72 | 17.3 ± 0.6, 0.77 | 21.6 ± 1.1, 0.80 | 14.0 ± 0.3, 0.83 |
| dense | 21.3, 0.68 | 29.9 ± 1.1, 0.46 | 21.3 ± 0.5, 0.53 | - | - |
| GRU, parameter-matched | 30.6, 0.41 | 37.7 ± 0.8, 0.32 | 27.9 ± 0.3, 0.50 | 49.4 ± 1.9, 0.08 | 39.8 ± 1.5, 0.25 |

Over 16 dark steps the connectome keeps heading to within 18 degrees and position at R² 0.82 where the GRU is at 38
degrees and 0.32; the real wiring's margin over block-shuffled grows from 2 to 6 degrees as the dark window lengthens
(4 degrees on the NYC twin). The 4:24 window is easier than 8:24 because the probe sees more early, well-anchored steps.

## R3. The code supports behaviour: blind homing from a linear readout

The agent walks out under a scripted policy for 12 visible steps, then vision is cut and a controller steers from a
*linear readout of the state* (pose relative to the start) for 24 steps. The readout is fit on training episodes plus
300 closed-loop rollouts in fresh arenas. Success = within 1 arena unit (12 m on the NYC twin). 50 test episodes per
seed, 3 seeds; controls use the same controller:

| readout | arena: % closer | arena: success | NYC: % closer | NYC: success |
|---|---|---|---|---|
| oracle pose (controller ceiling) | 89 | 1.00 | 75 | 0.92 |
| dead reckoning from the action record | 93 | 1.00 | 76 | 0.92 |
| random actions (floor) | -67 | 0.00 | - | - |
| connectome state | 42 | 0.10 | 47 | 0.37 |
| connectome, unsigned | 47 | 0.13 | - | - |
| block-shuffled state | 44 | 0.15 | 53 | 0.47 |
| degree-shuffled state | 42 | 0.14 | - | - |
| random state | 34 | 0.09 | - | - |
| dense state | 36 | 0.11 | - | - |
| GRU state | 13 | 0.04 | 18 | 0.10 |
| GRU, recurrent-budget-matched, state | 29 | 0.15 | - | - |

Every connectome-sized sparse student roughly halves its distance to home from a linear readout; the parameter-matched
GRU barely moves and the recurrent-budget-matched GRU gets a third of the way. On the
NYC twin the connectome and block-shuffled students reach home in 37 to 47 percent of episodes against 10 percent for
the GRU. Here the real wiring has no edge over block-shuffled; the behavioural effect is carried by the sparse
connectome-sized recurrence, not by the synapse identity, and the paper says so. Homing is imperfect because the
decoded heading under-reads long turn bursts in the dark (controller trace in `results/homing_oldctl` for the
pre-fix controller, which could not move any agent; the fix and its parameters are stated once in the methods).

## R4. Cost

On the real graph (RTX A5000, batch 1): one connectome step costs 7.4 MFLOP and 0.28 ms, the same as the
parameter-matched GRU, with 91,663 recurrent weights against the GRU's 580,644. Dense costs 103 MFLOP and 0.65 ms.
The recurrent-budget-matched GRU (174 units, 90,828 recurrent weights) closes the comparison from the other side: at
the connectome's recurrent budget it predicts better (R1) but does not path-integrate (R2, R3). Its per-step latency was
not benchmarked separately; at B=1 every model here is launch-bound at about 0.3 ms.

## R5. Transfer to real handheld video

Eight phone clips (79 episodes, egomotion quantised to the five arena actions). Zero-shot every student is far off
distribution (MSE 9.6 to 9.9, cosine 0.2 to 0.3; appendix). With leave-one-clip-out fine-tuning all students converge
to MSE 1.18 to 1.26 and cosine 0.94; the GRU remains best. But the copy-last-visible baseline on the same episodes
scores 0.84, so **no student beats the trivial predictor on real video**, and the heading probe has negative R² there.
Real video is a sanity check on the pipeline, not evidence for the wiring; it is reported as a negative result.

## What we do not claim

* Teacher-free navigation. Student-model planning succeeds 0 to 3 percent while the same planner with oracle dynamics
  succeeds 100 percent; the students' 8-step rollouts are not accurate enough to plan with. Dropped from the claims.
* Goal-vector navigation from the state. A controller that turns toward a linearly decoded egocentric goal vector
  succeeds in 22 percent of arena episodes for every wiring (3 seeds; GRU 22, random wiring 21, dense 20) against
  14 percent for random actions and 92 percent for the true goal vector; 5 percent on the NYC twin against 0 and 28.
  Cutting vision for 3 or 5 steps changes nothing. The decoded goal vector (R² 0.42 on closed-loop rollouts) is too
  coarse to steer with, and no wiring is better than another. Reported in the appendix as a negative result.
* Any effect of the exact synapse list on behaviour: on the NYC twin block-shuffled homing (0.47) beats the real wiring
  (0.37), within seed noise.
* Emergent ring attractor in EPG. Negative, see R2.
* That the exact FlyWire synapse list is necessary. The code and the behaviour appear for any sparse wiring of this
  size and sign structure; the real wiring is the best of them by a small, consistent margin.

## Appendix table: subgraph ablation (seed 0, arena, blind MSE 8:16, real vs degree-shuffled of the same subgraph)

| subgraph | connectome | shuffled | margin |
|---|---|---|---|
| full graph (synapse threshold 5, 91,663 synapses) | 0.575 | 0.610 | 0.035 |
| threshold 1 (all synapses) | 0.618 | 0.661 | 0.043 |
| threshold 10 | 0.584 | 0.604 | 0.020 |
| central complex only | 0.588 | 0.608 | 0.020 |
| mushroom bodies removed | 0.582 | 0.609 | 0.027 |
| output neurons removed | 0.588 | 0.610 | 0.022 |

The real wiring beats its shuffled twin on every subgraph by 0.02 to 0.04 MSE (one seed each; the full-graph margin is
0.035 ± 0.005 over 5 seeds). No single region carries the margin: removing the mushroom bodies or the output neurons, or
keeping only the central complex, leaves it intact.

## Status

All numbers above are final for the 2026-11-02 run. The collision-rate metric that reported 33333 per 100 m in the fog
condition was a per-episode mean with a zero-metre episode; `eval_plan.py` now pools collisions over metres and
`summarize.py` blanks the stale value. Multi-brain scaling (K = 2, 3) and the data-efficiency curve are in
`results/summary.md` and change none of the conclusions.
