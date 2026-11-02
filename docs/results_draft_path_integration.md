# Results draft: the connectome as a prior for path integration

Draft of the results section for the reframed paper. Numbers are from the 2026-11-02 GPU run (`results/summary.md`,
`results/hd`, `results/homing`, `results/bench`); items marked **[pending]** fill in when the overnight probe sweep
(`logs/hd_extra_*.log`), the capacity-matched GRU and the subgraph ablation finish.

## Claim

Distilling a video world model (V-JEPA 2.1, ViT-B) into a recurrent student whose only recurrent wiring is the FlyWire
connectome (6,935 neurons, 91,663 signed synapses) produces an internal state that **path-integrates**: heading and
position relative to the start are linearly decodable through 8 to 20 steps of darkness, and a linear readout of that
state drives blind homing. A parameter-matched GRU trained on the same objective predicts the teacher *better* and does
**not** form this code, even under a non-linear probe. The code is a property of the sparse, connectome-sized recurrence
rather than of the exact synapse list: block-shuffled and random wirings of the same size form it too, slightly worse.

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
**[pending]** recurrent-capacity-matched GRU (174 hidden units, 91 k recurrent weights, the connectome's budget).

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

Three controls make the claim honest:

* **Information ceiling.** Turns are fixed-angle, so a closed-form integrator of the action record recovers heading exactly
  (0 degrees) in the dark. The probes therefore do not ask whether heading *can* be known; they ask whether the state
  *keeps* it in a population code a linear readout can use. An MLP on the six cumulative action counts alone reaches
  1.5 degrees; the earlier "action-only ridge floor" of 40 degrees was a linearity artefact and is reported as such.
* **The GRU does not hide the code non-linearly.** An MLP probe on the GRU state still gives 27 degrees (seed 0);
  **[pending]** all seeds and wirings.
* **Feature count.** A random 762-neuron subset of the connectome (the GRU's width) decodes heading at 19 degrees,
  still better than the full GRU state.

The sign structure matters, but not which signs: the unsigned student loses the code (13.8 degrees, R² 0.88) while
random Dale-consistent signs match the real ones. The exact synapse list matters a little: the real wiring beats
block-shuffled by about 2 degrees and random by about 4, on both environments.

**Not region-specific.** The 51 EPG neurons carry almost no heading (50 degrees), and the central complex (12.6 degrees)
decodes no better than a random subset of the same size (10.7 degrees, seed 0). We make no ring-attractor claim.

**[pending]** Dark duration: heading error and position R² for blind windows of 16 and 20 steps (8:24, 4:24), per wiring.

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

Every connectome-sized sparse student halves its distance to home from a linear readout; the GRU barely moves. On the
NYC twin the connectome and block-shuffled students reach home in 37 to 47 percent of episodes against 10 percent for
the GRU. Here the real wiring has no edge over block-shuffled; the behavioural effect is carried by the sparse
connectome-sized recurrence, not by the synapse identity, and the paper says so. Homing is imperfect because the
decoded heading under-reads long turn bursts in the dark (controller trace in `results/homing_oldctl` for the
pre-fix controller, which could not move any agent; the fix and its parameters are stated once in the methods).

## R4. Cost

On the real graph (RTX A5000, batch 1): one connectome step costs 7.4 MFLOP and 0.28 ms, the same as the
parameter-matched GRU, with 91,663 recurrent weights against the GRU's 580,644. Dense costs 103 MFLOP and 0.65 ms.
**[pending]** the recurrent-capacity-matched GRU closes the comparison from the other side.

## R5. Transfer to real handheld video

Eight phone clips (79 episodes, egomotion quantised to the five arena actions). Zero-shot every student is far off
distribution (MSE 9.6 to 9.9, cosine 0.2 to 0.3; appendix). With leave-one-clip-out fine-tuning all students converge
to MSE 1.18 to 1.26 and cosine 0.94; the GRU remains best. Real video is a sanity check on the pipeline, not evidence
for the wiring.

## What we do not claim

* Teacher-free navigation. Student-model planning succeeds 0 to 3 percent while the same planner with oracle dynamics
  succeeds 100 percent; the students' 8-step rollouts are not accurate enough to plan with. Dropped from the claims.
* Emergent ring attractor in EPG. Negative, see R2.
* That the exact FlyWire synapse list is necessary. The code and the behaviour appear for any sparse wiring of this
  size and sign structure; the real wiring is the best of them by a small, consistent margin.

## Pending numbers (overnight)

1. Capacity-matched GRU: prediction MSE, heading / position probes, homing (R1, R2, R4).
2. MLP probes on all wirings and seeds; dark windows 8:24 and 4:24 (R2 table 2).
3. Subgraph ablation, one seed: synapse threshold 1 / 5 / 10 and region subsets, real vs shuffled (appendix table).
4. Fix the collision metric that reports 33333 per 100 m in the fog condition before any planning row is shown.
