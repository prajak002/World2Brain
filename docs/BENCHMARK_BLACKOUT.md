# Blackout Navigation: the benchmark where the connectome stands out (spec, 2026-11-05)

Owners: Atharva (sweeps, part A) and Prajak (new variants, metric, figure, part B), in parallel. Issue #18.

## 1. The pitch, in CVPR words

Every embodied-vision agent loses its camera at some point: darkness, occlusion, a tunnel, a dropped frame, a sensor
fault. A world model that stops being useful the moment the image stops is not a world model of the world, only of the
image. **Blackout Navigation** measures how long a latent world model's *state* keeps an agent on course when vision is
cut, and which recurrent wiring buys the longest memory at a fixed parameter budget. The result we expect, and already
see on the NYC twin, is that a student wired like the fly brain keeps navigating for tens of steps in the dark while a
GRU, LSTM or transformer of the same size collapses within a few.

What is new: (i) the first benchmark that scores a *latent* world model by closed-loop behaviour under sensor dropout,
not by pixel or latent error; (ii) the first evidence that a biological connectome is a useful inductive bias for that
memory, with shuffled-wiring controls that say which part of the wiring matters; (iii) three worlds, from a synthetic
arena to a city twin to real handheld video, so the claim is not tied to one simulator.

## 2. Episode structure (identical across tasks)

```
 t = 0 ............ t_vis-1 | t_vis ................................ t_vis + k - 1 | (optional) re-entry
      vision ON              |                 vision OFF (black frames)            |   vision ON again
   agent is driven by a      |   agent acts from a readout of its own state;       |   one step; measure
   random outbound policy    |   actions are still fed to the student              |   latent re-entry error
```

* `t_vis = 12` steps of vision, as in training. Blackout length `k ∈ {8, 16, 24, 30, 40}`, fixed before any result is
  seen (pre-registered). 40 is beyond the training blackout window, which tests extrapolation.
* The student receives the true action at every step and a black (zero) frame during the blackout. No teacher, no
  depth, no odometry, no map. Only the student's recurrent state is available to the controller.
* The controller is **shared and fixed** across wirings: turn toward the decoded target when the decoded heading error
  exceeds `tol_deg`, else step forward; on collision use the bumper rule (two turns toward the target side, three steps
  forward). The readout that feeds it is fit per wiring on 300 oracle-driven rollouts (`--fit_rollouts 300`), ridge or
  MLP. Everything that differs between rows is the state.
* Continuous turn angles (`--turn_noise 0.3`): the action sequence alone no longer gives the pose, so an "action
  counter" cannot solve the task. The action-only dead-reckoning floor is reported in every plot.

## 3. The three tasks

| task | what the agent must do in the dark | success | why it is there |
|---|---|---|---|
| **T1 Return** (LoopNav, as in Wijmans et al. 2023) | get back to where the episode started | within 1 unit (arena) / 12 m (NYC) | pure integration of its own path; the one we already win on NYC |
| **T2 Continue** | keep heading to a goal that was visible before the blackout (goal pose shown as a frame at `t_vis-1`) | within the same radius | the natural navigation case: the goal is in memory, the path is not |
| **T3 Re-entry** | after `k` dark steps the camera returns for one frame; predict that latent before seeing it | latent MSE at re-entry vs copy-last-visible | ties the behavioural result back to the prediction objective the student was trained on |

T1 and T2 are scored with success rate and SPL (Anderson et al. 2018), T3 with MSE and its ratio to the
copy-last-visible baseline. Position drift (metres between decoded and true pose at the end of the blackout) is reported
for all three as the mechanistic quantity.

## 4. Rows (every row, every k, every world)

Students trained on the identical objective and data: connectome (FlyWire v783 subgraph), block-shuffled, random (ER),
dense, GRU parameter-matched, GRU recurrent-budget-matched, LSTM (Priyanshi, #24), transformer (K=1), and the
connectome with attention input. Controls that bound the table: oracle pose (ceiling), dead reckoning of actions
(floor under continuous turns), random actions.

Seeds: 5 per row (run 2 has 3; two more per row are 15 GPU-hours on the L4 for the arena and NYC sets). Episodes: 100
per seed per cell. Confidence intervals: bootstrap over episodes and seeds, 1000 resamples, 95%.

## 5. Worlds

1. **Arena** (`wm2cx/env.py`): fast, 5 seeds, every k, every row. The full table.
2. **NYC twin** (`wm2cx/nyc.py`, geographic train/test split): 3 seeds now, 5 after the top-up. Appearance shifts
   (dusk, overcast, fog, night) on the visible phase at k = 30 as a robustness column.
3. **Real handheld video** (later, Priyanshi #20): splat twins of the phone clips give a closed loop on real imagery;
   until then the real-video row is T3 only (open-loop re-entry on recorded clips, which needs no simulator).

## 6. The figures

* **Fig. 1 (hero):** success vs blackout length k, one line per wiring, three panels (T1, T2, T3), arena and NYC
  overlaid or side by side. Ceiling and floor as grey bands. This is the figure that has to show the GRU/transformer
  lines falling to the floor by k = 16 while the connectome line stays high to k = 30.
* **Fig. 2:** position drift vs k (the mechanism), with the heading-probe error from run 2 as an inset.
* **Table 1:** T1 and T2 at k = 30 for every row with bootstrap CIs, bold where a row beats every non-connectome
  baseline with non-overlapping CIs.

## 7. The claim we can make, and the one we cannot, written now

Claim: *sparse, signed, connectome-scale recurrence trained only on latent prediction keeps a usable navigational
state through long blackouts; dense recurrent and attention students of the same size do not.* Supported today on
NYC T1 at k = 30 (connectome 0.47 ± 0.08 vs GRU 0.12 ± 0.05, 3 × 50 episodes).

Not claimed unless the data turns: that the *real* synapse list beats block-shuffled wiring. They tie on T1 (0.47 vs
0.51). The honest test is k = 40 with continuous turns, where a 2-6° heading margin compounds into metres; if the
curves separate there, it becomes a second claim, otherwise the paper says the win belongs to the wiring class.

## 8. Work split

**Part A, Atharva (run-2 checkpoints, no new code):**
1. T1 on arena and NYC for every row at every k, `_tn` checkpoints with `--turn_noise 0.3`, readout `state` and `mlp`,
   `--fit_rollouts 300`. Script: `scripts/eval_homing.py` (flags exist). Write a `scripts/run_l4_blackout.sh` modelled
   on `scripts/run_l4_goalnav.sh` (xargs -P 2, skip finished JSONs).
2. Missing baselines on NYC: transformer, random, budget GRU; LSTM once #24 lands.
3. Bootstrap CIs: add `--bootstrap 1000` to `scripts/summarize.py` E9 table (per-episode success is already saved).

**Part B, Prajak (new code):**
1. T2 Continue: `scripts/eval_goalnav.py --goal_before_blackout` (goal frame shown at `t_vis-1`, blackout from `t_vis`,
   `--t_blind k`). Reuses the bumper controller.
2. T3 Re-entry: `scripts/eval_ckpt.py --reentry k` (one visible frame after k black frames; MSE vs copy-last).
3. `scripts/make_blackout_figs.py` for Fig. 1 and 2 from `results/homing`, `results/goalnav`, `results/runs`.
4. Seed top-up (s3, s4) for the arena and NYC core rows on the L4, together with the real-video stage.

Both parts run concurrently on the L4 (`xargs -P 2`); A needs about 6 GPU-hours, B about 10 with the seed top-up.

## 9. Related work to position against (all verified 2026-11-05)

Banino et al. 2018 (grid-cell LSTM path integration; `google-deepmind/grid-cells`), Cueva & Wei 2018 (RNN localisation
from velocity), Wijmans et al. 2023 (blind PointGoal / LoopNav agents, `erikwijmans/emergence-of-maps`; our T1 is their
LoopNav with a distilled world-model state in place of a PPO memory), Kim et al. 2017 (fly ring attractor), Vafidis et
al. 2022 (`panvaf/LearnPI`, one ring circuit learns heading integration by prediction), SEISMIC 2022
(`aplbrain/seismic`, hemibrain PEN synapses as a heading estimator), Lappalainen et al. 2024 (connectome-constrained
visual system). None scores a latent world model by closed-loop behaviour under blackout; none uses the whole central
complex subgraph; none has shuffled-wiring controls on behaviour.

## 10. Pilot, 2026-11-05 evening (Mac M1, old 2026-11-02 checkpoints, clean-trained, tested with turn noise 0.3, 50 episodes)

Arena, T1 Return, success [bootstrap 95% CI]; files in `results/pilot_blackout_mac/` (kept out of `results/homing/` so
the run-2 sweep on the L4 does not skip them as "done").

| row | k=8 | k=16 | k=24 | k=30 | k=40 |
|---|---|---|---|---|---|
| oracle pose (ceiling) | 0.32 | 0.84 | 1.00 | 1.00 | 1.00 |
| dead reckoning of actions | 0.28 | 0.74 | 0.90 | 0.92 | 0.92 |
| random actions (floor) | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| connectome (2 seeds) | 0.33 [0.24, 0.42] | 0.19 [0.11, 0.27] | 0.13 [0.07, 0.20] | 0.12 [0.06, 0.19] | 0.09 [0.04, 0.15] |
| dense (1 seed) | 0.20 | 0.20 | 0.18 | 0.22 | 0.18 |
| GRU, param.-matched (1 seed) | 0.10 | 0.08 | 0.10 | 0.06 | 0.10 |

What it says. (i) k = 8 is time-limited (oracle 0.32); the memory region is k >= 16. (ii) On the **arena** no student
wins: the connectome sits at 0.1-0.2 with dense, all far below dead reckoning, and the connectome does not separate
from dense. The arena is a 1-unit radius with 0.6-unit steps, so any decoded-pose error of a step or two is a miss.
(iii) The win we have is on the **NYC twin** (connectome 0.47 vs GRU 0.12 at k = 30, 12 m radius, 3 m steps). The hero
figure therefore has to be the NYC panel, and the arena panel will show a null unless the run-2 `_tn` checkpoints and
an MLP readout change it. Decision for the L4 sweep: run NYC first (all rows, all k, 3 seeds), arena second.
