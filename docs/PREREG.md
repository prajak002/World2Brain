# Pre-registration — full-brain distilled student on external and in-house benchmarks

Registered: 2026-11-10 (the git commit that adds this file is the timestamp). Frozen once any run in
`docs/RUNBOOK_GPU.md` batch B or C starts; later changes go in **Amendments** at the bottom, dated, with reasons.

## 1. Question
Does a recurrent student whose recurrence is the whole FlyWire v783 brain (139,248 neurons, 2,700,429 signed edges at
>= 5 synapses), distilled from a frozen video world model, form a better long-horizon memory than (a) the same student
with rewired graphs and (b) parameter-matched standard recurrent / attention students trained with the identical
objective and data — measured on a benchmark designed and published by others?

## 2. Benchmarks
**Primary (external): Memory Maze offline probing** — Pasukonis, Lillicrap, Hafner, *Evaluating Long-Term Memory in 3D
Mazes*, ICLR 2023 (arXiv 2210.13383, code github.com/jurgisp/memory-maze, MIT). Protocol used unchanged:
* data: offline datasets "Memory Maze 9x9 (30M)" and "15x15 (30M)": 30k trajectories x 1000 steps, 64x64 RGB, 4 Hz,
  scripted planner policy; 29k train / 1k eval trajectories as released.
* tasks: **9x9 Objects, 9x9 Walls** (primary), 15x15 Objects, 15x15 Walls (secondary, run only if 9x9 finishes).
* probe: frozen model state concatenated with `agent_pos` and `agent_dir`; MLP 4 x 1024, LayerNorm (eps 1e-3), ELU;
  Adam lr 3e-4, eps 1e-5; BCE on `maze_layout` (Walls), MSE on agent-centric `targets_vec` (Objects). No probe gradient
  into the model.
* metric: Walls accuracy (% of tiles), Objects MSE (mean over objects), averaged over steps 500–1000 of each of the
  1k eval trajectories.
* published reference rows (paper Table 4; 3 seeds): Constant 80.8 / 78.3 % / 23.9 / 64.8; VAE (no memory) 84.1 / 78.7 /
  17.2 / 56.2; RSSM 89.0 / 80.0 / 8.3 / 39.5; VAE+GRU (TBTT) 90.0 / 80.2 / 7.8 / 34.9; RSSM (TBTT) 94.0 / 80.5 / 6.2 /
  33.8; supervised oracle 99.7 / 88.5 / 0.3 / 9.2 (columns: 9x9 Walls, 15x15 Walls, 9x9 Objects, 15x15 Objects).

**Secondary (in-house, already in the paper):** E3 heading/position probes with the action-only floor, and the
Blackout Navigation benchmark (`docs/BENCHMARK_BLACKOUT.md`), rerun with the full-brain student. Reported as
secondary; they cannot rescue a failed primary result.

## 3. Conditions (identical data, objective, epochs, batch, optimiser; parameters matched within +-5 %)
| id | student | role |
|---|---|---|
| C | full-brain connectome, signed | the method |
| S | degree-preserving shuffle of C | wiring-identity control |
| B | block-shuffled C (cell-type blocks kept) | wiring-identity control |
| R | Erdos-Renyi, same N and edge count | sparsity control |
| G | GRU, hidden size matched to C's parameter count | standard recurrent baseline |
| T | causal transformer, matched parameters, context = TBTT window | attention baseline |
| N | probe on the teacher's per-frame latent only (no recurrence) | memory-free control (analogue of the paper's VAE row) |
| K | constant predictor (train mean) | metric sanity check; must reproduce 80.8 % / 23.9 within 1 point |

Training: next-step teacher-latent prediction from images + actions (the paper's objective), truncated BPTT over
windows of 48 steps carrying state across windows (as the benchmark's TBTT baselines), state reset only at trajectory
start. Hyper-parameters chosen on a validation split = last 1,000 of the 29k training trajectories; the 1k eval
trajectories are touched once per final configuration.

## 4. Hypotheses (directional, fixed now)
* **H1 (primary).** 9x9 Objects MSE: C < G.
* **H2.** 9x9 Objects MSE: C < S and C < B (wiring identity matters, not just sparsity/degree).
* **H3.** 9x9 Walls accuracy: C > G.
* **H4.** C beats N on both 9x9 tasks (the student adds memory beyond the teacher's frame features). If H4 fails, no
  memory claim is made for any condition.
* **H5 (placement).** C's 9x9 Objects MSE is below the published RSSM (TBTT) 6.2. Reported descriptively: our
  pipeline differs (teacher distillation, not pixel reconstruction), so this is context, not a controlled test.
* **H6 (secondary, in-house).** Full-brain C keeps the subcircuit's E3 heading advantage over G (blind heading error
  below G and below the action-only floor).

Known risk stated in advance: the probe receives true position and heading, so these tasks test map/object memory,
not path integration; the connectome's known advantage (path integration, E3) may not transfer. A failure of H1–H3 is
a publishable negative and is reported as such.

## 5. Statistics
* 5 seeds for C and G; 3 seeds for S, B, R, T, N (probe seeds: 1 per model seed).
* Unit of analysis: eval trajectory (score averaged over steps 500–1000), averaged over model seeds.
* Paired bootstrap (10,000 resamples of the 1k eval trajectories, seeds resampled within) for C minus each control;
  report mean difference and 95 % CI. Holm correction over the family {H1, H2 (two tests), H3, H4 (two tests)} at
  alpha = 0.05. "C better" requires the Holm-adjusted test to pass and the CI to exclude 0.
* Report every condition's mean +- std over seeds in the same format as the paper's Table 4.

## 6. Budget and stopping rules
* Training budget is fixed per condition before launch (same number of gradient steps for every condition; value set
  in `docs/RUNBOOK_GPU.md` B3 from the H200 throughput measured in B0, then frozen by amendment before B3 starts).
* No early stopping on eval; checkpoint selection only on the validation split (last checkpoint unless validation
  loss diverged, decided by a rule written in the amendment).
* If the deadline forces a cut, cut whole conditions or the 15x15 tasks, never seeds of a single condition, and say so.

## 7. What the paper may claim (decided now)
* H1 and H4 pass: "on Memory Maze offline probing, the distilled whole-brain student remembers objects better than a
  parameter-matched GRU trained identically".
* H2 passes: "and the real wiring beats degree- and block-preserving rewirings".
* Only H5 descriptive placement against published rows; never "state of the art".

## Amendments

### Amendment 1 — 2026-11-10, before any B3/B4 run (RUNBOOK B0)
Written after the B0 throughput measurements and the smoke tests on training data only (no eval trajectory has been used
by any model; the constant predictor K, which has no parameters to tune, was scored on eval once as the metric check).
Measurements: `results/memmaze/b0_bench.json`; smoke logs `logs/smoke_*.log` on the H200 box.

1. **Training pool.** The 9x9 release could only be partly downloaded: Google Drive's per-file quota blocks
   `train-part4`..`part9` (retried hourly). The pool is fixed now as **download parts 0-3 = 11,600 trajectories**:
   train = the first 10,600, validation = the last 1,000 (PREREG section 3 rule applied to this pool); eval = all
   1,000 released eval trajectories. Parts that arrive later are **not** added to any headline condition.
2. **Teacher input.** V-JEPA 2.1 ViT-B, causal 8-frame window ending at step t (frame 0 repeated before the start),
   each frame duplicated so one tubelet = one step (as `scripts/collect.py --causal 1`); 64x64 frames upsampled
   bilinearly to **128x128** (not the paper's 256: 1,370 vs 350 clips/s on the H200, i.e. ~2.5 h instead of ~10 h for
   the pool); **bf16 autocast**. Latent = mean token + 2x2 pooled tokens (3,840), PCA-256 fit on the 10,600 train
   trajectories only, whitened as `wm2cx.data.Latents`.
3. **Student input.** Frame mean-pooled to 16x16x3 (768 values, the same size as the paper's 32x8 fly eye), the
   previous action (6, one-hot), the visible flag. The student never sees agent_pos/agent_dir.
4. **Conditions as built** (`wm2cx/mm_models.py`): C/S/B/R = FlyBrain on the full brain (139,248 neurons,
   2,700,429 edges), 41,393,175 parameters; G = GRU hidden 3,308, 41,386,644 (-0.02 %); T = causal transformer width
   512, 13 layers, 41,543,424 (+0.36 %), context = 48 steps; N = probe on the teacher's PCA-256 latent at step t
   (this latent covers frames t-7..t, i.e. 2 s, the teacher's own window, not a single frame); K = train means.
5. **Optimisation (identical objective for every condition).** `scripts/train.py` loss unchanged (blind-step-weighted
   latent MSE under random 8-step blackouts in 70 % of windows + 1e-4 activity penalty); TBTT windows of 48 steps,
   state carried across the 1,001-step trajectory and reset only at its start; batch 32 trajectories; AdamW wd 1e-4,
   grad clip 1; learning rates = the repo's validated defaults (`scripts/run_l4.sh`): C/S/B/R/T 1e-3, G 2e-3; **linear
   warmup of 1,000 updates** then cosine decay, for every condition (added after the smoke test: without it the
   full-brain readout's loss jumps to 10-100x the GRU's in the first 100 updates). No further hyper-parameter search
   (deadline); validation is used only for the checkpoint rule.
6. **Budget.** **U = {U} parameter updates per condition** (one update per 48-step window; {U} updates =
   {EP} passes over the 10,600 train trajectories), equal for every condition. Set from the measured throughput:
   C alone 0.94 s/update (38.5 GB); {CONC}.
7. **Checkpoint rule.** Validation (250 val trajectories, latent MSE with one 8-step blackout per window) every 2,500
   updates; the run's checkpoint is the last one unless its validation blind MSE exceeds 1.5x the run's minimum, in
   which case the minimum-validation checkpoint is used.
8. **Probe budget.** Each probe pair (Walls, Objects; architecture and optimiser as section 2) is trained for
   {PP} pass(es) over the first {NP} train trajectories, all steps, one update per 32-trajectory x 48-step window
   (~{PU} updates), then scored once on the 1,000 eval trajectories. The probe input for C/S/B/R is the full
   139,248-dimensional state (+4); for G 3,308 (+4); for T 512 (+4); for N 256 (+4). Probe seed = model seed; N gets
   probe seeds 0-2; K is deterministic.
9. **B3 metric check, already done:** K on eval = 80.79 % Walls / 24.00 Objects MSE (paper 80.8 / 23.9), both within
   1 point; definitions used: Walls = mean tile accuracy over all 81 tiles, Objects = squared Euclidean error of each
   agent-centric object vector averaged over the 3 objects, both averaged over steps 500-1000.
10. **Order and cuts.** Wave order C s0-4 and G s0-4 first, then S, B, R s0-2, then T s0-2. If the deadline forces a
   cut, whole conditions are cut in reverse order (T, then R), never seeds of a condition (section 6).
