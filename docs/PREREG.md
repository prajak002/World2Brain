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
(none yet)
