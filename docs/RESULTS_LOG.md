# Results log (append-only)

One entry per batch, newest last. Never edit an old entry; add a correction entry that points to it.

Template:

```
## <YYYY-MM-DD> <batch id> — <one line>
- commit: <git hash the run used>          box: <GPU, host alias>          wall time: <h>
- command: <exact command line, with env vars>
- prereg items: <H-ids from docs/PREREG.md>  -> PASS / FAIL / INCONCLUSIVE (with the number and its 95 % CI)
- outputs: results/<dir>/<files>          logs: logs/<file>
- failures / deviations from the runbook: <what and why; "none">
```

## 2026-11-10 setup — agent handoff docs, H200 launcher
- commit: see PR adding AGENTS.md / docs/PREREG.md / docs/RUNBOOK_GPU.md
- state: full-brain run connectome_K1_s0_full exists on the L4 (blind MSE 0.63 vs 0.58 subcircuit); the other
  full-brain runs and every external-benchmark run are still to do. The L4 box is stopped; its causal episode data
  have not yet been copied off.

## 2026-11-11 batch-A-train — whole-brain students on the in-house tasks: 36 trainings done, evaluations in progress
- commit: 0c17b96 for the trainings (`exp/memmaze`); probe fixes aa6699c, 3f6c3b6, 95c8aa0 on `exp/batchA-2026-11-11`
  box: Vast instance 60172952, 2x H200 NVL (ssh -p 9973)          wall time: 2.6 h training (06:51-09:26 UTC), 12 concurrent
- command: `/root/batchA.sh` = arena data regenerated with `scripts/collect.py --causal 1` (run-2 commands, seeds 0/1),
  then `CUDA_VISIBLE_DEVICES=0 JOBS=6 OMP=2 STAGES="arena" bash scripts/run_l4_full.sh` and the same with
  `CUDA_VISIBLE_DEVICES=1 ... STAGES="tn"`; statistics `PYTHONPATH=. python scripts/full_brain_stats.py`
- prereg items: H6 (secondary). Blind MSE, 3 seeds, 400 test episodes, paired bootstrap over episodes, Holm over 5
  comparisons per task (`results/full_brain_summary.md`):
  * arena: connectome 0.6236; degree-shuffled +0.0792 [+0.0704, +0.0881], block-shuffled +0.0692 [+0.0614, +0.0771],
    random +0.0564 [+0.0492, +0.0636], transformer +0.0472 [+0.0371, +0.0576] (all worse than the connectome, Holm
    p = 0.0005); GRU -0.2072 [-0.2260, -0.1886] (better than the connectome).
  * continuous turns: connectome 0.6179; the same ordering, differences +0.0793 / +0.0658 / +0.0563 / +0.0401 and
    GRU -0.2030, all Holm p = 0.0005.
  * prediction part of H6: real wiring beats every rewired whole-brain control and the parameter-matched transformer
    -> PASS; it does not beat the GRU -> reported as such. Heading / homing / blackout parts: NOT YET EVALUATED.
- outputs: `results/runs/*_full*.json` (36), `results/full_brain_summary.{md,json}`, `results/hd/*_full*.json` (partial)
  logs: on the box, `logs/batchA.log`, `logs/h200_full_{arena,tn,evals,evals2}.log`, `logs/train_*_full*.log`
- failures / deviations from the runbook:
  1. The causal arena data were regenerated on the box instead of copied from the L4 (which is stopped); same commands
     and seeds as the Kaggle batches. The NYC stage was skipped (no `nyc_twin_*.npz` on this box): 36 of the 54 runs.
  2. `scripts/eval_hd.py` could not run on the whole-brain state: the ridge normal equations are 139,248^2 (155 GB)
     and the first attempts were killed for memory. Fixed by solving the same ridge in the dual form
     (`wm2cx/probes.py`, tested against the primal form in `tests/test_probes.py`).
  3. INVALID RESULTS, kept and not used: the first GPU version of the dual solve (e2e47f1) accumulated the Gram matrix
     under TF32 (the pipeline exports `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`) and produced meaningless probes (heading
     error 77-90 deg, negative R^2) for six files, moved to `results/hd_invalid_tf32_2026-11-11/`. The corrected
     version (float64, TF32 off, 3f6c3b6) reproduces the CPU result for `block_shuffled_K1_s0_full` exactly
     (13.1 deg, position R^2 0.915). The evaluation stage was restarted at 13:05 UTC; finished JSONs are skipped.
  4. The MLP probe on all 139,248 neurons does not fit (heading error 61-66 deg, R^2 about 0) for block-shuffled
     seeds 0-2: a probe failure at this width, to be reported, not a property of the state (the ridge probe on the same
     states gives 13 deg).
- early heading numbers (ridge, 8 dark steps, arena; not the full table): block-shuffled s0 13.1 deg / R^2 0.92;
  GRU 32.3, 34.5 deg / 0.32, 0.30; transformer 40.6, 43.0 deg / 0.28, 0.29; connectome not probed yet.

## 2026-11-11 B1/B2/B4-start — Memory Maze pool re-downloaded and re-encoded on the new box; B4 training started
- commit: 0c17b96 (+ launcher fix 9a09bdc)          box: Vast 60172952, 2x H200          wall time: encode 1.2 h
- command: `/root/mm_pipeline.sh` (download of parts 0-3 from copies of the released zips, `scripts/mm_encode.py
  --stage encode --splits train_all` on both GPUs), then `scripts/run_mm_b4.sh`
- prereg items: Amendment 1 pool = parts 0-3 (11,600 trajectories) + 1,000 eval: 126/126 shards encoded; PCA-256 on the
  106 train shards explains 0.876 of the variance. H1-H7: no result yet.
- outputs (on the box): `data/mm9/{meta,stats,lat}`, `data/mm9/pca.npz`; raw latents in `/dev/shm/mm9raw` (lost on stop)
- failures / deviations from the runbook:
  1. The old H200 instance (60061059) with the first encoding is unreachable; everything was redone on this box.
     Google Drive's per-file quota blocked the released train zips for 5 h; they were fetched from copies of the same
     four files made in a signed-in Drive account. Trajectory counts verified (4 x 2,900, eval 1,000).
  2. `scripts/run_mm_b4.sh` called the projection stage without `--pool`, it asserted, and all 22 B4 runs exited at
     once on missing `lat` files (12:26-12:27 UTC; logs kept in `logs/b4_failed_2026-11-11/`). No model was trained
     and nothing was evaluated in that attempt. Fixed in 9a09bdc; relaunched 12:45 UTC.
  3. The waves run in parallel instead of in sequence: wave 1 (C, G) on GPU 1 with JOBS=3, waves 2-3 (S, B, R, T) on
     GPU 0 with JOBS=2, because one connectome-family run takes about 10 h at this concurrency. G seed 0 finished in the
     first attempt after the fix and is kept (checkpoint rule chose update 12,000).

## 2026-11-11 batch-A-heading — whole-brain heading / position probes complete (H6, secondary)
- commit: probes 95c8aa0 (`exp/batchA-2026-11-11`)          box: Vast 60172952, GPU 0          wall time: 13:05-18:00 UTC
- command: `CUDA_VISIBLE_DEVICES=0 P=2 OMP=4 STAGES="hd homing blackout" bash scripts/run_l4_full.sh`
- prereg items: H6 heading part. Ridge probe on all units, 8 dark steps, arena, mean +- sd over 3 seeds (no per-episode
  errors are stored by `eval_hd.py`, so no paired test; per-seed values given):
  * connectome 11.1 +- 1.8 deg (9.8, 13.1, 10.4), position R^2 0.92; block-shuffled 14.1 +- 0.9 (13.1, 14.7, 14.5);
    random 14.3 +- 0.5; degree-shuffled 14.6 +- 0.4; GRU 33.1 +- 1.2; transformer 41.7 +- 1.2.
  * 16 dark steps (8:24): connectome 20.4 +- 2.4 (17.8, 22.5, 20.8); rewired 27.2-28.4 (every seed 26.0-28.9);
    GRU 39.7; transformer 55.4. Continuous turns, 8 steps: 16.7 / 18.2-18.8 / 34.3 / 41.6.
  * -> the whole-brain connectome beats GRU and transformer on every seed (PASS for "sparse wiring holds heading");
    against the rewired brains it is ahead on the mean at 8 steps (seed 1 ties block-shuffled seed 0 at 13.1) and on
    every seed at 16 steps.
  * NEGATIVE / caution: with a random 3,302-unit subset (the GRU's width) the connectome reads 27.0 +- 0.4 deg, better
    than the GRU (33.1) but WORSE than the rewired brains' subsets (20.7-24.9). The full-state code is concentrated: CX
    alone (2,875 units) gives 10.3 deg in connectome seed 0 and 13.1 in block-shuffled seed 0 (actions enter CX).
  * The MLP probe on all 139,248 units does not fit for any sparse wiring (heading 50-66 deg, R^2 about 0).
- outputs: `results/hd/*_full*.json` (108)          logs: box `logs/h200_full_evals2.log`
- failures / deviations: (1) whole-brain homing with the linear readout failed for all 18 sparse runs:
  `eval_homing.py` builds the 139,248^2 normal equations on the GPU (72 GB) while Memory Maze trainings occupy it; only
  the GRU / transformer homing files exist (8). Needs the dual form and a free GPU; not rerun yet. (2) The blackout
  stage was not run (GPU occupied); the evaluation session was stopped at 18:26 UTC. (3) TO CHECK: `eval_homing.py`
  forms its Gram matrix on the GPU in float32 while the pipelines export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1; the
  circuit-scale homing numbers already in the paper may carry TF32 error in the readout. Not yet tested.

## 2026-11-11 B4 — Memory Maze training: stopped once in error, relaunched at the registered settings
- commit: PREREG Amendment 3 2d8e3a2, Amendment 4 e2fad64          box: Vast 60172952, both GPUs
- what happened: at 13:58 UTC I stopped C s0-2 and S s0-1 at updates 600-1,100 because their training loss was rising
  through warmup (C 1.38 -> 3.38; S up to 80.6), and registered a learning-rate selection (Amendment 3). The selection
  runs (C seed 0: validation blind MSE 0.972 / 0.956 / 0.951 / 0.934 at nominal 1e-4 / 2e-4 / 3e-4 / 5e-4; G seed 0:
  0.631 / 0.582 / 0.538 / 0.512 at 5e-4 / 1e-3 / 2e-3 / 4e-3; 1,500 updates) showed no benefit from a lower rate, and
  the batch-A logs show the same early loss transient recovering in the in-house whole-brain runs. Amendment 4
  rescinds Amendment 3: the stop was premature. Cost: about 5 h of both GPUs (3 h of it idle after the selection runs).
- command (relaunch 18:21 UTC, identical to Amendment 1): `/root/mm_g1.sh` (wave 1 on GPU 1, JOBS=3), `/root/mm_g0.sh`
  (waves 2-3 on GPU 0, JOBS=2), `/root/mm_side.sh` (labelled side conditions `C_s0_lr2e-4`, `G_s{0,1,2}_lr4e-3`)
- prereg items: none yet; no eval trajectory has been read by any trained model
- outputs: `results/memmaze/runs/{G_s0,G_s1}.json` (registered runs), `*_lrsel*.json` (selection runs)
  logs: box `logs/b4_lr1e-3_diverged_2026-11-11/`, `logs/lrsel_2026-11-11/`, `logs/b4_*.log`

## 2026-11-11 (20:15-21:00 UTC) — box cleanup; homing readout in float64 / dual form; whole-brain homing started

Box: Vast 60172952 (2x H200). Code: a55f182 plus the follow-up commit that carries this entry.

**Cleanup (disk was 95 % full, 5.9 GB free, with ~28 GB of Memory Maze checkpoints still to be written).** Removed on the box:
`/root/mm9dl/x_{eval,part0-3}` (41 GB, the Memory Maze episode downloads; re-downloadable, the four train parts are also on
the lab Drive), `data/mm9/stats` (14 GB, PCA sufficient statistics; `pca.npz` is fitted and kept), `/dev/shm/mm9raw`
(91 GB of RAM, the un-projected encoder latents; all 126 shards are projected in `data/mm9/lat`, metadata in `data/mm9/meta`),
and the `_resume.pt` restart files of the 13 Memory Maze runs whose result JSON exists (6.7 GB; `train_mm.py` exits on
the JSON, so they can no longer be read). No result JSON, `_u<update>.pt` checkpoint, log, projected latent or metadata
file was removed. Consequence: the PCA cannot be refitted without re-downloading and re-encoding (about 6 h); the
pre-registration fixes the PCA, so no registered stage needs it. Disk after: 36 % used; RAM 99 / 251 GB.

**Homing readout.** `eval_homing.py` built its ridge normal equations in float32 on the GPU, under TF32 when the pipeline
exports `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (the mode that broke the first GPU heading probe), and needed a
139,248^2 matrix at whole-brain size. It now calls `wm2cx.probes.ridge_weights`: float64, TF32 off, primal when
features <= samples, dual otherwise (states held on the CPU, Gram products on the GPU in blocks, in-place Cholesky solve).
Test: `tests/test_probes.py::test_ridge_weights_primal_and_dual_match_the_closed_form`.

**Check of the old numbers (one run).** `gru_K1_s0_full`, arena, same seeds: old float32/TF32 readout success 0.06, final
distance 3.240, path efficiency 0.264, drift 3.524; float64 readout 0.06, 3.263, 0.263, 3.551
(`results/homing_checks/gru_K1_s0_full_arena_state_f64check.json`). The primal float32 readout was therefore not broken the
way the dual heading probe was, on this 3,302-unit model. The circuit-scale checkpoints (6,935 units) are not on this box, so
the circuit homing numbers themselves have not been re-run; that check is still owed before the camera-ready.

**Whole-brain homing (running, tmux `hom`, GPU 0 shared with training, ~16 min per sparse run, 36 runs).** First result,
`connectome_K1_s0_full`: readout R2 0.80-0.91 on the fit set, success 0.06, final distance 2.23 (GRU s0: 0.06, 3.26). Two
failed attempts before it (GPU and then host memory at 58,800 fit samples; logs `logs/hom_f64.log`, `logs/hom_f64_b.log`).
No claim until all seeds and wirings are in. The Blackout stage at whole-brain size has not been started.

## 2026-11-12 batch-A-homing — whole-brain linear-readout homing complete (H6, secondary); Blackout stage started

- commit: readout a55f182 / 7a625a7; statistics `scripts/full_brain_homing_stats.py` (this commit)
  box: Vast 60172952, GPU 0 (shared with Memory Maze training)          wall time: 2026-11-11 20:40 - 2026-11-12 03:55 UTC
- command: `/root/hom2.sh` (tmux `hom`; `eval_homing.py --readout state`, 12 visible + 24 dark steps, 50 episodes,
  300 fit rollouts, one run at a time), then `PYTHONPATH=. python scripts/full_brain_homing_stats.py`
- prereg items: H6 homing part. 3 seeds, paired bootstrap over the 50 shared episodes (seeds averaged per episode), Holm
  over 5 comparisons per metric and task (`results/full_brain_homing_summary.md`). Start distance 3.89 (arena).
  * final distance to the start, arena: connectome 2.27; degree-shuffled 2.24, block-shuffled 2.09, random 2.26
    (all "no difference": -0.03 [-0.25, +0.19], -0.18 [-0.39, +0.03], -0.01 [-0.27, +0.25]); GRU 3.47
    (+1.20 [+0.85, +1.57]) and transformer 4.48 (+2.21 [+1.76, +2.68]) worse, Holm p = 0.0005.
  * decoded-position drift at the end of the blackout, arena: connectome 2.75; degree-shuffled 4.43
    (+1.67 [+1.25, +2.11]), block-shuffled 3.95 (+1.19 [+0.79, +1.60]), random 3.12 (+0.37 [+0.03, +0.71], Holm
    p = 0.035), GRU 3.58, transformer 4.31: all worse than the connectome.
  * success within radius 1.0 is low for every model: connectome 0.07 (0.06, 0.12, 0.04), rewired 0.11-0.19,
    GRU 0.05, transformer 0.01. Continuous turns: the same pattern (connectome 0.16, rewired 0.11-0.15; final distance
    no difference to any rewired brain; drift lowest for the connectome, all Holm p = 0.0005).
  * -> PASS for "sparse wiring homes better than GRU / transformer" (final distance and drift).
    NEGATIVE for real wiring against rewired wiring on behaviour: final distance is not different and success is not
    higher (arena: lower on the mean). Only the decoded-position drift favours the real wiring.
- outputs: `results/homing/*_full_arena_state.json`, `results/homing/*_full_tn_arena_state_tn.json` (36),
  `results/full_brain_homing_summary.{md,json}`          logs: box `logs/hom_f64_c.log`
- failures / deviations: none in this stage beyond the two memory failures already logged on 2026-11-11.

**Whole-brain Blackout stage (started 07:12 UTC, tmux `bk`, `/root/bk.sh`, GPU 1 next to Memory Maze wave 1).** The
variables of `run_l4_full.sh` line 80 with `ENVS=arena TASKS="t1 t3"`: 213 jobs (Return at k = 8, 16, 24, 30, 40 with
the policy and state readouts, and open-loop Re-entry). NOT RUN: T2 (goal navigation), because `eval_goalnav.py` still
builds the 139,248^2 primal normal equations for the state readout; and the NYC world, which has no whole-brain
checkpoints on this box. No result yet.

**Memory Maze B4 status (07:10 UTC).** Registered runs finished: C s0-2, G s0-4, S s0-1 (10 of 22); training: C s3,
C s4, S s2, B s0 (updates 1,000-1,700 of 12,000, about 7 h each); queued: B s1-2, R s0-2, T s0-2. Side conditions
`C_s0_lr2e-4` and `G_s{0,1,2}_lr4e-3` finished. Run JSONs pulled into `results/memmaze/runs/`; no probe, lesion or
statistics stage has been run, so there is no H1-H7 result.

**Paper.** `paper/cvpr-polish-2026-11-11` f3ed226: the whole-brain table and text carry the homing result, including
the negative against the rewired brains; body is 8 pages.

## 2026-11-12 (07:12-08:25 UTC) — whole-brain Blackout stage: state-readout jobs failed in the Cholesky solve; relaunched

- commit: code 7a625a7 (unchanged)          box: Vast 60172952, GPU 1          command: `/root/bk.sh` (tmux `bk`)
- what failed: all 9 whole-brain state-readout Return jobs that ran (k = 8, sparse wirings) stopped with
  `AssertionError: dposv info 15993` in `wm2cx.probes.ridge_weights` (dual form, 54,000 fit samples). The same pivot
  for every wiring and seed. The policy-readout jobs and the baselines were not affected (21 JSONs written).
- diagnosis: not the data and not the GPU. The float64 Gram accumulation on the GPU matches direct dot products to
  1e-11 at n = 54,000 and 58,800 (`/root/gram_check.py`). A synthetic positive-definite matrix of the same size fails in
  scipy's OpenBLAS (0.3.31.dev) `dposv` with the identical `info 15993` under `OMP_NUM_THREADS=4` and solves with the
  default 32 threads (`logs/chol_check_2026-11-12.log`). `run_blackout.sh` exports `OMP_NUM_THREADS=4`; last night's
  homing queue (`/root/hom2.sh`) did not set it, which is why the 24-step homing runs were solved correctly.
- fix: launcher only. `/root/bk.sh` now exports `OPENBLAS_NUM_THREADS=32`, which OpenBLAS reads ahead of
  `OMP_NUM_THREADS`. No evaluation setting changed; the failed jobs had written no JSON and are rerun by the skip rule.
  Stage stopped and relaunched 08:22 UTC (box clock). Failure logs kept in `logs/bk_failed_dposv_2026-11-12/` (box).
- still open: `run_l4_full.sh` exports `OPENBLAS_NUM_THREADS=$OMP` (default 4), so the same failure would hit any
  whole-brain dual solve launched through it at these sizes; the solver call needs a thread-independent path or a
  residual check before the next whole-brain evaluation is launched that way. Sizes 56,400 to 63,600 (k = 16 to 40)
  are untested with 32 threads; the `info` assertion stops a job that fails.

## 2026-11-13 batch-A-blackout — whole-brain Blackout stage complete (T1 Return, T3 Re-entry; in-house, secondary)

- commit: evaluation code 7a625a7 (unchanged); statistics `scripts/full_brain_blackout_stats.py` (this commit)
  box: Vast 60172952, GPU 1          wall time: 2026-11-12 08:22 - 16:59 UTC (after the relaunch logged above)
- command: `/root/bk.sh` (tmux `bk`; `run_blackout.sh` with `SFX=_full ENVS=arena TASKS="t1 t3"`, `OPENBLAS_NUM_THREADS=32`),
  then `PYTHONPATH=. python scripts/full_brain_blackout_stats.py` on the laptop. 213 of 213 jobs wrote a JSON; no
  `dposv` failure after the relaunch, including the untested sizes k = 16 to 40.
- statistics: 3 seeds x 6 wirings, 100 shared episodes (Return) / 200 test episodes (Re-entry), paired bootstrap over
  episodes with seeds averaged per episode, 10,000 resamples, Holm over the 5 comparisons of each metric, readout and k.
  Not a registered hypothesis family (H6 names only the heading comparison); reported as the secondary benchmark.
  * **Return, state readout, success** at k = 8 / 16 / 24 / 30 / 40: connectome 0.26 / 0.70 / 0.50 / 0.48 / 0.39;
    degree-shuffled 0.25 / 0.60 / 0.47 / 0.37 / 0.27; block-shuffled 0.27 / 0.59 / 0.46 / 0.35 / 0.28; random
    0.24 / 0.58 / 0.46 / 0.39 / 0.26; GRU 0.02 / 0.14 / 0.07 / 0.07 / 0.09; transformer 0.12 / 0.28 / 0.04 / 0.04 / 0.02.
    Ceiling (oracle pose) 0.24 / 0.94 / 1.00 / 1.00 / 1.00; dead reckoning of actions 0.26 / 0.83 / 0.91 / 0.91 / 0.91.
    Connectome against the three rewired brains: higher at k = 16, 30, 40 (differences 0.09 to 0.13, Holm p <= 0.01);
    "no difference" at k = 8 and k = 24. Against GRU and transformer: higher at every k (Holm p = 0.0005).
  * **Return, state readout, final distance**: connectome lower than every rewired brain at k >= 16 (0.21 to 0.77 units,
    Holm p <= 0.03), "no difference" at k = 8; lower than GRU and transformer at every k.
  * **Return, policy readout: NOT INFORMATIVE for the sparse whole-brain students.** The connectome runs give the
    identical result for all 3 seeds and for k = 8, 16, 24 (success 0.14, final distance 2.794), and the rewired brains
    mostly the same numbers: the fitted policy head emits the same actions whatever the state. The GRU with the policy
    readout reaches 0.25 / 0.54 / 0.50 / 0.51 / 0.48, i.e. at k >= 24 as good as or better than the connectome's state
    readout (0.50 / 0.48 / 0.39). So "the connectome beats the GRU on Return" holds for the state readout only; taking
    each model's better readout there is no connectome advantage over the GRU at k >= 24. This goes into the paper as is.
  * **Re-entry, blind latent MSE over 8 dark steps**: GRU 0.482 < connectome 0.771 < transformer 0.827 < random 0.840
    < block-shuffled 0.860 < degree-shuffled 0.867 < copy-last 1.032 < constant 1.301. Every difference to the connectome
    has a CI excluding 0 (Holm p = 0.0007). Same order as the whole-brain prediction result: real wiring beats rewired
    wiring and the transformer, and loses to the GRU.
- outputs: `results/homing/*_full_tn_arena_{state,policy}_k*_tn.json` (180) and the oracle / actions / random rows (15),
  `results/runs/*_full_tn_reentry.json` (18), `results/baselines/*_reentry.json`,
  `results/full_brain_blackout_summary.{md,json}`          logs: `logs/h200_full_blackout.log`, box `logs/bk_*.log`
- failures / deviations: T2 (goal navigation) and the NYC world NOT RUN at whole-brain size (reasons in the entry of
  2026-11-12). The policy-readout degeneracy above is not investigated further; no setting was changed and nothing rerun.
  `results/blackout_table.md` (circuit scale) is untouched.

## 2026-11-13 memmaze B4-B6 — Memory Maze 9x9 complete: training (22 runs), probes, lesions, statistics. H1 FAIL, H4 FAIL

- commit: code on the box 95c8aa0 (`train_mm.py`, `eval_mm_probe.py`, `eval_mm_lesion.py`, `mm_stats.py`,
  `mm_lesion_stats.py`); PREREG sections 2-5 with Amendments 1, 2, 4          box: Vast 60172952, both GPUs
- wall time: training 2026-11-11 18:21 - 2026-11-13 10:30 UTC (C / S / B / R 7.2-12.0 h per run, G 0.2-0.3 h, T 0.1 h);
  probes 2026-11-12 15:45 - 2026-11-13 11:05; lesions 2026-11-12 19:35 - 2026-11-13 11:44; statistics 11:16 and 11:45.
- command: `JOBS=.. scripts/run_mm_b4.sh` (tmux `mm0`, `mm1`, `mm1b`), `/root/mm_probe.sh`, `/root/mm_lesion.sh`,
  `PYTHONPATH=. python scripts/mm_stats.py`, `PYTHONPATH=. python scripts/mm_lesion_stats.py`
- runs: C 5, G 5, S 3, B 3, R 3, T 3 seeds, all 12,000 updates, checkpoint chosen on validation (u 12,000 for every run).
  Parameters: C / S / B / R 41,393,175 (recurrent 2,700,429, input 2,627,258, output 35,647,744), G 41,386,644,
  T 41,543,424 (within 0.4 %). Published constant baseline reproduced by K: 80.79 % / 24.01 (published 80.8 / 23.9).
- result (1,000 eval trajectories, steps 500-1000; mean +- sd over seeds), `results/memmaze/summary.md`:

  | condition | seeds | 9x9 Walls acc % | 9x9 Objects MSE |
  |---|---|---|---|
  | C connectome | 5 | 80.7 +- 0.1 | 20.13 +- 0.30 |
  | S degree-shuffled | 3 | 80.7 +- 0.1 | 22.35 +- 0.02 |
  | B block-shuffled | 3 | 80.7 +- 0.1 | 22.49 +- 0.02 |
  | R random | 3 | 80.7 +- 0.1 | 22.33 +- 0.09 |
  | G GRU | 5 | 80.6 +- 0.2 | 12.66 +- 0.11 |
  | T transformer | 3 | 80.9 +- 0.1 | 12.55 +- 0.09 |
  | N teacher frame features, no memory | 3 | 80.8 +- 0.1 | 12.74 +- 0.07 |
  | K constant | 1 | 80.8 | 24.01 |

- prereg items (paired bootstrap, 10,000 resamples, Holm over {H1, H2a, H2b, H3, H4a, H4b}; diff > 0 = C better):
  * **H1 (primary), Objects C < G: FAIL.** -7.47 [-7.79, -7.17]: the GRU is better.
  * **H2, Objects C < S and C < B: PASS.** +2.22 [1.94, 2.49] and +2.37 [2.08, 2.64], Holm p = 0.0000. Not interpretable
    as memory (H4), and all four sparse students are worse than the frame features.
  * **H3, Walls C > G: INCONCLUSIVE.** +0.11 [-0.04, 0.28]. Every condition is at the constant predictor (80.8 %).
  * **H4, C beats N on both tasks: FAIL.** Objects -7.39 [-7.70, -7.10]; Walls -0.09 [-0.22, 0.07]. By section 4,
    **no memory claim is made for any condition.** G (12.66) and T (12.55) are within 0.2 of N (12.74) as well: no
    student added memory beyond the teacher's frame features at this budget.
  * **H5 (descriptive):** C 20.13 against published RSSM (TBTT) 6.2: not below. Best of ours (T, 12.55) is between
    the published VAE without memory (17.2) and RSSM (8.3).
  * **H7 (lesions, Amendment 2): VOID**, because H4 failed (precondition 6). Reported descriptively
    (`results/memmaze/lesion_summary.md`): in C a CX lesion removes 0.60 [0.53, 0.67] of the above-constant Objects
    performance against 0.14 for matched random lesions (H7a estimate 0.455, H7b 0.491, both would pass), but
    H7c FAILS (-0.444 [-0.552, -0.338]: the CX-specific effect is larger in the degree-shuffled student, 0.90, than in C),
    and the CXdeaf deficit (0.73) is as large as the CX lesion deficit, so by precondition 6 the effect is attributed to
    lost action input, not to the circuit. Negative; no double dissociation, no wiring-specific lesion claim.
- outputs: `results/memmaze/runs/*.json` (22 registered + 4 side-condition + 8 lr-selection runs),
  `results/memmaze/probe/*.json` (26), `results/memmaze/lesion/*.json` (11), `results/memmaze/summary.{md,json}`,
  `results/memmaze/lesion_summary.{md,json}`          logs: `logs/mm_stats.log`, `logs/mm_lesion_stats.log`, box `logs/b4_*.log`
- failures / deviations: none in these stages. The side conditions (`C_s0_lr2e-4`, `G_s{0,1,2}_lr4e-3`) were trained but
  not probed; they are reported with their losses only (Amendment 4). Checkpoints and probe weights are on the box only.
