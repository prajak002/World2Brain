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
