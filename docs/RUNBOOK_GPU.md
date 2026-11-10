# GPU runbook — batch queue for the CVPR push (deadline 2026-11-16)

Read `AGENTS.md` (rules) and `docs/PREREG.md` (what counts as a result) first. Batches run in the order below; each
has an **acceptance check** that must pass before the next batch starts. Log every batch in `docs/RESULTS_LOG.md`.

Boxes: the lab's AWS L4 (`ocr-rnd`, 24 GB, holds the causal episode data; stopped when idle) and a borrowed H200
(141 GB). All long jobs run inside `tmux` on the box; drive them from a laptop with `scripts/aws_l4_launch.sh`
(`L4_HOST`, `L4_USER`, `L4_KEY` select the box).

## 0. Access and data (once)

```bash
# 0.1 start the L4 from the AWS console (the laptop's AWS CLI token is invalid), then copy its data off:
export L4_HOST=63.183.17.217 L4_KEY=~/Downloads/ocr-rnd/ocr-rnd.pem
bash scripts/aws_l4_launch.sh pull-data          # causal arena/arrange, *_tn, nyc_twin_* -> data/box/, *_full runs -> results/runs/
# stop the L4 in the console afterwards

# 0.2 the full-brain graph (11 MB) is rebuilt from the FlyWire raw files if missing:
uv run python -m wm2cx.connectome --regions VIS CX MB OUT OTH --min_syn 5 --out data/connectome_full.npz

# 0.3 point the launcher at the H200 and push
export L4_HOST=<h200 host> L4_USER=<user> L4_KEY=<key>
bash scripts/aws_l4_launch.sh push-full
```
**Check:** `push-full` prints the GPU name/141 GB, and `data/` on the box lists `nav_train.npz`, `nav_tn_train.npz`,
`nyc_twin_train.npz`, `connectome_full.npz`. `uv run python -m pytest tests -q` passes on the box.

## A. Full-brain student on the in-house tasks (existing code; secondary results, H6)

```bash
JOBS=6 P=6 OMP=8 bash scripts/aws_l4_launch.sh run-full "arena tn nyc hd homing blackout"
bash scripts/aws_l4_launch.sh log-full           # progress; per-run logs are logs/train_<run>.log on the box
bash scripts/aws_l4_launch.sh pull && bash scripts/aws_l4_launch.sh pull-ckpt
```
* 3 seeds x {connectome, shuffled, block_shuffled, random, gru, transformer} x {arena, tn, nyc} = 54 trainings.
  L4 reference: one run ~45 min, ~13.6 GB. Start with `JOBS=6`; if `nvidia-smi` shows < 70 % memory used after the
  first wave, the next restart may use `JOBS=8`. A `FAILED <run>` line means read `logs/train_<run>.log`, fix, rerun
  (finished runs are skipped).
* Batch A can run **concurrently** with B1/B2 (download and encoding are I/O- and teacher-bound) if GPU memory allows.

**Check:** 54 `results/runs/*_full*.json` with `test_mse_per_ep`; `results/hd/*_full*.json` exist;
`uv run python scripts/summarize.py` runs; log H6 in `docs/RESULTS_LOG.md`.

## B. Memory Maze offline probing (primary, external; PREREG §2–§5)

Code to write (on branch `exp/memmaze`), each with a unit test in `tests/` on a 3-trajectory fixture:

| file | does |
|---|---|
| `wm2cx/memmaze.py` | load released trajectory NPZs (`image` (1001,64,64,3), `action` one-hot (6), `agent_pos`, `agent_dir`, `maze_layout`, `targets_vec`); the released 29k/1k split; validation = last 1k of train |
| `scripts/mm_encode.py` | teacher latents per step with the **causal** sliding window used by `scripts/collect.py --causal 1` (frames upsampled 64 -> teacher crop); fits PCA-256 on train only; writes sharded fp16 `data/mm9/lat_*.npy` + `pca.npz`; resumable per shard |
| `scripts/train_mm.py` (or `train.py --data mm`) | the existing students on (latent, action) streams; next-step latent prediction; TBTT windows of 48 carrying state, reset at trajectory start; `--wiring` as in `train.py`; same param matching |
| `scripts/eval_mm_probe.py` | run the frozen student over each whole 1000-step trajectory, collect state; concat `agent_pos`, `agent_dir`; train the PREREG probe (MLP 4x1024, LayerNorm eps 1e-3, ELU, Adam 3e-4 eps 1e-5; BCE walls / MSE objects) on train-trajectory states; score eval steps 500–1000; write per-trajectory scores JSON |
| `scripts/mm_stats.py` | PREREG §5 statistics (paired bootstrap, Holm) -> `results/memmaze/summary.md` + JSON |

* **B0 environment + throughput (1 h).** `uv add memory-maze dm_control mujoco` with pinned versions (repo last
  updated 2023; old `gym` API, so prefer the `dm_env` tasks); `MUJOCO_GL=egl`. Measure teacher clips/s and student
  steps/s on the H200. Write PREREG **Amendment 1**: number of training trajectories (largest that encodes in
  <= 24 h of H200 time, minimum 5,000, all 1,000 eval trajectories always) and the fixed gradient-step budget per
  condition. Commit the amendment before B2.
* **B1 download 9x9 (30M, ~100 GB).** Google Drive folder
  https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB (`gdown --folder` or `rclone`; Drive
  quotas can throttle: retry, do not switch datasets). Verify trajectory count and shapes; log checksums.
* **B2 encode** (`mm_encode.py`, background, resumable). Storage ~2 bytes x 256 x steps (15 GB for 30M).
* **B3 sanity before any real result:** condition K (constant) must reproduce the paper's 80.8 % (9x9 Walls) and
  23.9 (9x9 Objects) within 1 point; the N probe should land near the paper's no-memory VAE row (84.1 / 17.2). If
  K is off, the metric code is wrong: stop and fix. Also train C and G for 1 % of the budget on 100 trajectories
  and check the loss falls.
* **B4 train** C, G (5 seeds), S, B, R, T (3 seeds) at the frozen budget; `JOBS` as memory allows.
* **B5 probe + stats** for every checkpoint; `mm_stats.py`.

**Check:** `results/memmaze/summary.md` has a Table-4-style row per condition (mean +- std over seeds), the
published rows quoted beside it, and H1–H5 marked PASS / FAIL / INCONCLUSIVE with CIs. Logged.

## C. Extensions (only if B is done by 2026-11-14)
* 15x15 Objects/Walls (same pipeline, second 100 GB download).
* Graph density axis: `data/connectome_full_syn1.npz` (15.1 M edges at >= 1 synapse) for condition C only.

## D. Paper (2026-11-14 to 11-16)
`scripts/summarize.py` + `results/memmaze/summary.md` -> tables; numbers enter `paper_v2/` only through the result
macros; every claim checked against PREREG §7. The negative results stay in.

## Time plan
| date | work |
|---|---|
| 11-10/11 | §0 data off the L4, push to H200; batch A launched; B0 + Amendment 1; B1 download starts |
| 11-11/12 | B code + tests; B2 encoding; batch A finishes |
| 11-12/13 | B3 sanity, B4 training |
| 11-14 | B5 probes + stats; C if time |
| 11-15/16 | D paper |
