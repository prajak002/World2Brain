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

## 2026-11-11 kaggle-small-a — LSTM baseline (#24) and circuit-structured objectives (#21), subcircuit students
- commit: git archive of `exp/memmaze` uploaded as Kaggle dataset `popsensharma/w2b-small-code` on 2026-11-10 (script
  `scripts/run_kaggle_small.sh` as of 88161d7; the exact archive hash was not recorded)
  box: Kaggle GPU kernel `popsensharma/w2b-small-batch-a` (version 2)          wall time: 8.6 h (exit 0, 36 trainings)
- command: `STAGES="data lr anchor lstm obj hd homing" bash scripts/run_kaggle_small.sh`
- prereg items: none (ablations, 3 seeds; not in `docs/PREREG.md`). Numbers below are mean +- sd over seeds 0-2 on the
  blackout window 8:16; the rule-4 statistics (bootstrap CI over test episodes, paired tests, Holm) are NOT computed
  yet, so no difference is claimed.

  | condition | blind MSE | blind heading err (deg) | homing success |
  |---|---|---|---|
  | connectome | 0.579 +- 0.001 | 8.5 +- 1.5 | 0.13 +- 0.04 |
  | connectome + dead reckoning (`_dr`) | 0.576 +- 0.003 | 8.8 +- 1.6 | 0.14 +- 0.11 |
  | connectome + contrastive (`_con`) | 0.578 +- 0.001 | 8.3 +- 1.2 | 0.09 +- 0.05 |
  | block_shuffled | 0.588 +- 0.004 | 10.5 +- 0.6 | 0.18 +- 0.04 |
  | block_shuffled `_dr` / `_con` | 0.586 / 0.589 | 11.0 / 10.2 | 0.10 / 0.13 |
  | GRU | 0.434 +- 0.002 | 30.6 +- 1.2 | 0.08 +- 0.00 |
  | GRU `_dr` / `_con` | 0.444 / 0.449 | 30.7 / 30.8 | 0.07 / 0.08 |
  | LSTM (lr 2e-3 chosen on validation) | 0.460 +- 0.006 | 30.5 +- 0.4 | 0.05 +- 0.01 |
  | LSTM turn-noise (`_tn`) | 0.450 +- 0.001 | 32.1 +- 1.2 | not run |

  Reading: the LSTM behaves like the GRU (lower latent MSE than the connectome, ~30 deg heading error). The two
  circuit-structured losses moved no metric outside the seed spread for any wiring: reported as a negative result.
- outputs: `results/kaggle_small_a/{runs,hd,homing}/*.json` (87 JSONs)          logs: `logs/kaggle_small_batch_a.log`
- failures / deviations from the runbook: the arena data were regenerated on Kaggle (run-2 commands, same seeds), so the
  anchor rows of this batch differ numerically from `results/runs/connectome_K1_s*.json`; the batch is therefore kept
  in its own directory with its own anchors and nothing in `results/runs/` was overwritten. `scripts/summarize.py` does
  not read this directory yet.

## 2026-11-11 kaggle-small-gnn — graph-network student (#23), subcircuit
- commit: 0c17b96 (Kaggle dataset `popsensharma/w2b-gnn-code`)
  box: Kaggle GPU kernel `popsensharma/w2b-small-batch-gnn`          wall time: 6.8 h (exit 0, 24 trainings)
- command: `STAGES="data gnnlr gnn anchor hd homing" TN_DATA=0 timeout 39000 bash scripts/run_kaggle_small.sh`
- prereg items: none (ablation, 3 seeds). Mean +- sd over seeds 0-2; rule-4 statistics NOT computed yet.

  | condition | params | blind MSE | blind heading err (deg) | homing success |
  |---|---|---|---|---|
  | FlyBrain connectome (anchor) | 3,714,529 | 0.579 +- 0.000 | 8.5 +- 1.5 | 0.11 +- 0.05 |
  | FlyBrain block_shuffled (anchor) | 3,714,529 | 0.588 +- 0.003 | 10.5 +- 0.8 | 0.13 +- 0.04 |
  | GRU (anchor) | 3,711,196 | 0.434 +- 0.002 | 30.6 +- 1.2 | 0.08 +- 0.00 |
  | GNN d=1, real graph (`_c1`) | 3,714,987 | 0.665 +- 0.110 | 5.7 +- 0.7 | 0.17 +- 0.07 |
  | GNN d=8, real graph (`_c8`) | 3,702,803 | 0.575 +- 0.004 | not probed | not run |
  | GNN d=8, block_shuffled (`_bs8`) | 3,702,803 | 0.536 +- 0.009 | not probed | not run |
  | GNN d=8, random (`_r8`) | 3,702,803 | 0.524 +- 0.003 | not probed | not run |

  Reading: with d=8 the rewired graphs have LOWER blind MSE than the real graph (0.524 / 0.536 vs 0.575), i.e. no
  real-wiring advantage in latent prediction for the GNN student. The d=1 GNN has the lowest heading error so far but
  the worst and least stable MSE (seed sd 0.11).
- outputs: `results/kaggle_small_gnn/{runs,hd,homing}/*.json` (60 JSONs)          logs: `logs/kaggle_small_batch_gnn.log`
- failures / deviations from the runbook: heading and homing probes exist only for `_c1`; the d=8 runs (`_c8`, `_bs8`,
  `_r8`) were not probed, so the GNN heading comparison across wirings is missing. Same regenerated-data note as above.
